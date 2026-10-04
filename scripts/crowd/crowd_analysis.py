"""Mapping driven CROWD download, inference, and manual audit sampling."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from scripts.core.config import ProjectConfig
from scripts.crowd.crowd_source import (
    CrowdSegment,
    CrowdVideoDownloader,
    PreparedVideo,
    ftp_credentials,
    extract_video_segment,
    find_local_source_video,
    load_crowd_mapping,
    probe_video,
)
from scripts.crowd.crowd_tracks import find_local_track_file, load_crowd_tracks
from scripts.core.models import DecisionLabel, to_jsonable
from scripts.core.pipeline import JaywalkingPipeline
from scripts.core.tracking import save_observations_csv
from scripts.core.vlm import prompt_version_for_mode
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger


SOURCE_FIELDS = (
    "video_key",
    "video_id",
    "start_second",
    "end_second",
    "effective_end_second",
    "time_of_day",
    "vehicle_type",
    "mapping_row_number",
    "continent",
    "country",
    "iso3",
    "city",
    "state",
    "filename",
    "video_path",
    "download_source",
)

VIDEO_FIELDS = SOURCE_FIELDS + (
    "prediction",
    "tracked_people",
    "predicted_crossings",
    "jaywalking_people",
    "uncertain_people",
    "latency_seconds",
)

PERSON_FIELDS = SOURCE_FIELDS + (
    "person_id",
    "classifier_probability",
    "classifier_threshold",
    "gate_probability",
    "gate_threshold",
    "gate_precision_tier",
    "predicted_crossing",
    "rule_outcome",
    "transition_start_frame",
    "transition_end_frame",
    "track_start_frame",
    "track_end_frame",
    "track_frames",
    "track_duration_seconds",
    "track_x_range",
    "track_road_frames",
    "track_start_state",
    "track_end_state",
    "track_complete_transition",
    "jaywalking_label",
    "decision_reason",
    "marked_crosswalk",
    "permissive_pedestrian_signal",
    "authorised_crossing_sign",
    "crossing_guard_permission",
    "prohibitive_pedestrian_signal",
    "visibility",
    "evidence_summary",
)

AUDIT_FIELDS = PERSON_FIELDS + (
    "audit_stratum",
    "human_crossing",
    "human_jaywalking",
    "reviewer_notes",
)

ERROR_FIELDS = (
    "video_key",
    "video_id",
    "start_second",
    "end_second",
    "stage",
    "error_type",
    "error",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _metadata_value(segment: dict[str, Any], name: str) -> str:
    metadata = segment.get("metadata", {})
    lowered = {str(key).lower(): value for key, value in metadata.items()}
    return str(lowered.get(name.lower(), ""))


def _source_row(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    segment = payload["segment"]
    video_path = result["video_path"]
    trim = payload.get("trim", {})
    return {
        "video_key": payload["video_key"],
        "video_id": segment["video_id"],
        "start_second": segment["start_second"],
        "end_second": segment["end_second"],
        "effective_end_second": trim.get(
            "effective_end_second", segment["end_second"]
        ),
        "time_of_day": segment["time_of_day"],
        "vehicle_type": segment.get("vehicle_type", ""),
        "mapping_row_number": segment["row_number"],
        "continent": _metadata_value(segment, "continent"),
        "country": _metadata_value(segment, "country"),
        "iso3": _metadata_value(segment, "iso3"),
        "city": _metadata_value(segment, "city"),
        "state": _metadata_value(segment, "state"),
        "filename": Path(video_path).name,
        "video_path": video_path,
        "download_source": payload.get("download_source", ""),
    }


class CrowdAnalysisRunner:
    """Run the frozen end to end method over mapped CROWD video segments."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.settings = config.crowd_settings()
        self.output = Path(self.settings["results"])
        self.details = self.output / "details"
        self.tracking = self.output / "tracking"
        self.evidence = self.output / "evidence"
        self.segments = self.output / "segments"
        self._downloader_instance: CrowdVideoDownloader | None = None

    def run(self) -> dict[str, Any]:
        mapped_segments = load_crowd_mapping(Path(self.settings["mapping"]))
        mapped_segments = self._select_segments(mapped_segments)
        if not mapped_segments:
            raise ValueError("The selected CROWD mapping contains no video segments.")

        self.output.mkdir(parents=True, exist_ok=True)
        self.details.mkdir(parents=True, exist_ok=True)
        self.tracking.mkdir(parents=True, exist_ok=True)
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.segments.mkdir(parents=True, exist_ok=True)
        self._write_manifest(mapped_segments)

        grouped: dict[str, list[CrowdSegment]] = {}
        for segment in mapped_segments:
            grouped.setdefault(segment.video_id, []).append(segment)

        failures: list[dict[str, Any]] = []
        pipeline: JaywalkingPipeline | None = None
        completed = 0
        total = len(mapped_segments)
        for video_id, group in grouped.items():
            pending = [
                segment
                for segment in group
                if not (
                    self.settings["resume"]
                    and (self.details / f"{segment.video_key}.json").is_file()
                )
            ]
            if not pending:
                for segment in group:
                    completed += 1
                    logger.info("[{:05d}/{:05d}] {} [saved]", completed, total, segment.video_key)
                continue

            try:
                prepared = self._prepare_source_video(video_id)
            except Exception as error:
                pending_keys = {segment.video_key for segment in pending}
                for segment in group:
                    completed += 1
                    if segment.video_key not in pending_keys:
                        logger.info("[{:05d}/{:05d}] {} [saved]", completed, total, segment.video_key)
                        continue
                    failures.append(self._failure(segment, "download", error))
                    logger.info(
                        "[{:05d}/{:05d}] {} ERROR {}: {}",
                        completed,
                        total,
                        segment.video_key,
                        type(error).__name__,
                        error,
                    )
                continue

            for segment in group:
                completed += 1
                detail_path = self.details / f"{segment.video_key}.json"
                if self.settings["resume"] and detail_path.is_file():
                    logger.info("[{:05d}/{:05d}] {} [saved]", completed, total, segment.video_key)
                    continue

                logger.info("[{:05d}/{:05d}] {}", completed, total, segment.video_key)
                track_file: Path | None = None
                track_source = "tracking"
                if self.settings["tracks_source"] == "precomputed":
                    try:
                        # The track file name carries the source video's frame rate.
                        source_fps = probe_video(prepared.path)[0]
                        track_file, track_source = self._precomputed_track_file(segment, source_fps)
                    except Exception as error:
                        failures.append(self._failure(segment, "tracks", error))
                        logger.info("  ERROR {}: {}", type(error).__name__, error)
                        continue
                try:
                    segment_video = extract_video_segment(
                        source=prepared.path,
                        destination=self.segments / f"{segment.video_key}.mp4",
                        start_second=segment.start_second,
                        end_second=segment.end_second,
                        end_margin_seconds=float(
                            self.settings["trim_end_margin_seconds"]
                        ),
                    )
                    if pipeline is None:
                        pipeline = JaywalkingPipeline(self.config)
                    started = time.perf_counter()
                    if track_file is None:
                        fps, observations = pipeline.tracker.track(segment_video.path)
                    else:
                        fps = segment_video.source_fps
                        observations = load_crowd_tracks(
                            track_file, max_frames=segment_video.output_frames
                        )
                    tracking_path = save_observations_csv(
                        self.tracking / f"{segment.video_key}.csv", observations
                    )
                    result = pipeline.process_observations(
                        segment_video.path,
                        self.evidence,
                        fps,
                        observations,
                        started=started,
                    )
                    result = replace(
                        result,
                        latency_seconds=round(time.perf_counter() - started, 2),
                    )
                    payload = {
                        "video_key": segment.video_key,
                        "segment": segment.as_dict(),
                        "source_video_filename": prepared.path.name,
                        "download_source": prepared.source,
                        "fps": fps,
                        "tracking_csv": str(tracking_path),
                        "tracks_source": track_source,
                        "precomputed_tracks": str(track_file) if track_file else None,
                        "crossing_method": pipeline.crossing_method,
                        "trim": {
                            "requested_start_second": segment_video.requested_start_second,
                            "requested_end_second": segment_video.requested_end_second,
                            "effective_end_second": segment_video.effective_end_second,
                            "source_fps": segment_video.source_fps,
                            "output_frames": segment_video.output_frames,
                        },
                        "result": to_jsonable(result),
                    }
                    self._write_json(detail_path, payload)
                    if not self.settings["keep_segment_videos"]:
                        segment_video.path.unlink(missing_ok=True)
                except Exception as error:
                    failures.append(self._failure(segment, "inference", error))
                    logger.info("  ERROR {}: {}", type(error).__name__, error)

            if (
                prepared.downloaded_this_run
                and self.settings["delete_downloaded_base_videos"]
            ):
                prepared.path.unlink(missing_ok=True)

        video_rows, person_rows = self._rebuild_tables()
        audit_rows = stratified_audit_sample(
            person_rows,
            per_stratum=int(self.settings["audit_per_stratum"]),
            seed=int(self.settings["audit_random_seed"]),
        )
        _write_csv(self.output / "per_video_results.csv", VIDEO_FIELDS, video_rows)
        _write_csv(self.output / "per_person_results.csv", PERSON_FIELDS, person_rows)
        _write_csv(self.output / "audit_sample.csv", AUDIT_FIELDS, audit_rows)
        _write_csv(self.output / "errors.csv", ERROR_FIELDS, failures)

        summary = self._summary(video_rows, person_rows, audit_rows, failures)
        self._write_json(self.output / "summary.json", summary)
        self._print_summary(summary)
        return summary

    def _select_segments(self, segments: list[CrowdSegment]) -> list[CrowdSegment]:
        selected_video = str(self.config.get("crowd_video_id") or "").strip()
        if selected_video:
            selected_video = selected_video.removesuffix(".mp4")
            segments = [
                item
                for item in segments
                if item.video_id.removesuffix(".mp4") == selected_video
            ]
        vehicle_types = self.settings["vehicle_types"]
        if vehicle_types is not None:
            # The method looks down the road the camera car drives on, so footage filmed from
            # a bicycle, bus or two-wheeler is left out: on a bicycle, pedestrian squares and
            # shared streets pass the crossing check as roads.
            if segments and all(item.vehicle_type is None for item in segments):
                raise ValueError(
                    "The mapping has no vehicle_type values; set crowd_vehicle_types to null to process every type."
                )
            kept = [item for item in segments if item.vehicle_type in vehicle_types]
            logger.info(
                "crowd_vehicle_types {}: kept {} of {} segments ({} without a vehicle type)",
                vehicle_types,
                len(kept),
                len(segments),
                sum(item.vehicle_type is None for item in segments),
            )
            segments = kept
        maximum = int(self.settings["max_segments"])
        return segments[:maximum] if maximum > 0 else segments

    def _prepare_source_video(self, video_id: str) -> PreparedVideo:
        local = find_local_source_video(video_id, self.config.paths("videos"))
        if local is not None:
            probe_video(local)
            return PreparedVideo(local, "local", False)

        display_name = (
            video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"
        )
        logger.info("Downloading CROWD source video: {}", display_name)
        prepared = self._downloader().download(
            video_id,
            Path(self.settings["download_dir"]),
        )
        try:
            probe_video(prepared.path)
        except Exception:
            if prepared.source != "local":
                prepared.path.unlink(missing_ok=True)
            raise
        return prepared

    def _downloader(self) -> CrowdVideoDownloader:
        if self._downloader_instance is None:
            secrets = ftp_credentials()
            self._downloader_instance = CrowdVideoDownloader(
                base_url=self.settings["ftp_server"],
                username=secrets.username,
                password=secrets.password,
                token=secrets.token,
                aliases=self.settings["ftp_aliases"],
                timeout_seconds=int(self.settings["download_timeout_seconds"]),
                max_pages=int(self.settings["download_max_pages"]),
            )
        return self._downloader_instance

    def _precomputed_track_file(self, segment: CrowdSegment, fps: float) -> tuple[Path, str]:
        """The CROWD track CSV of one segment and where it came from.

        The configured local ``bbox`` folders are searched first, then the download
        folder of earlier runs, then the ``bbox`` folder on the file server. A segment
        without precomputed tracks is an error, never silently tracked again.
        """

        local = find_local_track_file(
            segment.video_id, segment.start_second, self.settings["bbox_dirs"], fps
        )
        if local is not None:
            return local, "precomputed_local"
        cached = find_local_track_file(
            segment.video_id, segment.start_second, [self.settings["bbox_download_dir"]], fps
        )
        if cached is not None:
            return cached, "precomputed_cache"
        if self.settings["bbox_ftp_folder"]:
            downloaded = self._downloader().download_track_file(
                segment.video_id,
                segment.start_second,
                fps,
                self.settings["bbox_ftp_folder"],
                self.settings["bbox_ftp_aliases"],
                self.settings["bbox_download_dir"],
            )
            if downloaded is not None:
                return downloaded, "precomputed_ftp"
        raise FileNotFoundError(
            f"No precomputed CROWD tracks {segment.video_id}_{segment.start_second}_*.csv "
            "in crowd_bbox_dirs or on the file server"
        )

    @staticmethod
    def _failure(
        segment: CrowdSegment,
        stage: str,
        error: Exception,
    ) -> dict[str, Any]:
        return {
            "video_key": segment.video_key,
            "video_id": segment.video_id,
            "start_second": segment.start_second,
            "end_second": segment.end_second,
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        }

    def _write_manifest(self, segments: list[CrowdSegment]) -> None:
        classifier = self.config.crossing_classifier_settings()
        prompt_version = prompt_version_for_mode(self.config.vlm_settings()["prompt_mode"])
        model_path = Path(classifier["model"])
        mapping_path = Path(self.settings["mapping"])
        tracking_model = Path(str(self.config.get("tracking_model")))
        if not tracking_model.is_absolute():
            tracking_model = (self.config.root / tracking_model).resolve()
        tracker_path = Path(str(self.config.tracking_settings()["tracker"]))
        if not tracker_path.is_absolute():
            tracker_path = (self.config.root / tracker_path).resolve()
        server = urlparse(self.settings["ftp_server"])
        manifest = {
            "source_mode": "crowd_mapping_http_file_server",
            "source_video_count": len({item.video_id for item in segments}),
            "segment_count": len(segments),
            "mapping_file": str(mapping_path),
            "mapping_sha256": _sha256(mapping_path),
            "file_server_origin": f"{server.scheme}://{server.netloc}",
            "file_server_aliases": list(self.settings["ftp_aliases"]),
            "tracks_source": self.settings["tracks_source"],
            "vehicle_types": self.settings["vehicle_types"],
            "bbox_ftp_folder": self.settings["bbox_ftp_folder"],
            "trim_end_margin_seconds": self.settings["trim_end_margin_seconds"],
            "crossing_decision_mode": classifier["decision_mode"],
            "crossing_classifier_model": str(model_path),
            "crossing_classifier_sha256": (
                _sha256(model_path) if model_path.is_file() else None
            ),
            "crossing_classifier_min_track_frames": classifier["min_track_frames"],
            "tracking_model": self.config.get("tracking_model"),
            "tracking_model_sha256": (
                _sha256(tracking_model) if tracking_model.is_file() else None
            ),
            "tracker_config": str(tracker_path),
            "tracker_config_sha256": (
                _sha256(tracker_path) if tracker_path.is_file() else None
            ),
            "config_fingerprint": self.config.fingerprint(prompt_version),
            "prompt_version": prompt_version,
        }
        existing = self.output / "run_manifest.json"
        if existing.is_file() and self.settings["resume"]:
            previous = json.loads(existing.read_text(encoding="utf-8"))
            provenance_keys = (
                "source_mode",
                "mapping_sha256",
                "config_fingerprint",
                "crossing_classifier_sha256",
                "tracking_model_sha256",
                "tracker_config_sha256",
            )
            if any(previous.get(key) != manifest.get(key) for key in provenance_keys):
                raise RuntimeError(
                    "The CROWD results directory belongs to different source, model, "
                    "or configuration files. Use a new crowd_results directory."
                )
        self._write_json(existing, manifest)

    def _rebuild_tables(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        videos: list[dict[str, Any]] = []
        people: list[dict[str, Any]] = []
        for detail_path in sorted(self.details.glob("*.json")):
            payload = json.loads(detail_path.read_text(encoding="utf-8"))
            if "segment" not in payload:
                continue
            result = payload["result"]
            source = _source_row(payload, result)
            decisions = {item["person_id"]: item for item in result["person_decisions"]}
            classifications = result.get("crossing_classifications", [])
            videos.append(
                {
                    **source,
                    "prediction": result["prediction"],
                    "tracked_people": len(classifications),
                    "predicted_crossings": sum(
                        bool(item["predicted_crossing"]) for item in classifications
                    ),
                    "jaywalking_people": sum(
                        item["label"] == DecisionLabel.JAYWALKING.value
                        for item in result["person_decisions"]
                    ),
                    "uncertain_people": sum(
                        item["label"] == DecisionLabel.UNCERTAIN.value
                        for item in result["person_decisions"]
                    ),
                    "latency_seconds": result["latency_seconds"],
                }
            )
            for item in classifications:
                event = item["event"]
                features = item["track_features"]
                decision = decisions.get(item["person_id"])
                context = decision["context"] if decision else {}
                people.append(
                    {
                        **source,
                        "person_id": item["person_id"],
                        "classifier_probability": item["probability"],
                        "classifier_threshold": item["threshold"],
                        "gate_probability": item.get("gate_probability"),
                        "gate_threshold": item.get("gate_threshold"),
                        "gate_precision_tier": item.get("gate_precision_tier"),
                        "predicted_crossing": item["predicted_crossing"],
                        "rule_outcome": item["rule_outcome"],
                        "transition_start_frame": event["transition_start_frame"],
                        "transition_end_frame": event["transition_end_frame"],
                        "track_start_frame": event["start_frame"],
                        "track_end_frame": event["end_frame"],
                        "track_frames": features.get("matched_track_frames", ""),
                        "track_duration_seconds": features.get("matched_track_duration_seconds", ""),
                        "track_x_range": features.get("matched_track_x_range", ""),
                        "track_road_frames": features.get("matched_track_road_frames", ""),
                        "track_start_state": features.get("matched_track_start_state", ""),
                        "track_end_state": features.get("matched_track_end_state", ""),
                        "track_complete_transition": features.get("matched_track_complete_transition", ""),
                        "jaywalking_label": (
                            decision["label"] if decision else "NOT_EVALUATED"
                        ),
                        "decision_reason": decision["reason"] if decision else "",
                        "marked_crosswalk": context.get("marked_crosswalk", ""),
                        "permissive_pedestrian_signal": context.get(
                            "permissive_pedestrian_signal", ""
                        ),
                        "authorised_crossing_sign": context.get(
                            "authorised_crossing_sign", ""
                        ),
                        "crossing_guard_permission": context.get(
                            "crossing_guard_permission", ""
                        ),
                        "prohibitive_pedestrian_signal": context.get(
                            "prohibitive_pedestrian_signal", ""
                        ),
                        "visibility": context.get("visibility", ""),
                        "evidence_summary": context.get("evidence_summary", ""),
                    }
                )
        return videos, people

    @staticmethod
    def _summary(
        videos: list[dict[str, Any]],
        people: list[dict[str, Any]],
        audit: list[dict[str, Any]],
        failures: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "processed_source_videos": len({row["video_id"] for row in videos}),
            "processed_segments": len(videos),
            "failed_segments": len(failures),
            "tracked_people": len(people),
            "predicted_crossings": sum(
                bool(row["predicted_crossing"]) for row in people
            ),
            "jaywalking_people": sum(
                row["jaywalking_label"] == DecisionLabel.JAYWALKING.value
                for row in people
            ),
            "uncertain_people": sum(
                row["jaywalking_label"] == DecisionLabel.UNCERTAIN.value
                for row in people
            ),
            "audit_sample_rows": len(audit),
        }

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        logger.info("\nCROWD jaywalking analysis")
        logger.info("Processed source videos: {}", summary['processed_source_videos'])
        logger.info("Processed segments: {}", summary['processed_segments'])
        logger.info("Failed segments: {}", summary['failed_segments'])
        logger.info("Tracked people: {}", summary['tracked_people'])
        logger.info("Predicted crossings: {}", summary['predicted_crossings'])
        logger.info("Predicted jaywalking people: {}", summary['jaywalking_people'])
        logger.info("Manual audit rows: {}", summary['audit_sample_rows'])

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)


def stratified_audit_sample(
    rows: list[dict[str, Any]],
    per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select deterministic high confidence, boundary, and negative cases."""

    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    positives = [row for row in rows if bool(row["predicted_crossing"])]
    negatives = [row for row in rows if not bool(row["predicted_crossing"])]
    selected: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []

    def add(candidates: list[dict[str, Any]], name: str) -> None:
        for row in candidates:
            key = (str(row["video_key"]), str(row["person_id"]))
            if key in selected:
                continue
            selected.add(key)
            output.append(
                {
                    **row,
                    "audit_stratum": name,
                    "human_crossing": "",
                    "human_jaywalking": "",
                    "reviewer_notes": "",
                }
            )
            if sum(item["audit_stratum"] == name for item in output) >= per_stratum:
                break

    distance = lambda row: abs(
        float(row["classifier_probability"]) - float(row["classifier_threshold"])
    )
    add(sorted(positives, key=distance), "crossing_near_threshold")
    add(
        sorted(
            positives,
            key=lambda row: float(row["classifier_probability"]),
            reverse=True,
        ),
        "crossing_high_confidence",
    )
    add(sorted(negatives, key=distance), "noncrossing_near_threshold")
    random_negatives = list(negatives)
    random.Random(seed).shuffle(random_negatives)
    add(random_negatives, "noncrossing_random")
    return output
