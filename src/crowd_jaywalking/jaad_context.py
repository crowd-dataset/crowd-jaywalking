"""Prepare ground truth crossing evidence for manual VLM context annotation."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any

from .config import ProjectConfig
from .evidence import make_evidence_views
from .jaad import JAADDataset, JAADPedestrianTrack, JAADVideoAnnotations


MANUAL_CONTEXT_FIELDS = (
    "marked_crosswalk",
    "permissive_pedestrian_signal",
    "authorised_crossing_sign",
    "crossing_guard_permission",
    "prohibitive_pedestrian_signal",
    "visibility",
    "annotator",
    "notes",
)

CONTEXT_AUDIT_FIELDS = (
    "video_id",
    "filename",
    "jaad_pedestrian_id",
    "split",
    "crossing_start_frame",
    "crossing_end_frame",
    "jaad_designated",
    "jaad_signalized",
    "jaad_ped_crossing",
    "jaad_ped_sign",
    "jaad_traffic_light",
    "sampling_stratum",
    *MANUAL_CONTEXT_FIELDS,
    "evidence_directory",
)


def context_sampling_stratum(metadata: dict[str, str]) -> str:
    """Assign an annotation sampling stratum from JAAD metadata only."""

    sign = str(metadata.get("jaad_ped_sign", "")).strip().upper()
    signalised = str(metadata.get("jaad_signalized", "")).strip().upper()
    traffic_lights = {
        item.strip().upper()
        for item in str(metadata.get("jaad_traffic_light", "")).split("|")
        if item.strip()
    }
    crossing = str(metadata.get("jaad_ped_crossing", "")).strip().upper()
    designated = str(metadata.get("jaad_designated", "")).strip().upper()

    if sign in {"1", "YES", "TRUE"}:
        return "pedestrian_sign_metadata"
    if signalised in {"1", "YES", "TRUE", "S"} or traffic_lights.difference(
        {"N/A", "NA", "NONE", "0", ""}
    ):
        return "signal_metadata"
    if crossing in {"1", "YES", "TRUE"} or designated in {
        "1",
        "YES",
        "TRUE",
        "D",
    }:
        return "crosswalk_metadata"
    return "no_reported_permission_metadata"


def select_context_candidates(
    candidates: list[tuple[JAADVideoAnnotations, JAADPedestrianTrack, dict[str, str]]],
    *,
    sample_size: int,
    seed: int,
    required_keys: set[tuple[str, str]] | None = None,
) -> list[tuple[JAADVideoAnnotations, JAADPedestrianTrack, dict[str, str]]]:
    """Select a reproducible round robin sample across metadata strata."""

    ordered = sorted(
        candidates,
        key=lambda item: (item[0].video_id, item[1].pedestrian_id),
    )
    if sample_size == 0 or sample_size >= len(ordered):
        return ordered

    required = required_keys or set()
    selected = [
        item
        for item in ordered
        if (item[0].video_id, item[1].pedestrian_id) in required
    ]
    selected_keys = {
        (item[0].video_id, item[1].pedestrian_id) for item in selected
    }
    if len(selected) >= sample_size:
        return selected

    groups: dict[
        str,
        list[tuple[JAADVideoAnnotations, JAADPedestrianTrack, dict[str, str]]],
    ] = defaultdict(list)
    for item in ordered:
        key = (item[0].video_id, item[1].pedestrian_id)
        if key not in selected_keys:
            groups[context_sampling_stratum(item[2])].append(item)

    generator = random.Random(seed)
    for group in groups.values():
        generator.shuffle(group)

    strata = sorted(groups)
    while len(selected) < sample_size and any(groups.values()):
        for stratum in strata:
            if groups[stratum] and len(selected) < sample_size:
                selected.append(groups[stratum].pop())

    return sorted(
        selected,
        key=lambda item: (item[0].video_id, item[1].pedestrian_id),
    )


class JAADContextAuditBuilder:
    """Create person focused evidence and a non-destructive annotation template."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.dataset = JAADDataset(config.path("jaad_root"))
        self.split = str(config.get("jaad_context_split")).strip().lower()
        self.output_dir = config.path("jaad_context_results") / self.split
        self.evidence_dir = self.output_dir / "evidence"
        self.annotations_csv = self.output_dir / "context_annotations.csv"
        self.sampling_manifest = self.output_dir / "sampling_manifest.json"
        sampling = config.jaad_context_settings()
        self.sample_size = int(sampling["sample_size"])
        self.sampling_seed = int(sampling["sampling_seed"])
        evidence = config.evidence_settings()
        self.sample_positions = [float(value) for value in evidence["sample_positions"]]
        self.context_seconds = float(evidence["context_seconds"])
        self.crop_margin = float(evidence["crop_margin"])
        self.max_dimension = int(evidence["max_dimension"])
        self.jpeg_quality = int(evidence["jpeg_quality"])
        self.trajectory_enabled = bool(evidence["trajectory_enabled"])
        self.road_crop_margin = float(evidence["road_crop_margin"])
        self.control_crop_bottom = float(evidence["control_crop_bottom"])
        self.control_crop_overlap = float(evidence["control_crop_overlap"])

    def run(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        existing = self._existing_rows()
        candidates: list[
            tuple[JAADVideoAnnotations, JAADPedestrianTrack, dict[str, str]]
        ] = []
        video_ids = self.dataset.video_ids(self.split)

        for index, video_id in enumerate(video_ids, start=1):
            print(f"Scanning [{index:03d}/{len(video_ids):03d}] {video_id}")
            annotations = self.dataset.load_video(video_id)
            crossing_tracks = [
                track for track in annotations.behaviour_tracks if track.is_crossing
            ]
            if not crossing_tracks:
                continue
            for track in crossing_tracks:
                candidates.append(
                    (annotations, track, self._context_metadata(annotations, track))
                )

        annotated_keys = {
            key
            for key, row in existing.items()
            if any(str(row.get(field, "")).strip() for field in MANUAL_CONTEXT_FIELDS)
        }
        selected = select_context_candidates(
            candidates,
            sample_size=self.sample_size,
            seed=self.sampling_seed,
            required_keys=annotated_keys,
        )
        rows: list[dict[str, Any]] = []
        for index, (annotations, track, metadata) in enumerate(selected, start=1):
            video_path = self.dataset.clip_path(annotations.video_id)
            print(
                f"Building [{index:03d}/{len(selected):03d}] "
                f"{annotations.video_id} {track.pedestrian_id}"
            )
            evidence_directory = self._build_evidence(video_path, annotations, track)
            key = (annotations.video_id, track.pedestrian_id)
            row = self._row(annotations, track, evidence_directory, metadata)
            for field in MANUAL_CONTEXT_FIELDS:
                row[field] = existing.get(key, {}).get(field, "")
            rows.append(row)

        with self.annotations_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CONTEXT_AUDIT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self._write_sampling_manifest(candidates, selected)
        print(f"Crossing events prepared: {len(rows)}")
        print(f"Saved: {self.annotations_csv}")
        return self.annotations_csv

    def _existing_rows(self) -> dict[tuple[str, str], dict[str, str]]:
        if not self.annotations_csv.is_file():
            return {}
        with self.annotations_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            return {
                (str(row.get("video_id", "")), str(row.get("jaad_pedestrian_id", ""))): row
                for row in csv.DictReader(handle)
            }

    def _row(
        self,
        annotations: JAADVideoAnnotations,
        track: JAADPedestrianTrack,
        evidence_directory: Path,
        metadata: dict[str, str],
    ) -> dict[str, Any]:
        crossing_frames = track.crossing_frames
        return {
            "video_id": annotations.video_id,
            "filename": f"{annotations.video_id}.mp4",
            "jaad_pedestrian_id": track.pedestrian_id,
            "split": self.split,
            "crossing_start_frame": crossing_frames[0],
            "crossing_end_frame": crossing_frames[-1],
            **metadata,
            "sampling_stratum": context_sampling_stratum(metadata),
            **{field: "" for field in MANUAL_CONTEXT_FIELDS},
            "evidence_directory": str(evidence_directory.relative_to(self.output_dir)),
        }

    @staticmethod
    def _context_metadata(
        annotations: JAADVideoAnnotations,
        track: JAADPedestrianTrack,
    ) -> dict[str, str]:
        crossing_frames = track.crossing_frames
        traffic = [
            annotations.traffic[frame]
            for frame in crossing_frames
            if frame in annotations.traffic
        ]

        def any_one(name: str) -> str:
            values = {item.get(name, "") for item in traffic}
            if "1" in values:
                return "1"
            if "0" in values:
                return "0"
            return ""

        traffic_lights = sorted(
            {
                item.get("traffic_light", "")
                for item in traffic
                if item.get("traffic_light", "")
            }
        )
        return {
            "jaad_designated": track.attributes.get("designated", ""),
            "jaad_signalized": track.attributes.get("signalized", ""),
            "jaad_ped_crossing": any_one("ped_crossing"),
            "jaad_ped_sign": any_one("ped_sign"),
            "jaad_traffic_light": "|".join(traffic_lights),
        }

    def _write_sampling_manifest(
        self,
        candidates: list[
            tuple[JAADVideoAnnotations, JAADPedestrianTrack, dict[str, str]]
        ],
        selected: list[
            tuple[JAADVideoAnnotations, JAADPedestrianTrack, dict[str, str]]
        ],
    ) -> None:
        population_counts = Counter(
            context_sampling_stratum(item[2]) for item in candidates
        )
        selected_counts = Counter(
            context_sampling_stratum(item[2]) for item in selected
        )
        payload = {
            "split": self.split,
            "sampling_method": "round_robin_across_JAAD_metadata_strata",
            "metadata_is_not_ground_truth": True,
            "sampling_seed": self.sampling_seed,
            "requested_sample_size": self.sample_size,
            "population_events": len(candidates),
            "selected_events": len(selected),
            "population_by_stratum": dict(sorted(population_counts.items())),
            "selected_by_stratum": dict(sorted(selected_counts.items())),
            "selected_keys": [
                {
                    "video_id": annotations.video_id,
                    "jaad_pedestrian_id": track.pedestrian_id,
                }
                for annotations, track, _ in selected
            ],
        }
        self.sampling_manifest.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _build_evidence(
        self,
        video_path: Path,
        annotations: JAADVideoAnnotations,
        track: JAADPedestrianTrack,
    ) -> Path:
        import cv2

        crossing_frames = track.crossing_frames
        event_directory = self.evidence_dir / annotations.video_id / track.pedestrian_id
        event_directory.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open JAAD video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        context_frames = max(0, int(round(self.context_seconds * fps)))
        evidence_start = max(track.frames[0], crossing_frames[0] - context_frames)
        evidence_end = min(track.frames[-1], crossing_frames[-1] + context_frames)
        candidate_frames = [
            frame for frame in track.frames if evidence_start <= frame <= evidence_end
        ]

        try:
            for position in self.sample_positions:
                requested = int(round(evidence_start + position * (evidence_end - evidence_start)))
                frame_index = min(candidate_frames, key=lambda frame: abs(frame - requested))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
                self._save_pair(image, track, frame_index, event_directory)
            self._save_target_preview(
                capture,
                track,
                evidence_start,
                evidence_end,
                fps,
                event_directory,
            )
        finally:
            capture.release()
        return event_directory

    def _save_target_preview(
        self,
        capture,
        track: JAADPedestrianTrack,
        start_frame: int,
        end_frame: int,
        fps: float,
        event_directory: Path,
    ) -> None:
        """Save a short review clip using the independent JAAD target boxes."""

        from fractions import Fraction

        import av
        import cv2

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not determine JAAD video dimensions")

        preview_path = event_directory / "target_preview.mp4"
        codec = next(
            (
                name
                for name in ("libx264", "h264")
                if self._video_encoder_available(av, name)
            ),
            None,
        )
        if codec is None:
            raise RuntimeError(
                "No H.264 encoder is available for browser compatible target previews"
            )
        container = av.open(str(preview_path), mode="w")
        stream = container.add_stream(
            codec,
            rate=Fraction(str(fps)).limit_denominator(1000),
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            stream.options = {"crf": "23", "preset": "fast"}

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        try:
            for frame_index in range(start_frame, end_frame + 1):
                ok, image = capture.read()
                if not ok or image is None:
                    raise RuntimeError(
                        f"Could not decode frame {frame_index} while creating {preview_path}"
                    )
                box = track.boxes.get(frame_index)
                if box is not None:
                    x1 = max(0, min(width - 1, int(round(box.x1 * width))))
                    y1 = max(0, min(height - 1, int(round(box.y1 * height))))
                    x2 = max(x1 + 1, min(width, int(round(box.x2 * width))))
                    y2 = max(y1 + 1, min(height, int(round(box.y2 * height))))
                    self._draw_target(
                        image,
                        x1,
                        y1,
                        x2,
                        y2,
                        track.pedestrian_id,
                    )
                video_frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
        finally:
            for packet in stream.encode():
                container.mux(packet)
            container.close()

    @staticmethod
    def _video_encoder_available(av_module, codec: str) -> bool:
        try:
            av_module.CodecContext.create(codec, "w")
            return True
        except Exception:
            return False

    def _save_pair(
        self,
        image,
        track: JAADPedestrianTrack,
        frame_index: int,
        event_directory: Path,
    ) -> None:
        import cv2

        box = track.boxes[frame_index]
        trajectory_boxes = [track.boxes[item] for item in track.crossing_frames]
        views = make_evidence_views(
            image,
            box,
            trajectory_boxes,
            track.pedestrian_id,
            crop_margin=self.crop_margin,
            road_crop_margin=self.road_crop_margin,
            control_crop_bottom=self.control_crop_bottom,
            control_crop_overlap=self.control_crop_overlap,
            maximum_dimension=self.max_dimension,
            trajectory_enabled=self.trajectory_enabled,
        )

        parameters = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        for name, view in views.items():
            path = event_directory / f"frame_{frame_index:06d}_{name}.jpg"
            if not cv2.imwrite(str(path), view, parameters):
                raise RuntimeError(f"Could not save evidence image: {path}")

    @staticmethod
    def _draw_target(
        image,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        pedestrian_id: str,
    ) -> None:
        import cv2

        colour = (0, 0, 255)
        thickness = max(2, int(round(max(image.shape[:2]) / 400)))
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
        cv2.putText(
            image,
            f"TARGET {pedestrian_id}",
            (max(0, x1), max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )

    def _resize(self, image):
        import cv2

        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.max_dimension:
            return image
        scale = self.max_dimension / longest
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
