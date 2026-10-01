"""Audit every person the pipeline claims on JAAD against JAAD's own labels.

A claim is a person labelled JAYWALKING: a crossing with no zebra crossing and no
traffic light. Each claimed track is matched to a JAAD pedestrian by box overlap.
The claim is confirmed when that pedestrian has a JAAD crossing label and JAAD's
per frame scene annotations show no pedestrian crossing and no traffic light on its
crossing frames. Bystanders without behaviour labels and unannotated detections
cannot be verified and are reported separately, never as confirmed.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.core.config import ProjectConfig
from scripts.jaad.jaad import JAADDataset, JAADPedestrianTrack, JAADVideoAnnotations
from scripts.jaad.jaad_benchmark import box_iou
from scripts.core.models import BoundingBox, to_jsonable
from scripts.core.tracking import load_observations_csv
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger

CONFIRMED = "confirmed: crossing, no zebra, no traffic light"


def clopper_pearson_lower(successes: int, trials: int, confidence: float = 0.95) -> float:
    """One sided lower confidence bound for a binomial proportion."""

    if trials == 0:
        return 0.0
    if successes == trials:
        return (1.0 - confidence) ** (1.0 / trials)
    from scipy.stats import beta

    return float(beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def match_pedestrian(
    annotations: JAADVideoAnnotations,
    track_boxes: dict[int, BoundingBox],
    iou_threshold: float,
    minimum_frames: int,
) -> JAADPedestrianTrack | None:
    """Return the JAAD pedestrian that overlaps the track in the most frames."""

    best: tuple[int, JAADPedestrianTrack] | None = None
    for pedestrian in annotations.tracks.values():
        matched = sum(
            1
            for frame in pedestrian.frames
            if frame in track_boxes and box_iou(pedestrian.boxes[frame], track_boxes[frame]) >= iou_threshold
        )
        if matched >= minimum_frames and (best is None or matched > best[0]):
            best = (matched, pedestrian)
    return None if best is None else best[1]


def infrastructure_on_crossing(
    annotations: JAADVideoAnnotations,
    frames: list[int] | tuple[int, ...],
) -> tuple[bool, bool]:
    """Whether JAAD marks a pedestrian crossing or a traffic light on these frames."""

    zebra = any(annotations.traffic.get(frame, {}).get("ped_crossing") == "1" for frame in frames)
    light = any(
        annotations.traffic.get(frame, {}).get("traffic_light") not in ("n/a", "", None)
        for frame in frames
    )
    return zebra, light


def claim_verdict(
    annotations: JAADVideoAnnotations,
    track_boxes: dict[int, BoundingBox],
    event: dict[str, Any],
    iou_threshold: float,
    minimum_frames: int,
) -> str:
    pedestrian = match_pedestrian(annotations, track_boxes, iou_threshold, minimum_frames)
    if pedestrian is None:
        return "unverifiable: no JAAD pedestrian at this track"
    if not pedestrian.behaviour_annotated:
        return "unverifiable: JAAD bystander without behaviour label"
    if not pedestrian.is_crossing:
        return "wrong: JAAD pedestrian is not crossing"
    frames = [
        frame
        for frame in pedestrian.crossing_frames
        if event["start_frame"] <= frame <= event["end_frame"]
    ] or list(pedestrian.crossing_frames)
    zebra, light = infrastructure_on_crossing(annotations, frames)
    if zebra or light:
        return f"wrong: JAAD marks {'a zebra crossing' if zebra else 'a traffic light'}"
    return CONFIRMED


def eligible_crossers(annotations: JAADVideoAnnotations) -> list[JAADPedestrianTrack]:
    """JAAD crossers with no pedestrian crossing and no traffic light on their crossing frames."""

    eligible = []
    for pedestrian in annotations.behaviour_tracks:
        if pedestrian.is_crossing and not any(
            infrastructure_on_crossing(annotations, pedestrian.crossing_frames)
        ):
            eligible.append(pedestrian)
    return eligible


class JAADPersonAudit:
    """Run the configured pipeline over one saved JAAD split and audit its claims."""

    def __init__(self, config: ProjectConfig, split: str, exclude_locked_test: bool | None = None) -> None:
        # The project's locked test videos were never used for tuning, so they belong in
        # the held-out JAAD test audit but are kept out of the train and val audits.
        if exclude_locked_test is None:
            exclude_locked_test = split != "test"
        self.config = config
        self.split = split
        self.dataset = JAADDataset(config.path("jaad_root"))
        self.tracks_dir = config.path("jaad_benchmark_results") / split / "tracks"
        self.output = config.path("results") / f"jaad_person_audit_{split}"
        self.iou = float(config.get("jaad_match_iou"))
        self.minimum_frames = int(config.get("jaad_min_match_frames"))
        self.excluded = self._locked_test_videos() if exclude_locked_test else set()

    def run(self) -> dict[str, Any]:
        import cv2

        from scripts.core.pipeline import JaywalkingPipeline

        if not self.tracks_dir.is_dir():
            raise FileNotFoundError(
                f"No saved JAAD tracks: {self.tracks_dir}. Run scripts/jaad/run_jaad_crossing_benchmark.py "
                f"with jaad_benchmark_split set to {self.split} first."
            )
        (self.output / "details").mkdir(parents=True, exist_ok=True)
        (self.output / "tracking").mkdir(parents=True, exist_ok=True)
        videos = sorted(p.stem for p in self.tracks_dir.glob("*.csv") if p.stem not in self.excluded)
        pipeline = None
        for index, video_id in enumerate(videos, start=1):
            details_path = self.output / "details" / f"{video_id}.json"
            if details_path.exists():
                continue
            if pipeline is None:
                pipeline = JaywalkingPipeline(self.config)
            capture = cv2.VideoCapture(str(self.dataset.clip_path(video_id)))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            capture.release()
            source = self.tracks_dir / f"{video_id}.csv"
            shutil.copy(source, self.output / "tracking" / source.name)
            motion_path = self.tracks_dir.parent.parent / "camera_motion" / f"{video_id}.csv"
            camera_motion = None
            if motion_path.is_file():
                from scripts.crossing.camera_motion import load_camera_motion_csv

                camera_motion = load_camera_motion_csv(motion_path)
            result = pipeline.process_observations(
                self.dataset.clip_path(video_id),
                self.output / "evidence",
                fps,
                load_observations_csv(source),
                camera_motion=camera_motion,
            )
            details_path.write_text(
                json.dumps({"video_id": video_id, "result": to_jsonable(result)}, indent=1),
                encoding="utf-8",
            )
            claims = sum(item.label.value == "JAYWALKING" for item in result.person_decisions)
            logger.info("[{:03d}/{:03d}] {}: claims {}", index, len(videos), video_id, claims)
        return self.audit(videos)

    def audit(self, videos: list[str]) -> dict[str, Any]:
        verdicts: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        eligible = 0
        for video_id in videos:
            payload = json.loads((self.output / "details" / f"{video_id}.json").read_text(encoding="utf-8"))
            annotations = self.dataset.load_video(video_id)
            eligible += len(eligible_crossers(annotations))
            claims = [p for p in payload["result"]["person_decisions"] if p["label"] == "JAYWALKING"]
            if not claims:
                continue
            boxes: dict[int, dict[int, BoundingBox]] = defaultdict(dict)
            for observation in load_observations_csv(self.output / "tracking" / f"{video_id}.csv"):
                if observation.class_id == 0:
                    boxes[observation.track_id][observation.frame_index] = observation.box
            sources = {c["person_id"]: c["source"] for c in payload["result"].get("crossing_checks", [])}
            tiers = {
                c["person_id"]: c.get("gate_precision_tier")
                for c in payload["result"]["crossing_classifications"]
            }
            for claim in claims:
                verdict = claim_verdict(
                    annotations, boxes[claim["person_id"]], claim["event"], self.iou, self.minimum_frames
                )
                verdicts[verdict] += 1
                rows.append(
                    {
                        "video_id": video_id,
                        "person_id": claim["person_id"],
                        "gate_precision_tier": tiers.get(claim["person_id"]),
                        "source": sources.get(claim["person_id"], "gate"),
                        "verdict": verdict,
                    }
                )
        with (self.output / "claims.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("video_id", "person_id", "gate_precision_tier", "source", "verdict")
            )
            writer.writeheader()
            writer.writerows(rows)
        claims_total = sum(verdicts.values())
        confirmed = verdicts[CONFIRMED]
        summary = {
            "split": self.split,
            "videos": len(videos),
            "excluded_locked_test_videos": len(self.excluded),
            "claims": claims_total,
            "confirmed": confirmed,
            "precision_percent": 100.0 * confirmed / claims_total if claims_total else None,
            "precision_lower_95_percent": 100.0 * clopper_pearson_lower(confirmed, claims_total),
            "eligible_crossers": eligible,
            "recall_percent": 100.0 * confirmed / eligible if eligible else None,
            "verdicts": dict(verdicts),
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("{}", json.dumps(summary, indent=2))
        return summary

    def _locked_test_videos(self) -> set[str]:
        path = self.config.data_file("annotations")
        if not path.is_file():
            return set()
        with path.open("r", encoding="utf-8", newline="") as handle:
            return {row["video_id"] for row in csv.DictReader(handle) if row.get("split") == "locked_test"}
