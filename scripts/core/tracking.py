"""YOLO and BoT SORT object tracking."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from scripts.core.models import BoundingBox, TrackObservation


TRACK_CSV_FIELDS = (
    "frame_index",
    "track_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
)


def save_observations_csv(
    path: str | Path,
    observations: list[TrackObservation],
) -> Path:
    """Persist normalised tracker output for reproducible validation."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_CSV_FIELDS)
        writer.writeheader()
        for observation in observations:
            writer.writerow(
                {
                    "frame_index": observation.frame_index,
                    "track_id": observation.track_id,
                    "class_id": observation.class_id,
                    "confidence": observation.confidence,
                    "x1": observation.box.x1,
                    "y1": observation.box.y1,
                    "x2": observation.box.x2,
                    "y2": observation.box.y2,
                }
            )
    return destination


def load_observations_csv(path: str | Path) -> list[TrackObservation]:
    """Load tracker observations previously written by save_observations_csv."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {source}")
    observations: list[TrackObservation] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(TRACK_CSV_FIELDS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Tracking CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            observations.append(
                TrackObservation(
                    frame_index=int(row["frame_index"]),
                    track_id=int(row["track_id"]),
                    class_id=int(row["class_id"]),
                    confidence=float(row["confidence"]),
                    box=BoundingBox(
                        x1=float(row["x1"]),
                        y1=float(row["y1"]),
                        x2=float(row["x2"]),
                        y2=float(row["y2"]),
                    ),
                )
            )
    return observations


class PersonTracker:
    """Track people and contextual objects across one video at a time."""

    def __init__(self, settings: dict[str, Any], project_root: Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(settings["model"]))
        tracker_path = Path(str(settings["tracker"]))
        self.tracker_path = tracker_path if tracker_path.is_absolute() else (project_root / tracker_path).resolve()
        self.confidence = float(settings.get("confidence", 0.25))
        self.iou = float(settings.get("iou", 0.50))
        self.device = settings.get("device")

    def track(self, video_path: str | Path) -> tuple[float, list[TrackObservation]]:
        """Return video FPS and all tracked object observations.

        All object classes are retained because rider rejection and camera motion
        validation require vehicles and static reference objects.
        """

        import cv2

        source = Path(video_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Video not found: {source}")

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.release()

        options: dict[str, Any] = {
            "source": str(source),
            "tracker": str(self.tracker_path),
            "stream": True,
            "persist": False,
            "verbose": False,
            "conf": self.confidence,
            "iou": self.iou,
        }
        if self.device is not None:
            options["device"] = self.device

        observations: list[TrackObservation] = []
        results = self.model.track(**options)

        for frame_index, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or boxes.id is None or len(boxes) == 0:
                continue

            track_ids = boxes.id.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            normalised_boxes = boxes.xyxyn.detach().cpu().tolist()

            for track_id, class_id, confidence, coordinates in zip(
                track_ids,
                class_ids,
                confidences,
                normalised_boxes,
            ):
                x1, y1, x2, y2 = (float(value) for value in coordinates)
                observations.append(
                    TrackObservation(
                        frame_index=frame_index,
                        track_id=int(track_id),
                        class_id=int(class_id),
                        confidence=float(confidence),
                        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        return fps, observations
