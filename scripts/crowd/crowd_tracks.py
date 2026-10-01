"""Read the precomputed CROWD tracks written by the CROWD tracking code (crowd-city).

Each segment has one file ``<video_id>_<start_second>_<fps>.csv`` (or ``.parquet``)
with one row per tracked box: ``yolo-id``, ``x-center``, ``y-center``, ``width`` and
``height`` normalised to the frame, ``unique-id``, ``confidence``, and ``frame-count``
counted from the segment start. The tracks were made with YOLO11x (confidence 0.0,
640 px input) and BoT-SORT with ``configs/botsort.yaml`` and a two second buffer, the
settings the crossing classifier and gate must be validated with before use.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from scripts.core.models import BoundingBox, TrackObservation

REQUIRED_COLUMNS = ("yolo-id", "x-center", "y-center", "width", "height", "unique-id", "frame-count")


@dataclass(frozen=True)
class CrowdTrackFile:
    """One segment's precomputed tracks and where the segment sits in its source video."""

    path: Path
    video_id: str
    start_second: int
    fps: float


def parse_track_filename(path: str | Path) -> CrowdTrackFile:
    source = Path(path)
    try:
        video_id, start_text, fps_text = source.stem.rsplit("_", 2)
        return CrowdTrackFile(source, video_id, int(start_text), float(fps_text))
    except ValueError as error:
        raise ValueError(
            f"CROWD track file name must be <video_id>_<start_second>_<fps>: {source.name}"
        ) from error


def _rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ImportError("Reading CROWD Parquet tracks needs pyarrow: uv add pyarrow") from error
        return parquet.read_table(path).to_pylist()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_crowd_tracks(path: str | Path, min_confidence: float = 0.0) -> list[TrackObservation]:
    """Convert one CROWD track file to TrackObservation rows, frame-count as frame index."""

    source = Path(path)
    rows = _rows(source)
    if rows:
        missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
        if missing:
            raise ValueError(f"CROWD track file {source.name} lacks columns: {missing}")
    observations: list[TrackObservation] = []
    for row in rows:
        track_id = row["unique-id"]
        if track_id in (None, "", "nan"):
            continue
        confidence = float(row.get("confidence") or 1.0)
        if confidence < min_confidence:
            continue
        x, y = float(row["x-center"]), float(row["y-center"])
        width, height = float(row["width"]), float(row["height"])
        observations.append(
            TrackObservation(
                frame_index=int(float(row["frame-count"])),
                track_id=int(float(track_id)),
                class_id=int(float(row["yolo-id"])),
                confidence=confidence,
                box=BoundingBox(
                    x1=max(0.0, x - width / 2),
                    y1=max(0.0, y - height / 2),
                    x2=min(1.0, x + width / 2),
                    y2=min(1.0, y + height / 2),
                ),
            )
        )
    observations.sort(key=lambda item: (item.frame_index, item.track_id))
    return observations
