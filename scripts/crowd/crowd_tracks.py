"""Read the precomputed CROWD tracks written by the CROWD tracking code (crowd).

Each segment has one file ``<video_id>_<start_second>_<fps>.csv`` (or ``.parquet``)
in a ``bbox`` folder, where ``fps`` is the source video's frame rate truncated to an
integer. It has one row per tracked box: ``yolo-id``, ``x-center``, ``y-center``,
``width`` and ``height`` normalised to the frame, ``unique-id``, ``confidence``, and
``frame-count``, which counts the frames of the segment from 1. The segment starts at
``start_second`` and ends one second before the mapped end, as
``crowd_trim_end_margin_seconds`` cuts it here. The tracks were made with YOLO11x
(confidence 0.0, 640 px input) and BoT-SORT with ``configs/botsort.yaml`` and a two
second buffer.
"""

from __future__ import annotations

import csv
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.core.models import BoundingBox, TrackObservation

REQUIRED_COLUMNS = ("yolo-id", "x-center", "y-center", "width", "height", "unique-id", "frame-count")
TRACK_SUFFIXES = (".csv", ".parquet")
# CROWD numbers the first frame of a segment 1; TrackObservation frames start at 0.
FIRST_FRAME_COUNT = 1


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


def is_track_file_for(name: str, video_id: str, start_second: int) -> bool:
    """Whether a file name is the track file of this video and segment start."""

    if Path(name).suffix.lower() not in TRACK_SUFFIXES:
        return False
    try:
        info = parse_track_filename(name)
    except ValueError:
        return False
    return info.video_id == video_id and info.start_second == start_second


def find_local_track_file(
    video_id: str,
    start_second: int,
    folders: Iterable[Path],
    fps: float | None = None,
) -> Path | None:
    """The track file of one segment in the given ``bbox`` folders, or None.

    The frame rate in the name is not needed to find it; when several frame rates
    exist, the one matching ``fps`` (truncated, as CROWD names them) is preferred.
    """

    found: list[Path] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        pattern = f"{glob.escape(video_id)}_{start_second}_*"
        found += sorted(
            path
            for path in folder.glob(pattern)
            if path.is_file() and path.stat().st_size > 0 and is_track_file_for(path.name, video_id, start_second)
        )
    if not found:
        return None
    if fps is not None:
        for path in found:
            if int(parse_track_filename(path).fps) == int(fps):
                return path
    return found[0]


def _rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ImportError("Reading CROWD Parquet tracks needs pyarrow: uv add pyarrow") from error
        return parquet.read_table(path).to_pylist()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_crowd_tracks(
    path: str | Path,
    min_confidence: float = 0.0,
    max_frames: int | None = None,
) -> list[TrackObservation]:
    """Convert one CROWD track file to TrackObservation rows.

    The frame index is ``frame-count - 1``, the frame of the segment video cut by
    ``extract_video_segment``. ``max_frames`` drops rows past the end of that video.
    """

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
        frame_index = int(float(row["frame-count"])) - FIRST_FRAME_COUNT
        if frame_index < 0 or (max_frames is not None and frame_index >= max_frames):
            continue
        x, y = float(row["x-center"]), float(row["y-center"])
        width, height = float(row["width"]), float(row["height"])
        observations.append(
            TrackObservation(
                frame_index=frame_index,
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
