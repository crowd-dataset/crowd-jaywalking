"""Stored SegFormer surface labels of crowd-city's road_crossing rule, one entry per CROWD segment.

Segmenting a segment is the slow part of stage 1, and the labels depend only on the video, the
tracks and the segmentation settings, so they are kept and reused. The layout follows crowd-city's
``utils/segmentation/store.py``: ``index/<stem>.parquet`` (one row per track and sampled frame) and
``manifest/<stem>.json`` (the settings and the track ids the index covers). ``<stem>`` is the name
of the segment's CROWD track CSV without the extension: ``<video_id>_<start_second>_<fps>``, the fps truncated
to an integer as CROWD does.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from scripts.crossing.crowd_city_surface import SurfaceSample
from scripts.crossing.crowd_city_surface_constants import CODE_TO_SURFACE

INDEX_SCHEMA = "crowd_surface_index_v4"
PRODUCER = "crowd-jaywalking"
INDEX_COLUMNS = {
    "unique-id": pl.Utf8,
    "frame-count": pl.Int64,
    "video_time_s": pl.Float64,
    "surface": pl.Utf8,
    "surface_code": pl.Int8,
    "confidence": pl.Float32,
}
_SEGMENT_NAME = re.compile(r"^crowd_(?P<video>.+)_s(?P<start>\d+)_e\d+$")


def segment_stem(video_path: Any, fps: float) -> str | None:
    """``<video_id>_<start_second>_<fps>`` for a CROWD segment video, None for any other clip."""

    match = _SEGMENT_NAME.match(Path(str(video_path)).stem)
    if match is None:
        return None
    return f"{match['video']}_{int(match['start'])}_{int(float(fps))}"


def track_signature(track: pl.DataFrame) -> str:
    """Identity of a track's boxes; labels are reused only for the same boxes."""

    frames = track.get_column("frame-count").cast(pl.Int64, strict=False)
    digest = hashlib.sha256()
    for part in (frames.to_list(), track.get_column("x-center").round(4).to_list(),
                 track.get_column("y-center").round(4).to_list()):
        digest.update(json.dumps(part).encode("utf-8"))
    return digest.hexdigest()[:16]


class SurfaceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_dir = self.root / "index"
        self.manifest_dir = self.root / "manifest"

    def _read(self, stem: str, settings: Mapping[str, Any]) -> tuple[dict[str, str], pl.DataFrame] | None:
        manifest_path = self.manifest_dir / f"{stem}.json"
        index_path = self.index_dir / f"{stem}.parquet"
        if not manifest_path.is_file() or not index_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frame = pl.read_parquet(index_path)
        except (OSError, ValueError, pl.exceptions.PolarsError):
            return None
        if manifest.get("schema") != INDEX_SCHEMA or manifest.get("settings") != dict(settings):
            return None
        return dict(manifest.get("track_signatures") or {}), frame

    def load(
        self, stem: str, settings: Mapping[str, Any], signatures: Mapping[str, str]
    ) -> dict[str, list[SurfaceSample]] | None:
        """The stored timelines of the tracks, or None when absent, stale or not covering all of them."""

        stored = self._read(stem, settings)
        if stored is None:
            return None
        known, frame = stored
        if any(known.get(track_id) != signature for track_id, signature in signatures.items()):
            return None
        timelines: dict[str, list[SurfaceSample]] = {}
        for row in frame.iter_rows(named=True):
            if row["unique-id"] in signatures:
                timelines.setdefault(row["unique-id"], []).append(
                    SurfaceSample(
                        row["unique-id"], int(row["frame-count"]), float(row["video_time_s"]),
                        CODE_TO_SURFACE[int(row["surface_code"])], float(row["confidence"]),
                    )
                )
        for samples in timelines.values():
            samples.sort(key=lambda item: item.frame)
        return timelines

    def save(
        self, stem: str, settings: Mapping[str, Any], signatures: Mapping[str, str],
        timelines: Mapping[str, Sequence[SurfaceSample]],
    ) -> None:
        """Store the timelines, adding to what earlier runs stored for the same settings."""

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        merged_signatures: dict[str, str] = {}
        frames = []
        previous = self._read(stem, settings)
        if previous is not None:
            known, old = previous
            merged_signatures.update(known)
            frames.append(old.filter(~pl.col("unique-id").is_in(list(signatures))))
        merged_signatures.update(signatures)
        rows = [sample.as_row() for samples in timelines.values() for sample in samples]
        frames.append(pl.DataFrame(rows, schema=INDEX_COLUMNS) if rows else pl.DataFrame(schema=INDEX_COLUMNS))
        frame = pl.concat(frames)
        index_path = self.index_dir / f"{stem}.parquet"
        temporary = index_path.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(index_path)
        manifest = {
            "schema": INDEX_SCHEMA,
            "producer": PRODUCER,
            "settings": dict(settings),
            "track_ids": sorted(merged_signatures),
            "track_signatures": merged_signatures,
        }
        manifest_path = self.manifest_dir / f"{stem}.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(manifest_path)
