"""Stored SegFormer surface labels of crowd-city's road_crossing rule, one entry per CROWD segment.

Segmenting a segment is the slow part of stage 1, and a track's labels depend only on the video,
the track's own boxes and the segmentation settings, so they are kept and reused. The store is
crowd-city's (``utils/segmentation/store.py``, schema v5) and is shared with it: a segment either
project has segmented is not segmented again by the other. The layout is ``index/<stem>.parquet``
(one row per track and sampled frame) and ``manifest/<stem>.json`` (the settings, the track ids
and a signature of each track's boxes). ``<stem>`` is the name of the segment's CROWD detection
file without the extension, ``<video_id>_<start_second>_<fps>``, the fps rounded to an integer as
CROWD names its files (29.97 is 30).

Keep this in step with crowd-city: INDEX_SCHEMA, the settings keys (``store_settings``),
``track_signature`` and ``canonical_track_id`` must match exactly, and ``video_time_s`` is seconds
from the start of the whole video, not of the segment.
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

INDEX_SCHEMA = "crowd_surface_index_v5"
PRODUCER = "crowd-jaywalking"
INDEX_COLUMNS = {
    "unique-id": pl.Utf8,
    "frame-count": pl.Int64,
    "video_time_s": pl.Float64,
    "surface": pl.Utf8,
    "surface_code": pl.Int8,
    "confidence": pl.Float32,
}
# Manifest keys this module owns; any other key (video id, start, frame rates) is a fact about the
# segment and is kept from the previous entry unless a save states it again.
_OWN_KEYS = {"schema", "settings", "track_ids", "track_signatures", "producer"}
_SEGMENT_NAME = re.compile(r"^crowd_(?P<video>.+)_s(?P<start>\d+)_e\d+$")


def segment_stem(video_path: Any, fps: float) -> str | None:
    """``<video_id>_<start_second>_<fps>`` for a CROWD segment video, None for any other clip.

    Prefer the detection file's own name when it is known; this rebuilds it the way CROWD names
    them, with the frame rate rounded (29.97 -> 30).
    """

    match = _SEGMENT_NAME.match(Path(str(video_path)).stem)
    if match is None:
        return None
    return f"{match['video']}_{int(match['start'])}_{int(round(float(fps)))}"


def store_settings(
    model: str, input_width: int, input_height: int, coarse_hz: float, refine_hz: float,
    footpoint_band_fraction: float, footpoint_width_fraction: float, minimum_confidence: float,
) -> dict[str, Any]:
    """Everything that changes the meaning of a stored label, keyed as crowd-city's SegmentationSettings."""

    return {
        "model_identifier": f"{model}@{int(input_width)}x{int(input_height)}",
        "coarse_hz": float(coarse_hz),
        "refine_hz": float(refine_hz),
        "footpoint_band_fraction": float(footpoint_band_fraction),
        "footpoint_width_fraction": float(footpoint_width_fraction),
        "minimum_confidence": float(minimum_confidence),
    }


def canonical_track_id(value: Any) -> str:
    """The form a track id is stored under: 2118, 2118.0 and "2118" are one id."""

    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def track_signature(track: pl.DataFrame) -> str:
    """Identity of a track's boxes; labels are reused only for the same boxes."""

    ordered = track.sort("frame-count")
    digest = hashlib.sha256()
    for part in (
        ordered.get_column("frame-count").cast(pl.Int64, strict=False).to_list(),
        ordered.get_column("x-center").cast(pl.Float64, strict=False).round(4).to_list(),
        ordered.get_column("y-center").cast(pl.Float64, strict=False).round(4).to_list(),
    ):
        digest.update(json.dumps(part).encode("utf-8"))
    return digest.hexdigest()[:16]


class SurfaceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_dir = self.root / "index"
        self.manifest_dir = self.root / "manifest"

    def _manifest(self, stem: str) -> dict[str, Any]:
        path = self.manifest_dir / f"{stem}.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _index(self, stem: str) -> pl.DataFrame | None:
        path = self.index_dir / f"{stem}.parquet"
        if not path.is_file():
            return None
        try:
            return pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError):
            return None

    def load(
        self, stem: str, settings: Mapping[str, Any], signatures: Mapping[str, str]
    ) -> dict[str, list[SurfaceSample]]:
        """The stored timelines of the requested tracks the store covers, keyed by canonical id.

        A track is covered when the entry has the same settings and the same signature for it.
        Tracks that are not covered are absent from the result; an empty result means none is.
        """

        wanted = {canonical_track_id(key): value for key, value in signatures.items()}
        manifest = self._manifest(stem)
        if manifest.get("schema") != INDEX_SCHEMA or manifest.get("settings") != dict(settings):
            return {}
        stored = manifest.get("track_signatures") or {}
        covered = {track_id for track_id, signature in wanted.items() if stored.get(track_id) == signature}
        frame = self._index(stem) if covered else None
        if frame is None:
            return {}
        timelines: dict[str, list[SurfaceSample]] = {}
        for row in frame.iter_rows(named=True):
            track_id = canonical_track_id(row["unique-id"])
            if track_id in covered:
                timelines.setdefault(track_id, []).append(
                    SurfaceSample(
                        track_id, int(row["frame-count"]), float(row["video_time_s"]),
                        CODE_TO_SURFACE[int(row["surface_code"])], float(row["confidence"]),
                    )
                )
        for samples in timelines.values():
            samples.sort(key=lambda item: item.frame)
        return timelines

    def save(
        self, stem: str, settings: Mapping[str, Any], signatures: Mapping[str, str],
        timelines: Mapping[str, Sequence[SurfaceSample]], extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Store the timelines of the tracks in ``signatures``, keeping the entry's other tracks.

        Tracks of an existing entry under the same settings that this call does not mention keep
        their rows; a track it mentions is replaced. An entry in another schema or under other
        settings is replaced whole.
        """

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        tracks = {canonical_track_id(key): value for key, value in signatures.items()}
        previous = self._manifest(stem)
        kept: dict[str, str] = {}
        frames = []
        if previous.get("schema") == INDEX_SCHEMA and previous.get("settings") == dict(settings):
            kept = {
                track_id: signature
                for track_id, signature in (previous.get("track_signatures") or {}).items()
                if track_id not in tracks
            }
            old = self._index(stem)
            if old is not None and kept:
                old = old.with_columns(
                    pl.col("unique-id").map_elements(canonical_track_id, return_dtype=pl.Utf8)
                )
                frames.append(old.filter(pl.col("unique-id").is_in(list(kept))).select(list(INDEX_COLUMNS)))
        rows = []
        for samples in timelines.values():
            for sample in samples:
                row = sample.as_row()
                row["unique-id"] = canonical_track_id(row["unique-id"])
                rows.append(row)
        frames.append(pl.DataFrame(rows, schema=INDEX_COLUMNS) if rows else pl.DataFrame(schema=INDEX_COLUMNS))
        frame = pl.concat([item.cast(INDEX_COLUMNS) for item in frames])
        index_path = self.index_dir / f"{stem}.parquet"
        temporary = index_path.with_suffix(".parquet.tmp")
        frame.write_parquet(temporary, compression="zstd")
        temporary.replace(index_path)

        merged = {**kept, **tracks}
        manifest = {key: value for key, value in previous.items() if key not in _OWN_KEYS}
        manifest.update(extra or {})
        manifest.update({
            "schema": INDEX_SCHEMA,
            "producer": PRODUCER,
            "settings": dict(settings),
            "track_ids": sorted(merged),
            "track_signatures": dict(sorted(merged.items())),
        })
        manifest_path = self.manifest_dir / f"{stem}.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(manifest_path)
