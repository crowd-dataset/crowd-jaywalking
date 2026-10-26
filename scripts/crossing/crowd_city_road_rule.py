"""crowd-city's ``road_crossing`` decision applied to one clip's tracks (stage 1).

crowd-city counts a pedestrian as crossing only when the pedestrian walks across the road
in front of the camera. Its default rule (``crossing_rule: road_crossing``) works in these
steps, all copied from crowd-city at commit 340875e and run here in the same order:

1. Broken tracks are joined (``crowd_city_track_joining``).
2. Box candidates: a person track that passes in front of the camera or emerges there,
   moves sideways enough, and changes box size slowly (``box_only_candidates``), then is
   not a rider and does not just move with the camera (``independent_motion_rejection``).
3. SegFormer reads the surface under the feet at sampled frames (1 Hz, 4 Hz around the
   moments the track enters and leaves the road). The feet must be on the road and the
   walking speed plausible (``road_crossing_flags``).
4. The picture itself must not show the camera turning the person across the image
   (``swept_by_turning_camera``) and an unverifiable fast mover is dropped
   (``unverifiable_crossing``).

For a CROWD segment the frames are read exactly as crowd-city reads them: ffmpeg decodes the
segment's original video (the same file crowd-city reads from the CROWD file server) at times
on the whole video's clock, with crowd-city's own filter chain (``SourceFrames``). The labels
then go into the surface store shared with crowd-city (``crowd_city_surface_store``), so a
segment segmented by either project is not segmented again by the other. Any other clip (JAAD,
a cut segment with no source given) is read with OpenCV and its labels are never stored. The
sampling grid, windows and padding follow ``utils/segmentation/pipeline.py``.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from custom_logger import CustomLogger
from scripts.core.models import CrossingClassification, TrackObservation
from scripts.crossing import crowd_city_road_crossing as rule
from scripts.crossing import crowd_city_track_metrics as metrics
from scripts.crossing.crowd_city_camera_shift import (
    CAMERA_SHIFT_HZ,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    background_shift,
)
from scripts.crossing.crowd_city_crossing import (
    CROSSING_PARAMETER_DEFAULTS,
    MIN_CONFIDENCE,
    PERSON_CLASS,
    STRIP_LEFT,
    STRIP_RIGHT,
    CrowdCityCrossingDetector,
    observations_frame,
)
from scripts.crossing.crowd_city_surface import (
    DEFAULT_FOOTPOINT_BAND_FRACTION,
    DEFAULT_FOOTPOINT_WIDTH_FRACTION,
    SurfaceSample,
    sample_surface,
    surface_timeline,
    transition_brackets,
)
from scripts.crossing.crowd_city_surface_store import (
    SurfaceStore,
    canonical_track_id,
    store_settings,
    track_signature,
)
from scripts.crossing.model_crossing import ModelCrossingDetectionResult

logger = CustomLogger(__name__)

# utils/segmentation/pipeline.py
WINDOW_MERGE_GAP_SECONDS = 2.0
WINDOW_PADDING_SECONDS = 2.0
TRACK_SPAN_GAP_SECONDS = 2.0
MAXIMUM_WINDOW_SECONDS = 120.0
BRACKET_PADDING_SECONDS = 0.25


@dataclass(frozen=True)
class FrameWindow:
    start_seconds: float
    duration_seconds: float


class ClipFrames:
    """Frames of one local clip, read with OpenCV and resized like crowd-city's ffmpeg call."""

    def __init__(self, path: Any) -> None:
        import cv2

        self._cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open the video: {path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._position = -1

    def close(self) -> None:
        self.capture.release()

    def frame(self, index: int, width: int, height: int) -> np.ndarray | None:
        """RGB frame ``index`` scaled to ``width`` x ``height``, or None past the end."""

        if index < 0 or (self.frame_count > 0 and index >= self.frame_count):
            return None
        if index != self._position + 1:
            self.capture.set(self._cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self.capture.read()
        if not ok:
            self._position = -1
            return None
        self._position = index
        image = self._cv2.resize(image, (width, height), interpolation=self._cv2.INTER_AREA)
        return self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)

    def window(self, window: FrameWindow, fps: float, hz: float, width: int, height: int) -> np.ndarray:
        """Frames at ``window.start_seconds + i / hz``, like extract_window_frames."""

        count = math.ceil(window.duration_seconds * hz - 1e-9)
        frames = []
        for step in range(max(count, 0)):
            image = self.frame(round((window.start_seconds + step / hz) * fps), width, height)
            if image is None:
                break
            frames.append(image)
        if not frames:
            return np.zeros((0, height, width, 3), dtype=np.uint8)
        return np.stack(frames)


def _frame_rate(text: Any) -> float | None:
    value = str(text or "").strip()
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            rate = float(numerator) / float(denominator)
        else:
            rate = float(value)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def probe_video_fps(path: Any) -> float | None:
    """The video's average frame rate from ffprobe, as crowd-city's frames.probe_video_fps."""

    command = [
        shutil.which("ffprobe") or "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=60, check=False)
        streams = json.loads(completed.stdout.decode("utf-8")).get("streams") or []
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if completed.returncode != 0 or not streams:
        return None
    return _frame_rate(streams[0].get("avg_frame_rate")) or _frame_rate(streams[0].get("r_frame_rate"))


class SourceFrames:
    """Frames of one CROWD segment read from its original video, exactly as crowd-city reads them.

    Windows are given in seconds from the segment start, like ``ClipFrames``; frames are decoded
    at ``start_seconds`` later on the original video with crowd-city's ffmpeg call
    (``utils/segmentation/frames.py``, ``extract_window_frames``), so the pixels SegFormer sees
    are the ones crowd-city's pipeline sees.
    """

    def __init__(self, path: Any, start_seconds: float) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise RuntimeError(f"Cannot open the video: {path}")
        self.start_seconds = float(start_seconds)
        # The detection frame-count advances once per decoded frame, so frames map to time at the
        # video's real rate (crowd-city's FrameClock).
        self.video_fps = probe_video_fps(self.path)
        self.ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    def close(self) -> None:
        pass

    def window(self, window: FrameWindow, fps: float, hz: float, width: int, height: int) -> np.ndarray:
        """Frames at ``window.start_seconds + i / hz`` after the segment start."""

        if window.duration_seconds <= 0 or hz <= 0:
            return np.zeros((0, height, width, 3), dtype=np.uint8)
        command = [
            self.ffmpeg, "-nostdin", "-loglevel", "error",
            "-ss", f"{float(self.start_seconds + window.start_seconds):.3f}",
            "-i", str(self.path),
            "-t", f"{float(window.duration_seconds):.3f}",
            "-an", "-sn",
            "-vf", f"fps={float(hz):g},scale={int(width)}:{int(height)}",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "-",
        ]
        completed = subprocess.run(command, capture_output=True, timeout=600, check=False)
        if completed.returncode != 0:
            # Raised rather than returning no frames, so nothing is stored as if the window were empty.
            problem = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg could not read {window.duration_seconds:.1f}s of {self.path}: {problem}")
        frame_bytes = int(width) * int(height) * 3
        usable = (len(completed.stdout) // frame_bytes) * frame_bytes
        if usable == 0:
            return np.zeros((0, height, width, 3), dtype=np.uint8)
        return np.frombuffer(completed.stdout[:usable], dtype=np.uint8).reshape(-1, int(height), int(width), 3)


def merge_spans(spans: list[tuple[float, float]]) -> list[FrameWindow]:
    """Collapse overlapping or nearly adjacent time spans into windows."""

    if not spans:
        return []
    ordered = sorted(spans)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= WINDOW_MERGE_GAP_SECONDS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [FrameWindow(start, end - start) for start, end in merged if end > start]


def nearest_frame(frames: np.ndarray, target: float, tolerance: float) -> int | None:
    """Index of the track row closest to ``target``, if close enough."""

    if frames.size == 0:
        return None
    position = int(np.searchsorted(frames, target))
    candidates = [index for index in (position - 1, position) if 0 <= index < frames.size]
    if not candidates:
        return None
    best = min(candidates, key=lambda index: abs(float(frames[index]) - target))
    if abs(float(frames[best]) - target) > max(float(tolerance), 1.0):
        return None
    return best


class RoadSurfaceReader:
    """Surface labels under each track's feet, with crowd-city's coarse then refined sampling."""

    def __init__(self, settings: Mapping[str, Any], segmenter: Any = None) -> None:
        self.settings = settings
        self.segmenter = segmenter
        self.coarse_hz = float(settings["coarse_hz"])
        self.refine_hz = float(settings["refine_hz"])
        self.minimum_confidence = float(settings["segmentation_min_confidence"])
        root = settings.get("segmentation_store")
        self.store = SurfaceStore(root) if root else None
        self.stem: str | None = None
        # Facts about the segment written to its manifest with the labels (video id, frame rates).
        self.segment_facts: dict[str, Any] = {}

    def _store_settings(self) -> dict[str, Any]:
        """Everything that changes the meaning of a stored surface label (crowd-city's settings)."""

        return store_settings(
            str(self.settings["segmentation_model"]),
            int(self.settings["segmentation_input_width"]),
            int(self.settings["segmentation_input_height"]),
            self.coarse_hz,
            self.refine_hz,
            DEFAULT_FOOTPOINT_BAND_FRACTION,
            DEFAULT_FOOTPOINT_WIDTH_FRACTION,
            self.minimum_confidence,
        )

    def _segmenter(self) -> Any:
        if self.segmenter is None:
            from scripts.crossing.crowd_city_segformer import SurfaceSegmenter

            self.segmenter = SurfaceSegmenter(
                model_name=self.settings["segmentation_model"],
                device=self.settings["segmentation_device"],
                batch_size=self.settings["segmentation_batch_size"],
                input_width=self.settings["segmentation_input_width"],
                input_height=self.settings["segmentation_input_height"],
            )
        return self.segmenter

    @staticmethod
    def boxes(
        tracks: Mapping[str, pl.DataFrame],
    ) -> dict[str, tuple[np.ndarray, dict[int, tuple[float, float, float, float]]]]:
        boxes = {}
        for track_id, track in tracks.items():
            ordered = track.sort("frame-count")
            frames = ordered.get_column("frame-count").cast(pl.Int64, strict=False).to_numpy()
            geometry = {
                index: (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                for index, row in enumerate(ordered.select(["x-center", "y-center", "width", "height"]).iter_rows())
            }
            if geometry:
                boxes[str(track_id)] = (frames, geometry)
        return boxes

    def merged_windows(self, boxes: Mapping[str, Any], fps: float) -> list[FrameWindow]:
        gap_frames = max(1, round(TRACK_SPAN_GAP_SECONDS * fps))
        spans: list[tuple[float, float]] = []
        for frames, _ in boxes.values():
            if frames.size == 0:
                continue
            ordered = np.sort(frames)
            run_start = run_end = ordered[0]
            for value in ordered[1:]:
                if value - run_end > gap_frames:
                    spans.append(
                        (max(0.0, run_start / fps - WINDOW_PADDING_SECONDS), run_end / fps + WINDOW_PADDING_SECONDS)
                    )
                    run_start = value
                run_end = value
            spans.append((max(0.0, run_start / fps - WINDOW_PADDING_SECONDS), run_end / fps + WINDOW_PADDING_SECONDS))
        kept: list[FrameWindow] = []
        for window in merge_spans(spans):
            if window.duration_seconds <= MAXIMUM_WINDOW_SECONDS:
                kept.append(window)
                continue
            pieces = math.ceil(window.duration_seconds / MAXIMUM_WINDOW_SECONDS)
            length = window.duration_seconds / pieces
            kept.extend(FrameWindow(window.start_seconds + index * length, length) for index in range(pieces))
        return kept

    def sample_windows(
        self, clip: ClipFrames, fps: float, windows: Sequence[FrameWindow], hz: float, boxes: Mapping[str, Any]
    ) -> list[SurfaceSample]:
        samples: list[SurfaceSample] = []
        tolerance = fps / (2.0 * hz)
        segmenter = self._segmenter()
        # Seconds from the start of the whole video to the start of this clip (0 for a local clip).
        offset_seconds = float(getattr(clip, "start_seconds", 0.0))
        for window in windows:
            # A fixed grid of the whole video's time, as crowd-city samples, so a track's samples
            # do not depend on its neighbours and match crowd-city's.
            grid_start = math.floor((offset_seconds + window.start_seconds) * hz) / hz - offset_seconds
            window = FrameWindow(grid_start, window.start_seconds + window.duration_seconds - grid_start)
            frames = clip.window(window, fps, hz, segmenter.input_width, segmenter.input_height)
            if len(frames) == 0:
                continue
            labels, confidence = [], []
            for first in range(0, len(frames), 16):
                part_labels, part_confidence = segmenter.segment(frames[first:first + 16])
                labels.append(part_labels)
                confidence.append(part_confidence)
            labels, confidence = np.concatenate(labels), np.concatenate(confidence)
            for offset in range(len(frames)):
                video_time = window.start_seconds + offset / hz
                detection_frame = video_time * fps
                for track_id, (track_frames, geometry) in boxes.items():
                    matched = nearest_frame(track_frames, detection_frame, tolerance)
                    if matched is None:
                        continue
                    x_center, y_center, width, height = geometry[matched]
                    surface, score = sample_surface(
                        labels[offset], confidence[offset], x_center, y_center, width, height,
                        band_fraction=DEFAULT_FOOTPOINT_BAND_FRACTION,
                        width_fraction=DEFAULT_FOOTPOINT_WIDTH_FRACTION,
                        minimum_confidence=self.minimum_confidence,
                    )
                    samples.append(
                        SurfaceSample(
                            track_id, int(track_frames[matched]), float(offset_seconds + video_time),
                            surface, float(score),
                        )
                    )
        return samples

    def timelines(
        self, clip: ClipFrames | SourceFrames, fps: float, tracks: Mapping[str, pl.DataFrame]
    ) -> dict[str, list[SurfaceSample]]:
        """Surface samples per track, segmenting only the tracks the shared store lacks."""

        # Detection frames map to time at the video's real rate when it is known (crowd-city's clock).
        fps = float(getattr(clip, "video_fps", None) or fps)
        if not (self.store is not None and self.stem is not None and tracks):
            return self._compute_timelines(clip, fps, tracks)
        settings = self._store_settings()
        # The store holds CROWD's own frame-count, which starts at 1 where this project's frames start
        # at 0 (crowd_tracks.FIRST_FRAME_COUNT); shift at the boundary so crowd-city reads the same frames.
        shift = int(self.segment_facts.get("first_frame_number", 0))
        original = {canonical_track_id(key): str(key) for key in tracks}
        signatures = {
            canonical_track_id(key): track_signature(value.with_columns(pl.col("frame-count") + shift))
            for key, value in tracks.items()
        }
        stored = self.store.load(self.stem, settings, signatures)
        missing = {key: value for key, value in tracks.items() if canonical_track_id(key) not in stored}
        computed = self._compute_timelines(clip, fps, missing) if missing else {}
        if missing:
            self.store.save(
                self.stem, settings,
                {canonical_track_id(key): signatures[canonical_track_id(key)] for key in missing},
                {
                    key: [SurfaceSample(s.track_id, s.frame + shift, s.video_time_s, s.surface, s.confidence)
                          for s in samples]
                    for key, samples in computed.items()
                },
                extra={**self.segment_facts, "video_fps": fps},
            )
        result: dict[str, list[SurfaceSample]] = {}
        for track_id, samples in stored.items():
            result[original[track_id]] = [
                SurfaceSample(original[track_id], s.frame - shift, s.video_time_s, s.surface, s.confidence)
                for s in samples
            ]
        result.update({str(key): list(value) for key, value in computed.items()})
        return result

    def _compute_timelines(
        self, clip: ClipFrames, fps: float, tracks: Mapping[str, pl.DataFrame]
    ) -> dict[str, list[SurfaceSample]]:
        boxes = self.boxes(tracks)
        if not boxes:
            return {}
        coarse_windows = self.merged_windows(boxes, fps)
        samples = self.sample_windows(clip, fps, coarse_windows, self.coarse_hz, boxes)
        spans: list[tuple[float, float]] = []
        for track_samples in surface_timeline(samples).values():
            entry, exit_bracket = transition_brackets(
                [sample.frame for sample in track_samples], [sample.surface for sample in track_samples]
            )
            for bracket in (entry, exit_bracket):
                if bracket is not None:
                    low, high = bracket
                    spans.append((max(0.0, low / fps - BRACKET_PADDING_SECONDS), high / fps + BRACKET_PADDING_SECONDS))
        refine_windows = merge_spans(spans)
        if refine_windows:
            samples.extend(self.sample_windows(clip, fps, refine_windows, self.refine_hz, boxes))
        return surface_timeline(samples)

    @staticmethod
    def camera_shift(
        clip: ClipFrames | SourceFrames, fps: float, first_frame: float, last_frame: float
    ) -> float | None:
        """How far the background slides sideways between two frames, in image widths."""

        fps = float(getattr(clip, "video_fps", None) or fps)
        start, end = first_frame / fps, last_frame / fps
        if end <= start:
            return 0.0
        frames = clip.window(FrameWindow(start, end - start), fps, CAMERA_SHIFT_HZ, FRAME_WIDTH, FRAME_HEIGHT)
        if len(frames) == 0:
            return None
        return background_shift(frames)


def crossing_tracks(
    df: pl.DataFrame, track_ids: Sequence[Any], bounds: Mapping[Any, tuple[int, int]]
) -> dict[str, pl.DataFrame]:
    """Detection rows of each candidate, restricted to its own frame range (crowd-city's _crossing_tracks)."""

    wanted = {str(value) for value in track_ids}
    ranges = {str(key): value for key, value in bounds.items()}
    tracks: dict[str, pl.DataFrame] = {}
    for track in df.partition_by("unique-id", maintain_order=True):
        track_id = str(track.get_column("unique-id")[0])
        if track_id not in wanted:
            continue
        track = track.sort("frame-count")
        if track_id in ranges:
            start, end = ranges[track_id]
            restricted = track.filter(pl.col("frame-count").cast(pl.Int64, strict=False).is_between(start, end))
            if restricted.height > 0:
                track = restricted
        tracks[track_id] = track
    return tracks


class CrowdCityRoadCrossingDetector(CrowdCityCrossingDetector):
    """crowd-city's ``road_crossing`` rule on one clip; needs the clip for SegFormer and camera shift."""

    needs_video = True

    def __init__(self, settings: Mapping[str, Any], segmenter: Any = None) -> None:
        super().__init__({"track_joining": True})
        self.rule_settings = dict(settings)
        self.reader = RoadSurfaceReader(self.rule_settings, segmenter)
        # Set by use_segment_source for the CROWD segment about to be detected; None for other clips.
        self.segment_source: dict[str, Any] | None = None

    def use_segment_source(
        self, path: Any, start_seconds: float, stem: str, facts: Mapping[str, Any] | None = None
    ) -> None:
        """Read the next clip's frames from its original video and share its labels under ``stem``.

        ``path`` is the whole CROWD video the segment was cut from, ``start_seconds`` the segment's
        start in it, ``stem`` the segment's CROWD detection file name without extension. Call
        ``clear_segment_source`` after the clip.
        """

        self.segment_source = {
            "path": path, "start_seconds": float(start_seconds), "stem": str(stem), "facts": dict(facts or {}),
        }

    def clear_segment_source(self) -> None:
        self.segment_source = None

    def detect(
        self, observations: list[TrackObservation], fps: float, video_path: Any = None
    ) -> ModelCrossingDetectionResult:
        confident = [o for o in observations if o.confidence >= MIN_CONFIDENCE]
        if video_path is None or not any(o.class_id == PERSON_CLASS for o in confident):
            return ModelCrossingDetectionResult([], [], [])
        fps = float(fps)
        confident = self._joined(confident, fps)
        df = observations_frame(confident)

        # Step 2: box candidates (crowd-city's _road_crossing_candidates).
        person_ids = df.filter(pl.col("yolo-id") == PERSON_CLASS).get_column("unique-id").unique().to_list()
        person_rows = df.filter(pl.col("unique-id").is_in(person_ids)).sort(["unique-id", "frame-count"])
        track_index = {
            track.get_column("unique-id")[0]: track
            for track in person_rows.partition_by("unique-id", maintain_order=True)
        }
        rows, scene_profile = metrics.rows_and_scene_profile(df, fps)
        person_tracks = metrics.group_tracks(row for row in rows if row.class_id == metrics.PERSON_CLASS_ID)
        trimmed = {}
        for track_id, track_rows in person_tracks.items():
            run = rule.longest_track_run([row.frame for row in track_rows], fps)
            if run is not None:
                trimmed[track_id] = metrics.trim_rows_to_frame_range(track_rows, *run)
        aspect_ratio = metrics.DEFAULT_ASPECT_RATIO
        features = metrics.contextual_track_features(trimmed, fps, "clip", float(aspect_ratio), scene_profile)
        camera_strip = (STRIP_LEFT, STRIP_RIGHT, rule.PASS_TOLERANCE)
        minimum_x_range = float(CROSSING_PARAMETER_DEFAULTS["min_crossing_x_range"])
        ids, bounds, size_rates = rule.box_only_candidates(track_index, features, fps, minimum_x_range, camera_strip)
        frame_column = df.get_column("frame-count")
        kept = []
        for track_id in ids:
            first, last = bounds[track_id]
            window = df.filter(frame_column.is_between(first, last))
            if rule.independent_motion_rejection(window, track_id, fps) is None:
                kept.append(track_id)
        ids = kept
        if not ids:
            return ModelCrossingDetectionResult([], [], [])

        # Steps 3 and 4: the feet on the road, walking, and the picture itself (crossing_pass.select_road_crossings).
        tracks = crossing_tracks(df, ids, bounds)
        source = self.segment_source
        if source is not None:
            # A CROWD segment with its original video: read as crowd-city reads it and share labels.
            clip = SourceFrames(source["path"], source["start_seconds"])
            self.reader.stem = source["stem"]
            self.reader.segment_facts = dict(source.get("facts") or {})
        else:
            clip = ClipFrames(video_path)
            self.reader.stem = None
            self.reader.segment_facts = {}
        try:
            timelines = self.reader.timelines(clip, fps, tracks)
            intervals = rule.road_intervals_for_tracks(timelines)
            rates = {str(key): value for key, value in size_rates.items()}
            selected: set[int] = set()
            reasons: dict[int, str] = {}
            for track_id in ids:
                track = tracks.get(str(track_id))
                if track is None:
                    continue
                flags = rule.road_crossing_flags(
                    track, intervals.get(str(track_id)), rates.get(str(track_id)), minimum_x_range,
                    surfaces=[sample.surface for sample in timelines.get(str(track_id)) or []],
                    fps=fps, aspect_ratio=aspect_ratio, camera_strip=camera_strip,
                )
                key = int(float(track_id))
                if not flags["road_crossing"]:
                    reasons[key] = "feet not on the road, or not walking at a plausible pace"
                    continue
                ordered = track.sort("frame-count")
                frames = ordered.get_column("frame-count")
                shift = self.reader.camera_shift(clip, fps, float(frames.min()), float(frames.max()))
                if shift is None:
                    reasons[key] = "camera motion could not be read"
                    continue
                if rule.swept_by_turning_camera(ordered.get_column("x-center").to_list(), shift):
                    reasons[key] = "carried across the picture by the turning camera"
                    continue
                window = df.filter(frame_column.is_between(frames.min(), frames.max()))
                if rule.unverifiable_crossing(track, window, track_id, fps, shift, flags.get("walking_speed_mps")):
                    reasons[key] = "cannot be verified against a static background"
                    continue
                selected.add(key)
        finally:
            clip.close()

        by_track: dict[int, list[TrackObservation]] = {}
        for item in confident:
            if item.class_id == PERSON_CLASS:
                by_track.setdefault(item.track_id, []).append(item)
        accepted, rejected, classifications = [], [], []
        bounds_by_id = {int(float(key)): value for key, value in bounds.items()}
        for track_id in sorted(int(float(value)) for value in ids):
            start, end = (int(v) for v in bounds_by_id[track_id])
            track_rows = sorted(
                (o for o in by_track.get(track_id, []) if start <= o.frame_index <= end),
                key=lambda o: o.frame_index,
            )
            if not track_rows:
                continue
            valid = track_id in selected
            event = self._event(track_id, track_rows, valid)
            (accepted if valid else rejected).append(event)
            classifications.append(
                CrossingClassification(
                    person_id=track_id,
                    probability=1.0 if valid else 0.0,
                    threshold=1.0,
                    predicted_crossing=valid,
                    rule_outcome="CROWD_ROAD_CROSSING" if valid else "CROWD_ROAD_FILTERED",
                    event=event,
                    track_features={
                        "segment_start_frame": track_rows[0].frame_index,
                        "segment_end_frame": track_rows[-1].frame_index,
                        "x_range": event.features.x_range,
                        "road_frames": event.features.road_frames,
                        "rejection": reasons.get(track_id),
                    },
                )
            )
        accepted.sort(key=lambda e: (e.start_frame, e.person_id))
        rejected.sort(key=lambda e: (e.start_frame, e.person_id))
        return ModelCrossingDetectionResult(accepted, rejected, classifications)
