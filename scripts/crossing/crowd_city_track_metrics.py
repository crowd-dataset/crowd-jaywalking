"""Track features of crowd-city's crossing metrics, vendored from utils/crossing/metrics.py.

Source: https://github.com/crowd-dataset/crowd-city (main, commit 340875e). Only the functions the
crossing rule needs are copied, each unchanged (track_features and what it uses,
contextual_track_features, rows_and_scene_profile, group_tracks, normalise_id).
"""

import bisect
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

_ROWS_AND_SCENE_CACHE: List[Any] = [None, None, None, None]

def rows_and_scene_profile(df: pl.DataFrame, fps: float) -> Tuple[List["BBoxRow"], "SceneMotionProfile"]:
    """Return the box rows of ``df`` and their scene-motion profile, built once per table."""
    cache = _ROWS_AND_SCENE_CACHE
    if cache[0] is df and cache[1] == float(fps):
        return cache[2], cache[3]
    rows = bbox_rows_from_polars(df)
    profile = build_scene_motion_profile(rows, float(fps))
    cache[:] = [df, float(fps), rows, profile]
    return rows, profile

PERSON_CLASS_ID = 0

DEFAULT_ASPECT_RATIO = 16.0 / 9.0

DEFAULT_PERSON_HEIGHT_M = 1.70

SCENE_MOTION_SETTINGS: Dict[str, float] = {
    "maximum_pair_gap_frames": 2,
    "window_radius_frames": 3,
    "minimum_reference_tracks": 3,
    "maximum_absolute_x_rate_per_second": 3.0,
    "outlier_mad_multiplier": 4.0,
}

FEATURE_TIME_WINDOWS_SECONDS: Dict[str, float] = {
    "smoothing": 0.3,
    "cleaning": 0.5,
    "rate_span": 0.1,
    "scene_pair_gap": 0.2,
    "scene_radius": 0.3,
}

FEATURE_REFERENCE_FPS = 10.0

def window_frames(seconds: float, fps: float, odd: bool = False) -> int:
    """Return the number of frames spanning ``seconds`` at ``fps`` (at least one)."""
    frames = max(1, int(round(float(seconds) * float(fps))))
    if odd and frames % 2 == 0:
        frames += 1
    return frames

def lagged_rates(time_values: np.ndarray, values: np.ndarray, lag: int) -> np.ndarray:
    """Return rates of change over ``lag`` frames, starting at every frame."""
    if len(values) <= lag:
        return np.zeros(0, dtype=float)
    delta_time = time_values[lag:] - time_values[:-lag]
    delta_value = values[lag:] - values[:-lag]
    valid = delta_time > 0.0
    return delta_value[valid] / delta_time[valid]

def reversal_fraction(x_values: np.ndarray, lag: int = 1) -> float:
    """Share of consecutive movement steps whose direction reverses.

    Each step spans ``lag`` frames. Every frame starts one sequence of
    back-to-back steps, and the sequences are pooled, so no frame is dropped.
    With ``lag`` 1 this is the original frame-to-frame measure.
    """
    changes = 0
    comparisons = 0
    for offset in range(max(1, int(lag))):
        series = x_values[offset::max(1, int(lag))]
        if len(series) < 4:
            continue
        delta = np.diff(series)
        noise = max(robust_scale(delta) * 0.25, 0.0005)
        signs = np.sign(delta[np.abs(delta) > noise])
        if len(signs) < 2:
            continue
        changes += int(np.sum(signs[1:] != signs[:-1]))
        comparisons += len(signs) - 1
    return float(changes / comparisons) if comparisons else 0.0

SOURCE_CONTEXT_SETTINGS: Dict[str, float] = {
    "minimum_reference_tracks": 3,
    "minimum_proxy_mps": 0.001,
    "maximum_absolute_log_ratio": 2.00,
}

BASE_GATES: Dict[str, float] = {
    "minimum_rows": 8,
    "minimum_duration_seconds": 0.50,
    "minimum_coverage": 0.50,
    "minimum_median_height": 0.01,
    "minimum_horizontal_range": 0.08,
    "minimum_x_fit_r2": 0.15,
    "maximum_height_ratio": 3.50,
    "maximum_edge_fraction": 0.60,
    "maximum_reversal_fraction": 0.50,
    "minimum_scene_motion_support": 3,
}

@dataclass(frozen=True)
class BBoxRow:
    class_id: int
    x: float
    y: float
    width: float
    height: float
    track_id: str
    confidence: float
    frame: int

@dataclass
class SceneMotionProfile:
    samples_by_frame: Dict[int, List[Tuple[float, str]]]
    # window_radius_frames at the reference rate; build_scene_motion_profile
    # sets it from FEATURE_TIME_WINDOWS_SECONDS for the video's frame rate.
    radius_frames: int = 3

    def rate_at(self, frame: float) -> Tuple[float, int]:
        radius = int(self.radius_frames)
        centre_frame = int(round(frame))
        samples: List[Tuple[float, str]] = []
        for candidate in range(centre_frame - radius, centre_frame + radius + 1):
            samples.extend(self.samples_by_frame.get(candidate, []))
        support = len({track_id for _, track_id in samples})
        if support < int(SCENE_MOTION_SETTINGS["minimum_reference_tracks"]):
            return 0.0, support
        rates = np.asarray([rate for rate, _ in samples], dtype=float)
        centre = float(np.median(rates))
        scale = robust_scale(rates)
        limit = float(SCENE_MOTION_SETTINGS["outlier_mad_multiplier"]) * scale
        retained = rates[np.abs(rates - centre) <= limit]
        if len(retained):
            centre = float(np.median(retained))
        return centre, support

@dataclass
class TrackFeatures:
    source_id: str
    prediction_track_id: str
    input_rows: int
    clean_rows: int
    removed_rows: int
    first_frame: int
    last_frame: int
    duration_seconds: float
    coverage: float
    gap_ratio: float
    median_confidence: float
    median_height: float
    median_width: float
    height_ratio: float
    horizontal_range: float
    vertical_range: float
    crosses_image_midline: bool
    overlaps_central_band: bool
    direction: str
    x_slope_per_second: float
    x_fit_r2: float
    q_slope_per_second: float
    q_fit_r2: float
    q_residual_mad: float
    log_height_rate_abs: float
    bottom_rate_abs: float
    edge_fraction: float
    truncation_fraction: float
    reversal_fraction: float
    raw_speed_proxy_mps: float
    q_speed_proxy_mps: float
    robust_speed_proxy_mps: float
    compensated_raw_speed_proxy_mps: float
    compensated_q_speed_proxy_mps: float
    compensated_robust_speed_proxy_mps: float
    scene_motion_rate_abs: float
    scene_motion_equivalent_speed_mps: float
    scene_motion_fraction: float
    scene_motion_support: float
    log_scene_motion_support: float
    compensated_proxy_disagreement_mps: float
    one_minus_x_r2: float
    log_height_ratio: float
    log_duration: float
    log_median_height: float
    source_context_tracks: float = 0.0
    source_context_available: float = 0.0
    source_relative_log_raw_proxy: float = 0.0
    source_relative_log_q_proxy: float = 0.0
    source_relative_log_robust_proxy: float = 0.0
    source_relative_log_compensated_robust_proxy: float = 0.0
    source_compensated_robust_percentile: float = 0.5

def safe_float(value: Any) -> Optional[float]:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None

def safe_int(value: Any) -> Optional[int]:
    output = safe_float(value)
    return int(round(output)) if output is not None else None

def normalise_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    numeric = safe_float(text)
    if numeric is not None and abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return text

def first_column(
    fieldnames: Sequence[str],
    aliases: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    if required:
        raise ValueError(
            "CSV is missing a required column. Expected one of: " + ", ".join(aliases)
        )
    return None

def bbox_rows_from_polars(dataframe: pl.DataFrame) -> List[BBoxRow]:
    """Convert the existing analysis DataFrame without creating a second CSV."""
    names = dataframe.columns
    class_col = first_column(names, ["yolo-id", "yolo_id", "class-id", "class_id", "class"])
    x_col = first_column(names, ["x-center", "x_center", "xcentre", "xcenter"])
    y_col = first_column(names, ["y-center", "y_center", "ycentre", "ycenter"])
    width_col = first_column(names, ["width", "bbox-width", "bbox_width", "w"])
    height_col = first_column(names, ["height", "bbox-height", "bbox_height", "h"])
    track_col = first_column(names, ["unique-id", "unique_id", "track-id", "track_id", "id"])
    frame_col = first_column(names, ["frame-count", "frame_count", "frame", "frame-id", "frame_id"])
    confidence_col = first_column(
        names,
        ["confidence", "conf", "score"],
        required=False,
    )
    output: List[BBoxRow] = []
    for raw in dataframe.to_dicts():
        class_id = safe_int(raw.get(class_col))
        x_value = safe_float(raw.get(x_col))
        y_value = safe_float(raw.get(y_col))
        width = safe_float(raw.get(width_col))
        height = safe_float(raw.get(height_col))
        frame = safe_int(raw.get(frame_col))
        track_id = normalise_id(raw.get(track_col))
        confidence = safe_float(raw.get(confidence_col)) if confidence_col else 1.0
        if None in {class_id, x_value, y_value, width, height, frame} or not track_id:
            continue
        if width <= 0.0 or height <= 0.0:
            continue
        output.append(
            BBoxRow(
                class_id=int(class_id),
                x=float(x_value),
                y=float(y_value),
                width=float(width),
                height=float(height),
                track_id=track_id,
                confidence=float(confidence if confidence is not None else 1.0),
                frame=int(frame),
            )
        )
    return output

def group_tracks(rows: Iterable[BBoxRow]) -> Dict[str, List[BBoxRow]]:
    grouped: Dict[str, List[BBoxRow]] = defaultdict(list)
    for row in rows:
        grouped[row.track_id].append(row)
    return dict(grouped)

def rolling_median(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Centred rolling median; the window shrinks at the ends of the series.

    Vectorised with a sliding window view: for long CROWD tracks at 30 fps the
    per-element Python loop dominated the whole detection pass. The result is
    identical to taking np.median over values[max(0, i - r):i + r + 1].
    """
    if len(values) < 3 or window <= 1:
        return values.astype(float, copy=True)
    values = np.asarray(values, dtype=float)
    radius = window // 2
    count = len(values)
    output = np.empty(count, dtype=float)
    if count > 2 * radius:
        output[radius:count - radius] = np.median(
            np.lib.stride_tricks.sliding_window_view(values, 2 * radius + 1), axis=1
        )
        edges = list(range(radius)) + list(range(count - radius, count))
    else:
        edges = range(count)
    for index in edges:
        output[index] = float(np.median(values[max(0, index - radius):min(count, index + radius + 1)]))
    return output

def robust_scale(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    return max(1.4826 * mad, 1e-9)

def robust_line(time_values: np.ndarray, values: np.ndarray) -> Tuple[float, float, float, float]:
    if len(time_values) < 2 or float(np.ptp(time_values)) <= 0.0:
        first = float(values[0]) if len(values) else 0.0
        return 0.0, first, 0.0, 0.0
    design = np.column_stack([np.ones(len(time_values)), time_values])
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    weights = np.ones(len(values), dtype=float)
    for _ in range(20):
        residual = values - design @ beta
        scale = robust_scale(residual)
        threshold = 1.345 * scale
        weights = np.ones(len(values), dtype=float)
        large = np.abs(residual) > threshold
        weights[large] = threshold / np.maximum(np.abs(residual[large]), 1e-12)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = values * np.sqrt(weights)
        new_beta = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
        if float(np.max(np.abs(new_beta - beta))) < 1e-9:
            beta = new_beta
            break
        beta = new_beta
    fitted = design @ beta
    residual = values - fitted
    weighted_mean = float(np.average(values, weights=weights))
    ss_res = float(np.sum(weights * residual * residual))
    ss_tot = float(np.sum(weights * (values - weighted_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return (
        float(beta[1]),
        float(beta[0]),
        float(max(-1.0, min(1.0, r2))),
        robust_scale(residual),
    )

def clean_track(
    rows: Sequence[BBoxRow],
    fps: float = FEATURE_REFERENCE_FPS,
) -> List[BBoxRow]:
    best_by_frame: Dict[int, BBoxRow] = {}
    for row in rows:
        if not (-0.25 <= row.x <= 1.25 and -0.25 <= row.y <= 1.25):
            continue
        if not (0.001 <= row.width <= 1.50 and 0.001 <= row.height <= 1.50):
            continue
        current = best_by_frame.get(row.frame)
        if current is None or row.confidence > current.confidence:
            best_by_frame[row.frame] = row
    ordered = [best_by_frame[key] for key in sorted(best_by_frame)]
    if len(ordered) < 7:
        return ordered
    values = np.asarray(
        [[row.x, row.y, row.width, row.height] for row in ordered], dtype=float
    )
    cleaning_window = window_frames(FEATURE_TIME_WINDOWS_SECONDS["cleaning"], fps, odd=True)
    baseline = np.column_stack(
        [rolling_median(values[:, column], cleaning_window) for column in range(4)]
    )
    residual = values - baseline
    keep = np.ones(len(ordered), dtype=bool)
    for column in range(4):
        scale = robust_scale(residual[:, column])
        if scale > 1e-8:
            keep &= np.abs(residual[:, column]) <= 6.0 * scale
    keep[0] = True
    keep[-1] = True
    return [row for index, row in enumerate(ordered) if bool(keep[index])]

def _scene_motion_pairs(rows: Sequence[BBoxRow], rate_span: int, maximum_gap: int):
    """Pair each reference detection with the first one ``rate_span`` or more frames later.

    Every detection starts a pair, so no frame is dropped; pairs further apart
    than ``maximum_gap`` frames are skipped. With a span of one frame this is
    the original consecutive-detection pairing.
    """
    frames = [row.frame for row in rows]
    for index, first in enumerate(rows):
        later = bisect.bisect_left(frames, first.frame + int(rate_span), index + 1)
        if later < len(rows) and rows[later].frame - first.frame <= int(maximum_gap):
            yield first, rows[later]

def build_scene_motion_profile(rows: Sequence[BBoxRow], fps: float) -> SceneMotionProfile:
    samples_by_frame: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
    grouped: Dict[Tuple[int, str], List[BBoxRow]] = defaultdict(list)
    for row in rows:
        if row.class_id != PERSON_CLASS_ID:
            grouped[(row.class_id, row.track_id)].append(row)
    maximum_gap = window_frames(FEATURE_TIME_WINDOWS_SECONDS["scene_pair_gap"], fps)
    rate_span = window_frames(FEATURE_TIME_WINDOWS_SECONDS["rate_span"], fps)
    maximum_rate = float(SCENE_MOTION_SETTINGS["maximum_absolute_x_rate_per_second"])
    for (class_id, track_id), track_rows in grouped.items():
        reference_id = f"{class_id}:{track_id}"
        cleaned = clean_track(track_rows, fps)
        for first, second in _scene_motion_pairs(cleaned, rate_span, maximum_gap):
            frame_gap = second.frame - first.frame
            rate = (second.x - first.x) / (frame_gap / fps)
            if not math.isfinite(rate) or abs(rate) > maximum_rate:
                continue
            midpoint = int(round((first.frame + second.frame) / 2.0))
            samples_by_frame[midpoint].append((float(rate), reference_id))
    return SceneMotionProfile(
        dict(samples_by_frame),
        radius_frames=window_frames(FEATURE_TIME_WINDOWS_SECONDS["scene_radius"], fps),
    )

def track_features(
    rows: Sequence[BBoxRow],
    fps: float,
    source_id: str,
    aspect_ratio: float,
    scene_motion_profile: SceneMotionProfile,
) -> Optional[TrackFeatures]:
    cleaned = clean_track(rows, fps)
    if len(cleaned) < 2:
        return None
    frames = np.asarray([row.frame for row in cleaned], dtype=float)
    x_raw = np.asarray([row.x for row in cleaned], dtype=float)
    y_raw = np.asarray([row.y for row in cleaned], dtype=float)
    width_raw = np.asarray([row.width for row in cleaned], dtype=float)
    height_raw = np.asarray([row.height for row in cleaned], dtype=float)
    confidence = np.asarray([row.confidence for row in cleaned], dtype=float)
    smoothing = window_frames(FEATURE_TIME_WINDOWS_SECONDS["smoothing"], fps, odd=True)
    rate_span = window_frames(FEATURE_TIME_WINDOWS_SECONDS["rate_span"], fps)
    x = rolling_median(x_raw, smoothing)
    y = rolling_median(y_raw, smoothing)
    width = rolling_median(width_raw, smoothing)
    height = np.maximum(rolling_median(height_raw, smoothing), 0.001)
    times = (frames - frames[0]) / fps
    duration = float(times[-1])
    if duration <= 0.0:
        return None
    median_height = float(np.median(height))
    median_width = float(np.median(width))
    bottom = y + 0.50 * height
    q_value = aspect_ratio * (x - 0.50) / height
    log_height = np.log(height)
    x_slope, _, x_r2, _ = robust_line(times, x)
    q_slope, _, q_r2, q_mad = robust_line(times, q_value)
    log_height_slope, _, _, _ = robust_line(times, log_height)
    bottom_slope, _, _, _ = robust_line(times, bottom)

    delta_time = np.diff(times)
    delta_x = np.diff(x)
    local_rates = (
        aspect_ratio * lagged_rates(times, x, rate_span) / max(median_height, 0.001)
    )
    raw_proxy = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * abs(x_slope)
        / max(median_height, 0.001)
    )
    q_proxy = DEFAULT_PERSON_HEIGHT_M * abs(q_slope)
    robust_proxy = (
        DEFAULT_PERSON_HEIGHT_M * abs(float(np.median(local_rates)))
        if len(local_rates)
        else 0.0
    )

    scene_rates: List[float] = []
    scene_support: List[int] = []
    for first_frame, second_frame in zip(frames, frames[1:]):
        rate, support = scene_motion_profile.rate_at((first_frame + second_frame) / 2.0)
        scene_rates.append(float(rate))
        scene_support.append(int(support))
    scene_array = np.asarray(scene_rates, dtype=float)
    corrected_delta_x = delta_x - scene_array * delta_time
    corrected_x = np.concatenate([[x[0]], x[0] + np.cumsum(corrected_delta_x)])
    corrected_x_slope, _, _, _ = robust_line(times, corrected_x)
    corrected_q = aspect_ratio * (corrected_x - 0.50) / height
    corrected_q_slope, _, _, _ = robust_line(times, corrected_q)
    corrected_local_rates = (
        aspect_ratio * lagged_rates(times, corrected_x, rate_span) / max(median_height, 0.001)
    )
    compensated_raw = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * abs(corrected_x_slope)
        / max(median_height, 0.001)
    )
    compensated_q = DEFAULT_PERSON_HEIGHT_M * abs(corrected_q_slope)
    compensated_robust = (
        DEFAULT_PERSON_HEIGHT_M * abs(float(np.median(corrected_local_rates)))
        if len(corrected_local_rates)
        else 0.0
    )
    supported_rates = [
        abs(rate)
        for rate, support in zip(scene_rates, scene_support)
        if support >= int(SCENE_MOTION_SETTINGS["minimum_reference_tracks"])
    ]
    scene_rate_abs = float(np.median(supported_rates)) if supported_rates else 0.0
    median_support = float(np.median(scene_support)) if scene_support else 0.0
    scene_equivalent = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * scene_rate_abs
        / max(median_height, 0.001)
    )
    scene_fraction = scene_equivalent / max(raw_proxy + scene_equivalent, 1e-9)

    first_frame = int(frames[0])
    last_frame = int(frames[-1])
    expected_rows = max(1, last_frame - first_frame + 1)
    coverage = min(1.0, len(cleaned) / expected_rows)
    left = x - 0.50 * width
    right = x + 0.50 * width
    top = y - 0.50 * height
    lower = y + 0.50 * height
    edge = (left <= 0.01) | (right >= 0.99)
    truncated = edge | (top <= 0.01) | (lower >= 0.99)
    height_low = max(float(np.quantile(height, 0.10)), 0.001)
    height_high = float(np.quantile(height, 0.90))
    direction = (
        "left_to_right"
        if x_slope > 0.0
        else "right_to_left"
        if x_slope < 0.0
        else "stationary"
    )
    disagreement = float(
        max(compensated_raw, compensated_q, compensated_robust)
        - min(compensated_raw, compensated_q, compensated_robust)
    )
    return TrackFeatures(
        source_id=source_id,
        prediction_track_id=cleaned[0].track_id,
        input_rows=len(rows),
        clean_rows=len(cleaned),
        removed_rows=len(rows) - len(cleaned),
        first_frame=first_frame,
        last_frame=last_frame,
        duration_seconds=duration,
        coverage=float(coverage),
        gap_ratio=float(1.0 - coverage),
        median_confidence=float(np.median(confidence)),
        median_height=median_height,
        median_width=median_width,
        height_ratio=float(height_high / height_low),
        horizontal_range=float(np.max(x) - np.min(x)),
        vertical_range=float(np.max(y) - np.min(y)),
        crosses_image_midline=bool(float(np.min(x)) <= 0.50 <= float(np.max(x))),
        overlaps_central_band=bool(float(np.min(x)) <= 0.60 and float(np.max(x)) >= 0.40),
        direction=direction,
        x_slope_per_second=float(x_slope),
        x_fit_r2=float(x_r2),
        q_slope_per_second=float(q_slope),
        q_fit_r2=float(q_r2),
        q_residual_mad=float(q_mad),
        log_height_rate_abs=abs(float(log_height_slope)),
        bottom_rate_abs=abs(float(bottom_slope)),
        edge_fraction=float(np.mean(edge)),
        truncation_fraction=float(np.mean(truncated)),
        reversal_fraction=reversal_fraction(x, rate_span),
        raw_speed_proxy_mps=float(raw_proxy),
        q_speed_proxy_mps=float(q_proxy),
        robust_speed_proxy_mps=float(robust_proxy),
        compensated_raw_speed_proxy_mps=float(compensated_raw),
        compensated_q_speed_proxy_mps=float(compensated_q),
        compensated_robust_speed_proxy_mps=float(compensated_robust),
        scene_motion_rate_abs=scene_rate_abs,
        scene_motion_equivalent_speed_mps=float(scene_equivalent),
        scene_motion_fraction=float(scene_fraction),
        scene_motion_support=median_support,
        log_scene_motion_support=float(math.log1p(median_support)),
        compensated_proxy_disagreement_mps=disagreement,
        one_minus_x_r2=float(max(0.0, 1.0 - x_r2)),
        log_height_ratio=float(math.log(max(height_high / height_low, 1.0))),
        log_duration=float(math.log1p(duration)),
        log_median_height=float(math.log(max(median_height, 1e-6))),
    )

def base_rejection_reason(features: TrackFeatures) -> str:
    checks = [
        (features.clean_rows < int(BASE_GATES["minimum_rows"]), "too_few_rows"),
        (
            features.duration_seconds < BASE_GATES["minimum_duration_seconds"],
            "duration_too_short",
        ),
        (features.coverage < BASE_GATES["minimum_coverage"], "track_too_fragmented"),
        (
            features.median_height < BASE_GATES["minimum_median_height"],
            "box_too_small",
        ),
        (
            features.horizontal_range < BASE_GATES["minimum_horizontal_range"],
            "insufficient_lateral_motion",
        ),
        (
            features.x_fit_r2 < BASE_GATES["minimum_x_fit_r2"],
            "nonlinear_or_unstable_motion",
        ),
        (
            features.height_ratio > BASE_GATES["maximum_height_ratio"],
            "excessive_scale_change",
        ),
        (
            features.edge_fraction > BASE_GATES["maximum_edge_fraction"],
            "track_truncated_at_image_edge",
        ),
        (
            features.reversal_fraction > BASE_GATES["maximum_reversal_fraction"],
            "too_many_direction_reversals",
        ),
        (
            features.scene_motion_support < BASE_GATES["minimum_scene_motion_support"],
            "insufficient_scene_motion_references",
        ),
    ]
    for rejected, reason in checks:
        if rejected:
            return reason
    return ""

def apply_source_context(
    features_by_track: Dict[str, TrackFeatures],
) -> Dict[str, TrackFeatures]:
    """Construct the V31 label-free context from one complete bbox CSV."""
    reference = [
        features
        for features in features_by_track.values()
        if base_rejection_reason(features) == ""
    ]
    reference_count = len(reference)
    minimum_count = int(SOURCE_CONTEXT_SETTINGS["minimum_reference_tracks"])
    context_available = reference_count >= minimum_count
    for features in features_by_track.values():
        features.source_context_tracks = float(reference_count)
        features.source_context_available = 1.0 if context_available else 0.0
        features.source_relative_log_raw_proxy = 0.0
        features.source_relative_log_q_proxy = 0.0
        features.source_relative_log_robust_proxy = 0.0
        features.source_relative_log_compensated_robust_proxy = 0.0
        features.source_compensated_robust_percentile = 0.5
    if not context_available:
        return features_by_track

    minimum_proxy = float(SOURCE_CONTEXT_SETTINGS["minimum_proxy_mps"])
    maximum_log_ratio = float(
        SOURCE_CONTEXT_SETTINGS["maximum_absolute_log_ratio"]
    )
    proxy_fields = {
        "source_relative_log_raw_proxy": "raw_speed_proxy_mps",
        "source_relative_log_q_proxy": "q_speed_proxy_mps",
        "source_relative_log_robust_proxy": "robust_speed_proxy_mps",
        "source_relative_log_compensated_robust_proxy": (
            "compensated_robust_speed_proxy_mps"
        ),
    }
    medians = {
        output_field: float(
            np.median(
                [
                    max(float(getattr(features, input_field)), minimum_proxy)
                    for features in reference
                ]
            )
        )
        for output_field, input_field in proxy_fields.items()
    }
    rank_values = np.asarray(
        [
            max(
                float(features.compensated_robust_speed_proxy_mps),
                minimum_proxy,
            )
            for features in reference
        ],
        dtype=float,
    )
    for features in features_by_track.values():
        for output_field, input_field in proxy_fields.items():
            value = max(float(getattr(features, input_field)), minimum_proxy)
            ratio = math.log(value / max(medians[output_field], minimum_proxy))
            setattr(
                features,
                output_field,
                float(np.clip(ratio, -maximum_log_ratio, maximum_log_ratio)),
            )
        rank_value = max(
            float(features.compensated_robust_speed_proxy_mps),
            minimum_proxy,
        )
        less = float(np.sum(rank_values < rank_value))
        equal = float(np.sum(rank_values == rank_value))
        features.source_compensated_robust_percentile = float(
            (less + 0.5 * equal) / max(len(rank_values), 1)
        )
    return features_by_track

def contextual_track_features(
    tracks: Dict[str, List[BBoxRow]],
    fps: float,
    source_id: str,
    aspect_ratio: float,
    scene_motion_profile: SceneMotionProfile,
) -> Dict[str, TrackFeatures]:
    features_by_track: Dict[str, TrackFeatures] = {}
    for track_id in sorted(tracks, key=str):
        features = track_features(
            tracks[track_id],
            fps,
            source_id,
            aspect_ratio,
            scene_motion_profile,
        )
        if features is not None:
            features_by_track[track_id] = features
    return apply_source_context(features_by_track)


def trim_rows_to_frame_range(
    rows: Sequence[BBoxRow],
    start_frame: int,
    end_frame: int,
) -> List[BBoxRow]:
    """Keep only the rows of one track inside ``[start_frame, end_frame]``."""
    low, high = int(start_frame), int(end_frame)
    return [row for row in rows if low <= row.frame <= high]
