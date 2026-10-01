"""Camera motion from BoT-SORT's own global motion compensation, and track motion relative to the scene.

BoT-SORT estimates a frame to frame affine transform of the image (its GMC module)
to keep identities stable while the camera moves. With the ``sparseOptFlow``
method the estimate depends only on the frames, so it is recomputed here with the
same module and settings. Removing the camera's own motion from a pedestrian's
foot point path separates people who really move across the scene from people who
only appear to move sideways because the car turns.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from scripts.core.models import TrackObservation

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class CameraMotion:
    """Per frame affine warps mapping frame t-1 image coordinates to frame t."""

    def __init__(self, width: int, height: int, warps: dict[int, np.ndarray]) -> None:
        self.width = int(width)
        self.height = int(height)
        self.warps = warps

    def warp(self, frame_index: int) -> np.ndarray:
        return self.warps.get(frame_index, IDENTITY)

    def composed(self, start_frame: int, end_frame: int) -> np.ndarray:
        """Warp from start_frame coordinates to end_frame coordinates."""

        total = np.vstack([IDENTITY, [0.0, 0.0, 1.0]])
        for frame in range(start_frame + 1, end_frame + 1):
            total = np.vstack([self.warp(frame), [0.0, 0.0, 1.0]]) @ total
        return total[:2]


def estimate_camera_motion(video_path: str | Path, method: str = "sparseOptFlow") -> CameraMotion:
    """Run BoT-SORT's GMC over every frame of a video."""

    import cv2
    from ultralytics.trackers.utils.gmc import GMC

    capture = cv2.VideoCapture(str(Path(video_path).resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for camera motion: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gmc = GMC(method=method)
    warps: dict[int, np.ndarray] = {}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            warp = gmc.apply(frame)
            warps[frame_index] = IDENTITY.copy() if frame_index == 0 else np.asarray(warp, dtype=float)
            frame_index += 1
    finally:
        capture.release()
    return CameraMotion(width, height, warps)


def _static_region_mask(video_path: Path, downscale: int, samples: int = 24) -> np.ndarray | None:
    """Pixels that barely change over the video: the camera car's bonnet, mirror, or mount."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in np.linspace(0, max(0, total - 1), num=min(samples, max(1, total))).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray, (gray.shape[1] // downscale, gray.shape[0] // downscale)).astype(np.float32))
    capture.release()
    if len(frames) < 4:
        return None
    static = np.std(np.stack(frames), axis=0) < 6.0
    # Keep only large static regions; isolated static pixels are ordinary texture.
    static = cv2.morphologyEx(static.astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    return static.astype(bool)


def estimate_masked_camera_motion(
    video_path: str | Path,
    observations: list[TrackObservation] | None = None,
    mask_static: bool = True,
    downscale: int = 2,
) -> CameraMotion:
    """BoT-SORT's sparse optical flow GMC with detections and the camera car masked out.

    Identical to ``GMC.apply_sparseoptflow`` except that feature points are not taken
    from YOLO detection boxes (moving people and vehicles) or from image regions that
    never change (parts of the camera car), which otherwise bias the estimate.
    """

    import cv2

    source = Path(video_path).resolve()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for camera motion: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    static = _static_region_mask(source, downscale) if mask_static else None
    boxes: dict[int, list[TrackObservation]] = {}
    for item in observations or []:
        boxes.setdefault(item.frame_index, []).append(item)
    feature_params = {"maxCorners": 1000, "qualityLevel": 0.01, "minDistance": 1, "blockSize": 3,
                      "useHarrisDetector": False, "k": 0.04}
    warps: dict[int, np.ndarray] = {}
    previous_frame = previous_points = None
    frame_index = 0
    try:
        while True:
            ok, raw = capture.read()
            if not ok or raw is None:
                break
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(gray, (width // downscale, height // downscale)) if downscale > 1 else gray
            mask = np.full(frame.shape, 255, dtype=np.uint8)
            if static is not None and static.shape == frame.shape:
                mask[static] = 0
            for item in boxes.get(frame_index, []):
                w, h = frame.shape[1], frame.shape[0]
                pad_x = 0.1 * (item.box.x2 - item.box.x1)
                pad_y = 0.1 * (item.box.y2 - item.box.y1)
                x1, x2 = int(max(0, (item.box.x1 - pad_x) * w)), int(min(w, (item.box.x2 + pad_x) * w))
                y1, y2 = int(max(0, (item.box.y1 - pad_y) * h)), int(min(h, (item.box.y2 + pad_y) * h))
                mask[y1:y2, x1:x2] = 0
            points = cv2.goodFeaturesToTrack(frame, mask=mask, **feature_params)
            warp = IDENTITY.copy()
            if previous_frame is not None and previous_points is not None:
                matched, status, _ = cv2.calcOpticalFlowPyrLK(previous_frame, frame, previous_points, None)
                good = status.ravel().astype(bool)
                if good.sum() > 4:
                    estimate, _ = cv2.estimateAffinePartial2D(previous_points[good], matched[good], cv2.RANSAC)
                    if estimate is not None:
                        warp = estimate
                        warp[0, 2] *= downscale
                        warp[1, 2] *= downscale
            warps[frame_index] = np.asarray(warp, dtype=float)
            previous_frame, previous_points = frame, points
            frame_index += 1
    finally:
        capture.release()
    return CameraMotion(width, height, warps)


def save_camera_motion_csv(path: str | Path, motion: CameraMotion) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "a11", "a12", "a13", "a21", "a22", "a23", "width", "height"])
        for frame, warp in sorted(motion.warps.items()):
            writer.writerow([frame, *[round(float(v), 6) for v in warp.ravel()], motion.width, motion.height])
    return destination


def load_camera_motion_csv(path: str | Path) -> CameraMotion:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Camera motion CSV not found: {source}")
    warps: dict[int, np.ndarray] = {}
    width = height = 0
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            warps[int(row["frame_index"])] = np.array(
                [[float(row["a11"]), float(row["a12"]), float(row["a13"])],
                 [float(row["a21"]), float(row["a22"]), float(row["a23"])]]
            )
            width, height = int(row["width"]), int(row["height"])
    return CameraMotion(width, height, warps)


COMPENSATED_FEATURES = (
    "scene_x_net",
    "scene_x_range",
    "scene_x_gross",
    "scene_x_direction_consistency",
    "scene_x_range_over_height",
    "scene_to_image_x_ratio",
    "camera_x_travel",
)


def compensated_features(track: list[TrackObservation], motion: CameraMotion) -> dict[str, float]:
    """Foot point motion after removing the camera's own motion, normalised by image width."""

    track = sorted(track, key=lambda item: item.frame_index)
    width, height = motion.width, motion.height
    if len(track) < 2 or width <= 0:
        return {name: 0.0 for name in COMPENSATED_FEATURES}
    feet = [np.array([item.box.centre_x * width, item.box.y2 * height]) for item in track]
    cumulative = [0.0]
    residuals = []
    camera_travel = 0.0
    for previous, current, p_prev, p_cur in zip(track, track[1:], feet, feet[1:]):
        warp = motion.composed(previous.frame_index, current.frame_index)
        predicted = warp @ np.array([p_prev[0], p_prev[1], 1.0])
        residual = float(p_cur[0] - predicted[0])
        residuals.append(residual)
        cumulative.append(cumulative[-1] + residual)
        camera_travel += abs(float(predicted[0] - p_prev[0]))
    gross = sum(abs(r) for r in residuals)
    image_range = (max(f[0] for f in feet) - min(f[0] for f in feet))
    median_height = float(np.median([(item.box.y2 - item.box.y1) * height for item in track]))
    scene_range = max(cumulative) - min(cumulative)
    return {
        "scene_x_net": abs(cumulative[-1]) / width,
        "scene_x_range": scene_range / width,
        "scene_x_gross": gross / width,
        "scene_x_direction_consistency": abs(cumulative[-1]) / gross if gross > 0 else 0.0,
        "scene_x_range_over_height": scene_range / max(median_height, 1.0),
        "scene_to_image_x_ratio": scene_range / image_range if image_range > 0 else 0.0,
        "camera_x_travel": camera_travel / width,
    }


def features_for_tracks(
    tracks: dict[int, list[TrackObservation]], motion: CameraMotion
) -> dict[int, dict[str, Any]]:
    return {track_id: compensated_features(track, motion) for track_id, track in tracks.items()}
