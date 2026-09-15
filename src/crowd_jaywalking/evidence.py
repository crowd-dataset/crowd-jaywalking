"""Generate multi-scale visual evidence for observable crossing context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import BoundingBox, CrossingEvent, EvidenceImage, TrackObservation


def _box_pixels(
    box: BoundingBox,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(box.x1 * width))))
    y1 = max(0, min(height - 1, int(round(box.y1 * height))))
    x2 = max(x1 + 1, min(width, int(round(box.x2 * width))))
    y2 = max(y1 + 1, min(height, int(round(box.y2 * height))))
    return x1, y1, x2, y2


def _resize(image, maximum_dimension: int):
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= maximum_dimension:
        return image
    scale = maximum_dimension / longest
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _draw_target(
    image,
    pixel_box: tuple[int, int, int, int],
    target_label: str,
) -> None:
    import cv2

    x1, y1, x2, y2 = pixel_box
    colour = (0, 0, 255)
    thickness = max(2, int(round(max(image.shape[:2]) / 400)))
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
    cv2.putText(
        image,
        f"TARGET {target_label}",
        (max(0, x1), max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        colour,
        2,
        cv2.LINE_AA,
    )


def _trajectory_points(
    boxes: list[BoundingBox],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    return [
        (
            max(0, min(width - 1, int(round(box.centre_x * width)))),
            max(0, min(height - 1, int(round(box.y2 * height)))),
        )
        for box in boxes
    ]


def _draw_trajectory(image, points: list[tuple[int, int]]) -> None:
    if len(points) < 2:
        return

    import cv2
    import numpy as np

    colour = (0, 255, 255)
    thickness = max(2, int(round(max(image.shape[:2]) / 500)))
    path = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [path], False, colour, thickness, cv2.LINE_AA)
    cv2.arrowedLine(
        image,
        points[-2],
        points[-1],
        colour,
        thickness,
        cv2.LINE_AA,
        tipLength=0.35,
    )
    label_x, label_y = points[0]
    cv2.putText(
        image,
        "TARGET PATH",
        (max(0, label_x - 20), max(25, label_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colour,
        2,
        cv2.LINE_AA,
    )


def _draw_view_label(image, label: str) -> None:
    import cv2

    cv2.rectangle(image, (0, 0), (min(image.shape[1], 420), 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _expanded_axis(start: int, end: int, minimum: int, limit: int) -> tuple[int, int]:
    current = end - start
    if current < minimum:
        missing = minimum - current
        start -= missing // 2
        end += missing - missing // 2
    if start < 0:
        end = min(limit, end - start)
        start = 0
    if end > limit:
        start = max(0, start - (end - limit))
        end = limit
    return max(0, start), min(limit, end)


def make_evidence_views(
    image,
    target_box: BoundingBox,
    trajectory_boxes: list[BoundingBox],
    target_label: str,
    *,
    crop_margin: float,
    road_crop_margin: float,
    control_crop_bottom: float,
    control_crop_overlap: float,
    maximum_dimension: int,
    trajectory_enabled: bool,
) -> dict[str, Any]:
    """Create clean context views plus a separate annotated trajectory view."""

    height, width = image.shape[:2]
    target_pixels = _box_pixels(target_box, width, height)
    x1, y1, x2, y2 = target_pixels
    points = _trajectory_points(trajectory_boxes, width, height)

    context = image.copy()
    _draw_target(context, target_pixels, target_label)

    trajectory = image.copy()
    if trajectory_enabled:
        _draw_trajectory(trajectory, points)
    _draw_target(trajectory, target_pixels, target_label)

    margin_x = int(round((x2 - x1) * crop_margin))
    margin_y = int(round((y2 - y1) * crop_margin))
    focus_x1 = max(0, x1 - margin_x)
    focus_y1 = max(0, y1 - margin_y)
    focus_x2 = min(width, x2 + margin_x)
    focus_y2 = min(height, y2 + margin_y)
    focus = context[focus_y1:focus_y2, focus_x1:focus_x2].copy()

    if points:
        point_x = [item[0] for item in points]
        point_y = [item[1] for item in points]
        road_x1 = min(point_x) - int(round(road_crop_margin * width))
        road_x2 = max(point_x) + int(round(road_crop_margin * width))
        road_y1 = min(point_y) - int(round(0.35 * height))
        road_y2 = max(point_y) + int(round(0.12 * height))
    else:
        road_x1 = x1 - int(round(road_crop_margin * width))
        road_x2 = x2 + int(round(road_crop_margin * width))
        road_y1 = y1 - int(round(0.15 * height))
        road_y2 = y2 + int(round(0.12 * height))
    road_x1, road_x2 = _expanded_axis(
        road_x1,
        road_x2,
        int(round(0.55 * width)),
        width,
    )
    road_y1, road_y2 = _expanded_axis(
        road_y1,
        road_y2,
        int(round(0.40 * height)),
        height,
    )
    road = context[road_y1:road_y2, road_x1:road_x2].copy()
    _draw_view_label(road, "TARGET ROAD AREA")

    control_y2 = max(1, min(height, int(round(control_crop_bottom * height))))
    half_overlap = control_crop_overlap / 2.0
    left_x2 = max(1, min(width, int(round((0.5 + half_overlap) * width))))
    right_x1 = max(0, min(width - 1, int(round((0.5 - half_overlap) * width))))
    control_left = context[0:control_y2, 0:left_x2].copy()
    control_right = context[0:control_y2, right_x1:width].copy()
    _draw_view_label(control_left, "CONTROL SEARCH LEFT")
    _draw_view_label(control_right, "CONTROL SEARCH RIGHT")

    return {
        "context": _resize(context, maximum_dimension),
        "focus": _resize(focus, maximum_dimension),
        "road": _resize(road, maximum_dimension),
        "trajectory": _resize(trajectory, maximum_dimension),
        "control_left": _resize(control_left, maximum_dimension),
        "control_right": _resize(control_right, maximum_dimension),
    }


class EvidenceBuilder:
    """Create chronological multi-scale evidence for one crossing person."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.sample_positions = [float(value) for value in settings["sample_positions"]]
        self.context_seconds = float(settings.get("context_seconds", 0.50))
        self.crop_margin = float(settings.get("crop_margin", 0.75))
        self.max_dimension = int(settings.get("max_dimension", 1280))
        self.jpeg_quality = int(settings.get("jpeg_quality", 90))
        self.trajectory_enabled = bool(settings.get("trajectory_enabled", True))
        self.road_crop_margin = float(settings.get("road_crop_margin", 0.12))
        self.control_crop_bottom = float(settings.get("control_crop_bottom", 0.78))
        self.control_crop_overlap = float(settings.get("control_crop_overlap", 0.20))

    def build(
        self,
        video_path: str | Path,
        event: CrossingEvent,
        observations: list[TrackObservation],
        output_root: str | Path,
        fps: float,
    ) -> list[EvidenceImage]:
        """Save crossing-centred evidence views for one event."""

        import cv2

        source = Path(video_path).resolve()
        context_frames = max(0, int(round(self.context_seconds * max(float(fps), 1.0))))
        evidence_start = max(event.start_frame, event.transition_start_frame - context_frames)
        evidence_end = min(event.end_frame, event.transition_end_frame + context_frames)
        event_dir = (
            Path(output_root).resolve()
            / source.stem
            / (
                f"person_{event.person_id}_transition_"
                f"{event.transition_start_frame}_{event.transition_end_frame}"
            )
        )
        event_dir.mkdir(parents=True, exist_ok=True)

        target_track = sorted(
            [
                item
                for item in observations
                if item.class_id == 0
                and item.track_id == event.person_id
                and evidence_start <= item.frame_index <= evidence_end
            ],
            key=lambda item: item.frame_index,
        )
        if not target_track:
            raise RuntimeError(f"No observations found for person {event.person_id}")
        trajectory_boxes = [item.box for item in target_track]

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for evidence generation: {source}")

        generated: list[EvidenceImage] = []
        try:
            for position in self.sample_positions:
                requested = int(round(evidence_start + position * (evidence_end - evidence_start)))
                target = min(target_track, key=lambda item: abs(item.frame_index - requested))
                capture.set(cv2.CAP_PROP_POS_FRAMES, target.frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not decode frame {target.frame_index} from {source}")

                views = make_evidence_views(
                    frame,
                    target.box,
                    trajectory_boxes,
                    f"PERSON {event.person_id}",
                    crop_margin=self.crop_margin,
                    road_crop_margin=self.road_crop_margin,
                    control_crop_bottom=self.control_crop_bottom,
                    control_crop_overlap=self.control_crop_overlap,
                    maximum_dimension=self.max_dimension,
                    trajectory_enabled=self.trajectory_enabled,
                )
                paths = {
                    name: event_dir / f"frame_{target.frame_index:06d}_{name}.jpg"
                    for name in views
                }
                parameters = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                for name, view in views.items():
                    if not cv2.imwrite(str(paths[name]), view, parameters):
                        raise RuntimeError(f"Could not save evidence image: {paths[name]}")

                generated.append(
                    EvidenceImage(
                        frame_index=target.frame_index,
                        context_path=paths["context"],
                        focus_path=paths["focus"],
                        road_path=paths["road"],
                        trajectory_path=paths["trajectory"],
                        control_left_path=paths["control_left"],
                        control_right_path=paths["control_right"],
                    )
                )
        finally:
            capture.release()

        return generated
