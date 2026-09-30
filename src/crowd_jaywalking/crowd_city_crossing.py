"""Potential crossings from the CROWD crossing detector of crowd-city.

The detector decides from bounding boxes alone: a person track must pass from
one side of a narrow vertical strip in the middle of the image to the other and
survive crowd-city's motion, rider and camera motion filters. It only proposes
snippets; the VLM then decides whether the person really crosses the road and
whether a zebra crossing or traffic light is there.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .crowd_city_detection import CROSSING_PARAMETER_DEFAULTS, Detection
from .model_crossing import ModelCrossingDetectionResult
from .models import (
    CrossingClassification,
    CrossingEvent,
    CrossingFeatures,
    RejectionReason,
    TrackObservation,
)

# crowd-city's strip and detection filter (analysis.py and default.config).
STRIP_LEFT = 0.45
STRIP_RIGHT = 0.55
MIN_CONFIDENCE = 0.7
PERSON_CLASS = 0


def observations_frame(observations: list[TrackObservation]) -> Any:
    """The CROWD bbox table (normalised centre, width and height) for these observations."""

    import polars as pl

    return pl.DataFrame(
        {
            "yolo-id": [o.class_id for o in observations],
            "x-center": [o.box.centre_x for o in observations],
            "y-center": [o.box.centre_y for o in observations],
            "width": [o.box.width for o in observations],
            "height": [o.box.height for o in observations],
            "unique-id": [float(o.track_id) for o in observations],
            "confidence": [o.confidence for o in observations],
            "frame-count": [o.frame_index for o in observations],
        },
        schema={
            "yolo-id": pl.Int64,
            "x-center": pl.Float64,
            "y-center": pl.Float64,
            "width": pl.Float64,
            "height": pl.Float64,
            "unique-id": pl.Float64,
            "confidence": pl.Float64,
            "frame-count": pl.Int64,
        },
    )


class CrowdCityCrossingDetector:
    """Run crowd-city's ``Detection.pedestrian_crossing`` on one clip's tracks."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        settings = settings or {}
        self.min_confidence = float(settings.get("min_confidence", MIN_CONFIDENCE))
        self.parameters = {**CROSSING_PARAMETER_DEFAULTS, **settings.get("parameters", {})}
        self.detection = Detection()

    def detect(self, observations: list[TrackObservation], fps: float) -> ModelCrossingDetectionResult:
        confident = [o for o in observations if o.confidence >= self.min_confidence]
        if not any(o.class_id == PERSON_CLASS for o in confident):
            return ModelCrossingDetectionResult([], [], [])
        ids, candidate_ids, bounds = self.detection.pedestrian_crossing(
            observations_frame(confident),
            "clip",
            None,
            STRIP_LEFT,
            STRIP_RIGHT,
            PERSON_CLASS,
            fps=float(fps),
            **self.parameters,
        )
        accepted_ids = {int(float(value)) for value in ids}
        by_track: dict[int, list[TrackObservation]] = {}
        for item in confident:
            if item.class_id == PERSON_CLASS:
                by_track.setdefault(item.track_id, []).append(item)

        accepted: list[CrossingEvent] = []
        rejected: list[CrossingEvent] = []
        classifications: list[CrossingClassification] = []
        bounds_by_id = {int(float(key)): value for key, value in bounds.items()}
        for track_id in sorted({int(float(value)) for value in candidate_ids} | accepted_ids):
            rows = sorted(by_track.get(track_id, []), key=lambda o: o.frame_index)
            if track_id in bounds_by_id:
                start, end = (int(v) for v in bounds_by_id[track_id])
                rows = [o for o in rows if start <= o.frame_index <= end]
            if not rows:
                continue
            valid = track_id in accepted_ids
            event = self._event(track_id, rows, valid)
            (accepted if valid else rejected).append(event)
            classifications.append(
                CrossingClassification(
                    person_id=track_id,
                    probability=1.0 if valid else 0.0,
                    threshold=1.0,
                    predicted_crossing=valid,
                    rule_outcome="CROWD_CROSSING" if valid else "CROWD_FILTERED",
                    event=event,
                    track_features={
                        "segment_start_frame": rows[0].frame_index,
                        "segment_end_frame": rows[-1].frame_index,
                        "x_range": event.features.x_range,
                        "road_frames": event.features.road_frames,
                    },
                )
            )
        accepted.sort(key=lambda e: (e.start_frame, e.person_id))
        rejected.sort(key=lambda e: (e.start_frame, e.person_id))
        return ModelCrossingDetectionResult(accepted, rejected, classifications)

    @staticmethod
    def _event(track_id: int, rows: list[TrackObservation], valid: bool) -> CrossingEvent:
        xs = [o.box.centre_x for o in rows]
        ys = [o.box.centre_y for o in rows]
        # The passage through the middle strip centres the VLM evidence.
        strip = [o.frame_index for o in rows if STRIP_LEFT <= o.box.centre_x <= STRIP_RIGHT]
        passage = (min(strip), max(strip)) if strip else (rows[0].frame_index, rows[-1].frame_index)
        duration = max(1, rows[-1].frame_index - rows[0].frame_index + 1)
        return CrossingEvent(
            person_id=track_id,
            start_frame=rows[0].frame_index,
            end_frame=rows[-1].frame_index,
            transition_start_frame=passage[0],
            transition_end_frame=passage[1],
            valid=valid,
            rejection_reason=RejectionReason.NONE if valid else RejectionReason.INSUFFICIENT_LATERAL_MOTION,
            features=CrossingFeatures(
                track_frames=len(rows),
                road_frames=len(strip),
                x_range=round(max(xs) - min(xs), 6),
                x_speed_per_frame=(max(xs) - min(xs)) / duration,
                y_gross_motion=float(np.sum(np.abs(np.diff(ys)))) if len(ys) > 1 else 0.0,
                median_width=float(np.median([o.box.width for o in rows])),
                median_height=float(np.median([o.box.height for o in rows])),
                static_shared_frames=0,
                static_x_range=0.0,
                relative_x_range=0.0,
                camera_motion_ratio=0.0,
            ),
        )
