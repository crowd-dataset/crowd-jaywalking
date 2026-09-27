"""Apply the frozen JAAD classifier to arbitrary YOLO person tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .crossing import CrossingDetectionResult, CrossingDetector
from .crossing_classifier import CrossingClassifier
from .models import (
    CrossingClassification,
    CrossingEvent,
    CrossingFeatures,
    RejectionReason,
    TrackObservation,
)
from .track_features import TrackFeatureExtractor, person_tracks


@dataclass(frozen=True)
class ModelCrossingDetectionResult:
    """Crossing events and scores produced by the frozen model."""

    valid_events: list[CrossingEvent]
    rejected_events: list[CrossingEvent]
    classifications: list[CrossingClassification]


class ModelCrossingDetector:
    """Score every sufficiently long person track with the frozen classifier.

    An optional gate must also accept a track before it counts as a crossing.
    """

    def __init__(
        self,
        classifier: CrossingClassifier,
        crossing_settings: dict[str, Any],
        min_track_frames: int = 5,
        gate: Any | None = None,
    ) -> None:
        if min_track_frames < 1:
            raise ValueError("min_track_frames must be positive")
        self.classifier = classifier
        self.gate = gate
        self.rule_detector = CrossingDetector(crossing_settings)
        self.extractor = TrackFeatureExtractor(crossing_settings)
        self.min_track_frames = int(min_track_frames)

    def detect(
        self,
        observations: list[TrackObservation],
        fps: float,
    ) -> ModelCrossingDetectionResult:
        tracks = {
            track_id: track
            for track_id, track in person_tracks(observations).items()
            if len(track) >= self.min_track_frames
        }
        if not tracks:
            return ModelCrossingDetectionResult([], [], [])

        rule_result = self.rule_detector.detect(observations, fps)
        rule_events = self._rule_events_by_person(rule_result)
        ordered_ids = sorted(tracks)
        rows = [
            self.extractor.extract(tracks[track_id], observations, fps)
            for track_id in ordered_ids
        ]
        probabilities = self.classifier.predict_probabilities(rows)
        gate_probabilities = (
            self.gate.predict_probabilities(rows)
            if self.gate is not None
            else [None] * len(rows)
        )

        accepted: list[CrossingEvent] = []
        rejected: list[CrossingEvent] = []
        classifications: list[CrossingClassification] = []
        for track_id, row, probability, gate_probability in zip(
            ordered_ids, rows, probabilities, gate_probabilities
        ):
            first_stage = float(probability) >= self.classifier.threshold
            predicted = first_stage and (
                gate_probability is None or float(gate_probability) >= self.gate.threshold
            )
            rule_event = rule_events.get(track_id)
            event = self._event(
                track_id,
                tracks[track_id],
                row,
                predicted,
                rule_event,
                RejectionReason.GATE_NEGATIVE
                if first_stage
                else RejectionReason.CLASSIFIER_NEGATIVE,
            )
            rule_outcome = self._rule_outcome(rule_event)
            classification = CrossingClassification(
                person_id=track_id,
                probability=round(float(probability), 8),
                threshold=round(float(self.classifier.threshold), 8),
                predicted_crossing=predicted,
                rule_outcome=rule_outcome,
                event=event,
                track_features=row,
                gate_probability=(
                    None if gate_probability is None else round(float(gate_probability), 8)
                ),
                gate_threshold=(
                    None if self.gate is None else round(float(self.gate.threshold), 8)
                ),
                gate_precision_tier=(
                    None if gate_probability is None else self.gate.tier(float(gate_probability))
                ),
            )
            classifications.append(classification)
            (accepted if predicted else rejected).append(event)

        accepted.sort(key=lambda item: (item.start_frame, item.person_id))
        rejected.sort(key=lambda item: (item.start_frame, item.person_id))
        classifications.sort(key=lambda item: (item.event.start_frame, item.person_id))
        return ModelCrossingDetectionResult(accepted, rejected, classifications)

    @staticmethod
    def _rule_events_by_person(
        result: CrossingDetectionResult,
    ) -> dict[int, CrossingEvent]:
        events: dict[int, CrossingEvent] = {}
        for event in result.rejected_events + result.valid_events:
            previous = events.get(event.person_id)
            rank = (event.valid, event.features.track_frames)
            previous_rank = (
                (previous.valid, previous.features.track_frames)
                if previous is not None
                else (False, -1)
            )
            if rank > previous_rank:
                events[event.person_id] = event
        return events

    @staticmethod
    def _rule_outcome(event: CrossingEvent | None) -> str:
        if event is None:
            return "NO_RULE_CANDIDATE"
        return "ACCEPTED" if event.valid else event.rejection_reason.value

    def _event(
        self,
        person_id: int,
        track: list[TrackObservation],
        row: dict[str, Any],
        predicted: bool,
        rule_event: CrossingEvent | None,
        negative_reason: RejectionReason = RejectionReason.CLASSIFIER_NEGATIVE,
    ) -> CrossingEvent:
        transition_start, transition_end = self._transition_frames(track, rule_event)
        duration = max(
            1,
            int(row["matched_track_end_frame"])
            - int(row["matched_track_start_frame"])
            + 1,
        )
        features = CrossingFeatures(
            track_frames=int(row["matched_track_frames"]),
            road_frames=int(row["matched_track_road_frames"]),
            x_range=float(row["matched_track_x_range"]),
            x_speed_per_frame=float(row["matched_track_x_range"]) / duration,
            y_gross_motion=float(row["matched_track_gross_y_motion"]),
            median_width=float(row["matched_track_median_width"]),
            median_height=float(row["matched_track_median_height"]),
            static_shared_frames=int(row["matched_track_static_shared_frames"]),
            static_x_range=float(row["matched_track_static_x_range"]),
            relative_x_range=float(row["matched_track_relative_x_range"]),
            camera_motion_ratio=float(row["matched_track_camera_motion_ratio"]),
        )
        return CrossingEvent(
            person_id=person_id,
            start_frame=track[0].frame_index,
            end_frame=track[-1].frame_index,
            transition_start_frame=transition_start,
            transition_end_frame=transition_end,
            valid=predicted,
            rejection_reason=RejectionReason.NONE if predicted else negative_reason,
            features=features,
        )

    def _transition_frames(
        self,
        track: list[TrackObservation],
        rule_event: CrossingEvent | None,
    ) -> tuple[int, int]:
        if rule_event is not None:
            return (
                rule_event.transition_start_frame,
                rule_event.transition_end_frame,
            )

        states = self.rule_detector.track_states(track)
        road_indices = [index for index, state in enumerate(states) if state == "ROAD"]
        if road_indices:
            start = max(0, road_indices[0] - 1)
            end = min(len(track) - 1, road_indices[-1] + 1)
            return track[start].frame_index, track[end].frame_index

        closest = min(
            range(len(track)),
            key=lambda index: self._corridor_distance(track[index]),
        )
        start = max(0, closest - 1)
        end = min(len(track) - 1, closest + 1)
        return track[start].frame_index, track[end].frame_index

    def _corridor_distance(self, observation: TrackObservation) -> float:
        left, right = self.rule_detector.corridor_bounds(observation)
        centre = (left + right) / 2.0
        return abs(observation.box.centre_x - centre)
