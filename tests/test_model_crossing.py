"""Tests for frozen classifier inference on raw person tracks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from crowd_jaywalking.crowd_analysis import stratified_audit_sample
from crowd_jaywalking.model_crossing import ModelCrossingDetector
from crowd_jaywalking.models import BoundingBox, RejectionReason, TrackObservation
from crowd_jaywalking.track_features import TrackFeatureExtractor


class _FakeClassifier:
    threshold = 0.57

    @staticmethod
    def predict_probabilities(rows):
        return [0.80 if row["matched_track_id_for_test"] == 1 else 0.20 for row in rows]


class _AcceptAllClassifier:
    threshold = 0.57

    @staticmethod
    def predict_probabilities(rows):
        return [0.90 for _ in rows]


class _FakeGate:
    threshold = 0.95

    @staticmethod
    def predict_probabilities(rows):
        return [0.99 if row["matched_track_id_for_test"] == 1 else 0.10 for row in rows]

    @staticmethod
    def tier(probability):
        return 0.98 if probability >= 0.98 else None


class ModelCrossingTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = json.loads(Path("default.config").read_text(encoding="utf-8"))
        self.settings = {key: value for key, value in raw.items()}

    def test_track_feature_extractor_produces_training_schema(self) -> None:
        observations = self._track(1, [0.10, 0.48, 0.90])
        row = TrackFeatureExtractor(self.settings).extract(
            observations, observations, fps=1.0
        )
        self.assertEqual(row["matched_track_frames"], 3)
        self.assertEqual(row["matched_track_start_state"], "LEFT")
        self.assertEqual(row["matched_track_end_state"], "RIGHT")
        self.assertAlmostEqual(row["matched_track_x_range"], 0.80)

    def test_classifier_scores_tracks_without_a_rule_candidate(self) -> None:
        observations = self._track(1, [0.10, 0.12, 0.14]) + self._track(
            2, [0.80, 0.82, 0.84]
        )
        detector = ModelCrossingDetector(_FakeClassifier(), self.settings, min_track_frames=3)
        original = detector.extractor.extract

        def extract(track, all_observations, fps):
            row = original(track, all_observations, fps)
            row["matched_track_id_for_test"] = track[0].track_id
            return row

        detector.extractor.extract = extract
        result = detector.detect(observations, fps=1.0)
        self.assertEqual([event.person_id for event in result.valid_events], [1])
        self.assertEqual([event.person_id for event in result.rejected_events], [2])
        self.assertEqual(
            result.rejected_events[0].rejection_reason,
            RejectionReason.CLASSIFIER_NEGATIVE,
        )
        self.assertEqual(result.classifications[0].rule_outcome, "NO_RULE_CANDIDATE")

    def test_gate_rejects_first_stage_acceptances_it_does_not_confirm(self) -> None:
        observations = self._track(1, [0.10, 0.12, 0.14]) + self._track(
            2, [0.80, 0.82, 0.84]
        )
        detector = ModelCrossingDetector(
            _AcceptAllClassifier(), self.settings, min_track_frames=3, gate=_FakeGate()
        )
        original = detector.extractor.extract

        def extract(track, all_observations, fps):
            row = original(track, all_observations, fps)
            row["matched_track_id_for_test"] = track[0].track_id
            return row

        detector.extractor.extract = extract
        result = detector.detect(observations, fps=1.0)
        self.assertEqual([event.person_id for event in result.valid_events], [1])
        self.assertEqual(
            result.rejected_events[0].rejection_reason,
            RejectionReason.GATE_NEGATIVE,
        )
        self.assertEqual(
            [item.gate_probability for item in result.classifications], [0.99, 0.10]
        )
        self.assertEqual(
            [item.gate_precision_tier for item in result.classifications], [0.98, None]
        )

    def test_audit_sample_includes_boundary_and_confident_cases(self) -> None:
        rows = []
        for index, probability in enumerate((0.58, 0.99, 0.56, 0.01), start=1):
            rows.append(
                {
                    "video_key": f"video_{index}",
                    "person_id": index,
                    "predicted_crossing": probability >= 0.57,
                    "classifier_probability": probability,
                    "classifier_threshold": 0.57,
                }
            )
        sample = stratified_audit_sample(rows, per_stratum=1, seed=42)
        self.assertEqual(len(sample), 4)
        self.assertEqual(
            {row["audit_stratum"] for row in sample},
            {
                "crossing_near_threshold",
                "crossing_high_confidence",
                "noncrossing_near_threshold",
                "noncrossing_random",
            },
        )

    @staticmethod
    def _track(track_id: int, centres: list[float]) -> list[TrackObservation]:
        return [
            TrackObservation(
                frame_index=frame,
                track_id=track_id,
                class_id=0,
                confidence=0.90,
                box=BoundingBox(x - 0.02, 0.30, x + 0.02, 0.70),
            )
            for frame, x in enumerate(centres)
        ]


if __name__ == "__main__":
    unittest.main()
