"""Tests for the VLM crossing check and rescue in the pipeline."""

import unittest
from types import SimpleNamespace

from crowd_jaywalking.models import (
    ContextAssessment,
    CrossingEvent,
    CrossingFeatures,
    DecisionLabel,
    InfrastructureMeasurement,
    RejectionReason,
    Ternary,
    Visibility,
)
from crowd_jaywalking.pipeline import JaywalkingPipeline
from crowd_jaywalking.policy import JaywalkingPolicy


def _event(person_id: int, valid: bool) -> CrossingEvent:
    features = CrossingFeatures(10, 5, 0.5, 0.05, 0.1, 0.05, 0.2, 0, 0.0, 0.0, 0.0)
    return CrossingEvent(
        person_id, 0, 9, 3, 6, valid,
        RejectionReason.NONE if valid else RejectionReason.CLASSIFIER_NEGATIVE, features,
    )


class _Classifier:
    def __init__(self, crossing_answers):
        self.crossing_answers = crossing_answers
        self.classified = []

    def ensure_ready(self):
        pass

    def confirm_crossing(self, evidence):
        return self.crossing_answers[evidence], "summary"

    def classify(self, evidence):
        self.classified.append(evidence)
        return ContextAssessment(
            marked_crosswalk=Ternary.NO,
            permissive_pedestrian_signal=None,
            authorised_crossing_sign=Ternary.NO,
            crossing_guard_permission=None,
            prohibitive_pedestrian_signal=None,
            visibility=Visibility.CLEAR,
            evidence_summary="none",
            traffic_light=Ternary.NO,
        )


class CrossingCheckPipelineTests(unittest.TestCase):
    def _pipeline(self, answers, rescue):
        gated = _event(1, True)
        rescued = _event(2, False)
        weak = _event(3, False)
        classification = lambda event, probability, gate: SimpleNamespace(
            event=event,
            predicted_crossing=event.valid,
            probability=probability,
            gate_probability=gate,
        )
        detection = SimpleNamespace(
            valid_events=[gated],
            rejected_events=[rescued, weak],
            classifications=[
                classification(gated, 0.9, 0.99),
                classification(rescued, 0.30, 0.99),
                # Gate is confident but the first stage is below the rescue floor.
                classification(weak, 0.10, 0.99),
            ],
        )
        pipeline = object.__new__(JaywalkingPipeline)
        pipeline.crossing_detector = SimpleNamespace(detect=lambda observations, fps: detection)
        pipeline.evidence_builder = SimpleNamespace(
            build=lambda video_path, event, observations, output_root, fps: event.person_id
        )
        pipeline.context_classifier = _Classifier(answers)
        pipeline._context_ready = True
        pipeline._vlm_check = True
        pipeline._rescue = rescue
        pipeline.segmenter = None
        pipeline.policy = JaywalkingPolicy({"strict_absence": True, "context_scope": "scene"})
        return pipeline

    def test_only_confirmed_candidates_reach_the_context_question(self):
        pipeline = self._pipeline({1: Ternary.NO, 2: Ternary.YES, 3: Ternary.YES}, (0.25, 0.98))
        result = pipeline.process_observations("video.mp4", "evidence", 30.0, [])
        self.assertEqual([item.person_id for item in result.person_decisions], [2])
        self.assertEqual(
            [(item.person_id, item.source, item.answer) for item in result.crossing_checks],
            [(1, "gate", Ternary.NO), (2, "rescue", Ternary.YES)],
        )
        self.assertEqual(result.prediction, DecisionLabel.JAYWALKING)

    def test_segmentation_veto_blocks_the_claim(self):
        pipeline = self._pipeline({1: Ternary.YES}, None)
        pipeline.segmenter = SimpleNamespace(
            measure=lambda source, event, observations, evidence: InfrastructureMeasurement(
                0, 0, 250, 6, True
            )
        )
        result = pipeline.process_observations("video.mp4", "evidence", 30.0, [])
        self.assertEqual(result.person_decisions[0].label, DecisionLabel.UNCERTAIN)
        self.assertIn("Segmentation", result.person_decisions[0].reason)
        self.assertEqual(result.person_decisions[0].segmentation.traffic_light_px, 250)

    def test_uncertain_crossing_check_is_not_accepted(self):
        pipeline = self._pipeline({1: Ternary.UNCERTAIN}, None)
        result = pipeline.process_observations("video.mp4", "evidence", 30.0, [])
        self.assertEqual(result.person_decisions, [])
        self.assertEqual(result.prediction, DecisionLabel.COMPLIANT)


if __name__ == "__main__":
    unittest.main()
