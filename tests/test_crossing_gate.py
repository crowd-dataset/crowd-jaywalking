"""Tests for the high precision crossing gate threshold selection."""

import unittest

import numpy as np

from scripts.crossing.crossing_gate import (
    GATE_ARTIFACT_TYPE,
    CrossingGate,
    gate_metrics,
    select_precision_threshold,
    tier_for,
)
from scripts.crossing.crossing_classifier import FEATURE_VERSION


class CrossingGateThresholdTests(unittest.TestCase):
    def test_selects_lowest_threshold_meeting_precision(self) -> None:
        labels = np.array([True, True, True, False, True, False])
        probabilities = np.array([0.99, 0.95, 0.90, 0.85, 0.80, 0.10])
        self.assertEqual(
            select_precision_threshold(labels, probabilities, 1.0, 1), 0.90
        )
        self.assertEqual(
            select_precision_threshold(labels, probabilities, 0.80, 1), 0.80
        )

    def test_ignores_thresholds_accepting_too_few_tracks(self) -> None:
        labels = np.array([True, False, False])
        probabilities = np.array([0.99, 0.50, 0.40])
        self.assertIsNone(select_precision_threshold(labels, probabilities, 1.0, 2))

    def test_tied_probabilities_are_accepted_together(self) -> None:
        labels = np.array([True, False, True])
        probabilities = np.array([0.90, 0.90, 0.10])
        self.assertIsNone(select_precision_threshold(labels, probabilities, 1.0, 1))

    def test_metrics_count_unverifiable_acceptances_as_wrong(self) -> None:
        metrics = gate_metrics(
            np.array([True, False, True, False]),
            np.array([True, True, False, False]),
        )
        self.assertEqual(metrics["worst_case_precision_percent"], 50.0)
        self.assertEqual(metrics["recall_percent"], 50.0)


    def test_tier_is_the_strictest_threshold_met(self) -> None:
        tiers = [(0.90, 0.75), (0.98, 0.98), (0.95, 0.95)]
        self.assertEqual(tier_for(tiers, 0.99), 0.98)
        self.assertEqual(tier_for(tiers, 0.96), 0.95)
        self.assertEqual(tier_for(tiers, 0.80), 0.90)
        self.assertIsNone(tier_for(tiers, 0.50))

    def test_gate_accepts_at_the_configured_tier_only(self) -> None:
        artifact = {
            "artifact_type": GATE_ARTIFACT_TYPE,
            "feature_version": FEATURE_VERSION,
            "tier_thresholds": {"0.98": 0.98, "0.9": 0.75},
            "pipeline": None,
        }
        self.assertEqual(CrossingGate(artifact, 0.90).threshold, 0.75)
        self.assertEqual(CrossingGate(artifact, 0.98).threshold, 0.98)
        with self.assertRaises(ValueError):
            CrossingGate(artifact, 0.95)


if __name__ == "__main__":
    unittest.main()
