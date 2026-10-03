"""Tests for the deterministic jaywalking definition."""

import unittest

from scripts.core.models import ContextAssessment, DecisionLabel, Ternary, Visibility
from scripts.core.policy import JaywalkingPolicy


def context(
    *,
    crosswalk: Ternary = Ternary.NO,
    signal: Ternary = Ternary.NO,
    sign: Ternary = Ternary.NO,
    guard: Ternary = Ternary.NO,
    prohibitive: Ternary = Ternary.NO,
    visibility: Visibility = Visibility.CLEAR,
) -> ContextAssessment:
    return ContextAssessment(
        marked_crosswalk=crosswalk,
        permissive_pedestrian_signal=signal,
        authorised_crossing_sign=sign,
        crossing_guard_permission=guard,
        prohibitive_pedestrian_signal=prohibitive,
        visibility=visibility,
        evidence_summary="test",
    )


class JaywalkingPolicyTests(unittest.TestCase):
    def test_no_permission_is_jaywalking(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(context())
        self.assertEqual(label, DecisionLabel.JAYWALKING)

    def test_marked_crosswalk_is_compliant(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(context(crosswalk=Ternary.YES))
        self.assertEqual(label, DecisionLabel.COMPLIANT)

    def test_permissive_signal_is_compliant(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(context(signal=Ternary.YES))
        self.assertEqual(label, DecisionLabel.COMPLIANT)

    def test_red_signal_overrides_crosswalk_by_default(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(
            context(crosswalk=Ternary.YES, prohibitive=Ternary.YES)
        )
        self.assertEqual(label, DecisionLabel.JAYWALKING)

    def test_partial_absence_is_uncertain(self) -> None:
        label, _ = JaywalkingPolicy({}).decide(context(visibility=Visibility.PARTIAL))
        self.assertEqual(label, DecisionLabel.UNCERTAIN)

    def test_partial_absence_can_be_decided(self) -> None:
        policy = JaywalkingPolicy({"partial_visibility_uncertain": False})
        label, _ = policy.decide(context(visibility=Visibility.PARTIAL))
        self.assertEqual(label, DecisionLabel.JAYWALKING)

    def test_unselected_cue_does_not_grant_permission(self) -> None:
        policy = JaywalkingPolicy(
            {"permission_cues": ["marked_crosswalk", "permissive_pedestrian_signal"]}
        )
        label, _ = policy.decide(context(sign=Ternary.YES, guard=Ternary.UNCERTAIN))
        self.assertEqual(label, DecisionLabel.JAYWALKING)

    def test_rejects_unknown_settings(self) -> None:
        with self.assertRaises(ValueError):
            JaywalkingPolicy({"permission_cues": ["zebra"]})
        with self.assertRaises(ValueError):
            JaywalkingPolicy({"context_scope": "video"})

    def test_person_scope_decides_each_person_alone(self) -> None:
        labels = [
            label
            for label, _ in JaywalkingPolicy({}).decide_all(
                [context(crosswalk=Ternary.YES), context()]
            )
        ]
        self.assertEqual(labels, [DecisionLabel.COMPLIANT, DecisionLabel.JAYWALKING])

    def test_scene_scope_shares_permission_across_people(self) -> None:
        policy = JaywalkingPolicy({"context_scope": "scene"})
        outcomes = policy.decide_all(
            [
                context(crosswalk=Ternary.YES),
                context(),
                context(visibility=Visibility.INSUFFICIENT),
            ]
        )
        self.assertEqual(
            [label for label, _ in outcomes],
            [DecisionLabel.COMPLIANT] * 3,
        )
        self.assertIn("another crossing", outcomes[1][1])

    def test_scene_scope_keeps_person_prohibitive_signal(self) -> None:
        policy = JaywalkingPolicy({"context_scope": "scene"})
        outcomes = policy.decide_all(
            [context(crosswalk=Ternary.YES), context(prohibitive=Ternary.YES)]
        )
        self.assertEqual(outcomes[1][0], DecisionLabel.JAYWALKING)


    def test_strict_absence_blocks_positive_on_any_infrastructure_doubt(self) -> None:
        policy = JaywalkingPolicy(
            {
                "permission_cues": ["marked_crosswalk", "permissive_pedestrian_signal"],
                "prohibitive_signal_overrides_crosswalk": False,
                "strict_absence": True,
            }
        )
        for blocker in (
            context(sign=Ternary.YES),
            context(guard=Ternary.UNCERTAIN),
            context(prohibitive=Ternary.YES),
        ):
            label, _ = policy.decide(blocker)
            self.assertEqual(label, DecisionLabel.UNCERTAIN)
        label, _ = policy.decide(context())
        self.assertEqual(label, DecisionLabel.JAYWALKING)

    def test_strict_absence_checks_every_person_in_scene_scope(self) -> None:
        policy = JaywalkingPolicy({"context_scope": "scene", "strict_absence": True})
        labels = [
            label
            for label, _ in policy.decide_all([context(), context(sign=Ternary.UNCERTAIN)])
        ]
        self.assertEqual(labels, [DecisionLabel.UNCERTAIN, DecisionLabel.UNCERTAIN])
        labels = [label for label, _ in policy.decide_all([context(), context()])]
        self.assertEqual(labels, [DecisionLabel.JAYWALKING, DecisionLabel.JAYWALKING])

    def test_scope_window_only_joins_crossings_close_in_time(self) -> None:
        settings = {"context_scope": "scene", "strict_absence": True, "context_scope_window_seconds": 10}
        policy = JaywalkingPolicy(settings)
        contexts = [context(crosswalk=Ternary.YES), context(), context(), context(sign=Ternary.UNCERTAIN)]
        # Intervals in seconds: 6 s, 20 s, and 3 s apart in turn.
        times = [(100.0, 102.0), (108.0, 109.0), (129.0, 130.0), (133.0, 134.0)]
        labels = [label for label, _ in policy.decide_all(contexts, times)]
        self.assertEqual(
            labels,
            [DecisionLabel.COMPLIANT, DecisionLabel.COMPLIANT, DecisionLabel.UNCERTAIN, DecisionLabel.UNCERTAIN],
        )
        alone = [label for label, _ in policy.decide_all(contexts[:3], times[:3])]
        self.assertEqual(alone[2], DecisionLabel.JAYWALKING)
        # Without a window every crossing of the video is one scene, as before.
        whole = JaywalkingPolicy({**settings, "context_scope_window_seconds": None})
        self.assertEqual(
            [label for label, _ in whole.decide_all(contexts[:3], times[:3])], [DecisionLabel.COMPLIANT] * 3
        )
        with self.assertRaises(ValueError):
            policy.decide_all(contexts)
        with self.assertRaises(ValueError):
            JaywalkingPolicy({**settings, "context_scope_window_seconds": -1})


if __name__ == "__main__":
    unittest.main()
