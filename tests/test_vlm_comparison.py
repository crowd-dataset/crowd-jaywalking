"""Tests for deterministic VLM model comparison."""

import unittest

from crowd_jaywalking.vlm_comparison import (
    candidate_slug,
    comparison_metrics,
    model_slug,
    select_candidate,
    select_model,
)


def benchmark_summary(context_f1: float, permission_f1: float | None = None) -> dict:
    fields = (
        "marked_crosswalk",
        "permissive_pedestrian_signal",
        "authorised_crossing_sign",
        "crossing_guard_permission",
        "prohibitive_pedestrian_signal",
        "visibility",
    )
    return {
        "evaluated_events": 20,
        "context_field_metrics": {
            field: {
                "accuracy_percent": context_f1 + 5.0,
                "macro_f1_percent": context_f1,
                "observed_classes": 2,
            }
            for field in fields
        },
        "derived_metrics": {
            "permission_present": {
                "accuracy_percent": context_f1 + 5.0,
                "macro_f1_percent": (
                    context_f1 if permission_f1 is None else permission_f1
                ),
            }
        },
    }


class VLMComparisonTests(unittest.TestCase):
    def test_comparison_metrics_averages_context_fields(self) -> None:
        result = comparison_metrics(benchmark_summary(70.0))
        self.assertEqual(result["evaluated_events"], 20)
        self.assertAlmostEqual(result["mean_context_macro_f1_percent"], 70.0)
        self.assertAlmostEqual(result["minimum_field_macro_f1_percent"], 70.0)

    def test_selects_stronger_context_model(self) -> None:
        weaker_context = {
            "model_id": "model-a",
            **comparison_metrics(benchmark_summary(70.0)),
        }
        stronger_context = {
            "model_id": "model-b",
            **comparison_metrics(benchmark_summary(75.0)),
        }
        self.assertEqual(select_model([weaker_context, stronger_context]), "model-b")

    def test_selection_prioritises_permission_present_f1(self) -> None:
        stronger_atomic_fields = {
            "model_id": "model-a",
            **comparison_metrics(benchmark_summary(80.0, permission_f1=60.0)),
        }
        stronger_permission = {
            "model_id": "model-b",
            **comparison_metrics(benchmark_summary(70.0, permission_f1=75.0)),
        }
        self.assertEqual(
            select_model([stronger_atomic_fields, stronger_permission]),
            "model-b",
        )

    def test_requires_equal_sample_counts(self) -> None:
        first = {
            "model_id": "model-a",
            **comparison_metrics(benchmark_summary(70.0)),
        }
        second = dict(first, model_id="model-b", evaluated_events=19)
        with self.assertRaises(ValueError):
            select_model([first, second])

    def test_selects_model_and_prompt_as_one_candidate(self) -> None:
        baseline = {
            "model_id": "model-a",
            "prompt_mode": "baseline_v3",
            **comparison_metrics(benchmark_summary(70.0, permission_f1=75.0)),
        }
        focused = {
            "model_id": "model-a",
            "prompt_mode": "focused_v5",
            **comparison_metrics(benchmark_summary(75.0, permission_f1=80.0)),
        }
        self.assertEqual(
            select_candidate([baseline, focused]),
            ("model-a", "focused_v5"),
        )

    def test_exact_tie_prefers_the_simpler_baseline(self) -> None:
        baseline = {
            "model_id": "model-a",
            "prompt_mode": "baseline_v3",
            **comparison_metrics(benchmark_summary(70.0)),
        }
        focused = dict(baseline, prompt_mode="focused_v5")
        self.assertEqual(
            select_candidate([baseline, focused]),
            ("model-a", "baseline_v3"),
        )

    def test_model_slug_is_readable_and_collision_resistant(self) -> None:
        first = model_slug("owner/model-a")
        second = model_slug("owner_model-a")
        self.assertTrue(first.startswith("owner_model_a_"))
        self.assertNotEqual(first, second)
        self.assertTrue(candidate_slug("owner/model-a", "focused_v5").endswith("__focused_v5"))


if __name__ == "__main__":
    unittest.main()
