"""Tests for strict structured VLM output validation."""

import json
from pathlib import Path
import unittest

from crowd_jaywalking.models import EvidenceImage, Ternary, Visibility
from crowd_jaywalking.vlm import (
    AUTHORISATION_PROMPT,
    BASELINE_CONTEXT_PROMPT,
    CONTEXT_PROMPT,
    FOCUSED_PROMPT_VERSION,
    PEDESTRIAN_SIGNAL_PROMPT,
    PROMPT_VERSION,
    ROAD_MARKING_PROMPT,
    HuggingFaceContextClassifier,
    VLMError,
    _local_image_reference,
    _sample_evidence,
    _view_paths,
    normalise_prompt_mode,
    prompt_version_for_mode,
)


def valid_payload() -> dict[str, str]:
    return {
        "marked_crosswalk": "NO",
        "permissive_pedestrian_signal": "UNCERTAIN",
        "authorised_crossing_sign": "NO",
        "crossing_guard_permission": "NO",
        "prohibitive_pedestrian_signal": "NO",
        "visibility": "PARTIAL",
        "evidence_summary": "The target is visible but the signal is distant.",
    }


class VLMValidationTests(unittest.TestCase):
    def test_version_five_separates_context_tasks(self) -> None:
        self.assertEqual(PROMPT_VERSION, FOCUSED_PROMPT_VERSION)
        self.assertEqual(PROMPT_VERSION, "global-context-v5")
        self.assertIn("TARGET ROAD AREA", ROAD_MARKING_PROMPT)
        self.assertIn("vehicle traffic lamps", PEDESTRIAN_SIGNAL_PROMPT)
        self.assertIn("paint on the road are never signs", AUTHORISATION_PROMPT)
        self.assertIn("authorised_crossing_sign", CONTEXT_PROMPT)

    def test_baseline_version_three_remains_selectable(self) -> None:
        self.assertEqual(normalise_prompt_mode(" BASELINE_V3 "), "baseline_v3")
        self.assertEqual(prompt_version_for_mode("baseline_v3"), "global-context-v3")
        self.assertIn("Return one JSON object", BASELINE_CONTEXT_PROMPT)
        with self.assertRaises(VLMError):
            normalise_prompt_mode("unknown")

    def test_even_evidence_sampling_preserves_endpoints(self) -> None:
        evidence = [
            EvidenceImage(index, Path(f"context-{index}"), Path(f"focus-{index}"))
            for index in range(6)
        ]
        selected = _sample_evidence(evidence, 4)
        self.assertEqual([item.frame_index for item in selected], [0, 2, 3, 5])

    def test_uses_specialised_views_when_available(self) -> None:
        item = EvidenceImage(
            1,
            Path("context.jpg"),
            Path("focus.jpg"),
            road_path=Path("road.jpg"),
            trajectory_path=Path("trajectory.jpg"),
            control_left_path=Path("left.jpg"),
            control_right_path=Path("right.jpg"),
        )
        selected = _view_paths(
            item,
            ("full scene", "trajectory map", "control search left"),
        )
        self.assertEqual(
            [path.name for _, path in selected],
            ["context.jpg", "trajectory.jpg", "left.jpg"],
        )

    def test_local_image_reference_is_a_native_absolute_path(self) -> None:
        reference = _local_image_reference(Path(__file__))
        self.assertTrue(Path(reference).is_absolute())
        self.assertFalse(reference.startswith("file:"))

    def test_recognises_supported_model_families(self) -> None:
        self.assertEqual(
            HuggingFaceContextClassifier._model_family(
                "Qwen/Qwen3-VL-8B-Instruct"
            ),
            "qwen",
        )
        self.assertEqual(
            HuggingFaceContextClassifier._model_family("google/gemma-4-12B-it"),
            "gemma",
        )

    def test_rejects_unsupported_model_family(self) -> None:
        with self.assertRaises(VLMError):
            HuggingFaceContextClassifier._model_family("text-only/model")

    def test_accepts_exact_json_schema(self) -> None:
        result = HuggingFaceContextClassifier._validate_response(
            json.dumps(valid_payload())
        )
        self.assertEqual(result.marked_crosswalk, Ternary.NO)
        self.assertEqual(result.permissive_pedestrian_signal, Ternary.UNCERTAIN)
        self.assertEqual(result.visibility, Visibility.PARTIAL)

    def test_accepts_markdown_json_fence(self) -> None:
        content = f"```json\n{json.dumps(valid_payload())}\n```"
        result = HuggingFaceContextClassifier._validate_response(content)
        self.assertEqual(result.authorised_crossing_sign, Ternary.NO)

    def test_rejects_missing_or_extra_keys(self) -> None:
        payload = valid_payload()
        payload["decision"] = "JAYWALKING"
        with self.assertRaises(VLMError):
            HuggingFaceContextClassifier._validate_response(json.dumps(payload))

    def test_rejects_mutually_exclusive_signal_states(self) -> None:
        payload = valid_payload()
        payload["permissive_pedestrian_signal"] = "YES"
        payload["prohibitive_pedestrian_signal"] = "YES"
        with self.assertRaisesRegex(VLMError, "mutually exclusive"):
            HuggingFaceContextClassifier._validate_response(json.dumps(payload))

    def test_control_visibility_no_forces_both_signal_states_to_no(self) -> None:
        payload = {
            "dedicated_pedestrian_signal_visible": "NO",
            "permissive_pedestrian_signal": "YES",
            "authorised_crossing_sign": "NO",
            "crossing_guard_permission": "NO",
            "prohibitive_pedestrian_signal": "UNCERTAIN",
            "evidence_summary": "Only circular vehicle lamps are visible.",
        }
        result = HuggingFaceContextClassifier._validate_control_response(
            json.dumps(payload)
        )
        self.assertEqual(result["permissive_pedestrian_signal"], Ternary.NO)
        self.assertEqual(result["prohibitive_pedestrian_signal"], Ternary.NO)

    def test_focused_response_schemas_are_strict(self) -> None:
        road = HuggingFaceContextClassifier._validate_road_response(
            json.dumps(
                {
                    "marked_crosswalk": "YES",
                    "evidence_summary": "Stripes intersect the path.",
                }
            )
        )
        self.assertEqual(road["marked_crosswalk"], Ternary.YES)
        with self.assertRaises(VLMError):
            HuggingFaceContextClassifier._validate_visibility_response(
                json.dumps(
                    {
                        "visibility": "CLEAR",
                        "extra": "not allowed",
                        "evidence_summary": "Clear view.",
                    }
                )
            )

    def test_authorisation_schema_is_independent_from_road_markings(self) -> None:
        result = HuggingFaceContextClassifier._validate_authorisation_response(
            json.dumps(
                {
                    "authorised_crossing_sign": "NO",
                    "crossing_guard_permission": "NO",
                    "evidence_summary": "Only painted zebra stripes are visible.",
                }
            )
        )
        self.assertEqual(result["authorised_crossing_sign"], Ternary.NO)


if __name__ == "__main__":
    unittest.main()
