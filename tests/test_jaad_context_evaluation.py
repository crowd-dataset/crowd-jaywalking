"""Tests for manual JAAD context label validation and metrics."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from crowd_jaywalking.jaad_context_evaluation import (
    JAADContextBenchmark,
    _macro_metrics,
    _normalise_ternary,
    _normalise_visibility,
    _permission_present,
)


class JAADContextEvaluationTests(unittest.TestCase):
    def test_normalises_manual_labels(self) -> None:
        self.assertEqual(_normalise_ternary("yes"), "YES")
        self.assertEqual(_normalise_ternary("Not Sure"), "UNCERTAIN")
        self.assertEqual(_normalise_visibility("partial"), "PARTIAL")

    def test_macro_metrics_reports_accuracy(self) -> None:
        result = _macro_metrics(
            ["YES", "NO", "UNCERTAIN"],
            ["YES", "YES", "UNCERTAIN"],
            ("YES", "NO", "UNCERTAIN"),
        )
        self.assertAlmostEqual(result["accuracy_percent"], 200.0 / 3.0)
        self.assertGreater(result["macro_f1_percent"], 0.0)
        self.assertEqual(result["per_class"]["YES"]["support"], 1)
        self.assertEqual(result["confusion_matrix"]["NO"]["YES"], 1)
        self.assertIn("macro_f1_ci95_percent", result)

    def test_macro_metrics_do_not_penalise_absent_classes(self) -> None:
        result = _macro_metrics(
            ["NO", "NO"],
            ["NO", "NO"],
            ("YES", "NO", "UNCERTAIN"),
        )
        self.assertEqual(result["observed_classes"], 1)
        self.assertEqual(result["macro_f1_percent"], 100.0)

    def test_permission_present_is_derived_without_legal_judgement(self) -> None:
        row = {
            "ground_truth_marked_crosswalk": "NO",
            "ground_truth_permissive_pedestrian_signal": "YES",
            "ground_truth_authorised_crossing_sign": "NO",
            "ground_truth_crossing_guard_permission": "NO",
        }
        self.assertEqual(_permission_present(row, "ground_truth_"), "YES")

        row["ground_truth_permissive_pedestrian_signal"] = "NO"
        row["ground_truth_marked_crosswalk"] = "UNCERTAIN"
        self.assertEqual(_permission_present(row, "ground_truth_"), "UNCERTAIN")

    def test_read_csv_removes_utf8_bom_from_first_header(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            path.write_text(
                "video_id,jaad_pedestrian_id\nvideo_0001,0_1_1b\n",
                encoding="utf-8-sig",
            )

            rows = JAADContextBenchmark._read_csv(path)

        self.assertEqual(rows[0]["video_id"], "video_0001")
        self.assertNotIn("\ufeffvideo_id", rows[0])

    def test_evidence_loader_requires_and_returns_version_five_views(self) -> None:
        with TemporaryDirectory() as directory:
            annotation_dir = Path(directory)
            evidence_dir = annotation_dir / "evidence" / "video_0001" / "0_1_1b"
            evidence_dir.mkdir(parents=True)
            for suffix in (
                "context",
                "focus",
                "road",
                "trajectory",
                "control_left",
                "control_right",
            ):
                (evidence_dir / f"frame_000012_{suffix}.jpg").touch()
            benchmark = object.__new__(JAADContextBenchmark)
            benchmark.annotation_dir = annotation_dir

            evidence = benchmark._evidence(
                {"evidence_directory": "evidence/video_0001/0_1_1b"}
            )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].frame_index, 12)
        self.assertEqual(evidence[0].road_path.name, "frame_000012_road.jpg")
        self.assertEqual(
            evidence[0].trajectory_path.name,
            "frame_000012_trajectory.jpg",
        )
        self.assertEqual(
            evidence[0].control_left_path.name,
            "frame_000012_control_left.jpg",
        )
        self.assertEqual(
            evidence[0].control_right_path.name,
            "frame_000012_control_right.jpg",
        )


if __name__ == "__main__":
    unittest.main()
