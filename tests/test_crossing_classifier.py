"""Tests for leakage free supervised crossing classification."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from config_helpers import load_config, template
from scripts.crossing.crossing_classifier import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    JAADCrossingClassifierEvaluator,
    JAADCrossingClassifierTrainer,
    NUMERIC_FEATURES,
    CrossingClassifier,
    feature_record,
    select_threshold,
)


class CrossingClassifierTests(unittest.TestCase):
    def test_feature_schema_excludes_annotation_and_matching_leakage(self) -> None:
        forbidden = {
            "video_id",
            "pedestrian_id",
            "ground_truth_crossing",
            "visible_ground_truth_frames",
            "matched_iou_frames",
            "track_coverage",
            "mean_matched_iou",
            "candidate_outcome",
            "ground_truth_start_frame",
            "ground_truth_end_frame",
            "correct",
        }
        self.assertTrue(forbidden.isdisjoint(ALL_FEATURES))

    def test_derived_ratios_are_calculated_from_track_features(self) -> None:
        row = {name: 0 for name in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
        row.update(
            {
                "matched_track_frames": 80,
                "matched_track_duration_frames": 100,
                "matched_track_road_frames": 40,
                "matched_track_longest_road_run": 20,
                "matched_track_start_state": "left",
                "matched_track_end_state": "road",
                "matched_track_complete_transition": True,
            }
        )
        features = feature_record(row)
        self.assertEqual(features["derived_track_observation_ratio"], 0.80)
        self.assertEqual(features["derived_road_frame_ratio"], 0.50)
        self.assertEqual(features["derived_longest_road_run_ratio"], 0.25)
        self.assertEqual(features["matched_track_start_state"], "LEFT")
        self.assertEqual(features["matched_track_complete_transition"], "TRUE")

    def test_threshold_selection_respects_precision_constraint(self) -> None:
        selection = select_threshold(
            [True, True, False, False],
            [0.90, 0.60, 0.70, 0.20],
            minimum_precision=1.0,
            step=0.10,
        )
        self.assertTrue(selection["precision_constraint_satisfied"])
        self.assertAlmostEqual(selection["threshold"], 0.80)
        self.assertEqual(selection["metrics"]["precision_percent"], 100.0)
        self.assertEqual(selection["metrics"]["recall_percent"], 50.0)

    def test_training_writes_loadable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "classifier"
            model_path = output / "crossing_classifier.joblib"
            train_rows = []
            for group_index in range(6):
                train_rows.append(self._row(f"video_{group_index:04d}", True, 0.80))
                train_rows.append(self._row(f"video_{group_index:04d}", False, 0.08))
            validation_rows = [
                self._row("video_0100", True, 0.75),
                self._row("video_0100", False, 0.05),
                self._row("video_0101", True, 0.70),
                self._row("video_0101", False, 0.10),
            ]
            self._write_rows(benchmark / "train" / "per_pedestrian.csv", train_rows)
            self._write_rows(benchmark / "val" / "per_pedestrian.csv", validation_rows)
            self._write_rows(benchmark / "test" / "per_pedestrian.csv", validation_rows)
            (benchmark / "val" / "summary.json").write_text(
                json.dumps(
                    {
                        "end_to_end_crossing_metrics": {
                            "accuracy_percent": 50.0,
                            "precision_percent": 50.0,
                            "recall_percent": 50.0,
                            "specificity_percent": 50.0,
                            "balanced_accuracy_percent": 50.0,
                            "f1_percent": 50.0,
                        }
                    }
                ),
                encoding="utf-8",
            )

            raw = template()
            raw.update(
                {
                    "jaad_benchmark_results": str(benchmark),
                    "crossing_classifier_results": str(output),
                    "crossing_classifier_model": str(model_path),
                    "crossing_classifier_cv_folds": 3,
                    "crossing_classifier_logistic_c_values": [1.0],
                    "crossing_classifier_gradient_learning_rates": [0.10],
                    "crossing_classifier_gradient_max_leaf_nodes": [3],
                }
            )
            config = load_config(raw, root=root)
            summary = JAADCrossingClassifierTrainer(config).run()

            classifier = CrossingClassifier.load(model_path)
            predictions = classifier.predict(validation_rows)
            test_summary = JAADCrossingClassifierEvaluator(config).run()
            self.assertEqual(predictions.tolist(), [True, False, True, False])
            self.assertEqual(summary["validation_rows"], 4)
            self.assertEqual(test_summary["end_to_end_crossing_metrics"]["tp"], 2)
            self.assertTrue((output / "model_selection.csv").is_file())
            self.assertTrue((output / "validation_predictions.csv").is_file())
            self.assertTrue((output / "test_predictions.csv").is_file())

    @staticmethod
    def _row(video_id: str, crossing: bool, x_range: float) -> dict[str, object]:
        row: dict[str, object] = {
            "video_id": video_id,
            "pedestrian_id": f"{video_id}_{'yes' if crossing else 'no'}",
            "ground_truth_crossing": crossing,
            "track_matched": True,
            "matched_track_id": 1 if crossing else 2,
            "predicted_crossing": False,
        }
        row.update({name: 0.0 for name in NUMERIC_FEATURES})
        row.update(
            {
                "matched_track_frames": 60,
                "matched_track_duration_frames": 60,
                "matched_track_duration_seconds": 2.0,
                "matched_track_longest_segment_frames": 60,
                "matched_track_x_range": x_range,
                "matched_track_net_x_displacement": x_range,
                "matched_track_signed_x_displacement": x_range,
                "matched_track_gross_x_motion": x_range,
                "matched_track_x_direction_consistency": 1.0,
                "matched_track_road_frames": 30 if crossing else 0,
                "matched_track_longest_road_run": 30 if crossing else 0,
                "matched_track_start_state": "LEFT",
                "matched_track_end_state": "RIGHT" if crossing else "LEFT",
                "matched_track_complete_transition": crossing,
            }
        )
        return row

    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
