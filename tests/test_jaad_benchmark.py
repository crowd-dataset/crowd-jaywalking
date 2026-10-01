"""Tests for JAAD to YOLO track matching and persisted tracking output."""

import tempfile
import unittest
from pathlib import Path

from scripts.jaad.jaad import JAADPedestrianTrack, JAADVideoAnnotations
from scripts.jaad.jaad_benchmark import (
    JAADCrossingBenchmark,
    _classification_metrics,
    box_iou,
    match_person_tracks,
)
from scripts.core.models import BoundingBox, TrackObservation
from scripts.core.tracking import load_observations_csv, save_observations_csv


def box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


class JAADBenchmarkTests(unittest.TestCase):
    def test_box_iou(self) -> None:
        self.assertAlmostEqual(box_iou(box(0.0, 0.0, 1.0, 1.0), box(0.0, 0.0, 1.0, 1.0)), 1.0)
        self.assertEqual(box_iou(box(0.0, 0.0, 0.2, 0.2), box(0.8, 0.8, 1.0, 1.0)), 0.0)

    def test_matches_independent_annotation_to_yolo_track(self) -> None:
        ground_truth_boxes = {
            frame: box(0.10 + frame * 0.05, 0.20, 0.30 + frame * 0.05, 0.80)
            for frame in range(3)
        }
        ground_truth = JAADPedestrianTrack(
            pedestrian_id="0_1_1b",
            label="pedestrian",
            frames=(0, 1, 2),
            boxes=ground_truth_boxes,
            occlusion={0: 0, 1: 0, 2: 0},
            crossing={0: False, 1: True, 2: True},
            attributes={},
        )
        annotations = JAADVideoAnnotations(
            video_id="video_0001",
            num_frames=3,
            width=100,
            height=50,
            tracks={ground_truth.pedestrian_id: ground_truth},
            traffic={},
            road_type="street",
        )
        observations = [
            TrackObservation(
                frame_index=frame,
                track_id=7,
                class_id=0,
                confidence=0.9,
                box=ground_truth_boxes[frame],
            )
            for frame in range(3)
        ]

        matches, unmatched = match_person_tracks(
            annotations,
            observations,
            iou_threshold=0.50,
            min_match_frames=2,
            min_track_coverage=0.50,
        )

        self.assertEqual(matches["0_1_1b"].track_id, 7)
        self.assertEqual(matches["0_1_1b"].coverage, 1.0)
        self.assertFalse(unmatched)

    def test_tracking_csv_round_trip(self) -> None:
        observations = [
            TrackObservation(
                frame_index=3,
                track_id=9,
                class_id=0,
                confidence=0.75,
                box=box(0.1, 0.2, 0.3, 0.8),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.csv"
            save_observations_csv(path, observations)
            loaded = load_observations_csv(path)

        self.assertEqual(loaded, observations)

    def test_single_class_metrics_are_reported_as_undefined(self) -> None:
        metrics = _classification_metrics(
            [
                {
                    "ground_truth_crossing": False,
                    "predicted_crossing": False,
                },
                {
                    "ground_truth_crossing": False,
                    "predicted_crossing": False,
                },
            ]
        )

        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["accuracy_percent"], 100.0)
        self.assertEqual(metrics["specificity_percent"], 100.0)
        self.assertIsNone(metrics["precision_percent"])
        self.assertIsNone(metrics["recall_percent"])
        self.assertIsNone(metrics["f1_percent"])
        self.assertIsNone(metrics["balanced_accuracy_percent"])

    def test_f1_is_zero_when_positive_crossings_are_missed(self) -> None:
        metrics = _classification_metrics(
            [
                {
                    "ground_truth_crossing": True,
                    "predicted_crossing": False,
                },
                {
                    "ground_truth_crossing": False,
                    "predicted_crossing": False,
                },
            ]
        )

        self.assertEqual(metrics["recall_percent"], 0.0)
        self.assertEqual(metrics["f1_percent"], 0.0)
        self.assertIsNone(metrics["precision_percent"])
        self.assertEqual(metrics["balanced_accuracy_percent"], 50.0)

    def test_track_diagnostics_include_motion_features(self) -> None:
        observations = [
            TrackObservation(
                frame_index=frame,
                track_id=7,
                class_id=0,
                confidence=0.9,
                box=box(0.10 + frame * 0.05, 0.20, 0.30 + frame * 0.05, 0.80),
            )
            for frame in range(3)
        ]

        diagnostics = JAADCrossingBenchmark._track_diagnostics(observations)

        self.assertEqual(diagnostics["matched_track_frames"], 3)
        self.assertEqual(diagnostics["matched_track_start_frame"], 0)
        self.assertEqual(diagnostics["matched_track_end_frame"], 2)
        self.assertAlmostEqual(diagnostics["matched_track_x_range"], 0.10)
        self.assertAlmostEqual(diagnostics["matched_track_net_x_displacement"], 0.10)


if __name__ == "__main__":
    unittest.main()
