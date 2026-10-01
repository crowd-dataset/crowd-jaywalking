"""Tests for camera compensated track motion."""

import unittest

import numpy as np

from scripts.crossing.camera_motion import CameraMotion, compensated_features
from scripts.crossing.crossing_gate import gate_matrix
from scripts.crossing.crossing_classifier import feature_matrix
from scripts.core.models import BoundingBox, TrackObservation


def track(xs):
    return [
        TrackObservation(frame, 1, 0, 0.9, BoundingBox(x - 0.02, 0.4, x + 0.02, 0.8))
        for frame, x in enumerate(xs)
    ]


class CameraMotionTests(unittest.TestCase):
    def test_turning_camera_removes_apparent_motion(self):
        # The camera pans so the whole image shifts 40 px left per frame; a standing
        # person therefore moves 40 px left in the image but not in the scene.
        width = 1000
        warps = {f: np.array([[1.0, 0.0, -40.0], [0.0, 1.0, 0.0]]) for f in range(1, 10)}
        motion = CameraMotion(width, 500, warps)
        xs = [0.9 - 0.04 * f for f in range(10)]
        features = compensated_features(track(xs), motion)
        self.assertLess(features["scene_x_range"], 0.01)
        self.assertGreater(features["camera_x_travel"], 0.3)

    def test_still_camera_keeps_real_crossing(self):
        motion = CameraMotion(1000, 500, {})
        xs = [0.1 + 0.08 * f for f in range(10)]
        features = compensated_features(track(xs), motion)
        self.assertAlmostEqual(features["scene_x_range"], 0.72, places=6)
        self.assertAlmostEqual(features["scene_to_image_x_ratio"], 1.0, places=6)

    def test_gate_matrix_places_extras_before_categoricals(self):
        row = {"matched_track_start_state": "LEFT", "matched_track_end_state": "RIGHT",
               "matched_track_complete_transition": True, "scene_x_range": 0.5}
        base = feature_matrix([row])
        matrix = gate_matrix([row], ["scene_x_range"])
        self.assertEqual(matrix.shape[1], base.shape[1] + 1)
        self.assertEqual(float(matrix[0, -4]), 0.5)
        self.assertEqual(list(matrix[0, -3:]), list(base[0, -3:]))
        self.assertEqual([str(v) for v in gate_matrix([row], [])[0]], [str(v) for v in base[0]])


if __name__ == "__main__":
    unittest.main()
