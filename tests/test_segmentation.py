"""Tests for the segmentation infrastructure measurement."""

import unittest

import numpy as np

from crowd_jaywalking.models import BoundingBox
from crowd_jaywalking.segmentation import measure_segmentation

THRESHOLDS = {
    "max_crosswalk_near_path_px": 50,
    "max_crosswalk_road_px": 1000,
    "max_traffic_light_px": 100,
}
PATH = [BoundingBox(0.45, 0.40, 0.50, 0.60), BoundingBox(0.50, 0.40, 0.55, 0.60)]


class SegmentationMeasurementTests(unittest.TestCase):
    def test_clean_scene_is_not_blocked(self):
        seg = np.full((100, 200), 13)  # road everywhere
        result = measure_segmentation([seg], PATH, THRESHOLDS)
        self.assertFalse(result.infrastructure_found)

    def test_crosswalk_at_the_feet_blocks(self):
        seg = np.full((100, 200), 13)
        seg[58:64, 90:112] = 23  # crosswalk paint under the foot points
        result = measure_segmentation([seg], PATH, THRESHOLDS)
        self.assertGreaterEqual(result.crosswalk_near_path_px, 50)
        self.assertTrue(result.infrastructure_found)

    def test_traffic_light_anywhere_blocks(self):
        seg = np.full((100, 200), 13)
        seg[5:20, 5:15] = 48
        self.assertTrue(measure_segmentation([seg], PATH, THRESHOLDS).infrastructure_found)

    def test_largest_frame_counts(self):
        clean = np.full((100, 200), 13)
        busy = clean.copy()
        busy[5:20, 5:15] = 48
        result = measure_segmentation([clean, busy], PATH, THRESHOLDS)
        self.assertEqual(result.traffic_light_px, 150)
        self.assertEqual(result.frames, 2)


if __name__ == "__main__":
    unittest.main()
