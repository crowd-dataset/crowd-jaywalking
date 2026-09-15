"""Tests for version 5 visual evidence generation."""

import unittest

import numpy as np

from crowd_jaywalking.evidence import make_evidence_views
from crowd_jaywalking.models import BoundingBox


class EvidenceViewTests(unittest.TestCase):
    def test_trajectory_does_not_cover_clean_context_or_road_pixels(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is not installed in this test environment")
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        boxes = [
            BoundingBox(0.20, 0.30, 0.30, 0.70),
            BoundingBox(0.45, 0.30, 0.55, 0.70),
            BoundingBox(0.70, 0.30, 0.80, 0.70),
        ]

        views = make_evidence_views(
            image,
            boxes[1],
            boxes,
            "PERSON 1",
            crop_margin=0.75,
            road_crop_margin=0.12,
            control_crop_bottom=0.78,
            control_crop_overlap=0.20,
            maximum_dimension=1280,
            trajectory_enabled=True,
        )

        def yellow_pixels(view) -> int:
            return int(
                np.count_nonzero(
                    (view[:, :, 0] == 0)
                    & (view[:, :, 1] > 150)
                    & (view[:, :, 2] > 150)
                )
            )

        self.assertEqual(yellow_pixels(views["context"]), 0)
        self.assertEqual(yellow_pixels(views["road"]), 0)
        self.assertGreater(yellow_pixels(views["trajectory"]), 0)


if __name__ == "__main__":
    unittest.main()
