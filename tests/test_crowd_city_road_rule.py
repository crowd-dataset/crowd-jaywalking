"""Tests for crowd-city's road_crossing rule applied to a clip (stage 1)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.core.models import BoundingBox, TrackObservation
from scripts.crossing.crowd_city_road_rule import CrowdCityRoadCrossingDetector

SETTINGS = {
    "segmentation_model": "stub",
    "segmentation_device": "cpu",
    "segmentation_batch_size": 4,
    "segmentation_input_width": 64,
    "segmentation_input_height": 32,
    "segmentation_min_confidence": 0.5,
    "coarse_hz": 1.0,
    "refine_hz": 4.0,
}


class StubSegmenter:
    """Labels every pixel with one Cityscapes class: 0 is road, 1 is sidewalk."""

    input_width, input_height = 64, 32

    def __init__(self, train_id):
        self.train_id = train_id

    def segment(self, frames):
        shape = (len(frames), 8, 16)
        return np.full(shape, self.train_id, dtype=np.int16), np.ones(shape, dtype=np.float32)


def write_clip(path, frames=120, fps=30):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    for index in range(frames):
        writer.write(np.full((90, 160, 3), 100 + index % 5, dtype=np.uint8))
    writer.release()


def walker(track_id=1, frames=120):
    xs = np.linspace(0.15, 0.85, frames)
    return [
        TrackObservation(frame, track_id, 0, 0.9, BoundingBox(x - 0.03, 0.5, x + 0.03, 0.8))
        for frame, x in enumerate(xs)
    ]


class RoadCrossingRuleTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.clip = Path(self.folder.name) / "clip.mp4"
        write_clip(self.clip)

    def tearDown(self):
        self.folder.cleanup()

    def test_pedestrian_on_the_road_is_accepted(self):
        detector = CrowdCityRoadCrossingDetector(SETTINGS, StubSegmenter(0))
        result = detector.detect(walker(), 30.0, self.clip)
        self.assertEqual([event.person_id for event in result.valid_events], [1])

    def test_pedestrian_on_the_pavement_is_not_a_crossing(self):
        detector = CrowdCityRoadCrossingDetector(SETTINGS, StubSegmenter(1))
        result = detector.detect(walker(), 30.0, self.clip)
        self.assertEqual(result.valid_events, [])
        self.assertEqual([event.person_id for event in result.rejected_events], [1])

    def test_without_a_clip_nothing_is_proposed(self):
        detector = CrowdCityRoadCrossingDetector(SETTINGS, StubSegmenter(0))
        self.assertEqual(detector.detect(walker(), 30.0, None).valid_events, [])


if __name__ == "__main__":
    unittest.main()
