"""Tests for the vendored crowd-city crossing detector adapter."""

import unittest

import numpy as np

from crowd_jaywalking.crowd_city_crossing import CrowdCityCrossingDetector
from crowd_jaywalking.models import BoundingBox, TrackObservation


def person(track_id, xs, height=0.3, confidence=0.9):
    return [
        TrackObservation(frame, track_id, 0, confidence, BoundingBox(x - 0.03, 0.8 - height, x + 0.03, 0.8))
        for frame, x in enumerate(xs)
    ]


class CrowdCityCrossingTests(unittest.TestCase):
    def test_track_across_the_strip_is_proposed_with_its_passage(self):
        result = CrowdCityCrossingDetector().detect(person(4, np.linspace(0.15, 0.85, 90)), 30.0)
        self.assertEqual([event.person_id for event in result.valid_events], [4])
        event = result.valid_events[0]
        self.assertTrue(event.start_frame <= event.transition_start_frame < event.transition_end_frame <= event.end_frame)

    def test_track_on_one_side_and_low_confidence_boxes_are_ignored(self):
        detector = CrowdCityCrossingDetector()
        self.assertEqual(detector.detect(person(1, np.linspace(0.05, 0.40, 90)), 30.0).valid_events, [])
        faint = person(2, np.linspace(0.15, 0.85, 90), confidence=0.5)
        self.assertEqual(detector.detect(faint, 30.0).valid_events, [])


if __name__ == "__main__":
    unittest.main()
