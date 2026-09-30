"""Tests for reading precomputed CROWD tracks."""

import tempfile
import unittest
from pathlib import Path

from crowd_jaywalking.crowd_tracks import load_crowd_tracks, parse_track_filename

CSV = """yolo-id,x-center,y-center,width,height,unique-id,confidence,frame-count
0,0.5,0.5,0.1,0.4,7.0,0.9,3
2,0.2,0.6,0.2,0.2,8,0.4,3
0,0.52,0.5,0.1,0.4,7,0.9,4
0,0.3,0.3,0.1,0.1,,0.9,4
"""


class CrowdTrackTests(unittest.TestCase):
    def test_filename_gives_video_start_and_fps(self):
        info = parse_track_filename("some/bbox/_e0RQL7NJZg_127_30.csv")
        self.assertEqual((info.video_id, info.start_second, info.fps), ("_e0RQL7NJZg", 127, 30.0))
        with self.assertRaises(ValueError):
            parse_track_filename("bad.csv")

    def test_rows_become_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vid_0_30.csv"
            path.write_text(CSV, encoding="utf-8")
            observations = load_crowd_tracks(path)
            self.assertEqual(len(observations), 3)  # the row without a track id is skipped
            first = observations[0]
            self.assertEqual((first.frame_index, first.track_id, first.class_id), (3, 7, 0))
            self.assertAlmostEqual(first.box.x1, 0.45)
            self.assertAlmostEqual(first.box.y2, 0.70)
            self.assertEqual(len(load_crowd_tracks(path, min_confidence=0.5)), 2)


if __name__ == "__main__":
    unittest.main()
