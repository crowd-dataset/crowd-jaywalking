"""Tests for reading precomputed CROWD tracks."""

import tempfile
import unittest
from pathlib import Path

from scripts.crowd.crowd_tracks import (
    find_local_track_file,
    is_track_file_for,
    load_crowd_tracks,
    parse_track_filename,
)

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
            # CROWD counts frames from 1; the segment video's frames start at 0.
            self.assertEqual((first.frame_index, first.track_id, first.class_id), (2, 7, 0))
            self.assertAlmostEqual(first.box.x1, 0.45)
            self.assertAlmostEqual(first.box.y2, 0.70)
            self.assertEqual(len(load_crowd_tracks(path, min_confidence=0.5)), 2)
            # Rows past the end of the cut segment video are dropped.
            self.assertEqual([o.frame_index for o in load_crowd_tracks(path, max_frames=3)], [2, 2])

    def test_finds_the_segment_file_without_knowing_the_frame_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            bbox = Path(folder) / "bbox"
            bbox.mkdir()
            for name in ("a_b_10_30.csv", "a_b_10_25.csv", "a_b_100_30.csv", "x_a_b_10_30.csv"):
                (bbox / name).write_text(CSV, encoding="utf-8")
            (bbox / "a_b_20_30.csv").write_text("", encoding="utf-8")  # empty files are skipped
            missing = Path(folder) / "absent"

            self.assertEqual(find_local_track_file("a_b", 10, [missing, bbox], fps=25.0).name, "a_b_10_25.csv")
            self.assertEqual(find_local_track_file("a_b", 10, [bbox], fps=29.97).name, "a_b_10_25.csv")
            self.assertEqual(find_local_track_file("a_b", 10, [bbox], fps=30.0).name, "a_b_10_30.csv")
            self.assertIsNone(find_local_track_file("a_b", 20, [bbox]))
            self.assertIsNone(find_local_track_file("a_b", 1, [bbox]))
            self.assertFalse(is_track_file_for("a_b_10_30.mp4", "a_b", 10))


if __name__ == "__main__":
    unittest.main()
