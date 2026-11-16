"""Tests for the keypoint step that follows the detection of crossings without a marking."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.crowd.run_keypoints import (
    KEYPOINTS,
    Person,
    best_pose,
    crop_window,
    frame_measures,
    iou,
    people_of_run,
    person_boxes,
    person_summary,
    runs_of,
)


def _pose(**points: tuple[float, float]) -> list[tuple[float, float, float]]:
    """17 keypoints, all unseen except the named ones."""

    return [(points[name][0], points[name][1], 0.9) if name in points else (0.0, 0.0, 0.0) for name in KEYPOINTS]


SIDE_ON_FACE = _pose(nose=(52, 10), left_eye=(50, 8), right_eye=(54, 8), left_shoulder=(48, 30),
                     right_shoulder=(56, 30), left_hip=(49, 70), right_hip=(55, 70))
SIDE_ON_BACK = _pose(left_ear=(50, 9), left_shoulder=(48, 30), right_shoulder=(56, 30),
                     left_hip=(49, 70), right_hip=(55, 70))
FRONTAL_FACE = _pose(nose=(50, 10), left_eye=(45, 8), right_eye=(55, 8), left_shoulder=(30, 30),
                     right_shoulder=(70, 30), left_hip=(38, 70), right_hip=(62, 70))


def _person(**changes) -> Person:
    values = dict(run="run", video_key="crowd_abc_s000010_e000100", video_id="abc", start_second=10,
                  person_id=7, label="JAYWALKING", country="Bangladesh", locality="Sylhet", fps=10.0,
                  start_frame=100, end_frame=119, road_start_frame=110, road_end_frame=114, track_file=None)
    values.update(changes)
    return Person(**values)


class GeometryTests(unittest.TestCase):
    def test_iou(self) -> None:
        self.assertAlmostEqual(iou((0, 0, 2, 2), (0, 0, 2, 2)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 2, 2), (1, 0, 3, 2)), 1 / 3)
        self.assertEqual(iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_crop_window_adds_a_margin_and_stays_in_the_frame(self) -> None:
        self.assertEqual(crop_window((0.4, 0.4, 0.6, 0.6), 100, 100, margin=0.5), (30, 30, 71, 71))
        self.assertEqual(crop_window((0.0, 0.0, 0.2, 0.2), 100, 100, margin=0.5), (0, 0, 31, 31))

    def test_best_pose_takes_the_largest_overlap_above_the_minimum(self) -> None:
        target = (0, 0, 10, 10)
        self.assertEqual(best_pose(target, [(20, 20, 30, 30), (1, 1, 10, 10), (0, 0, 10, 10)], 0.3)[0], 2)
        self.assertIsNone(best_pose(target, [(8, 8, 20, 20)], 0.3))


class MeasureTests(unittest.TestCase):
    def test_a_face_turned_to_the_camera_while_walking_across_is_a_glance(self) -> None:
        measures = frame_measures(SIDE_ON_FACE, 0.5)
        self.assertTrue(measures["face_to_camera"])
        self.assertTrue(measures["side_on"])
        self.assertTrue(measures["glance"])

    def test_a_side_view_without_the_face_is_no_glance(self) -> None:
        measures = frame_measures(SIDE_ON_BACK, 0.5)
        self.assertFalse(measures["face_to_camera"])
        self.assertTrue(measures["side_on"])
        self.assertFalse(measures["glance"])

    def test_a_person_facing_the_camera_is_not_side_on(self) -> None:
        measures = frame_measures(FRONTAL_FACE, 0.5)
        self.assertTrue(measures["face_to_camera"])
        self.assertFalse(measures["side_on"])
        self.assertFalse(measures["glance"])
        self.assertAlmostEqual(measures["head_yaw"], 0.0)

    def test_runs_of(self) -> None:
        self.assertEqual(runs_of([True, True, False, True, True, True], 2), [(0, 1), (3, 5)])
        self.assertEqual(runs_of([True, False, True], 2), [])

    def test_person_summary_counts_glances_before_the_road(self) -> None:
        person = _person()
        rows = []
        for frame in range(100, 120):
            glance = frame in (103, 104, 105, 116, 117)
            rows.append({"frame": frame, "pose_found": frame != 119, "face_to_camera": glance,
                         "side_on": True, "glance": glance, "head_yaw": 0.2})
        summary = person_summary(person, rows, min_glance_seconds=0.2)
        self.assertEqual(summary["track_frames"], 20)
        self.assertEqual(summary["pose_frames"], 19)
        self.assertEqual(summary["glances"], 2)
        self.assertIs(summary["glanced_before_road"], True)
        self.assertEqual(summary["first_glance_seconds_before_road"], "0.70")
        self.assertEqual(summary["glance_share"], f"{5 / 19:.3f}")


class InputTests(unittest.TestCase):
    def test_people_of_run_keeps_the_chosen_labels(self) -> None:
        event = {"start_frame": 100, "end_frame": 119, "transition_start_frame": 110, "transition_end_frame": 114}
        payload = {
            "video_key": "crowd_abc_s000010_e000100",
            "segment": {"video_id": "abc", "start_second": 10,
                        "metadata": {"country": "Bangladesh", "locality": "Sylhet"}},
            "fps": 25.0,
            "trim": {"source_fps": 25.0},
            "precomputed_tracks": "/x/abc_10_25.csv",
            "result": {"person_decisions": [
                {"person_id": 7, "label": "JAYWALKING", "event": event},
                {"person_id": 8, "label": "COMPLIANT", "event": event},
            ]},
        }
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            (run / "details").mkdir(parents=True)
            (run / "details" / "crowd_abc_s000010_e000100.json").write_text(json.dumps(payload))
            people = people_of_run(run, {"JAYWALKING"})
            both = people_of_run(run, {"JAYWALKING", "COMPLIANT"})
        self.assertEqual([p.person_id for p in people], [7])
        self.assertEqual(people[0].country, "Bangladesh")
        self.assertEqual(people[0].road_start_frame, 110)
        self.assertEqual(len(both), 2)

    def test_person_boxes_reads_the_person_within_the_track_and_counts_frames_from_zero(self) -> None:
        lines = ["yolo-id,x-center,y-center,width,height,unique-id,confidence,frame-count",
                 "0,0.5,0.5,0.2,0.4,7,0.9,101",   # frame index 100
                 "0,0.5,0.5,0.2,0.4,7,0.9,150",   # after the track
                 "0,0.5,0.5,0.2,0.4,8,0.9,101",   # another person
                 "2,0.5,0.5,0.2,0.4,7,0.9,102"]   # a car with the same id
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "abc_10_25.csv"
            path.write_text("\n".join(lines) + "\n")
            boxes = person_boxes(path, _person())
        self.assertEqual(list(boxes), [100])
        for got, expected in zip(boxes[100], (0.4, 0.3, 0.6, 0.7)):
            self.assertAlmostEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
