"""Tests for the approach review flags."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from scripts.review.approach_review import (
    approach_frame_indices,
    find_claims,
    run_approach_review,
)

SETTINGS = {"seconds": 6.0, "end_seconds": 0.5, "frames": 4, "crop_bottom": 0.12}


class _VLM:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def evaluate_approach(self, evidence, prompt):
        self.calls.append([item.frame_index for item in evidence])
        return {"zebra_crossing": self.answers.pop(0), "evidence_summary": "stripes ahead"}


def _decision(person_id, label, start_frame):
    return {"person_id": person_id, "label": label, "event": {"transition_start_frame": start_frame}}


def _write_video(path: Path, frames: int = 120) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for number in range(frames):
        writer.write(np.full((48, 64, 3), number % 255, dtype=np.uint8))
    writer.release()


def _write_details(root: Path, name: str, video: Path, decisions) -> None:
    folder = root / "details"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_key": name,
        "segment": {"start_second": 10},
        "fps": 10.0,
        "result": {"video_path": str(video), "person_decisions": decisions},
    }
    (folder / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


class ApproachFrameTests(unittest.TestCase):
    def test_frames_run_from_six_seconds_to_half_a_second_before_the_crossing(self) -> None:
        self.assertEqual(approach_frame_indices(100, 10.0, 6.0, 0.5, 4), [40, 58, 77, 95])

    def test_window_never_starts_before_the_first_frame(self) -> None:
        self.assertEqual(approach_frame_indices(50, 10.0, 6.0, 0.5, 4), [0, 15, 30, 45])
        # A crossing at the very start has nothing before it: one frame is all there is.
        self.assertEqual(approach_frame_indices(3, 10.0, 6.0, 0.5, 4), [0])
        self.assertEqual(approach_frame_indices(100, 10.0, 6.0, 0.5, 1), [95])


@unittest.skipUnless(cv2 is not None, "OpenCV is not installed in this test environment")
class ApproachReviewTests(unittest.TestCase):
    def test_claims_are_flagged_for_review_and_nothing_else_changes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.mp4"
            _write_video(video)
            decisions = [
                _decision(1, "JAYWALKING", 100),
                _decision(2, "COMPLIANT", 80),
                _decision(3, "JAYWALKING", 50),
            ]
            _write_details(root, "crowd_a", video, decisions)
            _write_details(root, "crowd_gone", root / "missing.mp4", [_decision(4, "JAYWALKING", 90)])
            before = (root / "details" / "crowd_a.json").read_text(encoding="utf-8")

            self.assertEqual([claim.person_id for claim in find_claims(root)], [1, 3, 4])
            vlm = _VLM("YES", "NO")
            summary = run_approach_review(vlm, root, SETTINGS)

            # Only the claims are reviewed, with the planned frames; a missing video is not asked.
            self.assertEqual(vlm.calls, [[40, 58, 77, 95], [0, 15, 30, 45]])
            self.assertEqual(summary["flags"], {"ZEBRA_AHEAD": 1, "CLEAR": 1, "NO_VIDEO": 1})
            self.assertEqual(summary["to_review"], ["./crowd_a/1"])
            rows = {row["person_id"]: row for row in json.loads("[" + ",".join(
                (root / "approach_review" / "flags.jsonl").read_text(encoding="utf-8").splitlines()) + "]")}
            self.assertEqual((rows[1]["video_time_s"], rows[1]["window_s"], rows[1]["frames"]), (20.0, 6.0, 4))
            self.assertEqual(rows[4]["flag"], "NO_VIDEO")
            self.assertTrue((root / "approach_review" / "flags.csv").is_file())
            self.assertTrue(any((root / "approach_review" / "frames").glob("*_road.jpg")))
            # Claims and decisions are not touched.
            self.assertEqual((root / "details" / "crowd_a.json").read_text(encoding="utf-8"), before)

            # A rerun skips reviewed claims; the review output itself is never read as a run.
            again = _VLM()
            self.assertEqual(run_approach_review(again, root, SETTINGS)["claims"], 3)
            self.assertEqual(again.calls, [])

    def test_unclear_answers_are_listed_and_unexpected_ones_stop_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.mp4"
            _write_video(video)
            _write_details(root, "crowd_a", video, [_decision(1, "JAYWALKING", 100), _decision(2, "JAYWALKING", 60)])
            summary = run_approach_review(_VLM("UNCERTAIN", "NO"), root, SETTINGS)
            self.assertEqual(summary["to_review"], ["./crowd_a/1"])
            self.assertEqual(summary["flags"], {"UNCLEAR": 1, "CLEAR": 1})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "clip.mp4"
            _write_video(video)
            _write_details(root, "crowd_a", video, [_decision(1, "JAYWALKING", 100)])
            with self.assertRaises(ValueError):
                run_approach_review(_VLM("MAYBE"), root, SETTINGS)


if __name__ == "__main__":
    unittest.main()
