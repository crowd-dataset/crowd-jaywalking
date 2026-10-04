"""Tests for finding the precomputed CROWD tracks of a segment."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.crowd.crowd_analysis import CrowdAnalysisRunner
from scripts.crowd.crowd_source import CrowdSegment


def _runner(root: Path, ftp_folder: str = "data/bbox") -> CrowdAnalysisRunner:
    runner = object.__new__(CrowdAnalysisRunner)
    runner.settings = {
        "bbox_dirs": [root / "local"],
        "bbox_download_dir": root / "cache",
        "bbox_ftp_folder": ftp_folder,
        "bbox_ftp_aliases": ["tue4"],
    }
    runner._downloader_instance = mock.Mock()
    runner._downloader_instance.download_track_file.return_value = None
    return runner


SEGMENT = CrowdSegment(1, "abc", 12, 40, "day", {"iso3": "CAN"})


class PrecomputedTrackTests(unittest.TestCase):
    def test_local_folders_come_before_the_cache_and_the_file_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = _runner(root)
            server = runner._downloader_instance.download_track_file

            with self.assertRaises(FileNotFoundError):
                runner._precomputed_track_file(SEGMENT, 30.0)
            server.assert_called_once_with("abc", 12, 30.0, "data/bbox", ["tue4"], root / "cache")

            server.return_value = root / "cache" / "abc_12_30.csv"
            self.assertEqual(runner._precomputed_track_file(SEGMENT, 30.0)[1], "precomputed_ftp")

            for folder, source in (("cache", "precomputed_cache"), ("local", "precomputed_local")):
                (root / folder).mkdir()
                (root / folder / "abc_12_30.csv").write_text("x", encoding="utf-8")
                path, found = runner._precomputed_track_file(SEGMENT, 30.0)
                self.assertEqual((path.parent.name, found), (folder, source))
            self.assertEqual(server.call_count, 2)

    def test_no_file_server_lookup_without_a_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = _runner(Path(directory), ftp_folder="")
            with self.assertRaises(FileNotFoundError):
                runner._precomputed_track_file(SEGMENT, 30.0)
            runner._downloader_instance.download_track_file.assert_not_called()


def _selecting_runner(vehicle_types, max_segments=0):
    runner = object.__new__(CrowdAnalysisRunner)
    runner.config = mock.Mock()
    runner.config.get.return_value = ""
    runner.settings = {"vehicle_types": vehicle_types, "max_segments": max_segments}
    return runner


def _segments(*types):
    return [CrowdSegment(index, f"v{index}", 0, 10, "0", {}, vehicle) for index, vehicle in enumerate(types)]


class VehicleFilterTests(unittest.TestCase):
    def test_only_the_listed_vehicle_types_are_kept(self) -> None:
        segments = _segments(0, 4, None, 0, 1, 3)
        kept = _selecting_runner([0])._select_segments(segments)
        self.assertEqual([s.video_id for s in kept], ["v0", "v3"])
        self.assertEqual(len(_selecting_runner([0, 1])._select_segments(segments)), 3)
        # The filter comes before the segment cap, so the cap counts kept segments.
        self.assertEqual([s.video_id for s in _selecting_runner([0], 1)._select_segments(segments)], ["v0"])
        self.assertEqual([s.video_id for s in _selecting_runner([4], 1)._select_segments(segments)], ["v1"])

    def test_null_keeps_every_type_and_a_mapping_without_types_is_an_error(self) -> None:
        segments = _segments(0, 4, None)
        self.assertEqual(len(_selecting_runner(None)._select_segments(segments)), 3)
        with self.assertRaises(ValueError):
            _selecting_runner([0])._select_segments(_segments(None, None))


if __name__ == "__main__":
    unittest.main()
