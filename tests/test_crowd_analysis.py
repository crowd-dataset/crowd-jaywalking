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


if __name__ == "__main__":
    unittest.main()
