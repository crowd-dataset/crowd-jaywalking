"""Tests for CROWD mapping, secrets, downloads, and segment preparation."""

import json
import tempfile
import unittest
from unittest import mock

import common
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from scripts.crowd.crowd_source import (
    AuthenticationError,
    CrowdVideoDownloader,
    ftp_credentials,
    extract_video_segment,
    load_crowd_mapping,
)


class _FakeResponse:
    def __init__(self, url: str, status_code: int, body: bytes) -> None:
        self.url = url
        self.status_code = status_code
        self.body = body
        self.headers = {"content-length": str(len(body))}
        self.text = body.decode("utf-8", errors="replace")
        self.closed = False

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code))

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, responses: dict[str, tuple[int, bytes]]) -> None:
        self.responses = responses
        self.auth = None
        self.headers: dict[str, str] = {}
        self.requests: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get(self, url: str, **kwargs):
        del kwargs
        self.requests.append(url)
        status, body = self.responses.get(url, (404, b""))
        return _FakeResponse(url, status, body)


class CrowdSourceTests(unittest.TestCase):
    def test_credentials_come_from_the_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "secret").write_text(
                json.dumps({"ftp_username": "active", "ftp_password": "private", "ftp_token": "token"}),
                encoding="utf-8",
            )
            with mock.patch.object(common, "root_dir", str(root)):
                credentials = ftp_credentials()
            (root / "secret").write_text(json.dumps({"ftp_username": "", "ftp_password": ""}), encoding="utf-8")
            with mock.patch.object(common, "root_dir", str(root)), self.assertRaises(ValueError):
                ftp_credentials()

        self.assertEqual((credentials.username, credentials.password, credentials.token), ("active", "private", "token"))

    def test_loads_nested_crowd_mapping(self) -> None:
        mapping = (
            "iso3,city,videos,start_time,end_time,time_of_day\n"
            'NLD,Eindhoven,"[abc,-def]","[[0, 10], [20]]",'
            '"[[5, 15], [30]]","[[\'day\', \'night\'], [\'day\']]"\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.csv"
            path.write_text(mapping, encoding="utf-8")
            segments = load_crowd_mapping(path)

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].video_id, "abc")
        self.assertEqual(segments[0].start_second, 0)
        self.assertEqual(segments[1].time_of_day, "night")
        self.assertEqual(segments[2].video_id, "-def")
        self.assertEqual(segments[2].metadata["iso3"], "NLD")

    def test_direct_download_uses_basic_auth_and_atomic_destination(self) -> None:
        url = "https://files.example/v/tue4/files/example.mp4"
        session = _FakeSession({url: (200, b"video-bytes")})
        downloader = CrowdVideoDownloader(
            base_url="https://files.example/",
            username="user",
            password="password",
            token=None,
            aliases=["tue4", "tue5"],
            timeout_seconds=20,
            max_pages=10,
            session_factory=lambda: session,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = downloader.download("example", Path(directory))
            content = result.path.read_bytes()
            partial_exists = (Path(directory) / "example.mp4.part").exists()

        self.assertEqual(session.auth, ("user", "password"))
        self.assertEqual(content, b"video-bytes")
        self.assertFalse(partial_exists)
        self.assertEqual(result.source, "ftp_direct")
        self.assertTrue(result.downloaded_this_run)

    def test_authentication_error_does_not_include_credentials(self) -> None:
        url = "https://files.example/v/tue4/files/example.mp4"
        session = _FakeSession({url: (401, b"")})
        downloader = CrowdVideoDownloader(
            base_url="https://files.example/",
            username="sensitive-user",
            password="sensitive-password",
            token=None,
            aliases=["tue4"],
            timeout_seconds=20,
            max_pages=10,
            session_factory=lambda: session,
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AuthenticationError) as caught:
                downloader.download("example", Path(directory))

        message = str(caught.exception)
        self.assertNotIn("sensitive-user", message)
        self.assertNotIn("sensitive-password", message)

    def test_download_uses_browse_fallback(self) -> None:
        direct = "https://files.example/v/tue4/files/example.mp4"
        browse = "https://files.example/v/tue4/browse"
        nested = "https://files.example/v/tue4/browse/folder"
        file_url = "https://files.example/v/tue4/files/folder/example.mp4"
        session = _FakeSession(
            {
                direct: (404, b""),
                browse: (200, b'<a href="/v/tue4/browse/folder">folder</a>'),
                nested: (
                    200,
                    b'<a href="/v/tue4/files/folder/example.mp4">example.mp4</a>',
                ),
                file_url: (200, b"video-from-crawl"),
            }
        )
        downloader = CrowdVideoDownloader(
            base_url="https://files.example/",
            username="user",
            password="password",
            token=None,
            aliases=["tue4"],
            timeout_seconds=20,
            max_pages=10,
            session_factory=lambda: session,
        )

        with tempfile.TemporaryDirectory() as directory:
            result = downloader.download("example", Path(directory))
            content = result.path.read_bytes()

        self.assertEqual(content, b"video-from-crawl")
        self.assertEqual(result.source, "ftp_crawl")
        self.assertIn(nested, session.requests)

    @staticmethod
    def _track_downloader(session: "_FakeSession") -> CrowdVideoDownloader:
        return CrowdVideoDownloader(
            base_url="https://files.example/",
            username="user",
            password="password",
            token=None,
            aliases=["tue4"],
            timeout_seconds=20,
            max_pages=10,
            session_factory=lambda: session,
        )

    def test_track_file_download_tries_the_frame_rate_names_in_the_bbox_folder(self) -> None:
        folder = "https://files.example/v/alam/files/pedestrians_in-youtube/data/bbox"
        session = _FakeSession({f"{folder}/abc_12_30.csv": (200, b"tracks")})
        downloader = self._track_downloader(session)

        with tempfile.TemporaryDirectory() as directory:
            path = downloader.download_track_file(
                "abc", 12, 29.97, "/pedestrians_in-youtube/data/bbox/", ["tue4", "alam"], Path(directory)
            )
            content = path.read_bytes()

        self.assertEqual((path.name, content), ("abc_12_30.csv", b"tracks"))
        # int(29.97) is tried before round(29.97), on every alias in turn.
        self.assertEqual(
            session.requests[:3],
            [
                "https://files.example/v/tue4/files/pedestrians_in-youtube/data/bbox/abc_12_29.csv",
                "https://files.example/v/tue4/files/pedestrians_in-youtube/data/bbox/abc_12_30.csv",
                f"{folder}/abc_12_29.csv",
            ],
        )

    def test_track_file_download_lists_the_bbox_folder_once(self) -> None:
        browse = "https://files.example/v/tue4/browse/data/bbox"
        listing = (
            b"<a href='/v/tue4/browse/data'>..</a>"
            b'<a href="/v/tue4/files/data/bbox/abc_12_25.csv">a</a>'
            b'<a href="/v/tue4/files/data/bbox/abc_120_25.csv">b</a>'
            b'<a href="/v/tue4/files/data/bbox/xabc_12_25.csv">c</a>'
        )
        session = _FakeSession(
            {
                browse: (200, listing),
                "https://files.example/v/tue4/files/data/bbox/abc_12_25.csv": (200, b"tracks"),
            }
        )
        downloader = self._track_downloader(session)

        with tempfile.TemporaryDirectory() as directory:
            found = downloader.download_track_file("abc", 12, 30.0, "data/bbox", ["tue4"], Path(directory))
            missing = downloader.download_track_file("abc", 13, 30.0, "data/bbox", ["tue4"], Path(directory))
            self.assertEqual(found.read_bytes(), b"tracks")

        self.assertEqual(found.name, "abc_12_25.csv")
        self.assertIsNone(missing)
        self.assertEqual(session.requests.count(browse), 1)
        # The link back to the parent folder is never followed.
        self.assertNotIn("https://files.example/v/tue4/browse/data", session.requests)

    def test_transient_failures_are_retried_but_missing_files_are_not(self) -> None:
        url = "https://files.example/v/tue4/files/example.mp4"
        session = _FakeSession({url: (200, b"video-bytes")})
        original_get = session.get
        failures = iter([True, False])

        def flaky_get(request_url: str, **kwargs):
            if request_url == url and next(failures, False):
                session.requests.append(request_url)
                raise ConnectionError("reset by peer")
            return original_get(request_url, **kwargs)

        session.get = flaky_get
        downloader = self._track_downloader(session)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.crowd.crowd_source.RETRY_DELAY_SECONDS", 0
        ):
            result = downloader.download("example", Path(directory))
            self.assertEqual(result.path.read_bytes(), b"video-bytes")
            self.assertEqual(session.requests.count(url), 2)

            before = len(session.requests)
            self.assertIsNone(downloader.download_track_file("abc", 1, 30.0, "bbox", ["tue4"], Path(directory)))
            # A 404 on the one name and an empty listing: one pass, no retries.
            self.assertEqual(len(session.requests) - before, 2)

    def test_track_file_download_rejects_unsafe_folders(self) -> None:
        downloader = self._track_downloader(_FakeSession({}))
        with tempfile.TemporaryDirectory() as directory:
            for folder in ("", "data/../secret"):
                with self.assertRaises(ValueError):
                    downloader.download_track_file("abc", 1, 30.0, folder, ["tue4"], Path(directory))
            with self.assertRaises(ValueError):
                downloader.download_track_file("a/b", 1, 30.0, "bbox", ["tue4"], Path(directory))

    @unittest.skipUnless(cv2 is not None, "OpenCV is not installed in this test environment")
    def test_extracts_mapped_segment_and_applies_end_margin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (32, 24),
            )
            self.assertTrue(writer.isOpened())
            for frame_number in range(20):
                frame = np.full((24, 32, 3), frame_number, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            result = extract_video_segment(
                source=source,
                destination=root / "segment.mp4",
                start_second=0,
                end_second=2,
                end_margin_seconds=1.0,
            )

            self.assertTrue(result.path.is_file())
            self.assertEqual(result.output_frames, 10)
            self.assertEqual(result.effective_end_second, 1.0)


if __name__ == "__main__":
    unittest.main()
