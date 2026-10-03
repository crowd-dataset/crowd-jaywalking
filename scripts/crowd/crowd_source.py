"""CROWD mapping, authenticated source video download, and clip extraction."""

from __future__ import annotations

import ast
import csv
import math
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable
from urllib.parse import quote, urljoin, urlparse
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger

# A browse page of a large folder (the CROWD bbox folder lists about 86,000 files,
# 13 MB of HTML) takes tens of seconds to arrive.
LISTING_READ_TIMEOUT_SECONDS = 300
DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10

ACTIVE_SECRET_NAME = "secret"
DEFAULT_SECRET_NAME = "default.secret"


class CrowdSourceError(RuntimeError):
    """Base error for CROWD source preparation."""


class MappingFormatError(CrowdSourceError):
    """Raised when a CROWD mapping row cannot be interpreted safely."""


class DownloadError(CrowdSourceError):
    """Raised when a CROWD source video cannot be downloaded."""


class TransientDownloadError(DownloadError):
    """A request that failed or broke off; trying again may succeed."""


class AuthenticationError(DownloadError):
    """Raised when the CROWD file server rejects the configured credentials."""


@dataclass(frozen=True)
class FtpCredentials:
    """CROWD file server credentials from the local 'secret' file."""

    username: str
    password: str
    token: str | None = None


def ftp_credentials() -> FtpCredentials:
    """Read the file server credentials with common.get_secrets."""

    import common

    def entry(name: str) -> str:
        try:
            return str(common.get_secrets(name) or "")
        except KeyError:
            return ""
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"No '{ACTIVE_SECRET_NAME}' file; copy '{DEFAULT_SECRET_NAME}' and fill it in"
            ) from error

    credentials = FtpCredentials(
        username=entry("ftp_username").strip(),
        password=entry("ftp_password"),
        token=entry("ftp_token").strip() or None,
    )
    if not credentials.username or not credentials.password:
        raise ValueError("Set ftp_username and ftp_password in the local 'secret' file.")
    return credentials


@dataclass(frozen=True)
class CrowdSegment:
    """One mapped CROWD interval from one source video."""

    row_number: int
    video_id: str
    start_second: int
    end_second: int
    time_of_day: str
    metadata: dict[str, str]

    @property
    def video_key(self) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", self.video_id).strip("_")
        safe_id = safe_id or "video"
        return f"crowd_{safe_id}_s{self.start_second:06d}_e{self.end_second:06d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_number": self.row_number,
            "video_id": self.video_id,
            "start_second": self.start_second,
            "end_second": self.end_second,
            "time_of_day": self.time_of_day,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PreparedVideo:
    """A local source video and how it became available."""

    path: Path
    source: str
    downloaded_this_run: bool


@dataclass(frozen=True)
class SegmentVideo:
    """Metadata for one extracted CROWD clip."""

    path: Path
    source_fps: float
    output_frames: int
    requested_start_second: int
    requested_end_second: int
    effective_end_second: float


def _literal_list(value: str, field: str, row_number: int) -> list[Any]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise MappingFormatError(
            f"Mapping row {row_number} has an invalid {field} list."
        ) from error
    if not isinstance(parsed, (list, tuple)):
        raise MappingFormatError(
            f"Mapping row {row_number} field {field} must contain a list."
        )
    return list(parsed)


def _video_ids(value: str, row_number: int) -> list[str]:
    text = str(value).strip()
    if not text:
        raise MappingFormatError(f"Mapping row {row_number} has no videos.")

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple)):
        values = [str(item).strip() for item in parsed]
    else:
        values = [part.strip().strip("'\"") for part in text.strip("[]").split(",")]

    output = [item for item in values if item]
    if not output:
        raise MappingFormatError(f"Mapping row {row_number} has no valid video IDs.")
    if any("/" in item or "\\" in item or "\x00" in item for item in output):
        raise MappingFormatError(
            f"Mapping row {row_number} contains an unsafe video ID."
        )
    return output


def _per_video_lists(
    value: str,
    field: str,
    video_count: int,
    row_number: int,
) -> list[list[Any]]:
    parsed = _literal_list(str(value), field, row_number)
    if video_count == 1 and (not parsed or not isinstance(parsed[0], (list, tuple))):
        parsed = [parsed]
    if len(parsed) != video_count:
        raise MappingFormatError(
            f"Mapping row {row_number} has {video_count} videos but "
            f"{len(parsed)} {field} groups."
        )
    output: list[list[Any]] = []
    for group in parsed:
        if not isinstance(group, (list, tuple)):
            raise MappingFormatError(
                f"Mapping row {row_number} field {field} must contain one list per video."
            )
        output.append(list(group))
    return output


def load_crowd_mapping(path: Path) -> list[CrowdSegment]:
    """Read the official CROWD nested mapping format and deduplicate intervals."""

    if not path.is_file():
        raise FileNotFoundError(f"CROWD mapping file not found: {path}")

    segments: list[CrowdSegment] = []
    seen: set[tuple[str, int, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"videos", "start_time", "end_time", "time_of_day"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise MappingFormatError(
                "CROWD mapping is missing columns: " + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            videos = _video_ids(row.get("videos", ""), row_number)
            starts = _per_video_lists(
                row.get("start_time", ""), "start_time", len(videos), row_number
            )
            ends = _per_video_lists(
                row.get("end_time", ""), "end_time", len(videos), row_number
            )
            times = _per_video_lists(
                row.get("time_of_day", ""), "time_of_day", len(videos), row_number
            )
            metadata = {
                str(key): str(value).strip()
                for key, value in row.items()
                if key not in required and value is not None and str(value).strip()
            }

            for video_id, start_group, end_group, time_group in zip(
                videos, starts, ends, times
            ):
                if not (len(start_group) == len(end_group) == len(time_group)):
                    raise MappingFormatError(
                        f"Mapping row {row_number} has unequal interval list lengths "
                        f"for video {video_id}."
                    )
                for raw_start, raw_end, raw_time in zip(
                    start_group, end_group, time_group
                ):
                    try:
                        start = int(raw_start)
                        end = int(raw_end)
                    except (TypeError, ValueError) as error:
                        raise MappingFormatError(
                            f"Mapping row {row_number} has non-integer segment times."
                        ) from error
                    if start < 0 or end <= start:
                        raise MappingFormatError(
                            f"Mapping row {row_number} has invalid interval {start}-{end}."
                        )
                    time_of_day = str(raw_time).strip()
                    key = (video_id, start, end)
                    if key in seen:
                        continue
                    seen.add(key)
                    segments.append(
                        CrowdSegment(
                            row_number=row_number,
                            video_id=video_id,
                            start_second=start,
                            end_second=end,
                            time_of_day=time_of_day,
                            metadata=dict(metadata),
                        )
                    )
    return segments


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value.strip())


class CrowdVideoDownloader:
    """Download source videos from the authenticated CROWD HTTP file server."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        token: str | None,
        aliases: Iterable[str],
        timeout_seconds: int,
        max_pages: int,
        session_factory: Callable[[], Any] | None = None,
    ) -> None:
        base = str(base_url).strip()
        parsed = urlparse(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("ftp_server must be an absolute HTTP or HTTPS URL")
        if parsed.username or parsed.password:
            raise ValueError("Do not place credentials inside ftp_server")
        clean_aliases = [str(alias).strip().strip("/") for alias in aliases]
        if not clean_aliases or any(not alias for alias in clean_aliases):
            raise ValueError("crowd_ftp_aliases must contain at least one alias")
        if timeout_seconds < 1 or max_pages < 1:
            raise ValueError("Download timeout and crawl page limit must be positive")

        self.base_url = base if base.endswith("/") else base + "/"
        self.origin = (parsed.scheme.lower(), parsed.netloc.lower())
        self.username = username
        self.password = password
        self.token = token
        self.aliases = tuple(clean_aliases)
        self.timeout_seconds = timeout_seconds
        self.max_pages = max_pages
        if session_factory is None:
            import requests

            session_factory = requests.Session
        self.session_factory = session_factory
        self._listings: dict[tuple[str, ...], list[str]] = {}

    def download(self, video_id: str, output_directory: Path) -> PreparedVideo:
        return self._retrying(lambda: self._download_once(video_id, output_directory))

    def _download_once(self, video_id: str, output_directory: Path) -> PreparedVideo:
        filename = video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"
        if (
            filename != Path(filename).name
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
        ):
            raise ValueError("CROWD video ID must not contain path separators")
        destination = output_directory / filename
        if destination.is_file() and destination.stat().st_size > 0:
            return PreparedVideo(destination.resolve(), "download_cache", False)

        output_directory.mkdir(parents=True, exist_ok=True)
        encoded_name = quote(filename, safe="._-")
        with self.session_factory() as session:
            if self.username and self.password:
                session.auth = (self.username, self.password)
            session.headers.update({"User-Agent": "crowd-jaywalking/1.8"})

            for alias in self.aliases:
                url = urljoin(
                    self.base_url,
                    f"v/{quote(alias, safe='_-')}/files/{encoded_name}",
                )
                response = self._get(session, url, stream=True, allow_not_found=True)
                if response is None:
                    continue
                try:
                    self._save_response(response, destination)
                finally:
                    response.close()
                return PreparedVideo(destination.resolve(), "ftp_direct", True)

            target = filename.lower()
            found = next(
                iter(
                    self._crawl_files(
                        session,
                        [urljoin(self.base_url, f"v/{quote(alias, safe='_-')}/browse") for alias in self.aliases],
                        lambda name: name.lower() == target,
                        first_only=True,
                    )
                ),
                None,
            )
            if found is not None:
                response = self._get(session, found, stream=True, allow_not_found=False)
                if response is None:
                    raise DownloadError(
                        f"CROWD video disappeared before download: {filename}"
                    )
                try:
                    self._save_response(response, destination)
                finally:
                    response.close()
                return PreparedVideo(destination.resolve(), "ftp_crawl", True)

        raise DownloadError(f"CROWD video was not found on the file server: {filename}")

    def _get(
        self,
        session: Any,
        url: str,
        stream: bool,
        allow_not_found: bool,
        read_timeout: int | None = None,
    ) -> Any | None:
        self._require_same_origin(url)
        try:
            response = session.get(
                url,
                timeout=(
                    self.timeout_seconds
                    if read_timeout is None
                    else (self.timeout_seconds, max(self.timeout_seconds, read_timeout))
                ),
                params={"token": self.token} if self.token else None,
                stream=stream,
            )
        except Exception as error:
            raise TransientDownloadError("CROWD file server request failed") from error

        if response.status_code in {401, 403}:
            response.close()
            raise AuthenticationError(
                "CROWD file server authentication failed. Check ftp_username and "
                "ftp_password in the local 'secret' file."
            )
        if response.status_code == 404 and allow_not_found:
            response.close()
            return None
        try:
            response.raise_for_status()
        except Exception as error:
            response.close()
            raise DownloadError(
                f"CROWD file server returned HTTP {response.status_code}."
            ) from error
        if not self._same_origin(response.url):
            response.close()
            raise DownloadError("CROWD file server redirected to an untrusted origin")
        return response

    def download_track_file(self, *args: Any, **kwargs: Any) -> Path | None:
        return self._retrying(lambda: self._download_track_file_once(*args, **kwargs))

    def _download_track_file_once(
        self,
        video_id: str,
        start_second: int,
        fps: float,
        remote_folder: str,
        aliases: Iterable[str],
        output_directory: Path,
    ) -> Path | None:
        """Download the precomputed tracks ``<video_id>_<start>_<fps>.csv`` of one segment.

        ``remote_folder`` is the ``bbox`` folder below each alias on the file server.
        The names with the video's truncated and rounded frame rate are tried first;
        otherwise the folder's browse pages are listed once and searched. Returns None
        when the segment has no track file on the server.
        """

        from scripts.crowd.crowd_tracks import is_track_file_for

        if any(separator in video_id for separator in ("/", "\\", "\x00")):
            raise ValueError("CROWD video ID must not contain path separators")
        folder = "/".join(part for part in str(remote_folder).replace("\\", "/").split("/") if part)
        if not folder or any(part == ".." for part in folder.split("/")):
            raise ValueError("crowd_bbox_ftp_folder must be a relative folder on the file server")
        clean_aliases = [str(alias).strip().strip("/") for alias in aliases]
        if not clean_aliases or any(not alias for alias in clean_aliases):
            raise ValueError("crowd_bbox_ftp_aliases must contain at least one alias")
        names = list(dict.fromkeys(f"{video_id}_{start_second}_{value}.csv" for value in (int(fps), round(fps))))
        output_directory.mkdir(parents=True, exist_ok=True)
        encoded_folder = quote(folder, safe="/._-")

        with self.session_factory() as session:
            if self.username and self.password:
                session.auth = (self.username, self.password)
            session.headers.update({"User-Agent": "crowd-jaywalking/1.8"})

            for alias in clean_aliases:
                for name in names:
                    url = urljoin(
                        self.base_url,
                        f"v/{quote(alias, safe='_-')}/files/{encoded_folder}/{quote(name, safe='._-')}",
                    )
                    response = self._get(session, url, stream=True, allow_not_found=True)
                    if response is None:
                        continue
                    destination = output_directory / name
                    try:
                        self._save_response(response, destination)
                    finally:
                        response.close()
                    return destination.resolve()

            starts = tuple(
                urljoin(self.base_url, f"v/{quote(alias, safe='_-')}/browse/{encoded_folder}")
                for alias in clean_aliases
            )
            if starts not in self._listings:
                # One listing of the bbox folder serves every later segment of the run.
                self._listings[starts] = self._crawl_files(
                    session, list(starts), lambda name: name.lower().endswith(".csv")
                )
            for url in self._listings[starts]:
                name = PurePosixPath(urlparse(url).path).name
                if not is_track_file_for(name, video_id, start_second):
                    continue
                response = self._get(session, url, stream=True, allow_not_found=True)
                if response is None:
                    continue
                destination = output_directory / name
                try:
                    self._save_response(response, destination)
                finally:
                    response.close()
                return destination.resolve()
        return None

    def _crawl_files(
        self,
        session: Any,
        start_urls: list[str],
        matches: Callable[[str], bool],
        first_only: bool = False,
    ) -> list[str]:
        """File URLs whose name matches, found by walking the browse pages."""

        found: list[str] = []
        visited: set[str] = set()
        stack = list(reversed(start_urls))
        roots = [urlparse(url).path.rstrip("/") for url in start_urls]
        pages = 0
        while stack:
            url = stack.pop()
            if url in visited:
                continue
            visited.add(url)
            pages += 1
            if pages > self.max_pages:
                raise DownloadError(
                    f"CROWD file search exceeded {self.max_pages} browse pages."
                )
            response = self._get(
                session, url, stream=False, allow_not_found=True, read_timeout=LISTING_READ_TIMEOUT_SECONDS
            )
            if response is None:
                continue
            try:
                parser = _Links()
                parser.feed(response.text)
            finally:
                response.close()

            for href in parser.hrefs:
                full = urljoin(url, href)
                if not self._same_origin(full):
                    continue
                path = urlparse(full).path
                if "/files/" in path and matches(PurePosixPath(path).name) and full not in found:
                    found.append(full)
                    if first_only:
                        return found
                # Only descend: every page links back to its parent folder.
                if (
                    href.startswith("/v/")
                    and "/browse" in href
                    and full not in visited
                    and any(path.startswith(root + "/") for root in roots)
                ):
                    stack.append(full)
        return found

    def _save_response(self, response: Any, destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".part")
        expected_value = response.headers.get("content-length")
        try:
            expected = int(expected_value) if expected_value else None
        except ValueError:
            expected = None
        written = 0
        next_progress_report = 100 * 1024 * 1024
        try:
            with temporary.open("wb") as handle:
                for chunk in self._chunks(response):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
                        if written >= next_progress_report:
                            if expected:
                                percentage = 100.0 * written / expected
                                logger.info(
                                    "  downloaded {:.0f} MiB of {:.0f} MiB ({:.1f}%)",
                                    written / (1024 ** 2),
                                    expected / (1024 ** 2),
                                    percentage,
                                )
                            else:
                                logger.info("  downloaded {:.0f} MiB", written / (1024 ** 2))
                            next_progress_report += 100 * 1024 * 1024
                handle.flush()
                os.fsync(handle.fileno())
            if written <= 0:
                raise DownloadError("CROWD file server returned an empty video")
            if expected is not None and written != expected:
                raise TransientDownloadError(
                    f"Incomplete CROWD download: expected {expected} bytes, "
                    f"received {written}."
                )
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _chunks(response: Any) -> Iterable[bytes]:
        try:
            yield from response.iter_content(chunk_size=1024 * 1024)
        except DownloadError:
            raise
        except Exception as error:
            raise TransientDownloadError("CROWD file server connection broke off") from error

    def _retrying(self, attempt: Callable[[], Any]) -> Any:
        """Run one download, trying again after a transient failure."""

        for number in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                return attempt()
            except TransientDownloadError as error:
                if number == DOWNLOAD_ATTEMPTS:
                    raise
                logger.warning("  {} ({}); attempt {} of {}", error, type(error.__cause__).__name__,
                               number + 1, DOWNLOAD_ATTEMPTS)
                time.sleep(RETRY_DELAY_SECONDS * number)
        raise AssertionError("unreachable")

    def _same_origin(self, url: str) -> bool:
        parsed = urlparse(url)
        return (parsed.scheme.lower(), parsed.netloc.lower()) == self.origin

    def _require_same_origin(self, url: str) -> None:
        if not self._same_origin(url):
            raise DownloadError("CROWD file server redirected to an untrusted origin")


def find_local_source_video(video_id: str, roots: Iterable[Path]) -> Path | None:
    """Locate an exact source video in the configured CROWD video directories."""

    filename = video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"
    for root in roots:
        candidate = root / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    return None


def probe_video(path: Path) -> tuple[float, int, int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise CrowdSourceError(f"Video cannot be opened: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise CrowdSourceError(f"Video metadata is invalid: {path}")
    return fps, frames, width, height


def extract_video_segment(
    source: Path,
    destination: Path,
    start_second: int,
    end_second: int,
    end_margin_seconds: float,
) -> SegmentVideo:
    """Extract one mapped interval with OpenCV using deterministic frame bounds."""

    import cv2

    if start_second < 0 or end_second <= start_second:
        raise ValueError("Segment interval must satisfy 0 <= start < end")
    if end_margin_seconds < 0:
        raise ValueError("crowd_trim_end_margin_seconds must be non-negative")

    effective_end = float(end_second) - float(end_margin_seconds)
    if effective_end <= float(start_second):
        effective_end = float(end_second)

    if destination.is_file() and destination.stat().st_size > 0:
        try:
            fps, frames, _, _ = probe_video(destination)
        except CrowdSourceError:
            destination.unlink(missing_ok=True)
        else:
            return SegmentVideo(
                path=destination.resolve(),
                source_fps=fps,
                output_frames=frames,
                requested_start_second=start_second,
                requested_end_second=end_second,
                effective_end_second=effective_end,
            )

    source_fps, source_frames, width, height = probe_video(source)
    start_frame = int(round(float(start_second) * source_fps))
    stop_frame = min(int(round(effective_end * source_fps)), source_frames)
    if start_frame >= source_frames or stop_frame <= start_frame:
        raise CrowdSourceError(
            f"Mapped interval {start_second}-{end_second}s is outside {source.name}."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".part" + destination.suffix)
    capture = cv2.VideoCapture(str(source))
    writer: Any | None = None
    written = 0
    try:
        try:
            if not capture.isOpened():
                raise CrowdSourceError(f"Video cannot be opened: {source}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            writer = cv2.VideoWriter(
                str(temporary),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise CrowdSourceError(f"Could not create segment video: {destination}")
            frame_index = start_frame
            while frame_index < stop_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    if written <= 0:
        temporary.unlink(missing_ok=True)
        raise CrowdSourceError(f"No frames were extracted from {source.name}.")
    temporary.replace(destination)
    probe_video(destination)
    return SegmentVideo(
        path=destination.resolve(),
        source_fps=source_fps,
        output_frames=written,
        requested_start_second=start_second,
        requested_end_second=end_second,
        effective_end_second=effective_end,
    )
