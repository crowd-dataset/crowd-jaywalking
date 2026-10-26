"""Share the stored surface labels with the CROWD server's disk, and with crowd-city through it.

The labels of a segment are made once (``crowd_city_segmentation_store``) and are the same store
crowd-city uses (``seg_data``), so whichever project segmented a segment first, the other reuses it:

    uv run python scripts/crowd/sync_segmentation.py pull   # before a run: fetch what exists
    uv run python scripts/crowd/sync_segmentation.py push   # after a run: merge in and upload what is new

A segment's entry is merged, not overwritten: the tracks one side has and the other lacks are added
to both, so neither project loses the other's tracks. Entries of another schema or under other
settings are left alone on the side that has them. Nothing is deleted. It uses SFTP from Python, so
it runs the same on Windows, Linux and macOS. The login password is the ``crowd_tue3_ssh_password``
entry of the local ``secret`` file.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

import paramiko

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import common  # noqa: E402
from scripts.crossing.crowd_city_surface_store import INDEX_SCHEMA, SurfaceStore  # noqa: E402

DEFAULT_HOST = "100.80.227.26"
DEFAULT_USER = "salam"
# crowd-city's store folder (its ``seg_data``): ``pedestrians_in-youtube/seg`` on the CROWD disk.
DEFAULT_REMOTE = "/media/salam/crowd-tue-3/pedestrians_in-youtube/seg"
SUBFOLDERS = ("index", "manifest")


class Remote(Protocol):
    def names(self, sub: str) -> dict[str, int]: ...
    def get(self, sub: str, name: str, target: Path) -> None: ...
    def put(self, source: Path, sub: str, name: str) -> None: ...


class SftpRemote:
    """The store folder on the server."""

    def __init__(self, sftp: paramiko.SFTPClient, root: str) -> None:
        self.sftp, self.root = sftp, root
        self._make(root)

    def _make(self, path: str) -> None:
        parent = posixpath.dirname(path)
        if parent and parent != path:
            try:
                self.sftp.stat(parent)
            except OSError:
                self._make(parent)
        try:
            self.sftp.stat(path)
        except OSError:
            self.sftp.mkdir(path)

    def names(self, sub: str) -> dict[str, int]:
        try:
            items = self.sftp.listdir_attr(posixpath.join(self.root, sub))
        except OSError:
            return {}
        return {item.filename: item.st_size for item in items if stat.S_ISREG(item.st_mode)}

    def get(self, sub: str, name: str, target: Path) -> None:
        self.sftp.get(posixpath.join(self.root, sub, name), str(target))

    def put(self, source: Path, sub: str, name: str) -> None:
        self._make(posixpath.join(self.root, sub))
        target = posixpath.join(self.root, sub, name)
        self.sftp.put(str(source), target + ".part")  # a finished file appears under its name at once
        self.sftp.posix_rename(target + ".part", target)


def local_store() -> Path:
    try:  # an entry in `config` overrides the method default of scripts/core/method.py
        folder = common.get_configs("crowd_city_segmentation_store")
    except KeyError:
        folder = "data/crowd_segmentation"
    return (ROOT / str(folder)).resolve()


def connect(host: str, user: str) -> paramiko.SSHClient:
    password = common.get_secrets("crowd_tue3_ssh_password")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, look_for_keys=False, allow_agent=False, timeout=30)
    return client


def _stems(names: dict[str, int], suffix: str) -> set[str]:
    return {name[: -len(suffix)] for name in names if name.endswith(suffix)}


def _manifest(store: SurfaceStore, stem: str) -> dict[str, Any]:
    return store._manifest(stem)  # noqa: SLF001 - the store's own reader, shared by both projects


def _entry_is_v5(manifest: dict[str, Any]) -> bool:
    return manifest.get("schema") == INDEX_SCHEMA


def _copy_down(remote: Remote, stem: str, target: Path) -> bool:
    """Fetch one entry (index then manifest, so a manifest never points at a missing index)."""

    (target / "index").mkdir(parents=True, exist_ok=True)
    (target / "manifest").mkdir(parents=True, exist_ok=True)
    for sub, name in (("index", f"{stem}.parquet"), ("manifest", f"{stem}.json")):
        partial = target / sub / (name + ".part")
        remote.get(sub, name, partial)
        partial.replace(target / sub / name)
    return True


def _upload(remote: Remote, local: Path, stem: str) -> None:
    remote.put(local / "index" / f"{stem}.parquet", "index", f"{stem}.parquet")
    remote.put(local / "manifest" / f"{stem}.json", "manifest", f"{stem}.json")


def sync(remote: Remote, local: Path, upload: bool) -> dict[str, int]:
    """Merge the server's entries into ``local`` and, when ``upload``, send ours to the server."""

    store = SurfaceStore(local)
    counts = {"downloaded": 0, "merged": 0, "uploaded": 0, "skipped": 0}
    remote_stems = _stems(remote.names("manifest"), ".json") & _stems(remote.names("index"), ".parquet")
    local_stems = {path.stem for path in (local / "manifest").glob("*.json")}
    with tempfile.TemporaryDirectory() as folder:
        scratch = Path(folder)
        for stem in sorted(remote_stems):
            _copy_down(remote, stem, scratch)
            theirs = _manifest(SurfaceStore(scratch), stem)
            if not _entry_is_v5(theirs):
                counts["skipped"] += 1
                continue
            ours = _manifest(store, stem) if stem in local_stems else {}
            if not _entry_is_v5(ours):
                (local / "index").mkdir(parents=True, exist_ok=True)
                (local / "manifest").mkdir(parents=True, exist_ok=True)
                for sub, name in (("index", f"{stem}.parquet"), ("manifest", f"{stem}.json")):
                    (scratch / sub / name).replace(local / sub / name)
                counts["downloaded"] += 1
                continue
            if ours.get("settings") != theirs.get("settings"):
                counts["skipped"] += 1  # other settings: neither side's labels can stand in for the other's
                continue
            ours_only = set(ours.get("track_signatures") or {}) - set(theirs.get("track_signatures") or {})
            theirs_only = {
                track_id: signature for track_id, signature in (theirs.get("track_signatures") or {}).items()
                if (ours.get("track_signatures") or {}).get(track_id) != signature
            }
            if theirs_only:
                found = SurfaceStore(scratch).load(stem, theirs["settings"], theirs_only)
                store.save(stem, ours["settings"], theirs_only, found)
                counts["merged"] += 1
            if upload and (ours_only or theirs_only):
                _upload(remote, local, stem)
                counts["uploaded"] += 1
        if upload:
            for stem in sorted(local_stems - remote_stems):
                if _entry_is_v5(_manifest(store, stem)) and (local / "index" / f"{stem}.parquet").is_file():
                    _upload(remote, local, stem)
                    counts["uploaded"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("direction", choices=["push", "pull"])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="crowd-city's store folder on the server")
    arguments = parser.parse_args()
    local = local_store()
    local.mkdir(parents=True, exist_ok=True)
    client = connect(arguments.host, arguments.user)
    try:
        with client.open_sftp() as sftp:
            counts = sync(SftpRemote(sftp, arguments.remote), local, upload=arguments.direction == "push")
    finally:
        client.close()
    print(f"{arguments.direction}: {json.dumps(counts)} ({local} <-> {arguments.host}:{arguments.remote})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
