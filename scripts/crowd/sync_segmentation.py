"""Copy the stored surface labels between this machine and the CROWD server's disk.

The labels of a segment are made once (``crowd_city_segmentation_store``) and shared:

    uv run python scripts/crowd/sync_segmentation.py pull   # before a run: fetch what exists
    uv run python scripts/crowd/sync_segmentation.py push   # after a run: upload what is new

Both only add files that are missing on the other side (the files never change once written, so a
file of the same size is skipped); nothing is deleted. It uses SFTP from Python, so it needs no
rsync or ssh program and runs the same on Windows, Linux and macOS. The login password is the
``crowd_tue3_ssh_password`` entry of the local ``secret`` file.
"""

from __future__ import annotations

import argparse
import posixpath
import stat
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import common  # noqa: E402

DEFAULT_HOST = "100.80.227.26"
DEFAULT_USER = "salam"
DEFAULT_REMOTE = "/media/salam/crowd-tue-3/pedestrians_in-youtube/data/segmentation_jaywalking"
SUBFOLDERS = ("index", "manifest")


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


def make_remote_folder(sftp: paramiko.SFTPClient, path: str) -> None:
    """Like ``mkdir -p`` on the server."""

    parent = posixpath.dirname(path)
    if parent and parent != path:
        try:
            sftp.stat(parent)
        except OSError:
            make_remote_folder(sftp, parent)
    try:
        sftp.stat(path)
    except OSError:
        sftp.mkdir(path)


def remote_files(sftp: paramiko.SFTPClient, folder: str) -> dict[str, int]:
    try:
        return {item.filename: item.st_size for item in sftp.listdir_attr(folder) if stat.S_ISREG(item.st_mode)}
    except OSError:
        return {}


def push(sftp: paramiko.SFTPClient, local: Path, remote: str) -> int:
    sent = 0
    for sub in SUBFOLDERS:
        source = local / sub
        if not source.is_dir():
            continue
        make_remote_folder(sftp, posixpath.join(remote, sub))
        existing = remote_files(sftp, posixpath.join(remote, sub))
        for path in sorted(source.iterdir()):
            if path.suffix == ".tmp" or not path.is_file() or existing.get(path.name) == path.stat().st_size:
                continue
            target = posixpath.join(remote, sub, path.name)
            sftp.put(str(path), target + ".part")  # a finished file appears under its name at once
            sftp.posix_rename(target + ".part", target)
            sent += 1
    return sent


def pull(sftp: paramiko.SFTPClient, local: Path, remote: str) -> int:
    fetched = 0
    for sub in SUBFOLDERS:
        (local / sub).mkdir(parents=True, exist_ok=True)
        for name, size in sorted(remote_files(sftp, posixpath.join(remote, sub)).items()):
            target = local / sub / name
            if name.endswith(".part") or (target.is_file() and target.stat().st_size == size):
                continue
            partial = target.with_name(target.name + ".part")
            sftp.get(posixpath.join(remote, sub, name), str(partial))
            partial.replace(target)
            fetched += 1
    return fetched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("direction", choices=["push", "pull"])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="folder on the server")
    arguments = parser.parse_args()
    local = local_store()
    local.mkdir(parents=True, exist_ok=True)
    client = connect(arguments.host, arguments.user)
    try:
        with client.open_sftp() as sftp:
            count = (push if arguments.direction == "push" else pull)(sftp, local, arguments.remote)
    finally:
        client.close()
    print(f"{arguments.direction}: {count} file(s) copied ({local} <-> {arguments.host}:{arguments.remote})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
