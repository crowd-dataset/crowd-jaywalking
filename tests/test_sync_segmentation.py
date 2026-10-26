import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.crossing.crowd_city_surface import SurfaceSample
from scripts.crossing.crowd_city_surface_store import SurfaceStore
from scripts.crowd.sync_segmentation import sync

SETTINGS = {"model_identifier": "m@1024x512", "coarse_hz": 1.0}


class FolderRemote:
    """A store folder on the same disk, standing in for the server."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def names(self, sub: str) -> dict[str, int]:
        folder = self.root / sub
        return {path.name: path.stat().st_size for path in folder.iterdir()} if folder.is_dir() else {}

    def get(self, sub: str, name: str, target: Path) -> None:
        shutil.copyfile(self.root / sub / name, target)

    def put(self, source: Path, sub: str, name: str) -> None:
        (self.root / sub).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, self.root / sub / name)


def _save(store: SurfaceStore, stem: str, track: str, signature: str, settings=SETTINGS) -> None:
    store.save(stem, settings, {track: signature}, {track: [SurfaceSample(track, 5, 1.0, "road", 0.9)]})


class SyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        base = Path(self.folder.name)
        self.local, self.server = SurfaceStore(base / "local"), SurfaceStore(base / "server")
        self.remote = FolderRemote(base / "server")

    def test_new_entries_go_both_ways(self) -> None:
        _save(self.local, "a_1_30", "7", "s7")
        _save(self.server, "b_2_30", "9", "s9")
        counts = sync(self.remote, self.local.root, upload=True)
        self.assertEqual((counts["downloaded"], counts["uploaded"]), (1, 1))
        self.assertEqual(sorted(self.local.load("b_2_30", SETTINGS, {"9": "s9"})), ["9"])
        self.assertEqual(sorted(self.server.load("a_1_30", SETTINGS, {"7": "s7"})), ["7"])

    def test_tracks_are_merged_not_overwritten(self) -> None:
        _save(self.local, "a_1_30", "7", "s7")
        _save(self.server, "a_1_30", "9", "s9")  # crowd-city segmented another track of the same segment
        sync(self.remote, self.local.root, upload=True)
        both = {"7": "s7", "9": "s9"}
        self.assertEqual(sorted(self.local.load("a_1_30", SETTINGS, both)), ["7", "9"])
        self.assertEqual(sorted(self.server.load("a_1_30", SETTINGS, both)), ["7", "9"])

    def test_pull_does_not_upload(self) -> None:
        _save(self.local, "a_1_30", "7", "s7")
        sync(self.remote, self.local.root, upload=False)
        self.assertEqual(self.remote.names("manifest"), {})

    def test_other_settings_are_left_alone(self) -> None:
        _save(self.local, "a_1_30", "7", "s7")
        _save(self.server, "a_1_30", "9", "s9", settings={"model_identifier": "other"})
        counts = sync(self.remote, self.local.root, upload=True)
        self.assertEqual(counts["skipped"], 1)
        self.assertEqual(self.server.load("a_1_30", {"model_identifier": "other"}, {"9": "s9"}).keys(), {"9"})

    def test_old_schema_entries_are_not_shared(self) -> None:
        (self.server.root / "manifest").mkdir(parents=True)
        (self.server.root / "index").mkdir(parents=True)
        (self.server.root / "manifest" / "old_1_30.json").write_text('{"schema": "crowd_surface_index_v4"}')
        (self.server.root / "index" / "old_1_30.parquet").write_bytes(b"x")
        counts = sync(self.remote, self.local.root, upload=True)
        self.assertEqual(counts["skipped"], 1)
        self.assertFalse((self.local.root / "manifest" / "old_1_30.json").exists())


if __name__ == "__main__":
    unittest.main()
