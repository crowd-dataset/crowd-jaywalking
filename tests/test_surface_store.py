import tempfile
import unittest
from pathlib import Path

import polars as pl

from scripts.crossing.crowd_city_surface import SurfaceSample
from scripts.crossing.crowd_city_surface_store import (
    SurfaceStore,
    segment_stem,
    track_signature,
)


def _track(offset: float = 0.0) -> pl.DataFrame:
    return pl.DataFrame(
        {"frame-count": [1, 2, 3], "x-center": [0.1 + offset, 0.2, 0.3], "y-center": [0.5, 0.5, 0.5]}
    )


class SurfaceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = SurfaceStore(Path(self.folder.name))
        self.settings = {"model": "m", "coarse_hz": 1.0}
        self.signatures = {"7.0": track_signature(_track())}
        self.timelines = {"7.0": [SurfaceSample("7.0", 2, 0.1, "road", 0.9)]}

    def test_round_trip(self) -> None:
        self.assertIsNone(self.store.load("v_1_30", self.settings, self.signatures))
        self.store.save("v_1_30", self.settings, self.signatures, self.timelines)
        loaded = self.store.load("v_1_30", self.settings, self.signatures)
        self.assertEqual([(s.frame, s.surface) for s in loaded["7.0"]], [(2, "road")])

    def test_stale_settings_or_boxes_are_not_reused(self) -> None:
        self.store.save("v_1_30", self.settings, self.signatures, self.timelines)
        self.assertIsNone(self.store.load("v_1_30", {"model": "other", "coarse_hz": 1.0}, self.signatures))
        self.assertIsNone(self.store.load("v_1_30", self.settings, {"7.0": track_signature(_track(0.05))}))

    def test_later_runs_add_tracks(self) -> None:
        self.store.save("v_1_30", self.settings, self.signatures, self.timelines)
        added = {"9.0": [SurfaceSample("9.0", 1, 0.0, "footpath", 0.8)]}
        self.store.save("v_1_30", self.settings, {"9.0": "x"}, added)
        loaded = self.store.load("v_1_30", self.settings, {**self.signatures, "9.0": "x"})
        self.assertEqual(sorted(loaded), ["7.0", "9.0"])

    def test_segment_stem(self) -> None:
        self.assertEqual(segment_stem("a/crowd_GauYw99e5C4_s001069_e001275.mp4", 29.97), "GauYw99e5C4_1069_29")
        self.assertIsNone(segment_stem("video_0001.mp4", 30))


if __name__ == "__main__":
    unittest.main()
