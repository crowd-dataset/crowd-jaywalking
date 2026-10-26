import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from scripts.crossing.crowd_city_surface import SurfaceSample
from scripts.crossing.crowd_city_surface_store import (
    INDEX_SCHEMA,
    SurfaceStore,
    canonical_track_id,
    segment_stem,
    store_settings,
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
        self.settings = store_settings("m", 1024, 512, 1.0, 4.0, 0.06, 0.25, 0.5)
        self.signatures = {"7.0": track_signature(_track())}
        self.timelines = {"7.0": [SurfaceSample("7.0", 2, 61.0, "road", 0.9)]}

    def test_round_trip(self) -> None:
        self.assertEqual(self.store.load("v_1_30", self.settings, self.signatures), {})
        self.store.save("v_1_30", self.settings, self.signatures, self.timelines, extra={"video_id": "v"})
        loaded = self.store.load("v_1_30", self.settings, self.signatures)
        self.assertEqual([(s.frame, s.surface, s.video_time_s) for s in loaded["7"]], [(2, "road", 61.0)])
        manifest = json.loads((Path(self.folder.name) / "manifest" / "v_1_30.json").read_text())
        self.assertEqual(manifest["schema"], INDEX_SCHEMA)
        self.assertEqual(manifest["track_ids"], ["7"])
        self.assertEqual(manifest["video_id"], "v")

    def test_ids_are_canonical(self) -> None:
        # crowd-city keys tracks "7", this project 7.0: one track, one stored entry.
        self.store.save("v_1_30", self.settings, self.signatures, self.timelines)
        loaded = self.store.load("v_1_30", self.settings, {"7": self.signatures["7.0"]})
        self.assertEqual(sorted(loaded), ["7"])
        self.assertEqual([canonical_track_id(v) for v in (7, 7.0, "7.0", " 7 ", "a7", "7.5")],
                         ["7", "7", "7", "7", "a7", "7.5"])

    def test_stale_settings_or_boxes_are_not_reused(self) -> None:
        self.store.save("v_1_30", self.settings, self.signatures, self.timelines)
        other = store_settings("other", 1024, 512, 1.0, 4.0, 0.06, 0.25, 0.5)
        self.assertEqual(self.store.load("v_1_30", other, self.signatures), {})
        self.assertEqual(self.store.load("v_1_30", self.settings, {"7.0": track_signature(_track(0.05))}), {})

    def test_later_runs_add_tracks_and_keep_facts(self) -> None:
        self.store.save("v_1_30", self.settings, self.signatures, self.timelines, extra={"video_fps": 29.97})
        added = {"9.0": [SurfaceSample("9.0", 1, 60.0, "footpath", 0.8)]}
        self.store.save("v_1_30", self.settings, {"9.0": "x"}, added)
        loaded = self.store.load("v_1_30", self.settings, {**self.signatures, "9.0": "x"})
        self.assertEqual(sorted(loaded), ["7", "9"])
        manifest = json.loads((Path(self.folder.name) / "manifest" / "v_1_30.json").read_text())
        self.assertEqual(manifest["video_fps"], 29.97)

    def test_settings_match_crowd_city(self) -> None:
        # crowd-city's SegmentationSettings.as_dict(); a different key set would share nothing.
        self.assertEqual(
            self.settings,
            {
                "model_identifier": "m@1024x512",
                "coarse_hz": 1.0,
                "refine_hz": 4.0,
                "footpoint_band_fraction": 0.06,
                "footpoint_width_fraction": 0.25,
                "minimum_confidence": 0.5,
            },
        )

    def test_segment_stem_rounds_like_crowd(self) -> None:
        self.assertEqual(segment_stem("a/crowd_GauYw99e5C4_s001069_e001275.mp4", 29.97), "GauYw99e5C4_1069_30")
        self.assertEqual(segment_stem("a/crowd_x_s000000_e000010.mp4", 25), "x_0_25")
        self.assertIsNone(segment_stem("video_0001.mp4", 30))


if __name__ == "__main__":
    unittest.main()
