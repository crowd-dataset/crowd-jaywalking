"""Tests for the CROWD style configuration file workflow."""

import unittest

from config_helpers import load_config, project_root, template
from scripts.core.config import ProjectConfig
from scripts.core.method import METHOD_SETTINGS


class ProjectConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = template()

    def test_active_config_overrides_default_config(self) -> None:
        active = dict(self.template, tracking_model="other.pt")
        with project_root(self.template, active) as root:
            config = ProjectConfig.load()

        self.assertEqual(config.get("tracking_model"), "other.pt")
        self.assertEqual(config.root, root.resolve())
        self.assertFalse(any(isinstance(value, dict) for value in config.raw.values()))

    def test_method_settings_come_from_the_code(self) -> None:
        # default.config holds only user settings; the method's values are in method.py.
        self.assertFalse(set(self.template) & set(METHOD_SETTINGS))
        config = load_config(self.template)
        self.assertEqual(set(config.raw), set(self.template) | set(METHOD_SETTINGS))
        self.assertEqual(config.crossing_settings()["partial_exit_min_x_range"], 0.38)
        self.assertEqual(config.policy_settings()["context_scope_window_seconds"], 10.0)
        self.assertEqual(config.vlm_settings()["prompt_mode"], "zebra_light_v5")
        self.assertEqual(config.crowd_settings()["trim_end_margin_seconds"], 1.0)
        self.assertEqual(config.crossing_classifier_settings()["decision_mode"], "crowd_city")

    def test_config_can_override_a_method_setting(self) -> None:
        active = dict(self.template, crossing_vlm_check=False, crossing_decision_mode="classifier", not_a_setting=1)
        with project_root(self.template, active), self.assertLogs("scripts.core.config", "WARNING") as logs:
            config = ProjectConfig.load()
        self.assertFalse(config.crossing_gate_settings()["vlm_check"])
        self.assertEqual(config.crossing_classifier_settings()["decision_mode"], "classifier")
        self.assertNotIn("not_a_setting", config.raw)
        text = " ".join(logs.output)
        self.assertIn("crossing_vlm_check", text)
        self.assertIn("not_a_setting", text)
        # An override is validated like any other value.
        with project_root(self.template, dict(self.template, context_scope="video")), self.assertRaises(ValueError):
            ProjectConfig.load()

    def test_config_must_hold_every_default_entry(self) -> None:
        active = dict(self.template)
        active.pop("tracking_model")
        active["extra"] = 1
        with project_root(self.template, active), self.assertRaises(KeyError):
            ProjectConfig.load()

    def test_approach_review_settings(self) -> None:
        settings = load_config(self.template).approach_review_settings()
        self.assertEqual(settings, {"seconds": 6.0, "end_seconds": 0.5, "frames": 4, "crop_bottom": 0.12})
        for invalid in (
            {"approach_review_end_seconds": 6.0},
            {"approach_review_end_seconds": -1.0},
            {"approach_review_frames": 0},
            {"approach_review_crop_bottom": 0.5},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_config(dict(self.template, **invalid))
        # The review runs after a run and changes no result, so it keeps the fingerprint.
        base = load_config(self.template).fingerprint("v")
        self.assertEqual(load_config(dict(self.template, approach_review_seconds=3.0)).fingerprint("v"), base)

    def test_crowd_vehicle_types_default_to_cars(self) -> None:
        self.assertEqual(load_config(self.template).crowd_settings()["vehicle_types"], [0])
        mixed = load_config(dict(self.template, crowd_vehicle_types=[4, 0, 4]))
        self.assertEqual(mixed.crowd_settings()["vehicle_types"], [0, 4])
        self.assertIsNone(load_config(dict(self.template, crowd_vehicle_types=None)).crowd_settings()["vehicle_types"])
        for invalid in ([], "0", [0, "1"], [True], [-1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_config(dict(self.template, crowd_vehicle_types=invalid))
        # Choosing which footage to process does not change a result, so it keeps the fingerprint.
        base = load_config(self.template).fingerprint("v")
        self.assertEqual(load_config(dict(self.template, crowd_vehicle_types=[0, 4])).fingerprint("v"), base)

    def test_crowd_precomputed_track_settings(self) -> None:
        with project_root(self.template) as root:
            settings = ProjectConfig.load().crowd_settings()
            self.assertEqual(settings["bbox_download_dir"], (root / "data" / "crowd_bbox").resolve())
        self.assertEqual(
            settings["bbox_dirs"][0].parts[-4:], ("crowd-tue-3", "pedestrians_in-youtube", "data", "bbox")
        )
        self.assertEqual(settings["bbox_ftp_aliases"], ["data"])

        for invalid in (
            {"crowd_tracks_source": "yolo"},
            {"crowd_bbox_dirs": [], "crowd_bbox_ftp_folder": ""},
            {"crowd_bbox_ftp_aliases": []},
            {"crowd_bbox_dirs": "bbox"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_config(dict(self.template, **invalid))
        # Local tracking needs neither.
        load_config(
            dict(self.template, crowd_tracks_source="tracking", crowd_bbox_dirs=[], crowd_bbox_ftp_folder="")
        )

    def test_fingerprint_ignores_run_selection_but_not_method(self) -> None:
        base = load_config(self.template).fingerprint("v")
        for name, value in (("crowd_video_id", "abc"), ("crowd_max_segments", 3), ("logger_level", "debug")):
            with self.subTest(name=name):
                self.assertEqual(load_config(dict(self.template, **{name: value})).fingerprint("v"), base)
        for name, value in (("crowd_tracks_source", "tracking"), ("vlm_prompt_mode", "zebra_light_v4")):
            with self.subTest(name=name):
                self.assertNotEqual(load_config(dict(self.template, **{name: value})).fingerprint("v"), base)


if __name__ == "__main__":
    unittest.main()
