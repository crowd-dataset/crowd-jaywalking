"""Tests for the CROWD style configuration file workflow."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from crowd_jaywalking.config import ProjectConfig


class ProjectConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = Path("default.config").read_text(encoding="utf-8")

    def test_uses_default_config_when_active_config_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "default.config").write_text(self.template, encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                config = ProjectConfig.load()
            finally:
                os.chdir(previous)

        self.assertEqual(config.source_path.name, "default.config")
        self.assertEqual(config.get("tracking_model"), "yolo26x.pt")
        self.assertFalse(any(isinstance(value, dict) for value in config.raw.values()))

    def test_active_config_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "default.config").write_text(self.template, encoding="utf-8")
            (root / "config").write_text(self.template, encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                config = ProjectConfig.load()
            finally:
                os.chdir(previous)

        self.assertEqual(config.source_path.name, "config")

    def test_partial_crossing_settings_have_backwards_compatible_defaults(self) -> None:
        raw = json.loads(self.template)
        for name in (
            "partial_crossing_enabled",
            "partial_exit_min_x_range",
            "partial_exit_min_direction_consistency",
            "perspective_corridor_enabled",
            "road_top_y",
            "road_bottom_y",
            "road_top_left",
            "road_top_right",
            "road_bottom_left",
            "road_bottom_right",
            "strong_complete_override_enabled",
            "strong_complete_min_seconds",
            "strong_complete_min_x_range",
            "strong_complete_min_direction_consistency",
            "camera_min_shared_track_ratio",
        ):
            raw.pop(name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").write_text(json.dumps(raw), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                crossing = ProjectConfig.load().crossing_settings()
            finally:
                os.chdir(previous)

        self.assertTrue(crossing["partial_crossing_enabled"])
        self.assertEqual(crossing["partial_exit_min_x_range"], 0.48)
        self.assertEqual(crossing["partial_exit_min_direction_consistency"], 0.0)
        self.assertFalse(crossing["perspective_corridor_enabled"])
        self.assertFalse(crossing["strong_complete_override_enabled"])
        self.assertEqual(crossing["camera_min_shared_track_ratio"], 0.0)

    def test_classifier_settings_have_backwards_compatible_defaults(self) -> None:
        raw = json.loads(self.template)
        for name in (
            "crossing_classifier_results",
            "crossing_classifier_model",
            "crossing_classifier_min_precision",
            "crossing_classifier_cv_folds",
            "crossing_classifier_threshold_step",
            "crossing_classifier_random_seed",
            "crossing_classifier_logistic_c_values",
            "crossing_classifier_gradient_learning_rates",
            "crossing_classifier_gradient_max_leaf_nodes",
            "crossing_decision_mode",
            "crossing_classifier_fallback_to_rules",
            "crossing_classifier_min_track_frames",
        ):
            raw.pop(name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").write_text(json.dumps(raw), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                settings = ProjectConfig.load().crossing_classifier_settings()
            finally:
                os.chdir(previous)

        self.assertEqual(settings["min_precision"], 0.90)
        self.assertEqual(settings["cv_folds"], 5)
        self.assertEqual(settings["logistic_c_values"], [0.10, 1.00, 10.00])
        self.assertEqual(settings["gradient_max_leaf_nodes"], [7, 15])
        self.assertEqual(settings["decision_mode"], "classifier")
        self.assertFalse(settings["fallback_to_rules"])
        self.assertEqual(settings["min_track_frames"], 5)

    def test_evidence_version_five_settings_have_backwards_compatible_defaults(self) -> None:
        raw = json.loads(self.template)
        for name in (
            "evidence_trajectory_enabled",
            "evidence_road_crop_margin",
            "evidence_control_crop_bottom",
            "evidence_control_crop_overlap",
            "vlm_task_max_frames",
            "vlm_prompt_mode",
        ):
            raw.pop(name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").write_text(json.dumps(raw), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                config = ProjectConfig.load()
                evidence = config.evidence_settings()
                vlm = config.vlm_settings()
            finally:
                os.chdir(previous)

        self.assertTrue(evidence["trajectory_enabled"])
        self.assertEqual(evidence["road_crop_margin"], 0.12)
        self.assertEqual(evidence["control_crop_bottom"], 0.78)
        self.assertEqual(evidence["control_crop_overlap"], 0.20)
        self.assertEqual(vlm["task_max_frames"], 4)
        self.assertEqual(vlm["prompt_mode"], "baseline_v3")

    def test_crowd_settings_have_backwards_compatible_defaults(self) -> None:
        raw = json.loads(self.template)
        for name in (
            "mapping",
            "ftp_server",
            "crowd_results",
            "crowd_resume",
            "crowd_ftp_aliases",
            "crowd_download_dir",
            "crowd_download_timeout_seconds",
            "crowd_download_max_pages",
            "crowd_trim_end_margin_seconds",
            "crowd_delete_downloaded_base_videos",
            "crowd_keep_segment_videos",
            "crowd_max_segments",
            "crowd_audit_random_seed",
            "crowd_audit_per_stratum",
        ):
            raw.pop(name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").write_text(json.dumps(raw), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                settings = ProjectConfig.load().crowd_settings()
            finally:
                os.chdir(previous)

        self.assertTrue(settings["resume"])
        self.assertEqual(settings["mapping"].name, "mapping.csv")
        self.assertEqual(settings["ftp_aliases"], ["tue4", "tue5"])
        self.assertEqual(settings["trim_end_margin_seconds"], 1.0)
        self.assertTrue(settings["keep_segment_videos"])
        self.assertEqual(settings["max_segments"], 0)
        self.assertEqual(settings["audit_per_stratum"], 50)

    def test_vlm_comparison_settings_have_backwards_compatible_defaults(self) -> None:
        raw = json.loads(self.template)
        raw.pop("vlm_comparison_models")
        raw.pop("vlm_comparison_prompt_modes")
        raw.pop("vlm_comparison_results")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").write_text(json.dumps(raw), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                settings = ProjectConfig.load().vlm_comparison_settings()
            finally:
                os.chdir(previous)

        self.assertEqual(
            settings["models"],
            ["Qwen/Qwen3-VL-8B-Instruct", "google/gemma-4-12B-it"],
        )
        self.assertEqual(
            settings["prompt_modes"],
            ["baseline_v3", "focused_v5"],
        )
        self.assertEqual(settings["results"].name, "jaad_vlm_comparison_v5")

    def test_context_sampling_settings_have_backwards_compatible_defaults(self) -> None:
        raw = json.loads(self.template)
        raw.pop("jaad_context_sample_size")
        raw.pop("jaad_context_sampling_seed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").write_text(json.dumps(raw), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                settings = ProjectConfig.load().jaad_context_settings()
            finally:
                os.chdir(previous)

        self.assertEqual(settings["sample_size"], 120)
        self.assertEqual(settings["sampling_seed"], 42)


if __name__ == "__main__":
    unittest.main()
