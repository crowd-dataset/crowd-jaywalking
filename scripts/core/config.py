"""CROWD style flat project configuration loading and validation.

The method's settings are in scripts/core/method.py; default.config (copied to
``config``) holds the user's settings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from custom_logger import CustomLogger
from scripts.core.method import METHOD_SETTINGS
from scripts.core.policy import JaywalkingPolicy

logger = CustomLogger(__name__)  # use custom logger


ACTIVE_CONFIG_NAME = "config"
DEFAULT_CONFIG_NAME = "default.config"

# The rule detector's settings, passed to it as one dictionary (earlier designs).
CROSSING_KEYS = (
    "road_left",
    "road_right",
    "boundary_tolerance",
    "perspective_corridor_enabled",
    "road_top_y",
    "road_bottom_y",
    "road_top_left",
    "road_top_right",
    "road_bottom_left",
    "road_bottom_right",
    "min_track_seconds",
    "min_road_seconds",
    "max_track_gap_seconds",
    "partial_crossing_enabled",
    "partial_exit_min_x_range",
    "partial_exit_min_direction_consistency",
    "strong_complete_override_enabled",
    "strong_complete_min_seconds",
    "strong_complete_min_x_range",
    "strong_complete_min_direction_consistency",
    "min_crossing_x_range",
    "max_crossing_speed_per_frame",
    "low_x_range",
    "low_x_min_road_seconds",
    "weak_x_range",
    "long_weak_road_seconds",
    "weak_y_jitter_x_range",
    "weak_y_jitter_motion",
    "weak_y_jitter_height",
    "jitter_road_seconds",
    "tiny_long_track_x_range",
    "tiny_long_track_height",
    "tiny_long_track_road_seconds",
    "tiny_no_static_height",
    "tiny_no_static_width",
    "tiny_no_static_min_road_seconds",
    "no_static_tiny_min_road_seconds",
    "no_static_tiny_fast_speed",
    "slender_track_width",
    "slender_track_height",
    "slender_track_min_road_seconds",
    "slender_track_max_road_seconds",
    "no_static_slender_height",
    "no_static_slender_max_road_seconds",
    "slender_static_min_relative_x_range",
    "large_lateral_x_range",
    "large_lateral_tiny_height",
    "min_static_shared_seconds",
    "camera_static_x_range",
    "camera_ratio_threshold",
    "camera_min_shared_track_ratio",
    "camera_static_relative_x_range",
    "camera_static_height",
    "camera_static_tiny_relative_x_range",
    "camera_static_tiny_height",
    "camera_tiny_height",
    "camera_min_road_seconds",
    "min_relative_x_range",
    "rider_min_shared_seconds",
    "rider_min_continuous_shared_seconds",
    "rider_shared_run_gap_seconds",
    "rider_min_vehicle_width_ratio",
    "rider_min_vehicle_width_ratio_frames",
    "rider_distance_relative_threshold",
    "rider_proximity_ratio",
    "rider_alpha_x",
    "rider_beta_y",
    "rider_gamma_y",
    "rider_colocation_ratio",
    "rider_similarity_threshold",
    "rider_similarity_ratio",
    "rider_min_motion_seconds",
    "rider_motion_colocation_min",
    "rider_short_shared_seconds",
    "rider_short_similarity_ratio",
    "rider_short_displacement",
)

CROWD_TRACK_SOURCES = ("precomputed", "tracking")

# Entries that choose which items a run processes, how much it logs, or where local
# copies of inputs are kept. They never change a result, so the fingerprint ignores
# them and one results folder can be filled video by video or on another machine.
RESULT_NEUTRAL_KEYS = (
    "logger_level",
    "resume",
    "crowd_resume",
    "crowd_video_id",
    "crowd_max_segments",
    "jaad_video_id",
    "smoke_test_video",
    "crowd_download_dir",
    "crowd_bbox_dirs",
    "crowd_bbox_download_dir",
    "crowd_delete_downloaded_base_videos",
    "crowd_keep_segment_videos",
    "vlm_cache_dir",
)


@dataclass(frozen=True)
class ProjectConfig:
    """Validated flat project configuration and its source path."""

    source_path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls) -> "ProjectConfig":
        """The method's settings from scripts/core/method.py plus every entry of default.config.

        As in the other CROWD repositories, ``config`` (a local copy of default.config)
        must exist next to common.py and is read through common.get_configs. An entry of
        ``config`` named like a method setting overrides it, and is logged.
        """

        import common

        root = Path(common.root_dir).resolve()
        template = root / DEFAULT_CONFIG_NAME
        try:
            names = list(json.loads(template.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read the entry names of {template}") from error
        raw = dict(METHOD_SETTINGS)
        for name in names:
            try:
                raw[name] = common.get_configs(name)
            except KeyError as error:
                raise KeyError(
                    f"'{ACTIVE_CONFIG_NAME}' lacks the entry '{name}'; update it from {DEFAULT_CONFIG_NAME}"
                ) from error
        active = root / ACTIVE_CONFIG_NAME
        local = json.loads(active.read_text(encoding="utf-8")) if active.is_file() else {}
        for name, value in local.items():
            if name in names:
                continue
            if name in METHOD_SETTINGS:
                if value != METHOD_SETTINGS[name]:
                    logger.warning(
                        "config overrides the method setting {}: {} instead of {}",
                        name, value, METHOD_SETTINGS[name],
                    )
                raw[name] = value
            else:
                logger.warning("config entry {} is not a setting and is ignored", name)
        config = cls(source_path=active, raw=raw)
        config.validate()
        return config

    @property
    def root(self) -> Path:
        return self.source_path.parent

    def get(self, name: str) -> Any:
        if name not in self.raw:
            raise KeyError(f"Missing configuration entry: {name}")
        return self.raw[name]

    def path(self, name: str) -> Path:
        value = self.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration entry must be a path string: {name}")
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

    def paths(self, name: str) -> list[Path]:
        values = self.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Configuration entry must be a non-empty path list: {name}")
        resolved: list[Path] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Configuration path list contains an invalid value: {name}")
            candidate = Path(value)
            resolved.append(
                candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
            )
        return resolved

    def data_file(self, name: str) -> Path:
        value = self.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration entry must name a data file: {name}")
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.paths("data")[0] / candidate).resolve()

    def tracking_settings(self) -> dict[str, Any]:
        return {
            "model": self.get("tracking_model"),
            "tracker": self.get("bbox_tracker"),
            "confidence": self.get("min_confidence"),
            "iou": self.get("iou"),
            "device": self.get("device"),
        }

    def crossing_settings(self) -> dict[str, Any]:
        return {key: self.get(key) for key in CROSSING_KEYS}

    def evidence_settings(self) -> dict[str, Any]:
        return {
            "sample_positions": self.get("evidence_sample_positions"),
            "context_seconds": self.get("evidence_context_seconds"),
            "crop_margin": self.get("evidence_crop_margin"),
            "max_dimension": self.get("evidence_max_dimension"),
            "jpeg_quality": self.get("evidence_jpeg_quality"),
            "trajectory_enabled": self.get("evidence_trajectory_enabled"),
            "road_crop_margin": self.get("evidence_road_crop_margin"),
            "control_crop_bottom": self.get("evidence_control_crop_bottom"),
            "control_crop_overlap": self.get("evidence_control_crop_overlap"),
            "infrastructure_span": str(
                self.get("evidence_infrastructure_span")
            ).strip().lower(),
        }

    def vlm_settings(
        self,
        model_id: str | None = None,
        prompt_mode: str | None = None,
    ) -> dict[str, Any]:
        return {
            "model_id": model_id or self.get("vlm_model"),
            "prompt_mode": prompt_mode
            or str(
                self.get("vlm_prompt_mode")
            ),
            "device_map": self.get("vlm_device_map"),
            "torch_dtype": self.get("vlm_torch_dtype"),
            "attn_implementation": self.get("vlm_attn_implementation"),
            "cache_dir": self.get("vlm_cache_dir"),
            "local_files_only": self.get("vlm_local_files_only"),
            "min_pixels": self.get("vlm_min_pixels"),
            "max_pixels": self.get("vlm_max_pixels"),
            "max_new_tokens": self.get("vlm_max_new_tokens"),
            "task_max_frames": self.get("vlm_task_max_frames"),
            # "4bit" loads larger VLMs with bitsandbytes; null loads full precision.
            "quantization": self.get("vlm_quantization"),
        }

    def vlm_comparison_settings(self) -> dict[str, Any]:
        """Return model candidates and output location for VLM selection."""

        models_value = self.get("vlm_comparison_models")
        if not isinstance(models_value, list):
            raise ValueError("vlm_comparison_models must be a list")
        prompt_modes_value = self.get("vlm_comparison_prompt_modes")
        if not isinstance(prompt_modes_value, list):
            raise ValueError("vlm_comparison_prompt_modes must be a list")
        return {
            "models": [str(item).strip() for item in models_value],
            "prompt_modes": [str(item).strip().lower() for item in prompt_modes_value],
            "results": self.path("vlm_comparison_results"),
        }

    def jaad_context_settings(self) -> dict[str, int]:
        """Return reproducible context audit sampling settings."""

        return {
            "sample_size": int(self.get("jaad_context_sample_size")),
            "sampling_seed": int(self.get("jaad_context_sampling_seed")),
        }

    def policy_settings(self) -> dict[str, Any]:
        permission_cues = self.get("permission_cues")
        if not isinstance(permission_cues, list):
            raise ValueError("permission_cues must be a list")
        return {
            "prohibitive_signal_overrides_crosswalk": self.get(
                "prohibitive_signal_overrides_crosswalk"
            ),
            "permission_cues": [str(item).strip().lower() for item in permission_cues],
            "partial_visibility_uncertain": bool(
                self.get("partial_visibility_uncertain")
            ),
            "context_scope": str(
                self.get("context_scope")
            ).strip().lower(),
            "strict_absence": bool(
                self.get("strict_absence")
            ),
            "context_scope_window_seconds": self.get("context_scope_window_seconds"),
        }

    def jaywalking_law_settings(self) -> dict[str, Any]:
        """Return settings for stage 3, the country specific jaywalking rule sets."""

        value = self.get
        return {
            "rules": self.path("jaywalking_law_rules"),
            "vlm": self.vlm_settings(model_id=value("jaywalking_law_model")),
            "country": value("jaywalking_law_country"),
            "state": value("jaywalking_law_state"),
            "locality": value("jaywalking_law_locality"),
        }

    def crossing_gate_settings(self) -> dict[str, Any]:
        """Return settings for the optional high precision crossing gate."""

        value = self.get
        results = self.path("crossing_gate_results")
        model = value("crossing_gate_model")
        return {
            "enabled": model is not None,
            "results": results,
            "model": (
                self.path("crossing_gate_model")
                if model is not None
                else results / "crossing_gate.joblib"
            ),
            "min_precision": float(value("crossing_gate_min_precision")),
            "precision_tiers": sorted(
                (float(item) for item in value("crossing_gate_precision_tiers")),
                reverse=True,
            ),
            "min_accepted": int(value("crossing_gate_min_accepted")),
            "cv_folds": int(value("crossing_gate_cv_folds")),
            "learning_rate": float(value("crossing_gate_learning_rate")),
            "max_leaf_nodes": int(value("crossing_gate_max_leaf_nodes")),
            "random_seed": int(value("crossing_gate_random_seed")),
            "camera_motion": bool(value("crossing_gate_camera_motion")),
            "min_scene_x_range": value("crossing_min_scene_x_range"),
            "vlm_check": bool(value("crossing_vlm_check")),
            "vlm_check_version": str(value("crossing_vlm_check_version")).strip().lower(),
            "rescue_min_first_stage": value("crossing_rescue_min_first_stage"),
            "rescue_min_gate": value("crossing_rescue_min_gate"),
        }

    def crossing_classifier_settings(self) -> dict[str, Any]:
        """Return effective settings for supervised crossing classification."""

        value = self.get
        return {
            "benchmark_results": self.path("jaad_benchmark_results"),
            "results": self.path("crossing_classifier_results"),
            "model": self.path("crossing_classifier_model"),
            "min_precision": float(value("crossing_classifier_min_precision")),
            "cv_folds": int(value("crossing_classifier_cv_folds")),
            "threshold_step": float(value("crossing_classifier_threshold_step")),
            "random_seed": int(value("crossing_classifier_random_seed")),
            "logistic_c_values": [
                float(item) for item in value("crossing_classifier_logistic_c_values")
            ],
            "gradient_learning_rates": [
                float(item)
                for item in value("crossing_classifier_gradient_learning_rates")
            ],
            "gradient_max_leaf_nodes": [
                int(item)
                for item in value("crossing_classifier_gradient_max_leaf_nodes")
            ],
            "decision_mode": str(value("crossing_decision_mode")).strip().lower(),
            "fallback_to_rules": bool(value("crossing_classifier_fallback_to_rules")),
            "min_track_frames": int(value("crossing_classifier_min_track_frames")),
        }

    def crowd_settings(self) -> dict[str, Any]:
        """Return batch execution and manual audit settings for CROWD."""

        value = self.get
        return {
            "mapping": self.path("mapping"),
            "ftp_server": str(value("ftp_server")).strip(),
            "results": self.path("crowd_results"),
            "resume": bool(value("crowd_resume")),
            "ftp_aliases": [
                str(item).strip() for item in value("crowd_ftp_aliases")
            ],
            "download_dir": self.path("crowd_download_dir"),
            "download_timeout_seconds": int(value("crowd_download_timeout_seconds")),
            "download_max_pages": int(value("crowd_download_max_pages")),
            "trim_end_margin_seconds": float(value("crowd_trim_end_margin_seconds")),
            "delete_downloaded_base_videos": bool(
                value("crowd_delete_downloaded_base_videos")
            ),
            "keep_segment_videos": bool(value("crowd_keep_segment_videos")),
            "max_segments": int(value("crowd_max_segments")),
            "audit_random_seed": int(value("crowd_audit_random_seed")),
            "audit_per_stratum": int(value("crowd_audit_per_stratum")),
            "tracks_source": str(value("crowd_tracks_source")).strip().lower(),
            "bbox_dirs": [
                candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
                for candidate in (Path(str(item)) for item in value("crowd_bbox_dirs") or [])
            ],
            "bbox_ftp_folder": str(value("crowd_bbox_ftp_folder") or "").strip(),
            "bbox_ftp_aliases": [
                str(item).strip() for item in value("crowd_bbox_ftp_aliases") or []
            ],
            "bbox_download_dir": self.path("crowd_bbox_download_dir"),
        }

    def fingerprint(self, prompt_version: str) -> str:
        effective_config = dict(self.raw)
        for key in RESULT_NEUTRAL_KEYS:
            effective_config.pop(key, None)
        payload = {
            "config": effective_config,
            "prompt_version": prompt_version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        missing = sorted(set(METHOD_SETTINGS).difference(self.raw))
        if missing:
            raise ValueError(f"Missing configuration entries: {', '.join(missing)}")
        nested = sorted(key for key, value in self.raw.items() if isinstance(value, dict))
        if nested:
            raise ValueError(
                "Configuration must be flat; nested objects found at: " + ", ".join(nested)
            )

        self.paths("data")
        self.paths("videos")
        self.data_file("source_annotations")
        self.data_file("annotations")
        self.path("results")

        left = float(self.get("road_left"))
        right = float(self.get("road_right"))
        if not 0.0 <= left < right <= 1.0:
            raise ValueError("road_left and road_right must satisfy 0 <= left < right <= 1")
        partial_crossing_enabled = self.get("partial_crossing_enabled")
        if not isinstance(partial_crossing_enabled, bool):
            raise ValueError("partial_crossing_enabled must be true or false")
        partial_exit_min_x_range = float(
            self.get("partial_exit_min_x_range")
        )
        if not 0.0 <= partial_exit_min_x_range <= 1.0:
            raise ValueError("partial_exit_min_x_range must be between 0 and 1")

        for name in (
            "partial_exit_min_direction_consistency",
            "strong_complete_min_direction_consistency",
            "camera_min_shared_track_ratio",
        ):
            value = float(self.get(name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        for name in (
            "perspective_corridor_enabled",
            "strong_complete_override_enabled",
        ):
            value = self.get(name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false")

        road_top_y = float(self.get("road_top_y"))
        road_bottom_y = float(
            self.get("road_bottom_y")
        )
        if not 0.0 <= road_top_y < road_bottom_y <= 1.0:
            raise ValueError("road_top_y and road_bottom_y must satisfy 0 <= top < bottom <= 1")
        for prefix in ("road_top", "road_bottom"):
            corridor_left = float(
                self.get(f"{prefix}_left")
            )
            corridor_right = float(
                self.get(f"{prefix}_right")
            )
            if not 0.0 <= corridor_left < corridor_right <= 1.0:
                raise ValueError(
                    f"{prefix}_left and {prefix}_right must satisfy 0 <= left < right <= 1"
                )

        if float(
            self.get("strong_complete_min_seconds")
        ) < 0.0:
            raise ValueError("strong_complete_min_seconds must be non-negative")
        strong_complete_min_x_range = float(
            self.get("strong_complete_min_x_range")
        )
        if not 0.0 <= strong_complete_min_x_range <= 1.0:
            raise ValueError("strong_complete_min_x_range must be between 0 and 1")

        positions = self.get("evidence_sample_positions")
        if not isinstance(positions, list) or not positions:
            raise ValueError("evidence_sample_positions must be a non-empty list")
        if any(not 0.0 <= float(position) <= 1.0 for position in positions):
            raise ValueError("Every evidence sample position must be between 0 and 1")
        if float(self.get("evidence_context_seconds")) < 0.0:
            raise ValueError("evidence_context_seconds must be non-negative")
        trajectory_enabled = self.get("evidence_trajectory_enabled")
        if not isinstance(trajectory_enabled, bool):
            raise ValueError("evidence_trajectory_enabled must be true or false")
        if not 0.0 <= float(
            self.get("evidence_road_crop_margin")
        ) <= 0.5:
            raise ValueError("evidence_road_crop_margin must be between 0 and 0.5")
        for name in ("evidence_control_crop_bottom", "evidence_control_crop_overlap"):
            value = float(self.get(name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be greater than 0 and at most 1")

        if not str(self.get("tracking_model")).strip():
            raise ValueError("tracking_model must name a YOLO model")
        if not str(self.get("bbox_tracker")).strip():
            raise ValueError("bbox_tracker must name a tracker configuration")
        if not str(self.get("vlm_model")).strip():
            raise ValueError("vlm_model must name a Hugging Face model")

        min_pixels = int(self.get("vlm_min_pixels"))
        max_pixels = int(self.get("vlm_max_pixels"))
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("VLM pixel limits must satisfy 0 < min_pixels <= max_pixels")
        if int(self.get("vlm_max_new_tokens")) <= 0:
            raise ValueError("vlm_max_new_tokens must be positive")
        if int(
            self.get("vlm_task_max_frames")
        ) <= 0:
            raise ValueError("vlm_task_max_frames must be positive")

        valid_prompt_modes = {
            "baseline_v3",
            "focused_v5",
            "zebra_light_v1",
            "zebra_light_v2",
            "zebra_light_v3",
            "zebra_light_v4",
            "zebra_light_v5",
        }
        prompt_mode = str(
            self.get("vlm_prompt_mode")
        ).strip().lower()
        if prompt_mode not in valid_prompt_modes:
            raise ValueError(
                "vlm_prompt_mode must be one of: " + ", ".join(sorted(valid_prompt_modes))
            )

        comparison = self.vlm_comparison_settings()
        if len(comparison["models"]) < 2:
            raise ValueError("vlm_comparison_models must contain at least two models")
        if any(not model for model in comparison["models"]):
            raise ValueError("vlm_comparison_models must not contain empty model IDs")
        if len(set(comparison["models"])) != len(comparison["models"]):
            raise ValueError("vlm_comparison_models must contain distinct model IDs")
        if not comparison["prompt_modes"]:
            raise ValueError("vlm_comparison_prompt_modes must not be empty")
        if len(set(comparison["prompt_modes"])) != len(comparison["prompt_modes"]):
            raise ValueError("vlm_comparison_prompt_modes must contain distinct values")
        if any(mode not in valid_prompt_modes for mode in comparison["prompt_modes"]):
            raise ValueError(
                "vlm_comparison_prompt_modes may contain only: "
                + ", ".join(sorted(valid_prompt_modes))
            )

        JaywalkingPolicy(self.policy_settings())
        gate = self.crossing_gate_settings()
        if not gate["precision_tiers"] or any(
            not 0.0 < tier <= 1.0 for tier in gate["precision_tiers"]
        ):
            raise ValueError(
                "crossing_gate_precision_tiers must be values greater than 0 and at most 1"
            )
        rescue = (gate["rescue_min_first_stage"], gate["rescue_min_gate"])
        if (rescue[0] is None) != (rescue[1] is None):
            raise ValueError(
                "Set both crossing_rescue_min_first_stage and crossing_rescue_min_gate, or neither"
            )
        if rescue[0] is not None and not gate["vlm_check"]:
            raise ValueError("Rescued crossings require crossing_vlm_check to confirm them")
        if self.evidence_settings()["infrastructure_span"] not in {"transition", "track"}:
            raise ValueError("evidence_infrastructure_span must be one of: transition, track")
        if gate["vlm_check_version"] not in {"v1", "v2", "v3"}:
            raise ValueError("crossing_vlm_check_version must be one of: v1, v2, v3")
        if gate["min_precision"] not in gate["precision_tiers"]:
            raise ValueError(
                "crossing_gate_min_precision must be one of crossing_gate_precision_tiers"
            )
        if gate["min_accepted"] < 1 or gate["cv_folds"] < 2:
            raise ValueError(
                "crossing_gate_min_accepted must be positive and crossing_gate_cv_folds at least 2"
            )

        self.path("jaad_root")
        self.path("jaad_benchmark_results")
        self.path("jaad_context_results")
        for name in ("jaad_benchmark_split", "jaad_context_split"):
            if str(self.get(name)).strip().lower() not in {"train", "val", "test"}:
                raise ValueError(f"{name} must be one of: train, val, test")
        if not 0.0 < float(self.get("jaad_match_iou")) <= 1.0:
            raise ValueError("jaad_match_iou must be greater than 0 and at most 1")
        if int(self.get("jaad_min_match_frames")) <= 0:
            raise ValueError("jaad_min_match_frames must be positive")
        if not 0.0 < float(self.get("jaad_min_track_coverage")) <= 1.0:
            raise ValueError("jaad_min_track_coverage must be greater than 0 and at most 1")
        if self.jaad_context_settings()["sample_size"] < 0:
            raise ValueError("jaad_context_sample_size must be zero or positive")

        classifier = self.crossing_classifier_settings()
        if classifier["decision_mode"] not in {"classifier", "rules", "crowd_city"}:
            raise ValueError("crossing_decision_mode must be one of: classifier, rules, crowd_city")
        fallback = self.get("crossing_classifier_fallback_to_rules")
        if not isinstance(fallback, bool):
            raise ValueError("crossing_classifier_fallback_to_rules must be true or false")
        if classifier["min_track_frames"] < 1:
            raise ValueError("crossing_classifier_min_track_frames must be positive")
        if not 0.0 < classifier["min_precision"] <= 1.0:
            raise ValueError("crossing_classifier_min_precision must be greater than 0 and at most 1")
        if classifier["cv_folds"] < 2:
            raise ValueError("crossing_classifier_cv_folds must be at least 2")
        if not 0.0 < classifier["threshold_step"] < 1.0:
            raise ValueError("crossing_classifier_threshold_step must be greater than 0 and less than 1")
        for name in ("logistic_c_values", "gradient_learning_rates"):
            values = classifier[name]
            if not values or any(value <= 0.0 for value in values):
                raise ValueError(f"crossing_classifier_{name} must contain positive values")
        leaf_nodes = classifier["gradient_max_leaf_nodes"]
        if not leaf_nodes or any(value < 2 for value in leaf_nodes):
            raise ValueError(
                "crossing_classifier_gradient_max_leaf_nodes must contain integers of at least 2"
            )

        crowd = self.crowd_settings()
        for name in (
            "crowd_resume",
            "crowd_delete_downloaded_base_videos",
            "crowd_keep_segment_videos",
        ):
            value = self.get(name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false")
        if not crowd["ftp_server"]:
            raise ValueError("ftp_server must not be empty")
        if not crowd["ftp_aliases"] or any(not item for item in crowd["ftp_aliases"]):
            raise ValueError("crowd_ftp_aliases must contain at least one alias")
        if crowd["download_timeout_seconds"] < 1:
            raise ValueError("crowd_download_timeout_seconds must be positive")
        if crowd["download_max_pages"] < 1:
            raise ValueError("crowd_download_max_pages must be positive")
        if crowd["trim_end_margin_seconds"] < 0.0:
            raise ValueError("crowd_trim_end_margin_seconds must be non-negative")
        if crowd["max_segments"] < 0:
            raise ValueError("crowd_max_segments must be zero or positive")
        if crowd["audit_per_stratum"] < 1:
            raise ValueError("crowd_audit_per_stratum must be positive")
        if crowd["tracks_source"] not in CROWD_TRACK_SOURCES:
            raise ValueError("crowd_tracks_source must be one of: " + ", ".join(CROWD_TRACK_SOURCES))
        if not isinstance(self.get("crowd_bbox_dirs"), list):
            raise ValueError("crowd_bbox_dirs must be a list of folders")
        if crowd["tracks_source"] == "precomputed" and crowd["bbox_ftp_folder"]:
            if any(not item for item in crowd["bbox_ftp_aliases"]) or not crowd["bbox_ftp_aliases"]:
                raise ValueError("crowd_bbox_ftp_aliases must contain at least one alias")
        if crowd["tracks_source"] == "precomputed" and not (crowd["bbox_dirs"] or crowd["bbox_ftp_folder"]):
            raise ValueError("Precomputed CROWD tracks need crowd_bbox_dirs or crowd_bbox_ftp_folder")
