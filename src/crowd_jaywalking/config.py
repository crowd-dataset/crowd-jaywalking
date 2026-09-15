"""CROWD style flat project configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTIVE_CONFIG_NAME = "config"
DEFAULT_CONFIG_NAME = "default.config"

OPTIONAL_CROSSING_DEFAULTS = {
    "partial_crossing_enabled": True,
    "partial_exit_min_x_range": 0.48,
    "partial_exit_min_direction_consistency": 0.0,
    "perspective_corridor_enabled": False,
    "road_top_y": 0.0,
    "road_bottom_y": 1.0,
    "road_top_left": 0.45,
    "road_top_right": 0.55,
    "road_bottom_left": 0.45,
    "road_bottom_right": 0.55,
    "strong_complete_override_enabled": False,
    "strong_complete_min_seconds": 4.0,
    "strong_complete_min_x_range": 0.45,
    "strong_complete_min_direction_consistency": 0.85,
    "camera_min_shared_track_ratio": 0.0,
}

OPTIONAL_CLASSIFIER_DEFAULTS = {
    "crossing_classifier_results": "results/jaad_crossing_classifier_v1",
    "crossing_classifier_model": (
        "results/jaad_crossing_classifier_v1/crossing_classifier.joblib"
    ),
    "crossing_classifier_min_precision": 0.90,
    "crossing_classifier_cv_folds": 5,
    "crossing_classifier_threshold_step": 0.01,
    "crossing_classifier_random_seed": 42,
    "crossing_classifier_logistic_c_values": [0.10, 1.00, 10.00],
    "crossing_classifier_gradient_learning_rates": [0.05, 0.10],
    "crossing_classifier_gradient_max_leaf_nodes": [7, 15],
    "crossing_decision_mode": "classifier",
    "crossing_classifier_fallback_to_rules": False,
    "crossing_classifier_min_track_frames": 5,
}

OPTIONAL_CROWD_DEFAULTS = {
    "mapping": "mapping.csv",
    "ftp_server": "https://files.mobility-squad.com/",
    "crowd_results": "results/crowd_jaywalking_v2",
    "crowd_resume": True,
    "crowd_ftp_aliases": ["tue4", "tue5"],
    "crowd_download_dir": "data/crowd_downloads",
    "crowd_download_timeout_seconds": 20,
    "crowd_download_max_pages": 500,
    "crowd_trim_end_margin_seconds": 1.0,
    "crowd_delete_downloaded_base_videos": False,
    "crowd_keep_segment_videos": True,
    "crowd_max_segments": 0,
    "crowd_audit_random_seed": 42,
    "crowd_audit_per_stratum": 50,
}

OPTIONAL_VLM_COMPARISON_DEFAULTS = {
    "vlm_comparison_models": [
        "Qwen/Qwen3-VL-8B-Instruct",
        "google/gemma-4-12B-it",
    ],
    "vlm_comparison_prompt_modes": ["baseline_v3", "focused_v5"],
    "vlm_comparison_results": "results/jaad_vlm_comparison_v5",
}

OPTIONAL_EVIDENCE_DEFAULTS = {
    "evidence_trajectory_enabled": True,
    "evidence_road_crop_margin": 0.12,
    "evidence_control_crop_bottom": 0.78,
    "evidence_control_crop_overlap": 0.20,
    "vlm_task_max_frames": 4,
    "vlm_prompt_mode": "baseline_v3",
}

OPTIONAL_CONTEXT_AUDIT_DEFAULTS = {
    "jaad_context_sample_size": 120,
    "jaad_context_sampling_seed": 42,
}

CROSSING_KEYS = (
    "road_left",
    "road_right",
    "boundary_tolerance",
    "min_track_seconds",
    "min_road_seconds",
    "max_track_gap_seconds",
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

REQUIRED_KEYS = {
    "data",
    "videos",
    "source_annotations",
    "annotations",
    "evaluation_split",
    "results",
    "tracking_model",
    "bbox_tracker",
    "min_confidence",
    "iou",
    "device",
    *CROSSING_KEYS,
    "evidence_sample_positions",
    "evidence_context_seconds",
    "evidence_crop_margin",
    "evidence_max_dimension",
    "evidence_jpeg_quality",
    "vlm_model",
    "vlm_device_map",
    "vlm_torch_dtype",
    "vlm_attn_implementation",
    "vlm_cache_dir",
    "vlm_local_files_only",
    "vlm_min_pixels",
    "vlm_max_pixels",
    "vlm_max_new_tokens",
    "prohibitive_signal_overrides_crosswalk",
    "resume",
    "split_seed",
    "development_fraction",
    "validation_fraction",
    "locked_test_fraction",
    "jaad_root",
    "jaad_benchmark_split",
    "jaad_benchmark_results",
    "jaad_match_iou",
    "jaad_min_match_frames",
    "jaad_min_track_coverage",
    "jaad_context_split",
    "jaad_context_results",
}


@dataclass(frozen=True)
class ProjectConfig:
    """Validated flat project configuration and its source path."""

    source_path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProjectConfig":
        """Load an explicit config, or use config with default.config as fallback."""

        if path is None:
            active = (Path.cwd() / ACTIVE_CONFIG_NAME).resolve()
            fallback = (Path.cwd() / DEFAULT_CONFIG_NAME).resolve()
            source = active if active.is_file() else fallback
        else:
            source = Path(path).resolve()

        if not source.is_file():
            if path is None:
                raise FileNotFoundError(
                    f"Neither '{ACTIVE_CONFIG_NAME}' nor '{DEFAULT_CONFIG_NAME}' was found "
                    f"in {Path.cwd().resolve()}"
                )
            raise FileNotFoundError(f"Configuration file not found: {source}")

        try:
            with source.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Configuration is badly formatted: {source}. "
                f"Update it using '{DEFAULT_CONFIG_NAME}' as the template."
            ) from error

        if not isinstance(raw, dict):
            raise ValueError(f"Configuration root must be an object: {source}")

        config = cls(source_path=source, raw=raw)
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
        settings = {key: self.get(key) for key in CROSSING_KEYS}
        settings.update(
            {
                key: self.raw.get(key, default)
                for key, default in OPTIONAL_CROSSING_DEFAULTS.items()
            }
        )
        return settings

    def evidence_settings(self) -> dict[str, Any]:
        return {
            "sample_positions": self.get("evidence_sample_positions"),
            "context_seconds": self.get("evidence_context_seconds"),
            "crop_margin": self.get("evidence_crop_margin"),
            "max_dimension": self.get("evidence_max_dimension"),
            "jpeg_quality": self.get("evidence_jpeg_quality"),
            "trajectory_enabled": self.raw.get(
                "evidence_trajectory_enabled",
                OPTIONAL_EVIDENCE_DEFAULTS["evidence_trajectory_enabled"],
            ),
            "road_crop_margin": self.raw.get(
                "evidence_road_crop_margin",
                OPTIONAL_EVIDENCE_DEFAULTS["evidence_road_crop_margin"],
            ),
            "control_crop_bottom": self.raw.get(
                "evidence_control_crop_bottom",
                OPTIONAL_EVIDENCE_DEFAULTS["evidence_control_crop_bottom"],
            ),
            "control_crop_overlap": self.raw.get(
                "evidence_control_crop_overlap",
                OPTIONAL_EVIDENCE_DEFAULTS["evidence_control_crop_overlap"],
            ),
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
                self.raw.get(
                    "vlm_prompt_mode",
                    OPTIONAL_EVIDENCE_DEFAULTS["vlm_prompt_mode"],
                )
            ),
            "device_map": self.get("vlm_device_map"),
            "torch_dtype": self.get("vlm_torch_dtype"),
            "attn_implementation": self.get("vlm_attn_implementation"),
            "cache_dir": self.get("vlm_cache_dir"),
            "local_files_only": self.get("vlm_local_files_only"),
            "min_pixels": self.get("vlm_min_pixels"),
            "max_pixels": self.get("vlm_max_pixels"),
            "max_new_tokens": self.get("vlm_max_new_tokens"),
            "task_max_frames": self.raw.get(
                "vlm_task_max_frames",
                OPTIONAL_EVIDENCE_DEFAULTS["vlm_task_max_frames"],
            ),
        }

    def vlm_comparison_settings(self) -> dict[str, Any]:
        """Return model candidates and output location for VLM selection."""

        models_value = self.raw.get(
            "vlm_comparison_models",
            OPTIONAL_VLM_COMPARISON_DEFAULTS["vlm_comparison_models"],
        )
        if not isinstance(models_value, list):
            raise ValueError("vlm_comparison_models must be a list")
        prompt_modes_value = self.raw.get(
            "vlm_comparison_prompt_modes",
            OPTIONAL_VLM_COMPARISON_DEFAULTS["vlm_comparison_prompt_modes"],
        )
        if not isinstance(prompt_modes_value, list):
            raise ValueError("vlm_comparison_prompt_modes must be a list")
        return {
            "models": [str(item).strip() for item in models_value],
            "prompt_modes": [str(item).strip().lower() for item in prompt_modes_value],
            "results": self._resolved_optional_path(
                "vlm_comparison_results",
                OPTIONAL_VLM_COMPARISON_DEFAULTS["vlm_comparison_results"],
            ),
        }

    def jaad_context_settings(self) -> dict[str, int]:
        """Return reproducible context audit sampling settings."""

        return {
            "sample_size": int(
                self.raw.get(
                    "jaad_context_sample_size",
                    OPTIONAL_CONTEXT_AUDIT_DEFAULTS["jaad_context_sample_size"],
                )
            ),
            "sampling_seed": int(
                self.raw.get(
                    "jaad_context_sampling_seed",
                    OPTIONAL_CONTEXT_AUDIT_DEFAULTS["jaad_context_sampling_seed"],
                )
            ),
        }

    def policy_settings(self) -> dict[str, Any]:
        return {
            "prohibitive_signal_overrides_crosswalk": self.get(
                "prohibitive_signal_overrides_crosswalk"
            )
        }

    def crossing_classifier_settings(self) -> dict[str, Any]:
        """Return effective settings for supervised crossing classification."""

        value = lambda name: self.raw.get(name, OPTIONAL_CLASSIFIER_DEFAULTS[name])
        return {
            "benchmark_results": self.path("jaad_benchmark_results"),
            "results": self._resolved_optional_path(
                "crossing_classifier_results",
                OPTIONAL_CLASSIFIER_DEFAULTS["crossing_classifier_results"],
            ),
            "model": self._resolved_optional_path(
                "crossing_classifier_model",
                OPTIONAL_CLASSIFIER_DEFAULTS["crossing_classifier_model"],
            ),
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

        value = lambda name: self.raw.get(name, OPTIONAL_CROWD_DEFAULTS[name])
        return {
            "mapping": self._resolved_optional_path(
                "mapping", OPTIONAL_CROWD_DEFAULTS["mapping"]
            ),
            "ftp_server": str(value("ftp_server")).strip(),
            "results": self._resolved_optional_path(
                "crowd_results", OPTIONAL_CROWD_DEFAULTS["crowd_results"]
            ),
            "resume": bool(value("crowd_resume")),
            "ftp_aliases": [
                str(item).strip() for item in value("crowd_ftp_aliases")
            ],
            "download_dir": self._resolved_optional_path(
                "crowd_download_dir", OPTIONAL_CROWD_DEFAULTS["crowd_download_dir"]
            ),
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
        }

    def _resolved_optional_path(self, name: str, default: str) -> Path:
        value = self.raw.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Configuration entry must be a path string: {name}")
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()

    def fingerprint(self, prompt_version: str) -> str:
        effective_config = dict(self.raw)
        for key, default in OPTIONAL_CROSSING_DEFAULTS.items():
            effective_config.setdefault(key, default)
        for key, default in OPTIONAL_CLASSIFIER_DEFAULTS.items():
            effective_config.setdefault(key, default)
        for key, default in OPTIONAL_CROWD_DEFAULTS.items():
            effective_config.setdefault(key, default)
        for key, default in OPTIONAL_VLM_COMPARISON_DEFAULTS.items():
            effective_config.setdefault(key, default)
        for key, default in OPTIONAL_EVIDENCE_DEFAULTS.items():
            effective_config.setdefault(key, default)
        for key, default in OPTIONAL_CONTEXT_AUDIT_DEFAULTS.items():
            effective_config.setdefault(key, default)
        payload = {
            "config": effective_config,
            "prompt_version": prompt_version,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        missing = sorted(REQUIRED_KEYS.difference(self.raw))
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
        partial_crossing_enabled = self.raw.get(
            "partial_crossing_enabled",
            OPTIONAL_CROSSING_DEFAULTS["partial_crossing_enabled"],
        )
        if not isinstance(partial_crossing_enabled, bool):
            raise ValueError("partial_crossing_enabled must be true or false")
        partial_exit_min_x_range = float(
            self.raw.get(
                "partial_exit_min_x_range",
                OPTIONAL_CROSSING_DEFAULTS["partial_exit_min_x_range"],
            )
        )
        if not 0.0 <= partial_exit_min_x_range <= 1.0:
            raise ValueError("partial_exit_min_x_range must be between 0 and 1")

        for name in (
            "partial_exit_min_direction_consistency",
            "strong_complete_min_direction_consistency",
            "camera_min_shared_track_ratio",
        ):
            value = float(self.raw.get(name, OPTIONAL_CROSSING_DEFAULTS[name]))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        for name in (
            "perspective_corridor_enabled",
            "strong_complete_override_enabled",
        ):
            value = self.raw.get(name, OPTIONAL_CROSSING_DEFAULTS[name])
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false")

        road_top_y = float(self.raw.get("road_top_y", OPTIONAL_CROSSING_DEFAULTS["road_top_y"]))
        road_bottom_y = float(
            self.raw.get("road_bottom_y", OPTIONAL_CROSSING_DEFAULTS["road_bottom_y"])
        )
        if not 0.0 <= road_top_y < road_bottom_y <= 1.0:
            raise ValueError("road_top_y and road_bottom_y must satisfy 0 <= top < bottom <= 1")
        for prefix in ("road_top", "road_bottom"):
            corridor_left = float(
                self.raw.get(f"{prefix}_left", OPTIONAL_CROSSING_DEFAULTS[f"{prefix}_left"])
            )
            corridor_right = float(
                self.raw.get(f"{prefix}_right", OPTIONAL_CROSSING_DEFAULTS[f"{prefix}_right"])
            )
            if not 0.0 <= corridor_left < corridor_right <= 1.0:
                raise ValueError(
                    f"{prefix}_left and {prefix}_right must satisfy 0 <= left < right <= 1"
                )

        if float(
            self.raw.get(
                "strong_complete_min_seconds",
                OPTIONAL_CROSSING_DEFAULTS["strong_complete_min_seconds"],
            )
        ) < 0.0:
            raise ValueError("strong_complete_min_seconds must be non-negative")
        strong_complete_min_x_range = float(
            self.raw.get(
                "strong_complete_min_x_range",
                OPTIONAL_CROSSING_DEFAULTS["strong_complete_min_x_range"],
            )
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
        trajectory_enabled = self.raw.get(
            "evidence_trajectory_enabled",
            OPTIONAL_EVIDENCE_DEFAULTS["evidence_trajectory_enabled"],
        )
        if not isinstance(trajectory_enabled, bool):
            raise ValueError("evidence_trajectory_enabled must be true or false")
        if not 0.0 <= float(
            self.raw.get(
                "evidence_road_crop_margin",
                OPTIONAL_EVIDENCE_DEFAULTS["evidence_road_crop_margin"],
            )
        ) <= 0.5:
            raise ValueError("evidence_road_crop_margin must be between 0 and 0.5")
        for name in ("evidence_control_crop_bottom", "evidence_control_crop_overlap"):
            value = float(self.raw.get(name, OPTIONAL_EVIDENCE_DEFAULTS[name]))
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
            self.raw.get(
                "vlm_task_max_frames",
                OPTIONAL_EVIDENCE_DEFAULTS["vlm_task_max_frames"],
            )
        ) <= 0:
            raise ValueError("vlm_task_max_frames must be positive")

        valid_prompt_modes = {"baseline_v3", "focused_v5"}
        prompt_mode = str(
            self.raw.get(
                "vlm_prompt_mode",
                OPTIONAL_EVIDENCE_DEFAULTS["vlm_prompt_mode"],
            )
        ).strip().lower()
        if prompt_mode not in valid_prompt_modes:
            raise ValueError(
                "vlm_prompt_mode must be one of: baseline_v3, focused_v5"
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
                "vlm_comparison_prompt_modes may contain only baseline_v3 and focused_v5"
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
        if classifier["decision_mode"] not in {"classifier", "rules"}:
            raise ValueError("crossing_decision_mode must be one of: classifier, rules")
        fallback = self.raw.get(
            "crossing_classifier_fallback_to_rules",
            OPTIONAL_CLASSIFIER_DEFAULTS["crossing_classifier_fallback_to_rules"],
        )
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
            value = self.raw.get(name, OPTIONAL_CROWD_DEFAULTS[name])
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
