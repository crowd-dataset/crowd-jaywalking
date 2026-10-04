"""The method's fixed settings: the values the results in the README were made with.

They live in the code, not in default.config, because they define the method rather than
a machine or a run. default.config keeps what a user sets: paths, which videos or split
to run, models, GPU memory, the CROWD file server, and logging.

An entry of the same name in the local ``config`` still overrides a value here, and the
run logs every such override. That is how the earlier designs in the README are
reproduced; for the current method, leave them out.
"""

from __future__ import annotations

from typing import Any

METHOD_SETTINGS: dict[str, Any] = {
    # Tracking: the BoT-SORT settings of crowd-city and the detector thresholds.
    "bbox_tracker": "configs/botsort.yaml",
    "min_confidence": 0.0,
    "iou": 0.7,

    # Stage 1 and the VLM crossing check. crossing_decision_mode "crowd_city" is the current
    # method; "classifier" and "rules" are the earlier designs (see the README).
    "crossing_decision_mode": "crowd_city",
    "crossing_vlm_check": True,
    "crossing_vlm_check_version": "v3",
    "crossing_rescue_min_first_stage": None,
    "crossing_rescue_min_gate": None,

    # Earlier designs: this project's rule detector and the track features it measures.
    # Fractions are of the image width (x) or height (y); seconds become frames at the video's rate.
    "road_left": 0.45,
    "road_right": 0.55,
    "boundary_tolerance": 0.0,
    "perspective_corridor_enabled": True,
    "road_top_y": 0.35,
    "road_bottom_y": 1.0,
    "road_top_left": 0.45,
    "road_top_right": 0.55,
    "road_bottom_left": 0.3,
    "road_bottom_right": 0.7,
    "min_track_seconds": 0.33,
    "min_road_seconds": 0.1,
    "max_track_gap_seconds": 1.0,
    "partial_crossing_enabled": True,
    "partial_exit_min_x_range": 0.38,
    "partial_exit_min_direction_consistency": 0.85,
    "strong_complete_override_enabled": True,
    "strong_complete_min_seconds": 4.0,
    "strong_complete_min_x_range": 0.45,
    "strong_complete_min_direction_consistency": 0.85,
    "min_crossing_x_range": 0.14,
    "max_crossing_speed_per_frame": None,
    "low_x_range": 0.3,
    "low_x_min_road_seconds": 0.67,
    "weak_x_range": 0.64,
    "long_weak_road_seconds": 3.0,
    "weak_y_jitter_x_range": 0.5,
    "weak_y_jitter_motion": 0.3,
    "weak_y_jitter_height": 0.22,
    "jitter_road_seconds": 1.333333,
    "tiny_long_track_x_range": 0.36,
    "tiny_long_track_height": 0.12,
    "tiny_long_track_road_seconds": 1.666667,
    "tiny_no_static_height": 0.12,
    "tiny_no_static_width": 0.026,
    "tiny_no_static_min_road_seconds": 0.333333,
    "no_static_tiny_min_road_seconds": 0.166667,
    "no_static_tiny_fast_speed": 0.006,
    "slender_track_width": 0.05,
    "slender_track_height": 0.26,
    "slender_track_min_road_seconds": 0.166667,
    "slender_track_max_road_seconds": 1.633333,
    "no_static_slender_height": 0.24,
    "no_static_slender_max_road_seconds": 0.666667,
    "slender_static_min_relative_x_range": 0.13,
    "large_lateral_x_range": 0.56,
    "large_lateral_tiny_height": 0.105,
    "min_static_shared_seconds": 0.27,
    "camera_static_x_range": 0.25,
    "camera_ratio_threshold": 0.6,
    "camera_min_shared_track_ratio": 0.25,
    "camera_static_relative_x_range": 0.18,
    "camera_static_height": 0.15,
    "camera_static_tiny_relative_x_range": 0.12,
    "camera_static_tiny_height": 0.19,
    "camera_tiny_height": 0.15,
    "camera_min_road_seconds": 0.166667,
    "min_relative_x_range": 0.01,
    "rider_min_shared_seconds": 0.133333,
    "rider_min_continuous_shared_seconds": 0.4,
    "rider_shared_run_gap_seconds": 0.066667,
    "rider_min_vehicle_width_ratio": 0.5,
    "rider_min_vehicle_width_ratio_frames": 0.65,
    "rider_distance_relative_threshold": 0.8,
    "rider_proximity_ratio": 0.7,
    "rider_alpha_x": 0.75,
    "rider_beta_y": 0.08,
    "rider_gamma_y": 1.4,
    "rider_colocation_ratio": 0.7,
    "rider_similarity_threshold": 0.4,
    "rider_similarity_ratio": 0.5,
    "rider_min_motion_seconds": 0.1,
    "rider_motion_colocation_min": 0.5,
    "rider_short_shared_seconds": 0.266667,
    "rider_short_similarity_ratio": 0.8,
    "rider_short_displacement": 0.12,

    # Earlier designs: the JAAD trained crossing classifier.
    "crossing_classifier_results": "results/jaad_crossing_classifier_yolo11x",
    "crossing_classifier_model": "results/jaad_crossing_classifier_yolo11x/crossing_classifier.joblib",
    "crossing_classifier_min_precision": 0.9,
    "crossing_classifier_cv_folds": 5,
    "crossing_classifier_threshold_step": 0.01,
    "crossing_classifier_random_seed": 42,
    "crossing_classifier_logistic_c_values": [0.1, 1.0, 10.0],
    "crossing_classifier_gradient_learning_rates": [0.05, 0.1],
    "crossing_classifier_gradient_max_leaf_nodes": [7, 15],
    "crossing_classifier_fallback_to_rules": False,
    "crossing_classifier_min_track_frames": 5,

    # Earlier designs: the high precision crossing gate.
    "crossing_gate_model": "results/jaad_crossing_gate_yolo11x/crossing_gate.joblib",
    "crossing_gate_results": "results/jaad_crossing_gate_yolo11x",
    "crossing_gate_min_precision": 0.95,
    "crossing_gate_precision_tiers": [0.98, 0.95, 0.9],
    "crossing_gate_camera_motion": True,
    "crossing_gate_min_accepted": 30,
    "crossing_gate_cv_folds": 5,
    "crossing_gate_learning_rate": 0.05,
    "crossing_gate_max_leaf_nodes": 15,
    "crossing_gate_random_seed": 42,
    "crossing_min_scene_x_range": 0.2,

    # Evidence images around each crossing, for the VLM.
    "evidence_sample_positions": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "evidence_context_seconds": 0.5,
    "evidence_crop_margin": 0.75,
    "evidence_max_dimension": 1280,
    "evidence_jpeg_quality": 90,
    "evidence_trajectory_enabled": True,
    "evidence_road_crop_margin": 0.12,
    "evidence_control_crop_bottom": 0.78,
    "evidence_control_crop_overlap": 0.2,
    "evidence_infrastructure_span": "transition",

    # VLM prompts and answer length (model and memory settings stay in default.config).
    "vlm_prompt_mode": "zebra_light_v5",
    "vlm_max_new_tokens": 300,
    "vlm_task_max_frames": 4,

    # Decision policy: a claim needs every infrastructure answer to be NO for every crossing
    # person within context_scope_window_seconds.
    "prohibitive_signal_overrides_crosswalk": False,
    "permission_cues": ["marked_crosswalk", "permissive_pedestrian_signal", "traffic_light"],
    "partial_visibility_uncertain": False,
    "context_scope": "scene",
    "strict_absence": True,
    "context_scope_window_seconds": 10.0,

    # JAAD evaluation: matching a track to a JAAD pedestrian, and the context benchmark sample.
    "jaad_match_iou": 0.5,
    "jaad_min_match_frames": 5,
    "jaad_min_track_coverage": 0.1,
    "jaad_context_sample_size": 120,
    "jaad_context_sampling_seed": 42,

    # Earlier designs: the labelled video evaluation splits and the VLM comparison.
    "split_seed": 42,
    "development_fraction": 0.6,
    "validation_fraction": 0.2,
    "locked_test_fraction": 0.2,
    "vlm_comparison_models": ["Qwen/Qwen3-VL-8B-Instruct", "google/gemma-4-12B-it"],
    "vlm_comparison_prompt_modes": ["baseline_v3", "focused_v5"],

    # Approach review (scripts/review): frames from this many seconds before a claimed crossing, down to
    # the end offset, to see a zebra crossing the crossing window itself misses. Review flag only.
    "approach_review_seconds": 6.0,
    "approach_review_end_seconds": 0.5,
    "approach_review_frames": 4,
    "approach_review_crop_bottom": 0.12,

    # CROWD: the segment cut must match CROWD's own tracking (one second off the end), the
    # file server crawl limits, and the manual audit sample.
    "crowd_trim_end_margin_seconds": 1.0,
    "crowd_download_timeout_seconds": 20,
    "crowd_download_max_pages": 500,
    "crowd_audit_random_seed": 42,
    "crowd_audit_per_stratum": 50,
}
