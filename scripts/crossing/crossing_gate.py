"""High precision second stage that confirms first stage crossing acceptances.

The first stage classifier was trained only on JAAD behaviour annotated
pedestrians, but at inference it scores every person track, including
bystanders it never saw during training. This gate is trained on every track the
first stage accepts. Its positive class is a track matched to a JAAD pedestrian
whose annotated crossing overlaps the track. Bystanders without behaviour labels
and unannotated detections are treated as negatives, so the precision it reports
is a lower bound on the true precision.

The gate stores one threshold per precision tier. The pipeline accepts people at
the loosest configured tier and records the strictest tier each person meets, so
one run can later be filtered to any tier without new inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from scripts.crossing.camera_motion import COMPENSATED_FEATURES, compensated_features, load_camera_motion_csv
from scripts.core.config import ProjectConfig
from scripts.crossing.crossing_classifier import (
    FEATURE_VERSION,
    CrossingClassifier,
    ModelCandidate,
    feature_matrix,
)
from scripts.crossing.track_features import person_tracks
from scripts.jaad.jaad import JAADDataset
from scripts.jaad.jaad_benchmark import box_iou
from scripts.crossing.model_crossing import ModelCrossingDetector
from scripts.core.tracking import load_observations_csv
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger


GATE_ARTIFACT_TYPE = "crowd_jaywalking_crossing_gate"
POSITIVE_CATEGORY = "labelled_crossing"
PREDICTION_FIELDS = (
    "jaad_split",
    "video_id",
    "track_id",
    "category",
    "target",
    "first_stage_probability",
    "gate_probability",
    "gate_precision_tier",
)


def select_precision_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum_precision: float,
    minimum_accepted: int,
) -> float | None:
    """Return the lowest threshold whose precision meets the target.

    Thresholds that accept fewer than ``minimum_accepted`` tracks are ignored,
    because a handful of lucky acceptances says little about precision.
    """

    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    true_positives = np.cumsum(labels[order].astype(int))
    best: float | None = None
    for index, threshold in enumerate(sorted_probabilities):
        # Only evaluate at the last occurrence of tied probabilities.
        if index + 1 < len(order) and sorted_probabilities[index + 1] == threshold:
            continue
        accepted = index + 1
        if accepted < minimum_accepted:
            continue
        if true_positives[index] / accepted >= minimum_precision:
            best = float(threshold)
    return best


def gate_matrix(rows: Sequence[dict[str, Any]], extra_features: Sequence[str]) -> np.ndarray:
    """First stage feature matrix with extra numeric features before the categorical ones."""

    base = feature_matrix(rows)
    if not extra_features:
        return base
    extra = np.array(
        [[float(row.get(name, np.nan)) for name in extra_features] for row in rows], dtype=object
    ).reshape(len(rows), len(extra_features))
    return np.hstack([base[:, :-3], extra, base[:, -3:]])


def _fit_gate(
    candidate: ModelCandidate,
    seed: int,
    matrix: np.ndarray,
    labels: np.ndarray,
    extra_count: int,
) -> Pipeline:
    numeric_count = matrix.shape[1] - 3
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(range(numeric_count))),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), list(range(numeric_count, matrix.shape[1]))),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=float(candidate.parameters["learning_rate"]),
                    max_leaf_nodes=int(candidate.parameters["max_leaf_nodes"]),
                    max_iter=200,
                    min_samples_leaf=10,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(matrix, labels, classifier__sample_weight=compute_sample_weight("balanced", labels))
    return model


def gate_metrics(labels: np.ndarray, accepted: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(labels & accepted))
    fp = int(np.sum(~labels & accepted))
    fn = int(np.sum(labels & ~accepted))
    return {
        "accepted_tracks": int(np.sum(accepted)),
        "true_crossings_accepted": tp,
        "other_tracks_accepted": fp,
        "true_crossings_rejected": fn,
        "worst_case_precision_percent": 100.0 * tp / (tp + fp) if tp + fp else None,
        "recall_percent": 100.0 * tp / (tp + fn) if tp + fn else None,
    }


class CrossingGate:
    """Load and apply the frozen crossing gate artifact."""

    def __init__(self, artifact: dict[str, Any], min_precision: float) -> None:
        if artifact.get("artifact_type") != GATE_ARTIFACT_TYPE:
            raise ValueError("The file is not a crossing gate artifact")
        if artifact.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Unsupported crossing gate feature version: {artifact.get('feature_version')!r}"
            )
        # Tiers are stored as {precision: threshold}; strictest first.
        self.tiers = sorted(
            ((float(precision), float(threshold))
             for precision, threshold in artifact["tier_thresholds"].items()),
            reverse=True,
        )
        accepted = [threshold for precision, threshold in self.tiers if precision == min_precision]
        if not accepted:
            raise ValueError(
                f"crossing_gate_min_precision {min_precision} is not a trained tier. "
                f"Available tiers: {', '.join(str(precision) for precision, _ in self.tiers)}"
            )
        self.artifact = artifact
        self.model = artifact["pipeline"]
        # Gates trained with camera compensated motion need those features at inference.
        self.extra_features = list(artifact.get("extra_features", []))
        self.min_precision = float(min_precision)
        self.threshold = accepted[0]

    @classmethod
    def load(cls, path: str | Path, min_precision: float) -> "CrossingGate":
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"Crossing gate model not found: {source}. Run scripts/crossing/train_crossing_gate.py first."
            )
        artifact = joblib.load(source)
        if not isinstance(artifact, dict):
            raise ValueError(f"Crossing gate artifact is invalid: {source}")
        return cls(artifact, min_precision)

    @property
    def needs_camera_motion(self) -> bool:
        return any(name in COMPENSATED_FEATURES for name in self.extra_features)

    def predict_probabilities(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        missing = [name for name in self.extra_features if rows and name not in rows[0]]
        if missing:
            raise ValueError(f"The crossing gate needs features that were not computed: {missing}")
        return self.model.predict_proba(gate_matrix(rows, self.extra_features))[:, 1]

    def tier(self, probability: float) -> float | None:
        """Return the strictest precision tier this probability meets."""

        return tier_for(self.tiers, probability)


def tier_for(tiers: Sequence[tuple[float, float]], probability: float) -> float | None:
    for precision, threshold in sorted(tiers, reverse=True):
        if probability >= threshold:
            return precision
    return None


class CrossingGateTrainer:
    """Train on JAAD train and val, then check once on JAAD test.

    Tier thresholds come from grouped out of fold predictions over train and val.
    Videos in the project's locked test split are excluded everywhere, so the
    locked end to end evaluation stays untouched.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.settings = config.crossing_gate_settings()
        classifier_settings = config.crossing_classifier_settings()
        self.benchmark_root = Path(classifier_settings["benchmark_results"])
        self.detector = ModelCrossingDetector(
            CrossingClassifier.load(classifier_settings["model"]),
            config.crossing_settings(),
            classifier_settings["min_track_frames"],
        )
        self.dataset = JAADDataset(config.path("jaad_root"))
        self.output_dir = Path(self.settings["results"])
        self.model_path = Path(self.settings["model"])
        self.extra_features = list(COMPENSATED_FEATURES) if self.settings["camera_motion"] else []
        self.camera_motion_dir = self.benchmark_root / "camera_motion"

    def run(self) -> dict[str, Any]:
        excluded = self._locked_test_videos()
        fit_rows = self._rows("train", excluded) + self._rows("val", excluded)
        check_rows = self._rows("test", excluded)
        fit_labels = np.array([row["target"] for row in fit_rows], dtype=bool)
        check_labels = np.array([row["target"] for row in check_rows], dtype=bool)
        if len(np.unique(fit_labels)) < 2:
            raise ValueError("The gate training data contains only one class")

        candidate = ModelCandidate(
            "hist_gradient_boosting",
            {
                "learning_rate": self.settings["learning_rate"],
                "max_leaf_nodes": self.settings["max_leaf_nodes"],
            },
        )
        seed = int(self.settings["random_seed"])
        extra_count = len(self.extra_features)
        matrix = gate_matrix(fit_rows, self.extra_features)
        groups = np.array([row["video_id"] for row in fit_rows])
        out_of_fold = np.zeros(len(fit_rows), dtype=float)
        for fit_index, score_index in GroupKFold(int(self.settings["cv_folds"])).split(
            matrix, fit_labels, groups
        ):
            model = _fit_gate(candidate, seed, matrix[fit_index], fit_labels[fit_index], extra_count)
            out_of_fold[score_index] = model.predict_proba(matrix[score_index])[:, 1]

        tier_thresholds: dict[str, float] = {}
        for precision in self.settings["precision_tiers"]:
            threshold = select_precision_threshold(
                fit_labels, out_of_fold, precision, int(self.settings["min_accepted"])
            )
            if threshold is None:
                raise ValueError(
                    f"No gate threshold reaches precision {precision} on the out of "
                    "fold JAAD train and val predictions. Remove that tier."
                )
            tier_thresholds[str(precision)] = threshold
        tiers = [(float(key), value) for key, value in tier_thresholds.items()]

        final_model = _fit_gate(candidate, seed, matrix, fit_labels, extra_count)
        check_probabilities = (
            final_model.predict_proba(gate_matrix(check_rows, self.extra_features))[:, 1]
            if check_rows
            else np.zeros(0)
        )
        tier_report = {
            key: {
                "threshold": threshold,
                "cross_validated": gate_metrics(fit_labels, out_of_fold >= threshold),
                "jaad_test_check": gate_metrics(check_labels, check_probabilities >= threshold),
            }
            for key, threshold in tier_thresholds.items()
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "artifact_type": GATE_ARTIFACT_TYPE,
            "feature_version": FEATURE_VERSION,
            "model_family": candidate.family,
            "parameters": candidate.parameters,
            "tier_thresholds": tier_thresholds,
            "tier_report": tier_report,
            "extra_features": self.extra_features,
            "fit_splits": ["train", "val"],
            "check_split": "test",
            "excluded_locked_test_videos": sorted(excluded),
            "pipeline": final_model,
        }
        joblib.dump(artifact, self.model_path)

        self._write_predictions(
            [*fit_rows, *check_rows],
            np.concatenate([out_of_fold, check_probabilities]),
            tiers,
        )
        summary = {
            "tiers": tier_report,
            "fit_first_stage_accepted_tracks": len(fit_rows),
            "fit_categories": dict(Counter(row["category"] for row in fit_rows)),
            "check_first_stage_accepted_tracks": len(check_rows),
            "check_categories": dict(Counter(row["category"] for row in check_rows)),
            "fit_first_stage_metrics": gate_metrics(
                fit_labels, np.ones(len(fit_rows), dtype=bool)
            ),
            "check_first_stage_metrics": gate_metrics(
                check_labels, np.ones(len(check_rows), dtype=bool)
            ),
            "model_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        self._print_summary(summary)
        return summary

    def _locked_test_videos(self) -> set[str]:
        path = self.config.data_file("annotations")
        if not path.is_file():
            raise FileNotFoundError(
                f"Split annotations not found: {path}. Run scripts/evaluation/prepare_splits.py first, "
                "so the locked test videos can be excluded from gate training."
            )
        with path.open("r", encoding="utf-8", newline="") as handle:
            return {
                row["video_id"]
                for row in csv.DictReader(handle)
                if row.get("split") == "locked_test"
            }

    def _rows(self, split: str, excluded: set[str]) -> list[dict[str, Any]]:
        """Label every first stage acceptance in one saved JAAD benchmark split."""

        import cv2

        tracks_dir = self.benchmark_root / split / "tracks"
        if not tracks_dir.is_dir():
            raise FileNotFoundError(
                f"No saved JAAD tracks: {tracks_dir}. "
                f"Run scripts/jaad/run_jaad_crossing_benchmark.py with jaad_benchmark_split set to {split}."
            )
        rows: list[dict[str, Any]] = []
        for path in sorted(tracks_dir.glob("*.csv")):
            video_id = path.stem
            if video_id in excluded:
                continue
            capture = cv2.VideoCapture(str(self.dataset.clip_path(video_id)))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            capture.release()
            observations = load_observations_csv(path)
            annotations = self.dataset.load_video(video_id)
            boxes: dict[int, dict[int, Any]] = defaultdict(dict)
            for observation in observations:
                if observation.class_id == 0:
                    boxes[observation.track_id][observation.frame_index] = observation.box
            result = self.detector.detect(observations, fps)
            scene = {}
            if self.extra_features:
                motion_path = self.camera_motion_dir / f"{video_id}.csv"
                if not motion_path.is_file():
                    raise FileNotFoundError(
                        f"No camera motion for {video_id}: {motion_path}. Compute it with "
                        "scripts.crossing.camera_motion before training a camera motion gate."
                    )
                motion = load_camera_motion_csv(motion_path)
                tracks = person_tracks(observations)
            for classification in result.classifications:
                if not classification.predicted_crossing:
                    continue
                if self.extra_features:
                    scene = compensated_features(tracks[classification.person_id], motion)
                category = self._category(
                    annotations, boxes[classification.person_id], classification.event
                )
                rows.append(
                    {
                        **classification.track_features,
                        **scene,
                        "jaad_split": split,
                        "video_id": video_id,
                        "track_id": classification.person_id,
                        "category": category,
                        "target": category == POSITIVE_CATEGORY,
                        "first_stage_probability": classification.probability,
                    }
                )
        return rows

    def _category(self, annotations, track_boxes: dict[int, Any], event) -> str:
        iou_threshold = float(self.config.get("jaad_match_iou"))
        minimum_frames = int(self.config.get("jaad_min_match_frames"))
        best = None
        for ground_truth in annotations.tracks.values():
            matched = sum(
                1
                for frame in ground_truth.frames
                if frame in track_boxes
                and box_iou(ground_truth.boxes[frame], track_boxes[frame]) >= iou_threshold
            )
            if matched >= minimum_frames and (best is None or matched > best[0]):
                best = (matched, ground_truth)
        if best is None:
            return "unannotated"
        ground_truth = best[1]
        if not ground_truth.behaviour_annotated:
            return "bystander"
        if not ground_truth.is_crossing:
            return "labelled_not_crossing"
        overlap = set(ground_truth.crossing_frames).intersection(
            range(event.start_frame, event.end_frame + 1)
        )
        return POSITIVE_CATEGORY if overlap else "labelled_crossing_other_time"

    def _write_predictions(
        self,
        rows: list[dict[str, Any]],
        probabilities: np.ndarray,
        tiers: Sequence[tuple[float, float]],
    ) -> None:
        path = self.output_dir / "gate_predictions.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for row, probability in zip(rows, probabilities):
                writer.writerow(
                    {
                        "jaad_split": row["jaad_split"],
                        "video_id": row["video_id"],
                        "track_id": row["track_id"],
                        "category": row["category"],
                        "target": row["target"],
                        "first_stage_probability": row["first_stage_probability"],
                        "gate_probability": round(float(probability), 8),
                        "gate_precision_tier": tier_for(tiers, float(probability)),
                    }
                )

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        def show(name: str, metrics: dict[str, Any]) -> None:
            precision = metrics["worst_case_precision_percent"]
            recall = metrics["recall_percent"]
            logger.info(
                "{:<34} accepted {:4d}  true crossings {:4d}  worst case precision {}  recall {}",
                name,
                metrics['accepted_tracks'],
                metrics['true_crossings_accepted'],
                'n/a' if precision is None else f'{precision:6.2f}%',
                'n/a' if recall is None else f'{recall:6.2f}%',
            )

        show("Train+val, first stage only", summary["fit_first_stage_metrics"])
        show("JAAD test, first stage only", summary["check_first_stage_metrics"])
        for key, tier in summary["tiers"].items():
            logger.info("Tier {:.0%} (threshold {:.4f})", float(key), tier['threshold'])
            show("  train+val, out of fold", tier["cross_validated"])
            show("  JAAD test check", tier["jaad_test_check"])
