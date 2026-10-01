"""Train and apply an inference safe classifier for pedestrian crossings.

The classifier deliberately uses only measurements that can be calculated from
YOLO tracks at inference time. JAAD matching quality, annotation intervals,
pedestrian identifiers, rule outcomes, and ground truth fields are excluded.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from scripts.core.config import ProjectConfig
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger


FEATURE_VERSION = "yolo_track_features_v1"

NUMERIC_FEATURES = (
    "matched_track_frames",
    "matched_track_duration_frames",
    "matched_track_duration_seconds",
    "matched_track_max_gap_frames",
    "matched_track_segment_count",
    "matched_track_longest_segment_frames",
    "matched_track_x_range",
    "matched_track_net_x_displacement",
    "matched_track_signed_x_displacement",
    "matched_track_gross_x_motion",
    "matched_track_x_direction_consistency",
    "matched_track_x_range_over_height",
    "matched_track_y_range",
    "matched_track_net_y_displacement",
    "matched_track_signed_y_displacement",
    "matched_track_gross_y_motion",
    "matched_track_y_direction_consistency",
    "matched_track_y_range_over_height",
    "matched_track_bottom_y_range",
    "matched_track_height_change_ratio",
    "matched_track_median_width",
    "matched_track_median_height",
    "matched_track_left_frames",
    "matched_track_road_frames",
    "matched_track_right_frames",
    "matched_track_longest_road_run",
    "matched_track_static_shared_frames",
    "matched_track_static_x_range",
    "matched_track_relative_x_range",
    "matched_track_camera_motion_ratio",
)

DERIVED_NUMERIC_FEATURES = (
    "derived_track_observation_ratio",
    "derived_longest_segment_ratio",
    "derived_left_frame_ratio",
    "derived_road_frame_ratio",
    "derived_right_frame_ratio",
    "derived_longest_road_run_ratio",
    "derived_static_shared_ratio",
)

CATEGORICAL_FEATURES = (
    "matched_track_start_state",
    "matched_track_end_state",
    "matched_track_complete_transition",
)

ALL_FEATURES = NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_SELECTION_FIELDS = (
    "selected",
    "family_cv_winner",
    "model_family",
    "parameters",
    "cv_threshold",
    "threshold",
    "validation_threshold_calibrated",
    "precision_constraint_satisfied",
    "validation_precision_constraint_satisfied",
    "cv_tp",
    "cv_tn",
    "cv_fp",
    "cv_fn",
    "cv_accuracy_percent",
    "cv_precision_percent",
    "cv_recall_percent",
    "cv_specificity_percent",
    "cv_balanced_accuracy_percent",
    "cv_f1_percent",
    "validation_tp",
    "validation_tn",
    "validation_fp",
    "validation_fn",
    "validation_accuracy_percent",
    "validation_precision_percent",
    "validation_recall_percent",
    "validation_specificity_percent",
    "validation_balanced_accuracy_percent",
    "validation_f1_percent",
)

VALIDATION_PREDICTION_FIELDS = (
    "video_id",
    "pedestrian_id",
    "track_matched",
    "matched_track_id",
    "ground_truth_crossing",
    "rule_prediction",
    "classifier_probability",
    "classifier_prediction",
    "correct",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in {"true", "1", "yes", "y"}:
        return True
    if normalised in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot interpret as a Boolean value: {value!r}")


def _as_float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Cannot interpret as a number: {value!r}") from error
    return parsed if np.isfinite(parsed) else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return 0.0
    return numerator / denominator


def feature_record(row: dict[str, Any]) -> dict[str, float | str]:
    """Build the leakage free feature record used by training and inference."""

    numeric = {name: _as_float(row.get(name)) for name in NUMERIC_FEATURES}
    frames = numeric["matched_track_frames"]
    duration = numeric["matched_track_duration_frames"]
    numeric.update(
        {
            "derived_track_observation_ratio": _ratio(frames, duration),
            "derived_longest_segment_ratio": _ratio(
                numeric["matched_track_longest_segment_frames"], frames
            ),
            "derived_left_frame_ratio": _ratio(
                numeric["matched_track_left_frames"], frames
            ),
            "derived_road_frame_ratio": _ratio(
                numeric["matched_track_road_frames"], frames
            ),
            "derived_right_frame_ratio": _ratio(
                numeric["matched_track_right_frames"], frames
            ),
            "derived_longest_road_run_ratio": _ratio(
                numeric["matched_track_longest_road_run"], frames
            ),
            "derived_static_shared_ratio": _ratio(
                numeric["matched_track_static_shared_frames"], frames
            ),
        }
    )
    result: dict[str, float | str] = dict(numeric)
    result["matched_track_start_state"] = (
        str(row.get("matched_track_start_state", "UNKNOWN")).strip().upper() or "UNKNOWN"
    )
    result["matched_track_end_state"] = (
        str(row.get("matched_track_end_state", "UNKNOWN")).strip().upper() or "UNKNOWN"
    )
    result["matched_track_complete_transition"] = (
        "TRUE" if _as_bool(row.get("matched_track_complete_transition", False)) else "FALSE"
    )
    return result


def feature_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Convert benchmark shaped rows into the stable model matrix."""

    matrix = np.empty((len(rows), len(ALL_FEATURES)), dtype=object)
    for row_index, row in enumerate(rows):
        features = feature_record(row)
        for column_index, name in enumerate(ALL_FEATURES):
            matrix[row_index, column_index] = features[name]
    return matrix


def classification_metrics(
    ground_truth: Sequence[bool] | np.ndarray,
    predictions: Sequence[bool] | np.ndarray,
) -> dict[str, float | int | None]:
    """Calculate complete binary classification metrics as percentages."""

    truth = np.asarray(ground_truth, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError("Ground truth and predictions must have the same shape")

    tp = int(np.sum(truth & predicted))
    tn = int(np.sum(~truth & ~predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))

    def percentage(numerator: float, denominator: float) -> float | None:
        return 100.0 * numerator / denominator if denominator else None

    precision = percentage(tp, tp + fp)
    recall = percentage(tp, tp + fn)
    specificity = percentage(tn, tn + fp)
    balanced = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_percent": percentage(tp + tn, len(truth)),
        "precision_percent": precision,
        "recall_percent": recall,
        "specificity_percent": specificity,
        "balanced_accuracy_percent": balanced,
        "f1_percent": percentage(2 * tp, 2 * tp + fp + fn),
    }


def _metric_value(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    return float(value) if value is not None else -1.0


def select_threshold(
    ground_truth: Sequence[bool] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    minimum_precision: float,
    step: float,
) -> dict[str, Any]:
    """Select a recall oriented threshold subject to minimum precision."""

    truth = np.asarray(ground_truth, dtype=bool)
    scores = np.asarray(probabilities, dtype=float)
    if truth.shape != scores.shape:
        raise ValueError("Ground truth and probabilities must have the same shape")
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be greater than 0 and at most 1")
    if not 0.0 < step < 1.0:
        raise ValueError("step must be greater than 0 and less than 1")

    thresholds = np.arange(step, 1.0 + step / 2.0, step)
    evaluated: list[dict[str, Any]] = []
    for raw_threshold in thresholds:
        threshold = min(1.0, round(float(raw_threshold), 10))
        metrics = classification_metrics(truth, scores >= threshold)
        precision = metrics["precision_percent"]
        feasible = precision is not None and precision >= 100.0 * minimum_precision
        evaluated.append(
            {
                "threshold": threshold,
                "precision_constraint_satisfied": feasible,
                "metrics": metrics,
            }
        )

    feasible = [item for item in evaluated if item["precision_constraint_satisfied"]]
    candidates = feasible or evaluated
    if feasible:
        rank = lambda item: (
            _metric_value(item["metrics"], "balanced_accuracy_percent"),
            _metric_value(item["metrics"], "recall_percent"),
            _metric_value(item["metrics"], "f1_percent"),
            _metric_value(item["metrics"], "precision_percent"),
            -float(item["threshold"]),
        )
    else:
        rank = lambda item: (
            _metric_value(item["metrics"], "precision_percent"),
            _metric_value(item["metrics"], "balanced_accuracy_percent"),
            _metric_value(item["metrics"], "recall_percent"),
            _metric_value(item["metrics"], "f1_percent"),
            -float(item["threshold"]),
        )
    return max(candidates, key=rank)


def load_benchmark_rows(path: Path) -> list[dict[str, str]]:
    """Load and validate one JAAD per pedestrian benchmark CSV."""

    if not path.is_file():
        raise FileNotFoundError(
            f"JAAD benchmark output not found: {path}. "
            "Run scripts/jaad/run_jaad_crossing_benchmark.py for this split first."
        )
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    required = {
        "video_id",
        "pedestrian_id",
        "ground_truth_crossing",
        "track_matched",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if not rows:
        raise ValueError(f"JAAD benchmark output contains no rows: {path}")
    return rows


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _labels(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_as_bool(row["ground_truth_crossing"]) for row in rows], dtype=bool)


def _matched_indices(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [index for index, row in enumerate(rows) if _as_bool(row["track_matched"])],
        dtype=int,
    )


def _preprocessor() -> ColumnTransformer:
    numeric_indices = list(range(len(NUMERIC_FEATURES) + len(DERIVED_NUMERIC_FEATURES)))
    categorical_indices = list(range(len(numeric_indices), len(ALL_FEATURES)))
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, numeric_indices), ("categorical", categorical, categorical_indices)],
        remainder="drop",
    )


@dataclass(frozen=True)
class ModelCandidate:
    family: str
    parameters: dict[str, float | int]


def _pipeline(candidate: ModelCandidate, random_seed: int) -> Pipeline:
    if candidate.family == "logistic_regression":
        classifier = LogisticRegression(
            C=float(candidate.parameters["C"]),
            max_iter=3000,
            solver="lbfgs",
            random_state=random_seed,
        )
    elif candidate.family == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=float(candidate.parameters["learning_rate"]),
            max_leaf_nodes=int(candidate.parameters["max_leaf_nodes"]),
            max_iter=200,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=random_seed,
        )
    else:
        raise ValueError(f"Unknown model family: {candidate.family}")
    return Pipeline([("preprocess", _preprocessor()), ("classifier", classifier)])


def _fit_model(
    candidate: ModelCandidate,
    random_seed: int,
    matrix: np.ndarray,
    labels: np.ndarray,
) -> Pipeline:
    if len(np.unique(labels)) < 2:
        raise ValueError("A training fold contains only one ground truth class")
    model = _pipeline(candidate, random_seed)
    weights = compute_sample_weight(class_weight="balanced", y=labels)
    model.fit(matrix, labels, classifier__sample_weight=weights)
    return model


def _probabilities_for_all_rows(
    model: Pipeline,
    rows: Sequence[dict[str, Any]],
) -> np.ndarray:
    probabilities = np.zeros(len(rows), dtype=float)
    matched = _matched_indices(rows)
    if len(matched):
        matched_rows = [rows[index] for index in matched]
        probabilities[matched] = model.predict_proba(feature_matrix(matched_rows))[:, 1]
    return probabilities


def _candidate_rank(record: dict[str, Any]) -> tuple[float, ...]:
    metrics = record["cv_metrics"]
    return (
        1.0 if record["precision_constraint_satisfied"] else 0.0,
        _metric_value(metrics, "balanced_accuracy_percent"),
        _metric_value(metrics, "recall_percent"),
        _metric_value(metrics, "f1_percent"),
        _metric_value(metrics, "precision_percent"),
    )


def _validation_rank(record: dict[str, Any], minimum_precision: float) -> tuple[float, ...]:
    metrics = record["validation_metrics"]
    precision = _metric_value(metrics, "precision_percent")
    return (
        1.0 if precision >= 100.0 * minimum_precision else 0.0,
        _metric_value(metrics, "balanced_accuracy_percent"),
        _metric_value(metrics, "recall_percent"),
        _metric_value(metrics, "f1_percent"),
        precision,
        _metric_value(record["cv_metrics"], "balanced_accuracy_percent"),
    )


class CrossingClassifier:
    """Load and apply the frozen crossing classifier artifact."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        if artifact.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                "Unsupported crossing classifier feature version: "
                f"{artifact.get('feature_version')!r}"
            )
        self.artifact = artifact
        self.model: Pipeline = artifact["pipeline"]
        self.threshold = float(artifact["threshold"])

    @classmethod
    def load(cls, path: str | Path) -> "CrossingClassifier":
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Crossing classifier model not found: {source}")
        artifact = joblib.load(source)
        if not isinstance(artifact, dict):
            raise ValueError(f"Crossing classifier artifact is invalid: {source}")
        return cls(artifact)

    def predict_probabilities(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return self.model.predict_proba(feature_matrix(rows))[:, 1]

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return self.predict_probabilities(rows) >= self.threshold


class JAADCrossingClassifierTrainer:
    """Train by grouped JAAD cross validation and evaluate once on JAAD validation."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.settings = config.crossing_classifier_settings()
        self.benchmark_root = Path(self.settings["benchmark_results"])
        self.output_dir = Path(self.settings["results"])
        self.model_path = Path(self.settings["model"])

    def run(self) -> dict[str, Any]:
        train_path = self.benchmark_root / "train" / "per_pedestrian.csv"
        validation_path = self.benchmark_root / "val" / "per_pedestrian.csv"
        train_rows = load_benchmark_rows(train_path)
        validation_rows = load_benchmark_rows(validation_path)
        self._validate_dataset(train_rows, "train")
        self._validate_dataset(validation_rows, "validation")

        candidates = self._candidates()
        records: list[dict[str, Any]] = []
        trained_models: list[Pipeline] = []
        validation_probabilities: list[np.ndarray] = []
        for candidate in candidates:
            record, model, probabilities = self._evaluate_candidate(
                candidate, train_rows, validation_rows
            )
            records.append(record)
            trained_models.append(model)
            validation_probabilities.append(probabilities)

        family_winner_indices: list[int] = []
        for family in sorted({record["model_family"] for record in records}):
            family_indices = [
                index for index, record in enumerate(records) if record["model_family"] == family
            ]
            winner = max(family_indices, key=lambda index: _candidate_rank(records[index]))
            family_winner_indices.append(winner)
            records[winner]["family_cv_winner"] = True
            calibrated = select_threshold(
                _labels(validation_rows),
                validation_probabilities[winner],
                float(self.settings["min_precision"]),
                float(self.settings["threshold_step"]),
            )
            records[winner]["threshold"] = float(calibrated["threshold"])
            records[winner]["validation_threshold_calibrated"] = True
            records[winner]["validation_metrics"] = calibrated["metrics"]
            records[winner]["validation_precision_constraint_satisfied"] = bool(
                calibrated["precision_constraint_satisfied"]
            )
        selected_index = max(
            family_winner_indices,
            key=lambda index: _validation_rank(
                records[index], float(self.settings["min_precision"])
            ),
        )
        selected = records[selected_index]
        selected_model = trained_models[selected_index]
        for index, record in enumerate(records):
            record["selected"] = index == selected_index

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        final_rows = train_rows + validation_rows
        final_matched = _matched_indices(final_rows)
        final_candidate = ModelCandidate(
            selected["model_family"], dict(selected["parameters"])
        )
        final_model = _fit_model(
            final_candidate,
            int(self.settings["random_seed"]),
            feature_matrix([final_rows[index] for index in final_matched]),
            _labels(final_rows)[final_matched],
        )
        artifact = {
            "artifact_type": "crowd_jaywalking_crossing_classifier",
            "feature_version": FEATURE_VERSION,
            "numeric_features": list(NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "model_family": selected["model_family"],
            "parameters": selected["parameters"],
            "threshold": selected["threshold"],
            "minimum_precision": float(self.settings["min_precision"]),
            "cv_metrics": selected["cv_metrics"],
            "validation_metrics": selected["validation_metrics"],
            "fit_splits": ["train", "val"],
            "pipeline": final_model,
        }
        joblib.dump(artifact, self.model_path)
        model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()

        self._write_model_selection(records)
        self._write_validation_predictions(selected_model, selected, validation_rows)
        summary = self._summary(
            train_path,
            validation_path,
            train_rows,
            validation_rows,
            records,
            selected,
            model_sha256,
        )
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._print_summary(summary)
        return summary

    def _candidates(self) -> list[ModelCandidate]:
        candidates = [
            ModelCandidate("logistic_regression", {"C": value})
            for value in self.settings["logistic_c_values"]
        ]
        candidates.extend(
            ModelCandidate(
                "hist_gradient_boosting",
                {"learning_rate": learning_rate, "max_leaf_nodes": leaf_nodes},
            )
            for learning_rate in self.settings["gradient_learning_rates"]
            for leaf_nodes in self.settings["gradient_max_leaf_nodes"]
        )
        return candidates

    def _evaluate_candidate(
        self,
        candidate: ModelCandidate,
        train_rows: list[dict[str, Any]],
        validation_rows: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], Pipeline, np.ndarray]:
        labels = _labels(train_rows)
        matched = _matched_indices(train_rows)
        matched_rows = [train_rows[index] for index in matched]
        matrix = feature_matrix(matched_rows)
        matched_labels = labels[matched]
        groups = np.asarray([train_rows[index]["video_id"] for index in matched], dtype=object)
        fold_count = min(int(self.settings["cv_folds"]), len(np.unique(groups)))
        if fold_count < 2:
            raise ValueError("At least two distinct training videos are required")

        oof_matched = np.zeros(len(matched), dtype=float)
        splitter = GroupKFold(n_splits=fold_count)
        for train_indices, holdout_indices in splitter.split(matrix, matched_labels, groups):
            model = _fit_model(
                candidate,
                int(self.settings["random_seed"]),
                matrix[train_indices],
                matched_labels[train_indices],
            )
            oof_matched[holdout_indices] = model.predict_proba(matrix[holdout_indices])[:, 1]

        oof_all = np.zeros(len(train_rows), dtype=float)
        oof_all[matched] = oof_matched
        threshold = select_threshold(
            labels,
            oof_all,
            float(self.settings["min_precision"]),
            float(self.settings["threshold_step"]),
        )
        full_model = _fit_model(
            candidate,
            int(self.settings["random_seed"]),
            matrix,
            matched_labels,
        )
        validation_probabilities = _probabilities_for_all_rows(full_model, validation_rows)
        validation_metrics = classification_metrics(
            _labels(validation_rows),
            validation_probabilities >= float(threshold["threshold"]),
        )
        return (
            {
                "selected": False,
                "family_cv_winner": False,
                "model_family": candidate.family,
                "parameters": candidate.parameters,
                "cv_threshold": float(threshold["threshold"]),
                "threshold": float(threshold["threshold"]),
                "validation_threshold_calibrated": False,
                "precision_constraint_satisfied": bool(
                    threshold["precision_constraint_satisfied"]
                ),
                "validation_precision_constraint_satisfied": (
                    validation_metrics["precision_percent"] is not None
                    and float(validation_metrics["precision_percent"])
                    >= 100.0 * float(self.settings["min_precision"])
                ),
                "cv_metrics": threshold["metrics"],
                "validation_metrics": validation_metrics,
            },
            full_model,
            validation_probabilities,
        )

    @staticmethod
    def _validate_dataset(rows: list[dict[str, Any]], name: str) -> None:
        labels = _labels(rows)
        matched = _matched_indices(rows)
        if not len(matched):
            raise ValueError(f"The {name} split contains no matched tracks")
        if len(np.unique(labels)) < 2:
            raise ValueError(f"The {name} split must contain crossing and non crossing labels")
        if len(np.unique(labels[matched])) < 2:
            raise ValueError(f"The matched {name} tracks must contain both ground truth classes")

    def _write_model_selection(self, records: list[dict[str, Any]]) -> None:
        rows: list[dict[str, Any]] = []
        for record in records:
            row: dict[str, Any] = {
                "selected": record["selected"],
                "family_cv_winner": record["family_cv_winner"],
                "model_family": record["model_family"],
                "parameters": json.dumps(record["parameters"], sort_keys=True),
                "cv_threshold": record["cv_threshold"],
                "threshold": record["threshold"],
                "validation_threshold_calibrated": record[
                    "validation_threshold_calibrated"
                ],
                "precision_constraint_satisfied": record[
                    "precision_constraint_satisfied"
                ],
                "validation_precision_constraint_satisfied": record[
                    "validation_precision_constraint_satisfied"
                ],
            }
            for prefix, metrics in (
                ("cv", record["cv_metrics"]),
                ("validation", record["validation_metrics"]),
            ):
                for name, value in metrics.items():
                    row[f"{prefix}_{name}"] = value
            rows.append(row)
        _write_csv(self.output_dir / "model_selection.csv", MODEL_SELECTION_FIELDS, rows)

    def _write_validation_predictions(
        self,
        model: Pipeline,
        selected: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> None:
        probabilities = _probabilities_for_all_rows(model, rows)
        predictions = probabilities >= float(selected["threshold"])
        output_rows = []
        for row, probability, prediction in zip(rows, probabilities, predictions):
            truth = _as_bool(row["ground_truth_crossing"])
            output_rows.append(
                {
                    "video_id": row["video_id"],
                    "pedestrian_id": row["pedestrian_id"],
                    "track_matched": _as_bool(row["track_matched"]),
                    "matched_track_id": row.get("matched_track_id", ""),
                    "ground_truth_crossing": truth,
                    "rule_prediction": _as_bool(row.get("predicted_crossing", False)),
                    "classifier_probability": round(float(probability), 8),
                    "classifier_prediction": bool(prediction),
                    "correct": truth == bool(prediction),
                }
            )
        _write_csv(
            self.output_dir / "validation_predictions.csv",
            VALIDATION_PREDICTION_FIELDS,
            output_rows,
        )

    def _summary(
        self,
        train_path: Path,
        validation_path: Path,
        train_rows: list[dict[str, Any]],
        validation_rows: list[dict[str, Any]],
        records: list[dict[str, Any]],
        selected: dict[str, Any],
        model_sha256: str,
    ) -> dict[str, Any]:
        baseline = self._baseline_validation_metrics()
        selected_validation = selected["validation_metrics"]
        deltas = None
        if baseline:
            deltas = {
                name: (
                    None
                    if baseline.get(name) is None or selected_validation.get(name) is None
                    else float(selected_validation[name]) - float(baseline[name])
                )
                for name in (
                    "accuracy_percent",
                    "precision_percent",
                    "recall_percent",
                    "specificity_percent",
                    "balanced_accuracy_percent",
                    "f1_percent",
                )
            }
        validation_precision_passed = bool(
            selected["validation_precision_constraint_satisfied"]
        )
        baseline_balanced_accuracy = (
            baseline.get("balanced_accuracy_percent") if baseline else None
        )
        baseline_recall = baseline.get("recall_percent") if baseline else None
        balanced_accuracy_improved = (
            True
            if baseline_balanced_accuracy is None
            else _metric_value(selected_validation, "balanced_accuracy_percent")
            > float(baseline_balanced_accuracy)
        )
        recall_improved = (
            True
            if baseline_recall is None
            else _metric_value(selected_validation, "recall_percent") > float(baseline_recall)
        )
        return {
            "feature_version": FEATURE_VERSION,
            "selection_protocol": (
                "Grouped cross validation by training video selects hyperparameters within "
                "each model family. The validation split calibrates one threshold per family "
                "winner and selects between the logistic regression and gradient boosting "
                "family winners."
            ),
            "test_split_used": False,
            "minimum_precision_percent": 100.0 * float(self.settings["min_precision"]),
            "cross_validation_folds": int(self.settings["cv_folds"]),
            "train_csv": str(train_path),
            "validation_csv": str(validation_path),
            "train_rows": len(train_rows),
            "train_matched_tracks": len(_matched_indices(train_rows)),
            "validation_rows": len(validation_rows),
            "validation_matched_tracks": len(_matched_indices(validation_rows)),
            "selected_model": {
                "family": selected["model_family"],
                "parameters": selected["parameters"],
                "threshold": selected["threshold"],
                "precision_constraint_satisfied_in_cross_validation": selected[
                    "precision_constraint_satisfied"
                ],
                "precision_constraint_satisfied_on_validation": selected[
                    "validation_precision_constraint_satisfied"
                ],
                "cv_metrics": selected["cv_metrics"],
                "validation_metrics": selected_validation,
            },
            "baseline_validation_metrics": baseline,
            "validation_delta_from_rule_baseline": deltas,
            "acceptance_gate": {
                "validation_precision_passed": validation_precision_passed,
                "balanced_accuracy_improved_over_rule_baseline": balanced_accuracy_improved,
                "recall_improved_over_rule_baseline": recall_improved,
                "approved_for_locked_test": (
                    validation_precision_passed
                    and balanced_accuracy_improved
                    and recall_improved
                ),
            },
            "model_candidates": records,
            "model_path": str(self.model_path),
            "model_fit_splits": ["train", "val"],
            "model_sha256": model_sha256,
            "model_selection_csv": str(self.output_dir / "model_selection.csv"),
            "validation_predictions_csv": str(
                self.output_dir / "validation_predictions.csv"
            ),
        }

    def _baseline_validation_metrics(self) -> dict[str, Any] | None:
        path = self.benchmark_root / "val" / "summary.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("end_to_end_crossing_metrics")
        return metrics if isinstance(metrics, dict) else None

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        selected = summary["selected_model"]
        cv = selected["cv_metrics"]
        validation = selected["validation_metrics"]
        logger.info("\nJAAD supervised crossing classifier")
        logger.info("Selected model: {}", selected['family'])
        logger.info("Decision threshold: {:.2f}", selected['threshold'])
        logger.info(
            "Grouped CV: precision={:.2f}% recall={:.2f}% balanced accuracy={:.2f}%",
            _metric_value(cv, 'precision_percent'),
            _metric_value(cv, 'recall_percent'),
            _metric_value(cv, 'balanced_accuracy_percent'),
        )
        logger.info(
            "Validation: TP={} TN={} FP={} FN={}",
            validation['tp'],
            validation['tn'],
            validation['fp'],
            validation['fn'],
        )
        logger.info("Validation precision: {:.2f}%", _metric_value(validation, 'precision_percent'))
        logger.info("Validation recall: {:.2f}%", _metric_value(validation, 'recall_percent'))
        logger.info("Validation balanced accuracy: {:.2f}%", _metric_value(validation, 'balanced_accuracy_percent'))
        approval = summary["acceptance_gate"]["approved_for_locked_test"]
        logger.info("Approved for locked test: {}", 'YES' if approval else 'NO')
        logger.info("Saved model: {}", summary['model_path'])


class JAADCrossingClassifierEvaluator:
    """Evaluate the frozen classifier once on the configured JAAD test split."""

    def __init__(self, config: ProjectConfig) -> None:
        self.settings = config.crossing_classifier_settings()
        self.benchmark_root = Path(self.settings["benchmark_results"])
        self.output_dir = Path(self.settings["results"])
        self.model_path = Path(self.settings["model"])

    def run(self) -> dict[str, Any]:
        test_path = self.benchmark_root / "test" / "per_pedestrian.csv"
        rows = load_benchmark_rows(test_path)
        classifier = CrossingClassifier.load(self.model_path)
        probabilities = np.zeros(len(rows), dtype=float)
        matched = _matched_indices(rows)
        if len(matched):
            matched_rows = [rows[index] for index in matched]
            probabilities[matched] = classifier.predict_probabilities(matched_rows)
        predictions = probabilities >= classifier.threshold
        truth = _labels(rows)
        metrics = classification_metrics(truth, predictions)
        matched_metrics = classification_metrics(truth[matched], predictions[matched])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        prediction_rows = []
        for row, probability, prediction in zip(rows, probabilities, predictions):
            expected = _as_bool(row["ground_truth_crossing"])
            prediction_rows.append(
                {
                    "video_id": row["video_id"],
                    "pedestrian_id": row["pedestrian_id"],
                    "track_matched": _as_bool(row["track_matched"]),
                    "matched_track_id": row.get("matched_track_id", ""),
                    "ground_truth_crossing": expected,
                    "rule_prediction": _as_bool(row.get("predicted_crossing", False)),
                    "classifier_probability": round(float(probability), 8),
                    "classifier_prediction": bool(prediction),
                    "correct": expected == bool(prediction),
                }
            )
        predictions_path = self.output_dir / "test_predictions.csv"
        _write_csv(predictions_path, VALIDATION_PREDICTION_FIELDS, prediction_rows)

        baseline = self._baseline_test_metrics()
        summary = {
            "feature_version": FEATURE_VERSION,
            "test_split_used": True,
            "test_csv": str(test_path),
            "test_rows": len(rows),
            "test_matched_tracks": len(matched),
            "track_match_recall_percent": 100.0 * len(matched) / len(rows),
            "model_family": classifier.artifact["model_family"],
            "parameters": classifier.artifact["parameters"],
            "threshold": classifier.threshold,
            "end_to_end_crossing_metrics": metrics,
            "crossing_metrics_when_track_matched": matched_metrics,
            "rule_baseline_test_metrics": baseline,
            "model_path": str(self.model_path),
            "model_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
            "test_predictions_csv": str(predictions_path),
        }
        summary_path = self.output_dir / "test_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._print_summary(summary)
        return summary

    def _baseline_test_metrics(self) -> dict[str, Any] | None:
        path = self.benchmark_root / "test" / "summary.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("end_to_end_crossing_metrics")
        return metrics if isinstance(metrics, dict) else None

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        metrics = summary["end_to_end_crossing_metrics"]
        logger.info("\nJAAD crossing classifier test")
        logger.info("Rows: {}", summary['test_rows'])
        logger.info("Track match recall: {:.2f}%", summary['track_match_recall_percent'])
        logger.info(
            "Confusion matrix: TP={} TN={} FP={} FN={}",
            metrics['tp'],
            metrics['tn'],
            metrics['fp'],
            metrics['fn'],
        )
        logger.info("Precision: {:.2f}%", _metric_value(metrics, 'precision_percent'))
        logger.info("Recall: {:.2f}%", _metric_value(metrics, 'recall_percent'))
        logger.info("F1: {:.2f}%", _metric_value(metrics, 'f1_percent'))
        logger.info("Balanced accuracy: {:.2f}%", _metric_value(metrics, 'balanced_accuracy_percent'))
        logger.info("Saved: {}", summary['test_predictions_csv'])
