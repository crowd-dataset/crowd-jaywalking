"""Evaluate local VLM context predictions against manual JAAD annotations."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any

from .config import ProjectConfig
from .jaad_context import MANUAL_CONTEXT_FIELDS
from .models import EvidenceImage
from .vlm import (
    HuggingFaceContextClassifier,
    normalise_prompt_mode,
    prompt_content_for_mode,
    prompt_version_for_mode,
)


CONTEXT_FIELDS = (
    "marked_crosswalk",
    "permissive_pedestrian_signal",
    "authorised_crossing_sign",
    "crossing_guard_permission",
    "prohibitive_pedestrian_signal",
)

PERMISSION_FIELDS = (
    "marked_crosswalk",
    "permissive_pedestrian_signal",
    "authorised_crossing_sign",
    "crossing_guard_permission",
)

PREDICTION_FIELDS = (
    "video_id",
    "jaad_pedestrian_id",
    *(f"ground_truth_{field}" for field in CONTEXT_FIELDS),
    *(f"predicted_{field}" for field in CONTEXT_FIELDS),
    "ground_truth_visibility",
    "predicted_visibility",
    "evidence_summary",
)


def _normalise_ternary(value: str) -> str:
    normalised = str(value).strip().upper()
    aliases = {"Y": "YES", "N": "NO", "U": "UNCERTAIN", "NOT SURE": "UNCERTAIN"}
    normalised = aliases.get(normalised, normalised)
    if normalised not in {"YES", "NO", "UNCERTAIN"}:
        raise ValueError(f"Context labels must be YES, NO, or UNCERTAIN: {value}")
    return normalised


def _normalise_visibility(value: str) -> str:
    normalised = str(value).strip().upper()
    if normalised not in {"CLEAR", "PARTIAL", "INSUFFICIENT"}:
        raise ValueError(
            f"Visibility must be CLEAR, PARTIAL, or INSUFFICIENT: {value}"
        )
    return normalised


def _permission_present(row: dict[str, str], prefix: str) -> str:
    values = [
        _normalise_ternary(row[f"{prefix}{field}"])
        for field in PERMISSION_FIELDS
    ]
    if "YES" in values:
        return "YES"
    if "UNCERTAIN" in values:
        return "UNCERTAIN"
    return "NO"


def _classification_metrics(
    truth: list[str],
    predictions: list[str],
    labels: tuple[str, ...],
    macro_labels: tuple[str, ...],
) -> dict[str, Any]:
    confusion = {
        actual: {
            predicted: sum(
                expected == actual and observed == predicted
                for expected, observed in zip(truth, predictions)
            )
            for predicted in labels
        }
        for actual in labels
    }
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(truth, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(truth, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(truth, predictions))
        support = sum(actual == label for actual in truth)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "predicted": sum(predicted == label for predicted in predictions),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision_percent": 100.0 * precision,
            "recall_percent": 100.0 * recall,
            "f1_percent": 100.0 * f1,
        }
        if label in macro_labels:
            f1_values.append(f1)
    correct = sum(actual == predicted for actual, predicted in zip(truth, predictions))
    return {
        "samples": len(truth),
        "accuracy_percent": 100.0 * correct / len(truth) if truth else 0.0,
        "macro_f1_percent": 100.0 * mean(f1_values) if f1_values else 0.0,
        "observed_classes": len(macro_labels),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_intervals(
    truth: list[str],
    predictions: list[str],
    labels: tuple[str, ...],
    macro_labels: tuple[str, ...],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if not truth or samples <= 0:
        return {}
    generator = random.Random(seed)
    accuracy: list[float] = []
    macro_f1: list[float] = []
    for _ in range(samples):
        indices = [generator.randrange(len(truth)) for _ in truth]
        metrics = _classification_metrics(
            [truth[index] for index in indices],
            [predictions[index] for index in indices],
            labels,
            macro_labels,
        )
        accuracy.append(float(metrics["accuracy_percent"]))
        macro_f1.append(float(metrics["macro_f1_percent"]))
    return {
        "accuracy_ci95_percent": {
            "lower": _percentile(accuracy, 0.025),
            "upper": _percentile(accuracy, 0.975),
        },
        "macro_f1_ci95_percent": {
            "lower": _percentile(macro_f1, 0.025),
            "upper": _percentile(macro_f1, 0.975),
        },
    }


def _macro_metrics(
    truth: list[str],
    predictions: list[str],
    labels: tuple[str, ...],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    if len(truth) != len(predictions):
        raise ValueError("Ground truth and prediction lengths differ")
    observed_labels = tuple(label for label in labels if label in truth)
    metrics = _classification_metrics(truth, predictions, labels, observed_labels)
    metrics.update(
        _bootstrap_intervals(
            truth,
            predictions,
            labels,
            observed_labels,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    )
    return metrics


class JAADContextBenchmark:
    """Run and evaluate the VLM only on independently labelled true crossings."""

    def __init__(
        self,
        config: ProjectConfig,
        *,
        model_id: str | None = None,
        prompt_mode: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.split = str(config.get("jaad_context_split")).strip().lower()
        self.model_id = model_id or str(config.get("vlm_model"))
        configured_mode = config.vlm_settings(self.model_id)["prompt_mode"]
        self.prompt_mode = normalise_prompt_mode(prompt_mode or configured_mode)
        self.annotation_dir = config.path("jaad_context_results") / self.split
        self.output_dir = (
            Path(output_dir).resolve() if output_dir is not None else self.annotation_dir
        )
        self.annotations_csv = self.annotation_dir / "context_annotations.csv"
        self.predictions_csv = self.output_dir / "vlm_predictions.csv"
        self.summary_json = self.output_dir / "vlm_summary.json"
        self.manifest_json = self.output_dir / "vlm_manifest.json"

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        annotation_rows = self._annotation_rows()
        complete, incomplete = self._complete_rows(annotation_rows)
        if not complete:
            raise ValueError(
                f"No complete manual context annotations were found in {self.annotations_csv}"
            )
        for row in complete:
            for field in CONTEXT_FIELDS:
                _normalise_ternary(row[field])
            _normalise_visibility(row["visibility"])
        self._prepare_manifest(complete)
        completed = self._existing_predictions()

        classifier: HuggingFaceContextClassifier | None = None
        write_header = not self.predictions_csv.exists() or self.predictions_csv.stat().st_size == 0
        try:
            with self.predictions_csv.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
                if write_header:
                    writer.writeheader()

                for index, row in enumerate(complete, start=1):
                    key = (row["video_id"], row["jaad_pedestrian_id"])
                    if key in completed:
                        print(f"[{index:03d}/{len(complete):03d}] {key}: already completed")
                        continue
                    if classifier is None:
                        classifier = HuggingFaceContextClassifier(
                            self.config.vlm_settings(
                                self.model_id,
                                self.prompt_mode,
                            )
                        )
                        classifier.ensure_ready()
                    print(f"[{index:03d}/{len(complete):03d}] {key}")
                    context = classifier.classify(self._evidence(row))
                    writer.writerow(
                        {
                            "video_id": row["video_id"],
                            "jaad_pedestrian_id": row["jaad_pedestrian_id"],
                            **{
                                f"ground_truth_{field}": _normalise_ternary(row[field])
                                for field in CONTEXT_FIELDS
                            },
                            **{
                                f"predicted_{field}": getattr(context, field).value
                                for field in CONTEXT_FIELDS
                            },
                            "ground_truth_visibility": _normalise_visibility(row["visibility"]),
                            "predicted_visibility": context.visibility.value,
                            "evidence_summary": context.evidence_summary,
                        }
                    )
                    handle.flush()
        finally:
            if classifier is not None:
                classifier.close()

        prediction_rows = self._read_csv(self.predictions_csv)
        summary = self._summarise(prediction_rows, len(incomplete))
        with self.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self._print_summary(summary)
        return summary

    def _annotation_rows(self) -> list[dict[str, str]]:
        if not self.annotations_csv.is_file():
            raise FileNotFoundError(
                f"Context annotation sheet not found: {self.annotations_csv}. "
                "Run prepare_jaad_context.py first."
            )
        rows = self._read_csv(self.annotations_csv)
        required = set(MANUAL_CONTEXT_FIELDS).difference({"annotator", "notes"})
        missing = required.difference(rows[0] if rows else {})
        if missing:
            raise ValueError(
                "Context annotation sheet is missing columns: " + ", ".join(sorted(missing))
            )
        return rows

    @staticmethod
    def _complete_rows(
        rows: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        required = [*CONTEXT_FIELDS, "visibility"]
        complete = [row for row in rows if all(str(row.get(field, "")).strip() for field in required)]
        incomplete = [row for row in rows if row not in complete]
        return complete, incomplete

    def _prepare_manifest(self, annotated_rows: list[dict[str, str]]) -> None:
        annotation_payload = [
            {
                "video_id": row["video_id"],
                "jaad_pedestrian_id": row["jaad_pedestrian_id"],
                **{
                    field: str(row.get(field, "")).strip().upper()
                    for field in (*CONTEXT_FIELDS, "visibility")
                },
            }
            for row in sorted(
                annotated_rows,
                key=lambda item: (item["video_id"], item["jaad_pedestrian_id"]),
            )
        ]
        annotation_sha256 = hashlib.sha256(
            json.dumps(annotation_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        prompt_version = prompt_version_for_mode(self.prompt_mode)
        prompt_content = prompt_content_for_mode(self.prompt_mode)
        fingerprint = self.config.fingerprint(
            f"{prompt_version}\n{prompt_content}\njaad-context-v5\n"
            f"{self.model_id}\n{annotation_sha256}"
        )
        if self.manifest_json.is_file():
            existing = json.loads(self.manifest_json.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "The VLM context results use another configuration or prompt. "
                    "Choose a new jaad_context_results path before rerunning."
                )
            return
        payload = {
            "pipeline_version": "1.11.0",
            "prompt_mode": self.prompt_mode,
            "prompt_version": prompt_version,
            "fingerprint": fingerprint,
            "split": self.split,
            "annotated_events": len(annotated_rows),
            "annotation_sha256": annotation_sha256,
            "vlm_model": self.model_id,
        }
        self.manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _existing_predictions(self) -> set[tuple[str, str]]:
        if not self.predictions_csv.is_file():
            return set()
        return {
            (row["video_id"], row["jaad_pedestrian_id"])
            for row in self._read_csv(self.predictions_csv)
        }

    def _evidence(self, row: dict[str, str]) -> list[EvidenceImage]:
        directory = self.annotation_dir / row["evidence_directory"]
        if not directory.is_dir():
            raise FileNotFoundError(f"Evidence directory not found: {directory}")
        evidence: list[EvidenceImage] = []
        for context_path in sorted(directory.glob("frame_*_context.jpg")):
            frame_text = context_path.name.split("_")[1]
            focus_path = directory / context_path.name.replace("_context.jpg", "_focus.jpg")
            road_path = directory / context_path.name.replace("_context.jpg", "_road.jpg")
            trajectory_path = directory / context_path.name.replace(
                "_context.jpg", "_trajectory.jpg"
            )
            control_left_path = directory / context_path.name.replace(
                "_context.jpg", "_control_left.jpg"
            )
            control_right_path = directory / context_path.name.replace(
                "_context.jpg", "_control_right.jpg"
            )
            required = (
                focus_path,
                road_path,
                trajectory_path,
                control_left_path,
                control_right_path,
            )
            missing = [path for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Version 5 evidence is incomplete. Run prepare_jaad_context.py again. "
                    "Missing: " + ", ".join(str(path) for path in missing)
                )
            evidence.append(
                EvidenceImage(
                    frame_index=int(frame_text),
                    context_path=context_path,
                    focus_path=focus_path,
                    road_path=road_path,
                    trajectory_path=trajectory_path,
                    control_left_path=control_left_path,
                    control_right_path=control_right_path,
                )
            )
        if not evidence:
            raise FileNotFoundError(f"No evidence images found: {directory}")
        return evidence

    def _summarise(
        self,
        rows: list[dict[str, str]],
        incomplete_rows: int,
    ) -> dict[str, Any]:
        field_metrics = {
            field: _macro_metrics(
                [row[f"ground_truth_{field}"] for row in rows],
                [row[f"predicted_{field}"] for row in rows],
                ("YES", "NO", "UNCERTAIN"),
            )
            for field in CONTEXT_FIELDS
        }
        field_metrics["visibility"] = _macro_metrics(
            [row["ground_truth_visibility"] for row in rows],
            [row["predicted_visibility"] for row in rows],
            ("CLEAR", "PARTIAL", "INSUFFICIENT"),
        )
        permission_metrics = _macro_metrics(
            [_permission_present(row, "ground_truth_") for row in rows],
            [_permission_present(row, "predicted_") for row in rows],
            ("YES", "NO", "UNCERTAIN"),
        )
        return {
            "split": self.split,
            "vlm_model": self.model_id,
            "prompt_mode": self.prompt_mode,
            "prompt_version": prompt_version_for_mode(self.prompt_mode),
            "evaluated_events": len(rows),
            "incomplete_annotation_rows": incomplete_rows,
            "context_field_metrics": field_metrics,
            "derived_metrics": {
                "permission_present": permission_metrics,
            },
        }

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        # ``utf-8-sig`` also accepts ordinary UTF-8 while removing the BOM
        # commonly added by Excel and PowerShell CSV exports. Without this,
        # the first column is read as ``\ufeffvideo_id`` instead of ``video_id``.
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        print("\nJAAD VLM context benchmark")
        print(f"Split: {summary['split']}")
        print(f"Model: {summary['vlm_model']}")
        print(
            f"Prompt: {summary['prompt_mode']} "
            f"({summary['prompt_version']})"
        )
        print(f"Evaluated events: {summary['evaluated_events']}")
        for field, metrics in summary["context_field_metrics"].items():
            interval = metrics.get("macro_f1_ci95_percent", {})
            support = ", ".join(
                f"{label}={values['support']}"
                for label, values in metrics.get("per_class", {}).items()
            )
            print(
                f"{field}: accuracy={metrics['accuracy_percent']:.2f}% "
                f"macro-F1={metrics['macro_f1_percent']:.2f}% "
                f"CI95=[{float(interval.get('lower', 0.0)):.2f}, "
                f"{float(interval.get('upper', 0.0)):.2f}] support[{support}]"
            )
        permission = summary["derived_metrics"]["permission_present"]
        print(
            "permission_present: "
            f"accuracy={permission['accuracy_percent']:.2f}% "
            f"macro-F1={permission['macro_f1_percent']:.2f}%"
        )
