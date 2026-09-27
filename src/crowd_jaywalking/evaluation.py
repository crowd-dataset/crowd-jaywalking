"""JAAD annotation loading, resumable evaluation, and metrics."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .models import DecisionLabel, VideoResult, to_jsonable
from .tracking import save_observations_csv
from .vlm import prompt_content_for_mode, prompt_version_for_mode


RESULT_FIELDS = [
    "video_id",
    "filename",
    "ground_truth",
    "prediction",
    "correct",
    "person_decisions",
    "rejected_candidates",
    "latency_seconds",
    "details_json",
]


@dataclass(frozen=True)
class Annotation:
    """One valid human-labelled video record."""

    video_id: str
    filename: str
    ground_truth: DecisionLabel


def load_annotations(path: str | Path, selected_split: str | None = None) -> tuple[list[Annotation], int]:
    """Load Yes and No labels and exclude uncertain human annotations."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Annotations file not found: {source}")

    annotations: list[Annotation] = []
    excluded = 0
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotations are missing columns: {', '.join(sorted(missing))}")
        if selected_split and "split" not in (reader.fieldnames or []):
            raise ValueError(
                "The annotations file has no split column. Run 'uv run python prepare_splits.py' first."
            )

        for index, row in enumerate(reader, start=2):
            label = str(row.get("label", "")).strip().lower()
            if label == "yes":
                ground_truth = DecisionLabel.JAYWALKING
            elif label == "no":
                ground_truth = DecisionLabel.COMPLIANT
            else:
                excluded += 1
                continue

            if selected_split and str(row.get("split", "")).strip().lower() != selected_split.lower():
                continue

            filename = str(row.get("filename", "")).strip()
            if not filename:
                raise ValueError(f"Missing filename on annotation row {index}")
            video_id = str(row.get("video_id", "")).strip() or Path(filename).stem
            annotations.append(
                Annotation(
                    video_id=video_id,
                    filename=filename,
                    ground_truth=ground_truth,
                )
            )

    if not annotations:
        raise ValueError(f"No Yes or No annotations were found for split: {selected_split or 'all'}")
    video_ids = [item.video_id for item in annotations]
    duplicates = sorted({value for value in video_ids if video_ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"Duplicate selected video IDs: {', '.join(duplicates)}")
    return annotations, excluded


class EvaluationRunner:
    """Run a restartable evaluation without mixing different configurations."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.annotations_path = config.data_file("annotations")
        self.selected_split = str(config.get("evaluation_split")).strip() or None
        self.video_dirs = config.paths("videos")
        self.results_dir = config.path("results")
        self.details_dir = self.results_dir / "details"
        self.evidence_dir = self.results_dir / "evidence"
        self.tracking_dir = self.results_dir / "tracking"
        self.results_csv = self.results_dir / "per_video_results.csv"
        self.run_manifest = self.results_dir / "run_manifest.json"
        self.summary_json = self.results_dir / "summary.json"
        self.resume = bool(config.get("resume"))

    def run(self) -> dict[str, Any]:
        """Evaluate every definite annotation and return summary metrics."""

        annotations, excluded = load_annotations(self.annotations_path, self.selected_split)
        self._prepare_run(excluded, len(annotations))
        completed = self._completed_video_ids()
        from .pipeline import JaywalkingPipeline

        pipeline = JaywalkingPipeline(self.config)

        write_header = not self.results_csv.exists() or self.results_csv.stat().st_size == 0
        with self.results_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
            if write_header:
                writer.writeheader()
                handle.flush()

            for index, annotation in enumerate(annotations, start=1):
                if annotation.video_id in completed:
                    print(f"[{index:03d}/{len(annotations):03d}] {annotation.filename}: already completed")
                    continue

                video_path = self._find_video(annotation.filename)
                if video_path is None:
                    raise FileNotFoundError(
                        f"Annotated video is missing from every configured videos directory: "
                        f"{annotation.filename}. The run stopped instead of silently skipping it."
                    )

                print(f"[{index:03d}/{len(annotations):03d}] Processing {annotation.filename}")
                # Keep the exact tracks, so person level results can be audited later.
                started = time.perf_counter()
                fps, observations = pipeline.tracker.track(video_path)
                save_observations_csv(
                    self.tracking_dir / f"{annotation.video_id}.csv", observations
                )
                result = pipeline.process_observations(
                    video_path, self.evidence_dir, fps, observations, started=started
                )
                details_path = self._save_details(annotation, result)
                row = {
                    "video_id": annotation.video_id,
                    "filename": annotation.filename,
                    "ground_truth": annotation.ground_truth.value,
                    "prediction": result.prediction.value,
                    "correct": result.prediction == annotation.ground_truth,
                    "person_decisions": len(result.person_decisions),
                    "rejected_candidates": len(result.rejected_candidates),
                    "latency_seconds": result.latency_seconds,
                    "details_json": str(details_path.relative_to(self.results_dir)),
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"    GT={annotation.ground_truth.value} Pred={result.prediction.value} "
                    f"People={len(result.person_decisions)} Time={result.latency_seconds:.2f}s"
                )

        rows = self._read_results()
        summary = calculate_metrics(rows)
        summary["excluded_uncertain_human_labels"] = excluded
        with self.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self._print_summary(summary)
        return summary

    def _prepare_run(self, excluded: int, total: int) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.details_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        mode = self.config.vlm_settings()["prompt_mode"]
        prompt_version = prompt_version_for_mode(mode)
        fingerprint = self.config.fingerprint(
            f"{prompt_version}\n{prompt_content_for_mode(mode)}"
        )

        if self.run_manifest.exists():
            with self.run_manifest.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "The results directory contains a different configuration or prompt version. "
                    "Choose a new results path before running."
                )
            if not self.resume and self.results_csv.exists():
                raise RuntimeError("Existing results found while resume is false")
            return

        payload = {
            "pipeline_version": "1.1.0",
            "prompt_version": prompt_version,
            "fingerprint": fingerprint,
            "annotation_count": total,
            "evaluation_split": self.selected_split,
            "excluded_uncertain_human_labels": excluded,
            "vlm_model": self.config.get("vlm_model"),
            "tracking_model": self.config.get("tracking_model"),
        }
        with self.run_manifest.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _completed_video_ids(self) -> set[str]:
        if not self.resume or not self.results_csv.exists():
            return set()
        return {row["video_id"] for row in self._read_results()}

    def _find_video(self, filename: str) -> Path | None:
        matches = [
            directory / filename
            for directory in self.video_dirs
            if (directory / filename).is_file()
        ]
        if len(matches) > 1:
            locations = ", ".join(str(path) for path in matches)
            raise RuntimeError(f"Video exists in more than one configured directory: {locations}")
        return matches[0] if matches else None

    def _read_results(self) -> list[dict[str, str]]:
        if not self.results_csv.exists():
            return []
        with self.results_csv.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _save_details(self, annotation: Annotation, result: VideoResult) -> Path:
        path = self.details_dir / f"{annotation.video_id}.json"
        payload = {
            "video_id": annotation.video_id,
            "filename": annotation.filename,
            "ground_truth": annotation.ground_truth.value,
            "result": to_jsonable(result),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        print("\nEvaluation summary")
        print(f"Videos: {summary['total_videos']}")
        print(f"Coverage: {summary['coverage_percent']:.2f}%")
        print(f"Overall accuracy: {summary['overall_accuracy_percent']:.2f}%")
        print(f"Decided accuracy: {summary['decided_accuracy_percent']:.2f}%")
        print(f"Precision: {summary['precision_percent']:.2f}%")
        print(f"Recall: {summary['recall_percent']:.2f}%")
        print(f"Specificity: {summary['specificity_percent']:.2f}%")
        print(f"F1: {summary['f1_percent']:.2f}%")
        print(
            f"Confusion: TP={summary['tp']} TN={summary['tn']} "
            f"FP={summary['fp']} FN={summary['fn']} Uncertain={summary['uncertain']}"
        )


def calculate_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Calculate conservative overall and decided-only metrics."""

    total = len(rows)
    uncertain = sum(row["prediction"] == DecisionLabel.UNCERTAIN.value for row in rows)
    decided = [row for row in rows if row["prediction"] != DecisionLabel.UNCERTAIN.value]

    tp = sum(
        row["ground_truth"] == DecisionLabel.JAYWALKING.value
        and row["prediction"] == DecisionLabel.JAYWALKING.value
        for row in decided
    )
    tn = sum(
        row["ground_truth"] == DecisionLabel.COMPLIANT.value
        and row["prediction"] == DecisionLabel.COMPLIANT.value
        for row in decided
    )
    fp = sum(
        row["ground_truth"] == DecisionLabel.COMPLIANT.value
        and row["prediction"] == DecisionLabel.JAYWALKING.value
        for row in decided
    )
    fn = sum(
        row["ground_truth"] == DecisionLabel.JAYWALKING.value
        and row["prediction"] == DecisionLabel.COMPLIANT.value
        for row in decided
    )
    correct = tp + tn

    def percentage(numerator: float, denominator: float) -> float:
        return 100.0 * numerator / denominator if denominator else 0.0

    precision = percentage(tp, tp + fp)
    recall = percentage(tp, tp + fn)
    specificity = percentage(tn, tn + fp)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "total_videos": total,
        "decided_videos": len(decided),
        "uncertain": uncertain,
        "coverage_percent": percentage(len(decided), total),
        "overall_accuracy_percent": percentage(correct, total),
        "decided_accuracy_percent": percentage(correct, len(decided)),
        "precision_percent": precision,
        "recall_percent": recall,
        "specificity_percent": specificity,
        "f1_percent": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
