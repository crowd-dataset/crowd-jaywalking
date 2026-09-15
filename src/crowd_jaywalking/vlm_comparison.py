"""Compare supported VLMs on the same manually labelled context events."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from .config import ProjectConfig
from .jaad_context_evaluation import CONTEXT_FIELDS, JAADContextBenchmark


COMPARISON_FIELDS = (
    "selected",
    "model_id",
    "prompt_mode",
    "evaluated_events",
    "mean_context_macro_f1_percent",
    "minimum_field_macro_f1_percent",
    "mean_context_accuracy_percent",
    "informative_context_fields",
    "permission_present_macro_f1_percent",
    "permission_present_accuracy_percent",
    "result_directory",
)


def model_slug(model_id: str) -> str:
    """Return a readable collision resistant directory name for a model."""

    readable = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def candidate_slug(model_id: str, prompt_mode: str) -> str:
    """Return a unique directory name for one model and prompt combination."""

    return f"{model_slug(model_id)}__{prompt_mode}"


def comparison_metrics(summary: dict[str, Any]) -> dict[str, float | int]:
    """Reduce one benchmark summary to model selection metrics."""

    context = summary["context_field_metrics"]
    fields = (*CONTEXT_FIELDS, "visibility")
    informative = [
        field
        for field in fields
        if int(context[field].get("observed_classes", 2)) >= 2
    ]
    scoring_fields = informative or list(fields)
    macro_f1_values = [
        float(context[field]["macro_f1_percent"]) for field in scoring_fields
    ]
    accuracy_values = [
        float(context[field]["accuracy_percent"]) for field in scoring_fields
    ]
    permission = summary.get("derived_metrics", {}).get("permission_present")
    if permission is None:
        permission = {
            "macro_f1_percent": mean(macro_f1_values),
            "accuracy_percent": mean(accuracy_values),
        }
    return {
        "evaluated_events": int(summary["evaluated_events"]),
        "mean_context_macro_f1_percent": mean(macro_f1_values),
        "minimum_field_macro_f1_percent": min(macro_f1_values),
        "mean_context_accuracy_percent": mean(accuracy_values),
        "informative_context_fields": len(scoring_fields),
        "permission_present_macro_f1_percent": float(
            permission["macro_f1_percent"]
        ),
        "permission_present_accuracy_percent": float(
            permission["accuracy_percent"]
        ),
    }


def select_candidate(rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Select the strongest model and prompt pair with deterministic tie breaks."""

    if len(rows) < 2:
        raise ValueError("At least two completed model evaluations are required")
    sample_counts = {int(row["evaluated_events"]) for row in rows}
    if len(sample_counts) != 1:
        raise ValueError("Every comparison model must be evaluated on the same events")
    selected = max(
        rows,
        key=lambda row: (
            float(row["permission_present_macro_f1_percent"]),
            float(row["mean_context_macro_f1_percent"]),
            float(row["minimum_field_macro_f1_percent"]),
            float(row["mean_context_accuracy_percent"]),
            str(row.get("prompt_mode", "baseline_v3")) == "baseline_v3",
            str(row["model_id"]),
        ),
    )
    return (
        str(selected["model_id"]),
        str(selected.get("prompt_mode", "baseline_v3")),
    )


def select_model(rows: list[dict[str, Any]]) -> str:
    """Return only the model ID for compatibility with earlier callers."""

    return select_candidate(rows)[0]


class VLMModelComparison:
    """Run each configured VLM sequentially and select one on development data."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.split = str(config.get("jaad_context_split")).strip().lower()
        settings = config.vlm_comparison_settings()
        self.model_ids = list(settings["models"])
        self.prompt_modes = list(settings["prompt_modes"])
        self.output_dir = Path(settings["results"]) / self.split
        self.comparison_csv = self.output_dir / "model_comparison.csv"
        self.summary_json = self.output_dir / "comparison_summary.json"
        self.selected_json = self.output_dir / "selected_model.json"

    def run(self) -> dict[str, Any]:
        """Evaluate all candidates and save a transparent selection record."""

        if self.split == "test":
            raise ValueError(
                "VLM model selection cannot use the locked test split. "
                "Set jaad_context_split to train or val."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        summaries: dict[str, dict[str, Any]] = {}
        candidates = [
            (model_id, prompt_mode)
            for model_id in self.model_ids
            for prompt_mode in self.prompt_modes
        ]
        for index, (model_id, prompt_mode) in enumerate(candidates, start=1):
            result_dir = self.output_dir / candidate_slug(model_id, prompt_mode)
            print(
                f"\nVLM candidate {index}/{len(candidates)}: "
                f"{model_id} [{prompt_mode}]"
            )
            summary = JAADContextBenchmark(
                self.config,
                model_id=model_id,
                prompt_mode=prompt_mode,
                output_dir=result_dir,
            ).run()
            candidate_key = f"{model_id}::{prompt_mode}"
            summaries[candidate_key] = summary
            rows.append(
                {
                    "model_id": model_id,
                    "prompt_mode": prompt_mode,
                    **comparison_metrics(summary),
                    "result_directory": str(result_dir),
                }
            )

        selected_model, selected_prompt_mode = select_candidate(rows)
        for row in rows:
            row["selected"] = (
                row["model_id"] == selected_model
                and row["prompt_mode"] == selected_prompt_mode
            )

        with self.comparison_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "selection_split": self.split,
            "selection_rule": (
                "Highest derived permission-present macro F1, then mean macro F1 over "
                "context fields with at least two observed ground-truth classes, minimum "
                "field macro F1, mean context accuracy, the simpler baseline on an exact "
                "tie, and model ID"
            ),
            "selected_model": selected_model,
            "selected_prompt_mode": selected_prompt_mode,
            "models": rows,
            "model_summaries": summaries,
        }
        self.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.selected_json.write_text(
            json.dumps(
                {
                    "selected_model": selected_model,
                    "selected_prompt_mode": selected_prompt_mode,
                    "selection_split": self.split,
                    "config_entries": {
                        "vlm_model": selected_model,
                        "vlm_prompt_mode": selected_prompt_mode,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._print_summary(rows, selected_model, selected_prompt_mode)
        return payload

    @staticmethod
    def _print_summary(
        rows: list[dict[str, Any]],
        selected_model: str,
        selected_prompt_mode: str,
    ) -> None:
        print("\nJAAD VLM model comparison")
        for row in rows:
            marker = (
                "SELECTED"
                if row["model_id"] == selected_model
                and row["prompt_mode"] == selected_prompt_mode
                else ""
            )
            print(
                f"{row['model_id']} [{row['prompt_mode']}]: context macro F1="
                f"{float(row['mean_context_macro_f1_percent']):.2f}% "
                f"permission F1="
                f"{float(row['permission_present_macro_f1_percent']):.2f}% "
                f"minimum field F1={float(row['minimum_field_macro_f1_percent']):.2f}% "
                f"context accuracy={float(row['mean_context_accuracy_percent']):.2f}% {marker}"
            )
        print(f"Selected model: {selected_model}")
        print(f"Selected prompt mode: {selected_prompt_mode}")
