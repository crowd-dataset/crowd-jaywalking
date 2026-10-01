"""Attribute end-to-end evaluation errors to a pipeline stage.

This reads the JSON details already written by scripts/evaluation/run_evaluation.py. It performs no
inference, so it needs no GPU and no model weights. The report separates three
questions that the single video level accuracy number confuses:

1. Is the VLM reporting observable context the way a human annotator would?
2. Is the deterministic policy discarding decidable cases as UNCERTAIN?
3. Is the video level OR over people turning per person errors into video errors?
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.core.config import ProjectConfig
from scripts.core.models import ContextAssessment, Ternary, Visibility
from scripts.core.policy import JaywalkingPolicy
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


CONTEXT_FIELDS = (
    "marked_crosswalk",
    "permissive_pedestrian_signal",
    "authorised_crossing_sign",
    "crossing_guard_permission",
    "prohibitive_pedestrian_signal",
    "traffic_light",
    "visibility",
)


def load_details(results_dir: Path) -> list[dict[str, Any]]:
    """Load every per video details file written by the evaluation runner."""

    details_dir = results_dir / "details"
    if not details_dir.is_dir():
        raise FileNotFoundError(
            f"No details directory found: {details_dir}. "
            "Run scripts/evaluation/run_evaluation.py first, or point 'results' at the finished run."
        )
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(details_dir.glob("*.json"))
    ]
    if not payloads:
        raise FileNotFoundError(f"No details JSON files in {details_dir}")
    return payloads


def video_label(labels: list[str]) -> str:
    """Reproduce the pipeline's video level aggregation over person labels."""

    if not labels:
        return "COMPLIANT"
    if "JAYWALKING" in labels:
        return "JAYWALKING"
    if "UNCERTAIN" in labels:
        return "UNCERTAIN"
    return "COMPLIANT"


def accuracy(payloads: list[dict[str, Any]], predictions: list[str]) -> float:
    correct = sum(
        payload["ground_truth"] == prediction
        for payload, prediction in zip(payloads, predictions)
    )
    return 100.0 * correct / len(payloads) if payloads else 0.0


def report_context_fields(payloads: list[dict[str, Any]]) -> None:
    """Show what the VLM actually reported across every evaluated person."""

    people = [
        person
        for payload in payloads
        for person in payload["result"]["person_decisions"]
    ]
    logger.info("{}", f"\n{'=' * 72}\nVLM CONTEXT REPORTED ACROSS {len(people)} PERSON DECISIONS\n{'=' * 72}")
    if not people:
        logger.info("No person decisions were recorded.")
        return
    for field in CONTEXT_FIELDS:
        counts = Counter(str(person["context"].get(field)) for person in people)
        parts = [
            f"{value}={count} ({100.0 * count / len(people):.0f}%)"
            for value, count in counts.most_common()
        ]
        logger.info("  {:32s} {}", field, '  '.join(parts))

    logger.info("\nPolicy reasons:")
    for reason, count in Counter(person["reason"] for person in people).most_common():
        logger.info("  {:4d}  {}", count, reason)


def report_per_video(payloads: list[dict[str, Any]]) -> None:
    """Show where each video's label came from, ordered by people count."""

    logger.info("{}", f"\n{'=' * 72}\nPER VIDEO OUTCOMES\n{'=' * 72}")
    logger.info("{:<14}{:>7}  {:<13}{:<13}result", 'video', 'people', 'ground truth', 'prediction')
    rows = sorted(
        payloads,
        key=lambda payload: len(payload["result"]["person_decisions"]),
    )
    for payload in rows:
        people = payload["result"]["person_decisions"]
        prediction = payload["result"]["prediction"]
        truth = payload["ground_truth"]
        outcome = "correct" if prediction == truth else "WRONG"
        logger.info("{:<14}{:>7}  {:<13}{:<13}{}", payload['video_id'], len(people), truth, prediction, outcome)


def report_people_count_effect(payloads: list[dict[str, Any]]) -> None:
    """Test whether video errors grow with the number of accepted crossings."""

    logger.info("{}", f"\n{'=' * 72}\nEFFECT OF ACCEPTED CROSSING COUNT\n{'=' * 72}")
    buckets: dict[str, list[dict[str, Any]]] = {"1-2 people": [], "3-4 people": [], "5+ people": []}
    for payload in payloads:
        count = len(payload["result"]["person_decisions"])
        key = "1-2 people" if count <= 2 else "3-4 people" if count <= 4 else "5+ people"
        buckets[key].append(payload)

    logger.info("{:<14}{:>7}{:>9}{:>10}   predictions", 'bucket', 'videos', 'correct', 'accuracy')
    for name, group in buckets.items():
        if not group:
            continue
        correct = sum(
            payload["ground_truth"] == payload["result"]["prediction"]
            for payload in group
        )
        spread = Counter(payload["result"]["prediction"] for payload in group)
        summary = " ".join(f"{label}={count}" for label, count in sorted(spread.items()))
        logger.info("{:<14}{:>7}{:>9}{:>9.0f}%   {}", name, len(group), correct, 100.0 * correct / len(group), summary)


def report_threshold_sweep(payloads: list[dict[str, Any]]) -> None:
    """Simulate a stricter crossing threshold without rerunning the VLM.

    Raising the threshold can only remove accepted people, and the context of
    every person who survives is already known, so the video label for any
    higher threshold can be recomputed exactly from the saved results.
    """

    logger.info("{}", f"\n{'=' * 72}\nCROSSING THRESHOLD SWEEP (recomputed, no inference)\n{'=' * 72}")
    usable = [
        payload
        for payload in payloads
        if payload["result"].get("crossing_classifications")
    ]
    if not usable:
        logger.info("No classifier scores were saved; the run used rule based detection.")
        return

    current = usable[0]["result"]["crossing_classifications"][0]["threshold"]
    logger.info("Current threshold: {}", current)
    logger.info("{:>10}{:>10}{:>5}{:>5}{:>5}{:>5}{:>6}", 'threshold', 'accuracy', 'TP', 'TN', 'FP', 'FN', 'UNC')
    for step in range(0, 9):
        threshold = round(float(current) + 0.05 * step, 2)
        if threshold > 1.0:
            break
        predictions: list[str] = []
        for payload in usable:
            probability = {
                item["person_id"]: float(item["probability"])
                for item in payload["result"]["crossing_classifications"]
            }
            labels = [
                person["label"]
                for person in payload["result"]["person_decisions"]
                if probability.get(person["person_id"], 1.0) >= threshold
            ]
            predictions.append(video_label(labels))

        counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "UNC": 0}
        for payload, prediction in zip(usable, predictions):
            truth = payload["ground_truth"]
            if prediction == "UNCERTAIN":
                counts["UNC"] += 1
            elif truth == "JAYWALKING" and prediction == "JAYWALKING":
                counts["TP"] += 1
            elif truth == "COMPLIANT" and prediction == "COMPLIANT":
                counts["TN"] += 1
            elif truth == "COMPLIANT":
                counts["FP"] += 1
            else:
                counts["FN"] += 1
        logger.info(
            "{:>10.2f}{:>9.1f}%{:>5}{:>5}{:>5}{:>5}{:>6}",
            threshold,
            accuracy(usable, predictions),
            counts['TP'],
            counts['TN'],
            counts['FP'],
            counts['FN'],
            counts['UNC'],
        )


def report_counterfactuals(payloads: list[dict[str, Any]]) -> None:
    """Quantify how much each policy rule costs on this split."""

    logger.info("{}", f"\n{'=' * 72}\nPOLICY COUNTERFACTUALS\n{'=' * 72}")
    baseline = [payload["result"]["prediction"] for payload in payloads]
    logger.info("  {:<46}{:>6.1f}%", 'as run', accuracy(payloads, baseline))

    treat_uncertain_compliant = [
        "COMPLIANT" if prediction == "UNCERTAIN" else prediction
        for prediction in baseline
    ]
    logger.info("  {:<46}{:>6.1f}%", 'UNCERTAIN counted as COMPLIANT', accuracy(payloads, treat_uncertain_compliant))

    treat_uncertain_jaywalking = [
        "JAYWALKING" if prediction == "UNCERTAIN" else prediction
        for prediction in baseline
    ]
    logger.info("  {:<46}{:>6.1f}%", 'UNCERTAIN counted as JAYWALKING', accuracy(payloads, treat_uncertain_jaywalking))

    majority = []
    for payload in payloads:
        labels = [person["label"] for person in payload["result"]["person_decisions"]]
        jaywalkers = sum(label == "JAYWALKING" for label in labels)
        majority.append("JAYWALKING" if jaywalkers >= 2 else video_label(
            [label for label in labels if label != "JAYWALKING"]
        ))
    logger.info("  {:<46}{:>6.1f}%", 'require 2+ jaywalking people per video', accuracy(payloads, majority))


POLICY_VARIANTS = (
    ("legacy: four cues, person scope", {}),
    (
        "zebra and light, person scope",
        {
            "permission_cues": ["marked_crosswalk", "permissive_pedestrian_signal"],
            "partial_visibility_uncertain": False,
        },
    ),
    (
        "zebra and light, scene scope",
        {
            "permission_cues": ["marked_crosswalk", "permissive_pedestrian_signal"],
            "partial_visibility_uncertain": False,
            "context_scope": "scene",
        },
    ),
    (
        "four cues, scene scope",
        {"partial_visibility_uncertain": False, "context_scope": "scene"},
    ),
    (
        "zebra and light, scene, strict absence",
        {
            "permission_cues": ["marked_crosswalk", "permissive_pedestrian_signal"],
            "prohibitive_signal_overrides_crosswalk": False,
            "partial_visibility_uncertain": False,
            "context_scope": "scene",
            "strict_absence": True,
        },
    ),
)


def _ternary(value: Any) -> Ternary | None:
    return None if value is None else Ternary(value)


def _assessment(context: dict[str, Any]) -> ContextAssessment:
    return ContextAssessment(
        marked_crosswalk=Ternary(context["marked_crosswalk"]),
        permissive_pedestrian_signal=_ternary(context["permissive_pedestrian_signal"]),
        authorised_crossing_sign=_ternary(context["authorised_crossing_sign"]),
        crossing_guard_permission=_ternary(context["crossing_guard_permission"]),
        prohibitive_pedestrian_signal=_ternary(context["prohibitive_pedestrian_signal"]),
        visibility=Visibility(context["visibility"]),
        evidence_summary=str(context.get("evidence_summary", "")),
        traffic_light=_ternary(context.get("traffic_light")),
    )


def report_policy_variants(payloads: list[dict[str, Any]]) -> None:
    """Re-apply alternative policy settings to the saved VLM context."""

    logger.info("{}", f"\n{'=' * 72}\nPOLICY VARIANTS (recomputed, no inference)\n{'=' * 72}")
    logger.info("  {:<40}{:>9}{:>5}{:>5}{:>5}{:>5}{:>6}", 'variant', 'accuracy', 'TP', 'TN', 'FP', 'FN', 'UNC')
    for name, settings in POLICY_VARIANTS:
        policy = JaywalkingPolicy(settings)
        predictions = []
        for payload in payloads:
            contexts = [
                _assessment(person["context"])
                for person in payload["result"]["person_decisions"]
            ]
            labels = [label.value for label, _ in policy.decide_all(contexts)]
            predictions.append(video_label(labels))
        counts = Counter(
            "UNC" if prediction == "UNCERTAIN"
            else ("T" if (prediction == payload["ground_truth"]) else "F")
            + ("P" if prediction == "JAYWALKING" else "N")
            for payload, prediction in zip(payloads, predictions)
        )
        logger.info(
            "  {:<40}{:>8.1f}%{:>5}{:>5}{:>5}{:>5}{:>6}",
            name,
            accuracy(payloads, predictions),
            counts['TP'],
            counts['TN'],
            counts['FP'],
            counts['FN'],
            counts['UNC'],
        )


def main() -> None:
    config = ProjectConfig.load()
    results_dir = config.path("results")
    payloads = load_details(results_dir)
    logger.info("Loaded {} videos from {}", len(payloads), results_dir)

    report_context_fields(payloads)
    report_per_video(payloads)
    report_people_count_effect(payloads)
    report_threshold_sweep(payloads)
    report_counterfactuals(payloads)
    report_policy_variants(payloads)


if __name__ == "__main__":
    main()
