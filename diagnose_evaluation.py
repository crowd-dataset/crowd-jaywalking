"""Attribute end-to-end evaluation errors to a pipeline stage.

This reads the JSON details already written by run_evaluation.py. It performs no
inference, so it needs no GPU and no model weights. The report separates three
questions that the single video level accuracy number confuses:

1. Is the VLM reporting observable context the way a human annotator would?
2. Is the deterministic policy discarding decidable cases as UNCERTAIN?
3. Is the video level OR over people turning per person errors into video errors?
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from crowd_jaywalking.config import ProjectConfig


CONTEXT_FIELDS = (
    "marked_crosswalk",
    "permissive_pedestrian_signal",
    "authorised_crossing_sign",
    "crossing_guard_permission",
    "prohibitive_pedestrian_signal",
    "visibility",
)


def load_details(results_dir: Path) -> list[dict[str, Any]]:
    """Load every per video details file written by the evaluation runner."""

    details_dir = results_dir / "details"
    if not details_dir.is_dir():
        raise FileNotFoundError(
            f"No details directory found: {details_dir}. "
            "Run run_evaluation.py first, or point 'results' at the finished run."
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
    print(f"\n{'=' * 72}\nVLM CONTEXT REPORTED ACROSS {len(people)} PERSON DECISIONS\n{'=' * 72}")
    if not people:
        print("No person decisions were recorded.")
        return
    for field in CONTEXT_FIELDS:
        counts = Counter(str(person["context"][field]) for person in people)
        parts = [
            f"{value}={count} ({100.0 * count / len(people):.0f}%)"
            for value, count in counts.most_common()
        ]
        print(f"  {field:32s} {'  '.join(parts)}")

    print("\nPolicy reasons:")
    for reason, count in Counter(person["reason"] for person in people).most_common():
        print(f"  {count:4d}  {reason}")


def report_per_video(payloads: list[dict[str, Any]]) -> None:
    """Show where each video's label came from, ordered by people count."""

    print(f"\n{'=' * 72}\nPER VIDEO OUTCOMES\n{'=' * 72}")
    print(f"{'video':<14}{'people':>7}  {'ground truth':<13}{'prediction':<13}result")
    rows = sorted(
        payloads,
        key=lambda payload: len(payload["result"]["person_decisions"]),
    )
    for payload in rows:
        people = payload["result"]["person_decisions"]
        prediction = payload["result"]["prediction"]
        truth = payload["ground_truth"]
        outcome = "correct" if prediction == truth else "WRONG"
        print(
            f"{payload['video_id']:<14}{len(people):>7}  "
            f"{truth:<13}{prediction:<13}{outcome}"
        )


def report_people_count_effect(payloads: list[dict[str, Any]]) -> None:
    """Test whether video errors grow with the number of accepted crossings."""

    print(f"\n{'=' * 72}\nEFFECT OF ACCEPTED CROSSING COUNT\n{'=' * 72}")
    buckets: dict[str, list[dict[str, Any]]] = {"1-2 people": [], "3-4 people": [], "5+ people": []}
    for payload in payloads:
        count = len(payload["result"]["person_decisions"])
        key = "1-2 people" if count <= 2 else "3-4 people" if count <= 4 else "5+ people"
        buckets[key].append(payload)

    print(f"{'bucket':<14}{'videos':>7}{'correct':>9}{'accuracy':>10}   predictions")
    for name, group in buckets.items():
        if not group:
            continue
        correct = sum(
            payload["ground_truth"] == payload["result"]["prediction"]
            for payload in group
        )
        spread = Counter(payload["result"]["prediction"] for payload in group)
        summary = " ".join(f"{label}={count}" for label, count in sorted(spread.items()))
        print(
            f"{name:<14}{len(group):>7}{correct:>9}"
            f"{100.0 * correct / len(group):>9.0f}%   {summary}"
        )


def report_threshold_sweep(payloads: list[dict[str, Any]]) -> None:
    """Simulate a stricter crossing threshold without rerunning the VLM.

    Raising the threshold can only remove accepted people, and the context of
    every person who survives is already known, so the video label for any
    higher threshold can be recomputed exactly from the saved results.
    """

    print(f"\n{'=' * 72}\nCROSSING THRESHOLD SWEEP (recomputed, no inference)\n{'=' * 72}")
    usable = [
        payload
        for payload in payloads
        if payload["result"].get("crossing_classifications")
    ]
    if not usable:
        print("No classifier scores were saved; the run used rule based detection.")
        return

    current = usable[0]["result"]["crossing_classifications"][0]["threshold"]
    print(f"Current threshold: {current}")
    print(f"{'threshold':>10}{'accuracy':>10}{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}{'UNC':>6}")
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
        print(
            f"{threshold:>10.2f}{accuracy(usable, predictions):>9.1f}%"
            f"{counts['TP']:>5}{counts['TN']:>5}{counts['FP']:>5}"
            f"{counts['FN']:>5}{counts['UNC']:>6}"
        )


def report_counterfactuals(payloads: list[dict[str, Any]]) -> None:
    """Quantify how much each policy rule costs on this split."""

    print(f"\n{'=' * 72}\nPOLICY COUNTERFACTUALS\n{'=' * 72}")
    baseline = [payload["result"]["prediction"] for payload in payloads]
    print(f"  {'as run':<46}{accuracy(payloads, baseline):>6.1f}%")

    treat_uncertain_compliant = [
        "COMPLIANT" if prediction == "UNCERTAIN" else prediction
        for prediction in baseline
    ]
    print(
        f"  {'UNCERTAIN counted as COMPLIANT':<46}"
        f"{accuracy(payloads, treat_uncertain_compliant):>6.1f}%"
    )

    treat_uncertain_jaywalking = [
        "JAYWALKING" if prediction == "UNCERTAIN" else prediction
        for prediction in baseline
    ]
    print(
        f"  {'UNCERTAIN counted as JAYWALKING':<46}"
        f"{accuracy(payloads, treat_uncertain_jaywalking):>6.1f}%"
    )

    majority = []
    for payload in payloads:
        labels = [person["label"] for person in payload["result"]["person_decisions"]]
        jaywalkers = sum(label == "JAYWALKING" for label in labels)
        majority.append("JAYWALKING" if jaywalkers >= 2 else video_label(
            [label for label in labels if label != "JAYWALKING"]
        ))
    print(
        f"  {'require 2+ jaywalking people per video':<46}"
        f"{accuracy(payloads, majority):>6.1f}%"
    )

    ignore_partial = []
    for payload in payloads:
        labels = []
        for person in payload["result"]["person_decisions"]:
            context = person["context"]
            if context["visibility"] == "INSUFFICIENT":
                labels.append("UNCERTAIN")
                continue
            if context["prohibitive_pedestrian_signal"] == "YES":
                labels.append("JAYWALKING")
                continue
            permissions = [
                context[field]
                for field in CONTEXT_FIELDS[:4]
            ]
            if "YES" in permissions:
                labels.append("COMPLIANT")
            elif "UNCERTAIN" in permissions:
                labels.append("UNCERTAIN")
            else:
                labels.append("JAYWALKING")
        ignore_partial.append(video_label(labels))
    print(
        f"  {'PARTIAL visibility no longer forces UNCERTAIN':<46}"
        f"{accuracy(payloads, ignore_partial):>6.1f}%"
    )


def main() -> None:
    config = ProjectConfig.load(os.environ.get("CROWD_JAYWALKING_CONFIG"))
    results_dir = config.path("results")
    payloads = load_details(results_dir)
    print(f"Loaded {len(payloads)} videos from {results_dir}")

    report_context_fields(payloads)
    report_per_video(payloads)
    report_people_count_effect(payloads)
    report_threshold_sweep(payloads)
    report_counterfactuals(payloads)


if __name__ == "__main__":
    main()
