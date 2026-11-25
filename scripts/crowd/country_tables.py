"""The country tables of the README, merged from the run summaries of one or more machines.

Each summary is a JSON written by summarise_runs.py where the runs are. A segment is counted once, however
many runs or machines finished it; a planned segment that no run finished counts as failed.

    uv run python scripts/crowd/country_tables.py summary_spark.json summary_wsl5090.json --output-dir docs

Writes four tables:

* crossings_by_country.csv: per country the segments, hours, crossings detected (stage 1), at a zebra crossing
  or traffic light, rejected by the VLM, without a marking (jaywalking) and jaywalking per hour.
* jaywalking_rates.csv: the same with a 95% interval of the rate (bootstrap over segments, 4,000 resamples,
  seeded) and of the share of judged crossings without a marking (Wilson), and the share rejected by the VLM.
* crossing_movement.csv: per country and group (without a marking, at a marking) the people whose movement
  could be measured and the movement measures with 95% intervals; `compared` is True where there are at least
  20 people without a marking.
* infrastructure_by_country.csv: per country the street scenes (segments) where someone crossed and the marking
  check judged it, the scenes among them without crossing infrastructure (no crossing was judged at a zebra
  crossing, traffic light or crossing sign), and the crossings without a marking. Both counts come from the same
  marking check, so they are not independent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["Bangladesh", "Vietnam", "France", "Denmark", "Sweden", "South Korea", "China", "Japan", "India",
         "Brazil", "Indonesia", "Poland", "South Africa", "Thailand", "Türkiye", "United States"]
TABLE_ORDER = ["China", "Japan", "Bangladesh", "France", "Sweden", "Denmark", "South Korea", "Vietnam", "India",
               "Brazil", "Indonesia", "Poland", "South Africa", "Thailand", "Türkiye", "United States"]
MIN_COMPARED = 20
SEED = 20261116
COUNTS = ("hours", "detected", "rejected", "compliant", "jaywalking", "uncertain")
GROUPS = (("JAYWALKING", "without_marking"), ("COMPLIANT", "at_marking"))


def merge(paths: list[Path]) -> dict:
    """Segments counted once, planned segments as a union, and the movement aggregates summed."""

    data = {"segments": [], "planned": defaultdict(set), "movement": {}, "flags": {}, "bins": None}
    seen = set()
    for path in paths:
        part = json.loads(path.read_text(encoding="utf-8"))
        if "planned_keys" not in part:
            raise SystemExit(f"{path} has no planned_keys: write it again with summarise_runs.py")
        for segment in part["segments"]:
            if segment["video_key"] not in seen:
                seen.add(segment["video_key"])
                data["segments"].append(segment)
        for country, keys in part["planned_keys"].items():
            data["planned"][country].update(keys)
        data["bins"] = part["bins"]
        for key, hist in part["movement"].items():
            if key not in data["movement"]:
                data["movement"][key] = hist
                continue
            for metric, h in hist.items():
                m = data["movement"][key][metric]
                m["n"] += h["n"]
                m["sum"] += h["sum"]
                m["sumsq"] += h["sumsq"]
                m["counts"] = [a + b for a, b in zip(m["counts"], h["counts"])]
        for key, f in part["flags"].items():
            mine = data["flags"].setdefault(key, {k: 0 for k in f})
            for k, v in f.items():
                mine[k] += v
    return data


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def bootstrap_rate(rng, segments: list[dict], field: str, reps: int = 4000) -> tuple[float, float, float]:
    counts = np.array([s[field] for s in segments], float)
    hours = np.array([s["hours"] for s in segments], float)
    index = rng.integers(0, len(segments), size=(reps, len(segments)))
    rates = counts[index].sum(1) / hours[index].sum(1)
    return counts.sum() / hours.sum(), np.percentile(rates, 2.5), np.percentile(rates, 97.5)


def mean_ci(h: dict) -> tuple[float, float, float]:
    n = h["n"]
    if n < 2:
        return math.nan, math.nan, math.nan
    m = h["sum"] / n
    se = math.sqrt(max(0.0, (h["sumsq"] - n * m * m) / (n - 1)) / n)
    return m, m - 1.96 * se, m + 1.96 * se


def hist_quantile(h: dict, lo: float, step: float, q: float) -> float:
    total = sum(h["counts"])
    if total == 0:
        return math.nan
    target, run = q * total, 0
    for i, c in enumerate(h["counts"]):
        if run + c >= target and c > 0:
            return lo + step * (i + (target - run) / c)
        run += c
    return lo + step * len(h["counts"])


def summarise(data: dict) -> dict[str, dict]:
    by_country = defaultdict(list)
    for segment in data["segments"]:
        by_country[segment["country"]].append(segment)
    rng = np.random.default_rng(SEED)
    summary = {}
    for country in [c for c in ORDER if c in by_country]:
        segments = by_country[country]
        totals = {k: sum(s[k] for s in segments) for k in COUNTS}
        rate, low, high = bootstrap_rate(rng, segments, "jaywalking")
        bootstrap_rate(rng, segments, "detected")  # keeps the random draws the same as in earlier tables
        share, share_low, share_high = wilson(totals["jaywalking"], totals["jaywalking"] + totals["compliant"])
        planned = len(data["planned"].get(country, ()))
        judged = [s for s in segments if s["jaywalking"] + s["compliant"] > 0]
        summary[country] = dict(totals, segments=len(segments), planned=planned, rate=rate, low=low, high=high,
                                share=share, share_low=share_low, share_high=share_high, scenes_judged=len(judged),
                                scenes_without_infrastructure=sum(s["compliant"] == 0 for s in judged))
    return summary


def write_tables(data: dict, summary: dict[str, dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "crossings_by_country.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["country", "segments", "hours", "crossings_detected", "at_zebra_or_light", "rejected_by_vlm",
                         "crossings_without_zebra_or_light", "jaywalking_per_hour"])
        for c in [c for c in TABLE_ORDER if c in summary]:
            s = summary[c]
            writer.writerow([c, s["segments"], f"{s['hours']:.3f}", s["detected"], s["compliant"], s["rejected"],
                             s["jaywalking"], f"{s['jaywalking'] / s['hours']:.2f}"])

    with (output_dir / "jaywalking_rates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["country", "segments", "hours", "crossings_detected", "at_zebra_or_light", "rejected_by_vlm",
                         "crossings_without_zebra_or_light", "uncertain", "jaywalking_per_hour", "per_hour_low",
                         "per_hour_high", "share_without_pct", "share_low_pct", "share_high_pct", "rejected_pct"])
        for c, s in summary.items():
            writer.writerow([c, s["segments"], f"{s['hours']:.3f}", s["detected"], s["compliant"], s["rejected"],
                             s["jaywalking"], s["uncertain"], f"{s['rate']:.3f}", f"{s['low']:.3f}",
                             f"{s['high']:.3f}", f"{100 * s['share']:.2f}", f"{100 * s['share_low']:.2f}",
                             f"{100 * s['share_high']:.2f}", f"{100 * s['rejected'] / max(1, s['detected']):.2f}"])

    with (output_dir / "infrastructure_by_country.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["country", "segments", "hours", "scenes_judged", "scenes_without_infrastructure",
                         "crossings_without_zebra_or_light"])
        for c, s in summary.items():
            writer.writerow([c, s["segments"], f"{s['hours']:.3f}", s["scenes_judged"],
                             s["scenes_without_infrastructure"], s["jaywalking"]])

    lo, _, step = data["bins"]["lateral_speed"]
    with (output_dir / "crossing_movement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["country", "group", "people", "compared", "lateral_speed_median", "lateral_speed_mean",
                         "lateral_speed_low", "lateral_speed_high", "stopped_pct", "stopped_low_pct",
                         "stopped_high_pct", "with_others_pct", "with_others_low_pct", "with_others_high_pct",
                         "vehicles_mean", "vehicles_low", "vehicles_high"])
        for c in summary:
            compared = data["flags"].get(f"{c}|JAYWALKING", {}).get("n", 0) >= MIN_COMPARED
            for label, group in GROUPS:
                flags, movement = data["flags"].get(f"{c}|{label}"), data["movement"].get(f"{c}|{label}")
                if not flags or flags["n"] == 0:
                    continue
                row = [c, group, flags["n"], compared,
                       f"{hist_quantile(movement['lateral_speed'], lo, step, 0.5):.3f}"]
                row += [f"{x:.3f}" for x in mean_ci(movement["lateral_speed"])]
                for key in ("stopped", "in_group"):
                    row += [f"{100 * x:.2f}" for x in wilson(flags[key], flags["n"])]
                row += [f"{x:.3f}" for x in mean_ci(movement["vehicles"])]
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+", type=Path, help="JSONs written by summarise_runs.py")
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    arguments = parser.parse_args()
    data = merge(sorted(arguments.summaries))
    summary = summarise(data)
    write_tables(data, summary, arguments.output_dir)
    done = sum(s["segments"] for s in summary.values())
    planned = sum(s["planned"] for s in summary.values())
    print(f"wrote 4 tables to {arguments.output_dir}: {done} of {planned} planned segments, {planned - done} failed")
    for c, s in summary.items():
        print(f"  {c}: {s['segments']}/{s['planned']} segments, {s['hours']:.1f} h, {s['jaywalking']} jaywalking")


if __name__ == "__main__":
    main()
