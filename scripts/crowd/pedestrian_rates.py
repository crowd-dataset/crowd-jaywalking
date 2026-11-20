"""Crossings per pedestrian in view, per country, next to the rate per hour of footage.

A rate per hour of footage grows with the number of people on the street, so busy streets look like
places where people jaywalk more. This table divides by the people instead, using the counts of
count_pedestrians.py (person tracks seen for at least a second, and person-hours in view):

- jaywalking per 1,000 pedestrians in view, and per 100 pedestrian-hours in view;
- jaywalking per 100 crossing candidates (people stage 1 judged as possible crossers);
- all detected crossings per 1,000 pedestrians, for comparison;

each with a 95% interval from a bootstrap over segments (4,000 resamples, seed as in country_tables.py).
Segments come from the summaries (each counted once, as in the country tables) and are joined to the
pedestrian counts by video key.

    uv run python scripts/crowd/pedestrian_rates.py paper/stats/summary_*.json \
        --pedestrians paper/stats/pedestrians_*.csv --output docs/jaywalking_per_pedestrian.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.crowd.country_tables import SEED, TABLE_ORDER, merge  # noqa: E402

REPS = 4000
# (column, numerator field, denominator field, scale)
MEASURES = (
    ("jaywalking_per_1000_pedestrians", "jaywalking", "pedestrians", 1000),
    ("jaywalking_per_100_pedestrian_hours", "jaywalking", "pedestrian_hours", 100),
    ("jaywalking_per_100_candidates", "jaywalking", "candidates", 100),
    ("crossings_per_1000_pedestrians", "detected", "pedestrians", 1000),
    ("jaywalking_per_hour", "jaywalking", "hours", 1),
)


def ratio_ci(rng, num: np.ndarray, den: np.ndarray) -> tuple[float, float, float]:
    index = rng.integers(0, len(num), size=(REPS, len(num)))
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = num[index].sum(1) / den[index].sum(1)
    return num.sum() / den.sum(), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--pedestrians", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    counts: dict[str, dict] = {}
    for path in arguments.pedestrians:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                counts.setdefault(row["video_key"], row)
    segments = []
    unmatched = 0
    for segment in merge(arguments.summaries)["segments"]:
        row = counts.get(segment["video_key"])
        if row is None:
            unmatched += 1
            continue
        segments.append({**segment, "pedestrians": float(row["pedestrians"]),
                         "pedestrian_hours": float(row["pedestrian_hours"]), "candidates": float(row["candidates"])})
    countries = [c for c in TABLE_ORDER if any(s["country"] == c for s in segments)]
    countries += sorted({s["country"] for s in segments} - set(countries))

    rng = np.random.default_rng(SEED)
    rows = []
    for country in countries + ["All"]:
        mine = [s for s in segments if country == "All" or s["country"] == country]
        row = {"country": country, "segments": len(mine), "hours": round(sum(s["hours"] for s in mine), 2),
               "pedestrians": int(sum(s["pedestrians"] for s in mine)),
               "pedestrian_hours": round(sum(s["pedestrian_hours"] for s in mine), 1),
               "candidates": int(sum(s["candidates"] for s in mine)),
               "jaywalking": int(sum(s["jaywalking"] for s in mine)),
               "detected": int(sum(s["detected"] for s in mine))}
        for name, num, den, scale in MEASURES:
            value, low, high = ratio_ci(rng, np.array([s[num] for s in mine], float),
                                        np.array([s[den] for s in mine], float))
            row[name] = round(scale * value, 3)
            row[f"{name}_low"] = round(scale * low, 3)
            row[f"{name}_high"] = round(scale * high, 3)
        rows.append(row)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {arguments.output}: {len(segments)} segments joined, {unmatched} without a pedestrian count")


if __name__ == "__main__":
    main()
