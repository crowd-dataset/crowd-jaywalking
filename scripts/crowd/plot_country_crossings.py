"""Bar plot of the crossings found per country in a CROWD run.

For each country: the crossings detected (stage 1, pedestrians crossing the road) and those at a zebra crossing or traffic light, and those that
had no zebra crossing, traffic light or crossing sign (stage 2 claims, the label JAYWALKING).
The country law (stage 3) is not used.

    uv run python scripts/crowd/plot_country_crossings.py results/six_countries
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DETECTED_COLOUR = "#4C78A8"
MARKED_COLOUR = "#72B7B2"
UNMARKED_COLOUR = "#E45756"


def summarise(run: Path) -> list[dict[str, float | str]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with (run / "per_video_results.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            country = row["country"]
            totals[country]["hours"] += (float(row["effective_end_second"]) - float(row["start_second"])) / 3600
            totals[country]["segments"] += 1
            totals[country]["detected"] += int(row["predicted_crossings"])
            totals[country]["unmarked"] += int(row["jaywalking_people"])
    with (run / "per_person_results.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["predicted_crossing"] == "True":
                label = row["jaywalking_label"]
                key = {"COMPLIANT": "marked", "NOT_EVALUATED": "unconfirmed"}.get(label)
                if key:
                    totals[row["country"]][key] += 1
    return [{"country": name, **values} for name, values in sorted(totals.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="results folder of a CROWD run")
    parser.add_argument("--countries", nargs="*", help="order of the bars")
    arguments = parser.parse_args()
    rows = summarise(arguments.run)
    if arguments.countries:
        rows = sorted(rows, key=lambda r: arguments.countries.index(r["country"]) if r["country"] in arguments.countries else 99)
    with (arguments.run / "crossings_by_country.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["country", "segments", "hours", "crossings_detected", "at_zebra_or_light", "not_confirmed_by_vlm", "crossings_without_zebra_or_light"])
        for r in rows:
            writer.writerow([r["country"], int(r["segments"]), f"{r['hours']:.2f}", int(r["detected"]), int(r["marked"]), int(r["unconfirmed"]), int(r["unmarked"])])

    names = [str(r["country"]) for r in rows]
    figure, (counts, rates) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    width = 0.27
    positions = range(len(names))
    for axis, scale, title, label in (
        (counts, lambda r: 1.0, "Crossings in the sample", "Number of crossings"),
        (rates, lambda r: 1.0 / max(float(r["hours"]), 1e-9), "Crossings per hour of footage", "Crossings per hour"),
    ):
        series = (
            ("detected", "Crossings detected", DETECTED_COLOUR, -width),
            ("marked", "At a zebra crossing or traffic light", MARKED_COLOUR, 0.0),
            ("unmarked", "Without zebra crossing or traffic light", UNMARKED_COLOUR, width),
        )
        for key, name, colour, shift in series:
            values = [float(r[key]) * scale(r) for r in rows]
            bars = axis.bar([p + shift for p in positions], values, width, label=name, color=colour)
            axis.bar_label(bars, fmt="%g" if scale(rows[0]) == 1.0 else "%.2f", padding=2, fontsize=8)
        axis.set_xticks(list(positions), [f"{n}\n{float(r['hours']):.1f} h" for n, r in zip(names, rows)])
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    counts.legend(frameon=False)
    figure.suptitle("Pedestrians crossing the road, daytime dashcam footage")
    output = arguments.run / "crossings_by_country.png"
    figure.savefig(output, dpi=160)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
