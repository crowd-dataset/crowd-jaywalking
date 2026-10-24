"""Scatter plot of total crossings against jaywalking crossings, one point per country.

X axis: crossings without a zebra crossing, traffic light or crossing sign (stage 2 claims, the label
JAYWALKING). Y axis: all crossings detected (stage 1). Results folders of several runs can be combined.

    uv run python scripts/crowd/plot_crossing_scatter.py results/six_countries results/korea_vietnam \
        --output results/crossings_scatter.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.crowd.plot_country_crossings import summarise  # noqa: E402

COLOUR = "#4C78A8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="results folders of CROWD runs")
    parser.add_argument("--output", type=Path, default=Path("crossings_scatter.png"))
    parser.add_argument("--per-hour", action="store_true", help="divide both axes by the hours of footage")
    arguments = parser.parse_args()

    rows = [row for run in arguments.runs for row in summarise(run)]
    scale = (lambda row: 1.0 / float(row["hours"])) if arguments.per_hour else (lambda row: 1.0)
    x = [float(row["unmarked"]) * scale(row) for row in rows]
    y = [float(row["detected"]) * scale(row) for row in rows]

    figure, axis = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    axis.scatter(x, y, s=70, color=COLOUR, zorder=3)
    for row, px, py in zip(rows, x, y):
        axis.annotate(f"{row['country']} ({float(row['hours']):.0f} h)", (px, py), textcoords="offset points",
                      xytext=(7, 5), fontsize=10)
    unit = " per hour of footage" if arguments.per_hour else ""
    axis.set_xlabel(f"Jaywalking crossings{unit} (no zebra crossing or traffic light)")
    axis.set_ylabel(f"Total crossings{unit}")
    axis.set_title("Total crossings against jaywalking crossings, by country")
    axis.set_xlim(left=-0.03 * max(x + [1e-9]))
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=160)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
