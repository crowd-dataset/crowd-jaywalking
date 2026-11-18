"""Scatter plot of total crossings against jaywalking crossings, one point per country.

X axis: crossings without a zebra crossing, traffic light or crossing sign (stage 2 claims, the label
JAYWALKING). Y axis: all crossings detected (stage 1). Results folders of several runs can be combined, or a
country table (crossings_by_country.csv) plotted instead; --html also writes an interactive version.

    uv run python scripts/crowd/plot_crossing_scatter.py results/six_countries results/korea_vietnam \
        --output results/crossings_scatter.png
    uv run --with plotly python scripts/crowd/plot_crossing_scatter.py --table docs/crossings_by_country.csv \
        --output docs/images/crossings_scatter.png --html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.crowd.plot_country_crossings import read_table, summarise  # noqa: E402

COLOUR = "#4C78A8"
TITLE = "Total crossings against jaywalking crossings, by country"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="results folders of CROWD runs")
    parser.add_argument("--table", type=Path, help="plot this crossings_by_country.csv instead of results folders")
    parser.add_argument("--output", type=Path, default=Path("crossings_scatter.png"))
    parser.add_argument("--per-hour", action="store_true", help="divide both axes by the hours of footage")
    parser.add_argument("--html", action="store_true", help="also write an interactive .html next to the PNG")
    arguments = parser.parse_args()
    if bool(arguments.runs) == bool(arguments.table):
        parser.error("give either results folders or --table")

    rows = read_table(arguments.table) if arguments.table else [row for run in arguments.runs for row in summarise(run)]
    scale = (lambda row: 1.0 / float(row["hours"])) if arguments.per_hour else (lambda row: 1.0)
    x = [float(row["unmarked"]) * scale(row) for row in rows]
    y = [float(row["detected"]) * scale(row) for row in rows]
    labels = [f"{row['country']} ({float(row['hours']):.0f} h)" for row in rows]
    unit = " per hour of footage" if arguments.per_hour else ""
    x_title = f"Jaywalking crossings{unit} (no zebra crossing or traffic light)"
    y_title = f"Total crossings{unit}"

    figure, axis = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    axis.scatter(x, y, s=70, color=COLOUR, zorder=3)
    for label, px, py in zip(labels, x, y):
        axis.annotate(label, (px, py), textcoords="offset points", xytext=(7, 5), fontsize=10)
    axis.set_xlabel(x_title)
    axis.set_ylabel(y_title)
    axis.set_title(TITLE)
    axis.set_xlim(left=-0.03 * max(x + [1e-9]), right=1.2 * max(x + [1e-9]))
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=160)
    print(f"wrote {arguments.output}")

    if arguments.html:
        import plotly.graph_objects as go

        digits = 2 if arguments.per_hour else 0
        page = go.Figure(go.Scatter(
            x=[round(v, digits) for v in x], y=[round(v, digits) for v in y], text=labels,
            mode="markers+text", textposition="top right", marker=dict(size=12, color=COLOUR),
            hovertemplate="%{text}<br>jaywalking %{x}<br>total %{y}<extra></extra>",
        ))
        page.update_layout(title=TITLE, xaxis_title=x_title, yaxis_title=y_title, template="simple_white",
                           height=620, xaxis=dict(rangemode="tozero"), yaxis=dict(rangemode="tozero"))
        html = arguments.output.with_suffix(".html")
        page.write_html(html, include_plotlyjs=True)
        print(f"wrote {html}")


if __name__ == "__main__":
    main()
