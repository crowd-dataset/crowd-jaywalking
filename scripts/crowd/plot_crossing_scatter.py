"""Scatter plot of total crossings against jaywalking crossings, one point per country.

X axis: crossings without a zebra crossing, traffic light or crossing sign (stage 2 claims, the label
JAYWALKING). Y axis: all crossings detected (stage 1). Results folders of several runs can be combined, or a
country table (crossings_by_country.csv) plotted instead; --compare adds a second table (for example the 4-bit
run) as open circles with an arrow to each country's point; --html also writes an interactive version.

    uv run python scripts/crowd/plot_crossing_scatter.py results/six_countries results/korea_vietnam \
        --output results/crossings_scatter.png
    uv run --with plotly python scripts/crowd/plot_crossing_scatter.py --table docs/crossings_by_country.csv \
        --compare docs/crossings_by_country_4bit.csv --label "16-bit VLM" --compare-label "4-bit VLM" \
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
COMPARE_COLOUR = "#8C8C8C"
TITLE = "Total crossings against jaywalking crossings, by country"


def points(rows: list[dict], per_hour: bool) -> dict[str, tuple[float, float, float]]:
    """Per country: jaywalking (x), all crossings detected (y) and hours, optionally per hour of footage."""

    out = {}
    for row in rows:
        scale = 1.0 / float(row["hours"]) if per_hour else 1.0
        hours = float(row["hours"])
        out[str(row["country"])] = (float(row["unmarked"]) * scale, float(row["detected"]) * scale, hours)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="results folders of CROWD runs")
    parser.add_argument("--table", type=Path, help="plot this crossings_by_country.csv instead of results folders")
    parser.add_argument("--compare", type=Path, help="a second country table drawn as open circles, with an arrow "
                        "from each of its points to the same country's point (for example an earlier run)")
    parser.add_argument("--label", default="This run", help="legend name of the main points (with --compare)")
    parser.add_argument("--compare-label", default="Earlier run", help="legend name of the --compare points")
    parser.add_argument("--output", type=Path, default=Path("crossings_scatter.png"))
    parser.add_argument("--per-hour", action="store_true", help="divide both axes by the hours of footage")
    parser.add_argument("--html", action="store_true", help="also write an interactive .html next to the PNG")
    arguments = parser.parse_args()
    if bool(arguments.runs) == bool(arguments.table):
        parser.error("give either results folders or --table")

    if arguments.table:
        rows = read_table(arguments.table)
    else:
        rows = [row for run in arguments.runs for row in summarise(run)]
    main_points = points(rows, arguments.per_hour)
    other = points(read_table(arguments.compare), arguments.per_hour) if arguments.compare else {}
    other = {c: v for c, v in other.items() if c in main_points}
    names = list(main_points)
    x = [main_points[c][0] for c in names]
    y = [main_points[c][1] for c in names]
    labels = [f"{c} ({main_points[c][2]:.0f} h)" for c in names]
    unit = " per hour of footage" if arguments.per_hour else ""
    x_title = f"Jaywalking crossings{unit} (no zebra crossing or traffic light)"
    y_title = f"Total crossings{unit}"
    every_x = x + [v[0] for v in other.values()]
    # a label goes below its point when another point sits where the label would be (just above and to the right)
    span_x, span_y = 1.2 * max(every_x + [1e-9]), max(y + [1e-9])
    below = [any(0 < ox - px < 0.13 * span_x and 0 < oy - py < 0.03 * span_y for ox, oy in zip(x, y))
             for px, py in zip(x, y)]

    figure, axis = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    for c, (ox, oy, _) in other.items():
        axis.annotate("", xy=main_points[c][:2], xytext=(ox, oy), zorder=2,
                      arrowprops={"arrowstyle": "->", "color": COMPARE_COLOUR, "lw": 1, "shrinkA": 4, "shrinkB": 5})
    if other:
        axis.scatter([v[0] for v in other.values()], [v[1] for v in other.values()], s=55, facecolors="white",
                     edgecolors=COMPARE_COLOUR, linewidths=1.5, zorder=3, label=arguments.compare_label)
    axis.scatter(x, y, s=70, color=COLOUR, zorder=4, label=arguments.label if other else None)
    for label, px, py, low in zip(labels, x, y, below):
        axis.annotate(label, (px, py), textcoords="offset points", xytext=(7, -14) if low else (7, 5), fontsize=10)
    axis.set_xlabel(x_title)
    axis.set_ylabel(y_title)
    axis.set_title(TITLE)
    axis.set_xlim(left=-0.03 * max(every_x + [1e-9]), right=1.2 * max(every_x + [1e-9]))
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    if other:
        axis.legend(frameon=False, loc="lower right")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=160)
    print(f"wrote {arguments.output}")

    if arguments.html:
        import plotly.graph_objects as go

        digits = 2 if arguments.per_hour else 0
        page = go.Figure()
        if other:
            line_x, line_y = [], []
            for c, (ox, oy, _) in other.items():
                line_x += [ox, main_points[c][0], None]
                line_y += [oy, main_points[c][1], None]
            page.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", line={"color": COMPARE_COLOUR, "width": 1},
                                      hoverinfo="skip", showlegend=False))
            page.add_trace(go.Scatter(
                x=[round(v[0], digits) for v in other.values()], y=[round(v[1], digits) for v in other.values()],
                text=[f"{c} ({v[2]:.0f} h)" for c, v in other.items()], mode="markers", name=arguments.compare_label,
                marker={"size": 11, "color": "white", "line": {"color": COMPARE_COLOUR, "width": 2}},
                hovertemplate="%{text}<br>jaywalking %{x}<br>total %{y}<extra>" + arguments.compare_label + "</extra>",
            ))
        page.add_trace(go.Scatter(
            x=[round(v, digits) for v in x], y=[round(v, digits) for v in y], text=labels, name=arguments.label,
            mode="markers+text", textposition=["bottom right" if low else "top right" for low in below],
            marker={"size": 12, "color": COLOUR},
            showlegend=bool(other),
            hovertemplate="%{text}<br>jaywalking %{x}<br>total %{y}<extra>" + arguments.label + "</extra>",
        ))
        page.update_layout(title=TITLE, xaxis_title=x_title, yaxis_title=y_title, template="simple_white",
                           height=620, yaxis={"rangemode": "tozero"},
                           xaxis={"range": [-0.03 * max(every_x + [1e-9]), 1.25 * max(every_x + [1e-9])]},
                           legend={"x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"})
        html = arguments.output.with_suffix(".html")
        page.write_html(html, include_plotlyjs=True)
        print(f"wrote {html}")


if __name__ == "__main__":
    main()
