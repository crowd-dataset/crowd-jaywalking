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
from matplotlib.text import Text  # noqa: E402

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


# Where a label may go around its point: direction (x, y sign), alignment, and the matching Plotly text position;
# each direction is tried close to the point first, then further out with a leader line.
DIRECTIONS = [((1, 1), "left", "bottom", "top right"), ((1, -1), "left", "top", "bottom right"),
              ((-1, 1), "right", "bottom", "top left"), ((-1, -1), "right", "top", "bottom left"),
              ((1, 0), "left", "center", "middle right"), ((-1, 0), "right", "center", "middle left")]
DISTANCES = (6, 16, 28, 42, 60, 80)


def place_labels(figure, axis, labels, x, y, marker_x, marker_y, fontsize=10) -> list[str]:
    """Put each label at the nearest place around its point that overlaps no point, no label and no leader line.

    Labels are placed from the top of the plot down; when every place overlaps, the one with the least
    overlap is used. Returns the Plotly text position of each label.
    """

    figure.canvas.draw()  # settle the layout first, then keep it, so the pixel boxes below stay where they are
    figure.set_layout_engine("none")
    renderer = figure.canvas.get_renderer()
    to_pixels = axis.transData.transform
    radius = 4 * figure.dpi / 72
    markers = [to_pixels((mx, my)) for mx, my in zip(marker_x, marker_y)]
    inside = axis.get_window_extent(renderer)
    taken, leaders, positions = [], [], [""] * len(labels)

    def crosses(a, b, c, d) -> bool:
        def turn(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return turn(a, b, c) * turn(a, b, d) < 0 and turn(c, d, a) * turn(c, d, b) < 0

    def cost(box) -> float:
        area = sum(max(0, min(box.x1, t.x1) - max(box.x0, t.x0)) * max(0, min(box.y1, t.y1) - max(box.y0, t.y0))
                   for t in taken)
        hits = sum(box.x0 - radius < mx < box.x1 + radius and box.y0 - radius < my < box.y1 + radius
                   for mx, my in markers)
        outside = box.x1 > inside.x1 or box.x0 < inside.x0 or box.y0 < inside.y0 or box.y1 > inside.y1
        return 20 * area + 1e4 * hits + (1e6 if outside else 0)

    for i in sorted(range(len(labels)), key=lambda k: -y[k]):
        best = None
        for distance in DISTANCES:
            for (dx, dy), ha, va, position in DIRECTIONS:
                offset = (dx * distance, dy * distance * 0.8)
                arrow = {"arrowstyle": "-", "color": "0.55", "lw": 0.7, "shrinkA": 1, "shrinkB": 5} \
                    if distance > DISTANCES[0] else None
                text = axis.annotate(labels[i], (x[i], y[i]), textcoords="offset points", xytext=offset,
                                     fontsize=fontsize, ha=ha, va=va, arrowprops=arrow)
                text.update_positions(renderer)
                box = Text.get_window_extent(text, renderer)  # the text alone, without its leader line
                score = cost(box) + distance ** 2 / 10  # nearer places win; far ones only to avoid overlap
                start = to_pixels((x[i], y[i]))
                end = (min(max(start[0], box.x0), box.x1), min(max(start[1], box.y0), box.y1))
                if arrow:  # a leader line must not pass through another point, label or leader line
                    score += 5e3 * sum(crosses(start, end, a, b) for a, b in leaders)
                    for step in range(1, 10):
                        px = start[0] + (end[0] - start[0]) * step / 10
                        py = start[1] + (end[1] - start[1]) * step / 10
                        score += 5e3 * sum(t.x0 < px < t.x1 and t.y0 < py < t.y1 for t in taken)
                        score += 5e3 * sum(abs(px - mx) < radius and abs(py - my) < radius for mx, my in markers
                                           if abs(mx - start[0]) > 1 or abs(my - start[1]) > 1)
                if best is None or score < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (score, text, box, position, (start, end) if arrow else None)
                else:
                    text.remove()
            if best[0] <= distance ** 2 / 10:  # a place without overlap at this distance
                break
        taken.append(best[2])
        if best[4] is not None:
            leaders.append(best[4])
        positions[i] = best[3]
    return positions


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

    figure, axis = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    for c, (ox, oy, _) in other.items():
        axis.annotate("", xy=main_points[c][:2], xytext=(ox, oy), zorder=2,
                      arrowprops={"arrowstyle": "->", "color": COMPARE_COLOUR, "lw": 1, "shrinkA": 4, "shrinkB": 5})
    if other:
        axis.scatter([v[0] for v in other.values()], [v[1] for v in other.values()], s=55, facecolors="white",
                     edgecolors=COMPARE_COLOUR, linewidths=1.5, zorder=3, label=arguments.compare_label)
    axis.scatter(x, y, s=70, color=COLOUR, zorder=4, label=arguments.label if other else None)
    axis.set_xlabel(x_title)
    axis.set_ylabel(y_title)
    axis.set_title(TITLE)
    axis.set_xlim(left=-0.03 * max(every_x + [1e-9]), right=1.2 * max(every_x + [1e-9]))
    axis.set_ylim(bottom=0)
    positions = place_labels(figure, axis, labels, x, y, every_x, y + [v[1] for v in other.values()])
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
            mode="markers+text", textposition=positions,
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
