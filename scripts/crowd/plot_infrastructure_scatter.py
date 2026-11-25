"""Scatter plot of street scenes without crossing infrastructure against crossings without a marking, per country.

X axis: crossings without a zebra crossing, traffic light or crossing sign (stage 2 claims). Y axis: the street
scenes (segments) in which someone crossed and no crossing was judged at a zebra crossing, traffic light or
crossing sign. Both come from the same marking check, so part of their relation is built in. Reads the
infrastructure_by_country.csv written by country_tables.py; --exclude leaves countries out (for a zoomed plot
without an outlier), --label-only names only the given countries in the PNG (the others crowd near the origin
and are named in the zoom; the interactive version names them all on hover); --html also writes an
interactive version.

    uv run --with plotly python scripts/crowd/plot_infrastructure_scatter.py docs/infrastructure_by_country.csv \
        --label-only Bangladesh India Indonesia "South Africa" Vietnam Türkiye \
        --output docs/images/infrastructure_scatter.png --html
    uv run --with plotly python scripts/crowd/plot_infrastructure_scatter.py docs/infrastructure_by_country.csv \
        --exclude Bangladesh --output docs/images/infrastructure_scatter_zoom.png --html
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.crowd.plot_crossing_scatter import COLOUR, place_labels  # noqa: E402

TITLE = "Scenes without crossing infrastructure against unmarked crossings, by country"
X_TITLE = "Crossings without a zebra crossing or traffic light (total)"
Y_TITLE = "Street scenes with no crossing infrastructure (total)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", type=Path, help="infrastructure_by_country.csv from country_tables.py")
    parser.add_argument("--exclude", nargs="*", default=[], help="countries to leave out")
    parser.add_argument("--label-only", nargs="*", help="name only these countries in the PNG")
    parser.add_argument("--output", type=Path, default=Path("infrastructure_scatter.png"))
    parser.add_argument("--html", action="store_true", help="also write an interactive .html next to the PNG")
    arguments = parser.parse_args()

    with arguments.table.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["country"] not in arguments.exclude]
    names = [row["country"] for row in rows]
    x = [float(row["crossings_without_zebra_or_light"]) for row in rows]
    y = [float(row["scenes_without_infrastructure"]) for row in rows]
    judged = [int(row["scenes_judged"]) for row in rows]
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    left_out = f"; {', '.join(arguments.exclude)} left out" if arguments.exclude else ""
    note = (f"{len(names)} countries{left_out}\nPearson r = {r:.2f} (p = {p:.1g}), "
            f"Spearman ρ = {rho:.2f} (p = {p_rho:.1g})")
    print(note.replace("\n", ": "))
    slope, intercept = np.polyfit(x, y, 1)
    line_x = [0.0, 1.1 * max(x)]

    figure, axis = plt.subplots(figsize=(8.5, 7), constrained_layout=True)
    axis.plot(line_x, [slope * v + intercept for v in line_x], color="0.65", lw=1.5, zorder=1)
    axis.scatter(x, y, s=70, color=COLOUR, edgecolor="white", linewidth=1.2, zorder=4)
    axis.set_xlabel(X_TITLE)
    axis.set_ylabel(Y_TITLE)
    axis.set_title(TITLE)
    axis.set_xlim(-0.03 * max(x), 1.15 * max(x))
    axis.set_ylim(-0.03 * max(y), 1.15 * max(y))
    shown = [i for i, c in enumerate(names) if arguments.label_only is None or c in arguments.label_only]
    placed = place_labels(figure, axis, [names[i] for i in shown], [x[i] for i in shown], [y[i] for i in shown], x, y)
    positions = ["top right"] * len(names)
    for i, position in zip(shown, placed):
        positions[i] = position
    if arguments.label_only is not None:
        note += "\nThe other countries are named in the zoomed plot"
    axis.text(0.02, 0.97, note, transform=axis.transAxes, va="top", fontsize=10)
    axis.grid(alpha=0.3, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=160)
    print(f"wrote {arguments.output}")

    if arguments.html:
        import plotly.graph_objects as go

        page = go.Figure()
        page.add_trace(go.Scatter(x=line_x, y=[slope * v + intercept for v in line_x], mode="lines",
                                  line={"color": "#A6A6A6", "width": 1.5}, hoverinfo="skip", showlegend=False))
        page.add_trace(go.Scatter(
            x=x, y=y, text=[c if i in shown else "" for i, c in enumerate(names)], hovertext=names, customdata=judged,
            mode="markers+text", textposition=positions,
            marker={"size": 12, "color": COLOUR}, showlegend=False,
            hovertemplate="%{hovertext}<br>unmarked crossings %{x}<br>scenes without infrastructure %{y} "
                          "of %{customdata} judged<extra></extra>",
        ))
        page.update_layout(title=TITLE, xaxis_title=X_TITLE, yaxis_title=Y_TITLE, template="simple_white", height=620,
                           xaxis={"range": [-0.03 * max(x), 1.15 * max(x)]},
                           yaxis={"range": [-0.03 * max(y), 1.15 * max(y)]},
                           annotations=[{"text": note.replace("\n", "<br>"), "xref": "paper", "yref": "paper",
                                         "x": 0.02, "y": 0.98, "showarrow": False, "align": "left"}])
        html = arguments.output.with_suffix(".html")
        page.write_html(html, include_plotlyjs=True)
        print(f"wrote {html}")


if __name__ == "__main__":
    main()
