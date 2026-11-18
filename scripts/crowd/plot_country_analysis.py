"""Figures of the country analysis: jaywalking rates with 95% intervals, and how people move when they cross.

Reads two tables made from the run summaries (one row per country, and one per country and group):

* jaywalking_rates.csv: crossings without a zebra crossing, traffic light or crossing sign per hour of footage,
  with a 95% bootstrap interval over segments, and what the detected crossings were (at a marking, without one,
  rejected by the VLM as not crossing).
* crossing_movement.csv: for the people who crossed without a marking and those who crossed at one, the
  sideways speed, the share who stopped mid-crossing, the share who crossed together with another person, and
  the vehicles in view, each with a 95% interval. Only countries with `compared` True (at least 20 crossings
  without a marking) are drawn.

    uv run --with plotly python scripts/crowd/plot_country_analysis.py docs/jaywalking_rates.csv \
        docs/crossing_movement.csv --output-dir docs/images --html

Writes jaywalking_rate.png and crossing_movement.png (and .html with Plotly embedded when --html is given).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MARKED = "#72B7B2"
UNMARKED = "#E45756"
REJECTED = "#BAB0AC"
SHARES = (
    ("at_zebra_or_light", "At a zebra crossing or light", MARKED),
    ("crossings_without_zebra_or_light", "Without a marking (jaywalking)", UNMARKED),
    ("rejected_by_vlm", "Rejected by the VLM (not crossing)", REJECTED),
)
MEASURES = (
    ("lateral_speed_mean", "lateral_speed_low", "lateral_speed_high", "Sideways speed (body heights/s)"),
    ("stopped_pct", "stopped_low_pct", "stopped_high_pct", "Stopped mid-crossing (%)"),
    ("with_others_pct", "with_others_low_pct", "with_others_high_pct", "Crossed with others (%)"),
    ("vehicles_mean", "vehicles_low", "vehicles_high", "Vehicles in view (mean)"),
)
GROUPS = (("without_marking", "Without a marking", UNMARKED), ("at_marking", "At a marking", MARKED))


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_rates(rows: list[dict[str, str]], output: Path, html: bool) -> None:
    rows = sorted(rows, key=lambda r: float(r["jaywalking_per_hour"]))
    names = [r["country"] for r in rows]
    rate = [float(r["jaywalking_per_hour"]) for r in rows]
    low = [float(r["per_hour_low"]) for r in rows]
    high = [float(r["per_hour_high"]) for r in rows]
    shares = {key: [100 * float(r[key]) / float(r["crossings_detected"]) for r in rows] for key, _, _ in SHARES}

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True,
                                         gridspec_kw={"width_ratios": [1, 1.2]})
    y = range(len(rows))
    left.errorbar(rate, y, xerr=[[r - lo for r, lo in zip(rate, low)], [hi - r for r, hi in zip(rate, high)]],
                  fmt="o", color=UNMARKED, ms=7, capsize=3, lw=1.5)
    for yi, r, hi in zip(y, rate, high):
        left.annotate(f"{r:.2f}", (hi, yi), textcoords="offset points", xytext=(6, -3), fontsize=9)
    left.set_xscale("symlog", linthresh=0.1)
    left.set_xlim(0, 30)
    left.set_xticks([0, 0.1, 0.5, 1, 5, 10])
    left.set_xticklabels(["0", "0.1", "0.5", "1", "5", "10"])
    left.set_yticks(list(y), names)
    left.set_xlabel("Crossings without a marking per hour of footage (log scale above 0.1)")
    left.set_title("Jaywalking rate, with 95% interval", loc="left")
    left.grid(axis="x", alpha=0.3)

    start = [0.0] * len(rows)
    for key, label, colour in SHARES:
        right.barh(list(y), shares[key], left=start, color=colour, label=label, height=0.7, edgecolor="white",
                   linewidth=1.5)
        for yi, s0, v in zip(y, start, shares[key]):
            if v >= 8:
                right.text(s0 + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8.5)
        start = [s + v for s, v in zip(start, shares[key])]
    right.set_yticks(list(y), names)
    right.set_xlim(0, 100)
    right.set_xlabel("Share of the crossings stage 1 detected (%)")
    right.set_title("What the detected crossings were", loc="left")
    right.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"wrote {output}")

    if html:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        page = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.12,
                             subplot_titles=("Jaywalking rate, with 95% interval", "What the detected crossings were"))
        page.add_trace(go.Scatter(
            x=rate, y=names, mode="markers", marker=dict(size=11, color=UNMARKED), showlegend=False,
            error_x=dict(type="data", symmetric=False, array=[hi - r for r, hi in zip(rate, high)],
                         arrayminus=[r - lo for r, lo in zip(rate, low)], thickness=1.5),
            customdata=list(zip(low, high, [r["hours"] for r in rows])),
            hovertemplate="%{y}: %{x:.2f} per hour<br>95% interval %{customdata[0]:.2f} to %{customdata[1]:.2f}"
                          "<br>%{customdata[2]} hours of footage<extra></extra>",
        ), row=1, col=1)
        for key, label, colour in SHARES:
            counts = [int(r[key]) for r in rows]
            page.add_trace(go.Bar(
                x=[round(v, 1) for v in shares[key]], y=names, orientation="h", name=label, marker_color=colour,
                marker_line=dict(color="white", width=1.5), customdata=counts,
                hovertemplate="%{y}: %{x}% (%{customdata} crossings)<extra>" + label + "</extra>",
            ), row=1, col=2)
        ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
        page.update_xaxes(type="log", tickvals=ticks, ticktext=[f"{t:g}" for t in ticks],
                          title_text="Crossings without a marking per hour (log scale)", row=1, col=1)
        page.update_xaxes(range=[0, 100], title_text="Share of the crossings stage 1 detected (%)", row=1, col=2)
        page.update_layout(barmode="stack", template="simple_white", height=520,
                           legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", traceorder="normal"))
        target = output.with_suffix(".html")
        page.write_html(target, include_plotlyjs=True)
        print(f"wrote {target}")


def plot_movement(rows: list[dict[str, str]], output: Path, html: bool) -> None:
    rows = [r for r in rows if r["compared"] == "True"]
    order = list(dict.fromkeys(r["country"] for r in rows))
    by_key = {(r["country"], r["group"]): r for r in rows}
    y = list(range(len(order)))[::-1]

    figure, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True, constrained_layout=True)
    for axis, (mid, low, high, title) in zip(axes, MEASURES):
        for (group, label, colour), shift in zip(GROUPS, (0.15, -0.15)):
            values = [by_key[(c, group)] for c in order]
            m = [float(v[mid]) for v in values]
            lo = [float(v[low]) for v in values]
            hi = [float(v[high]) for v in values]
            axis.errorbar(m, [yi + shift for yi in y], xerr=[[a - b for a, b in zip(m, lo)],
                                                               [b - a for a, b in zip(m, hi)]],
                          fmt="o", ms=6, capsize=2, lw=1.2, color=colour, label=label)
        axis.set_title(title, fontsize=10)
        axis.set_yticks(y, order)
        axis.grid(axis="x", alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="outside lower center", ncol=2)
    figure.suptitle("How people cross without a marking and at one, per country (95% intervals)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"wrote {output}")

    if html:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        page = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.03,
                             subplot_titles=[title for *_, title in MEASURES])
        for column, (mid, low, high, title) in enumerate(MEASURES, start=1):
            for group, label, colour in GROUPS:
                values = [by_key[(c, group)] for c in order]
                m = [round(float(v[mid]), 2) for v in values]
                page.add_trace(go.Scatter(
                    x=m, y=order, mode="markers", name=label, legendgroup=group, showlegend=column == 1,
                    marker=dict(size=10, color=colour),
                    error_x=dict(type="data", symmetric=False, thickness=1.2,
                                 array=[float(v[high]) - float(v[mid]) for v in values],
                                 arrayminus=[float(v[mid]) - float(v[low]) for v in values]),
                    customdata=[(v["people"], float(v[low]), float(v[high])) for v in values],
                    hovertemplate="%{y}, " + label.lower() + ": %{x}<br>95% interval %{customdata[1]:.2f} to "
                                  "%{customdata[2]:.2f}<br>%{customdata[0]} people<extra>" + title + "</extra>",
                ), row=1, col=column)
        page.update_yaxes(autorange="reversed")
        page.update_layout(template="simple_white", height=480, scattermode="group",
                           title="How people cross without a marking and at one, per country (95% intervals)",
                           legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
        target = output.with_suffix(".html")
        page.write_html(target, include_plotlyjs=True)
        print(f"wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rates", type=Path, help="jaywalking_rates.csv")
    parser.add_argument("movement", type=Path, help="crossing_movement.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--html", action="store_true", help="also write interactive .html files")
    arguments = parser.parse_args()
    plot_rates(read(arguments.rates), arguments.output_dir / "jaywalking_rate.png", arguments.html)
    plot_movement(read(arguments.movement), arguments.output_dir / "crossing_movement.png", arguments.html)


if __name__ == "__main__":
    main()
