"""Bar plot of the crossings found per country in a CROWD run.

For each country: the crossings detected (stage 1, pedestrians crossing the road), those at a zebra
crossing or traffic light, and those that had no zebra crossing, traffic light or crossing sign (stage 2
claims, the label JAYWALKING). The country law (stage 3) is not used.

    uv run python scripts/crowd/plot_country_crossings.py results/six_countries

A country table written earlier (crossings_by_country.csv, for example one merged from several runs) can be
plotted instead of a results folder. --html also writes an interactive version with Plotly embedded:

    uv run --with plotly python scripts/crowd/plot_country_crossings.py --table docs/crossings_by_country.csv \
        --countries China Japan --output docs/images/six_countries_crossings.png --html
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
TITLE = "Pedestrians crossing the road, daytime dashcam footage"
SERIES = (
    ("detected", "Crossings detected", DETECTED_COLOUR),
    ("marked", "At a zebra crossing or traffic light", MARKED_COLOUR),
    ("unmarked", "Without zebra crossing or traffic light", UNMARKED_COLOUR),
)
TABLE_FIELDS = {
    "segments": "segments",
    "hours": "hours",
    "detected": "crossings_detected",
    "marked": "at_zebra_or_light",
    "unconfirmed": "rejected_by_vlm",
    "unmarked": "crossings_without_zebra_or_light",
}


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


def read_table(path: Path) -> list[dict[str, float | str]]:
    """Rows of a crossings_by_country.csv, in the form summarise() returns."""

    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {"country": row["country"], **{key: float(row[column]) for key, column in TABLE_FIELDS.items()}}
            for row in csv.DictReader(handle)
        ]


def choose(rows: list[dict[str, float | str]], countries: list[str] | None) -> list[dict[str, float | str]]:
    """The listed countries in the listed order; all countries when none are listed."""

    if not countries:
        return rows
    by_name = {row["country"]: row for row in rows}
    return [by_name[name] for name in countries if name in by_name]


def write_table(rows: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["country", *TABLE_FIELDS.values(), "jaywalking_per_hour"])
        for r in rows:
            hours = float(r["hours"])
            writer.writerow(
                [
                    r["country"],
                    int(r["segments"]),
                    f"{hours:.3f}",
                    int(r["detected"]),
                    int(r["marked"]),
                    int(r["unconfirmed"]),
                    int(r["unmarked"]),
                    f"{float(r['unmarked']) / hours:.2f}" if hours else "",
                ]
            )


def per_hour(row: dict[str, float | str]) -> float:
    return 1.0 / max(float(row["hours"]), 1e-9)


def plot(rows: list[dict[str, float | str]], output: Path) -> None:
    names = [str(r["country"]) for r in rows]
    figure, (counts, rates) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    width = 0.27
    positions = range(len(names))
    for axis, scale, title, label in (
        (counts, lambda r: 1.0, "Crossings in the sample", "Number of crossings"),
        (rates, per_hour, "Crossings per hour of footage", "Crossings per hour"),
    ):
        for (key, name, colour), shift in zip(SERIES, (-width, 0.0, width)):
            values = [float(r[key]) * scale(r) for r in rows]
            bars = axis.bar([p + shift for p in positions], values, width, label=name, color=colour)
            axis.bar_label(bars, fmt="%g" if scale(rows[0]) == 1.0 else "%.2f", padding=2, fontsize=8)
        axis.set_xticks(list(positions), [f"{n}\n{float(r['hours']):.1f} h" for n, r in zip(names, rows)])
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    counts.margins(y=0.25)
    counts.legend(frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0))
    figure.suptitle(TITLE)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(f"wrote {output}")


def plot_html(rows: list[dict[str, float | str]], output: Path) -> None:
    """The same bars with Plotly, embedded in the page so that it opens without a CDN."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = [f"{r['country']}<br>{float(r['hours']):.1f} h" for r in rows]
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Crossings in the sample", "Crossings per hour of footage"))
    for column, scale, digits in ((1, lambda r: 1.0, 0), (2, per_hour, 2)):
        for key, name, colour in SERIES:
            values = [round(float(r[key]) * scale(r), digits) for r in rows]
            figure.add_trace(
                go.Bar(x=names, y=values, name=name, marker_color=colour, legendgroup=key,
                       showlegend=column == 1, text=values, textposition="outside"),
                row=1, col=column,
            )
    figure.update_yaxes(title_text="Number of crossings", row=1, col=1)
    figure.update_yaxes(title_text="Crossings per hour", row=1, col=2)
    figure.update_layout(title=TITLE, barmode="group", template="simple_white", height=560,
                         legend=dict(orientation="h", y=-0.15))
    figure.write_html(output, include_plotlyjs=True)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?", help="results folder of a CROWD run")
    parser.add_argument("--table", type=Path, help="plot this crossings_by_country.csv instead of a results folder")
    parser.add_argument("--countries", nargs="*", help="countries to plot, in this order (default: all)")
    parser.add_argument("--output", type=Path, help="PNG to write (default: crossings_by_country.png in the run)")
    parser.add_argument("--html", action="store_true", help="also write an interactive .html next to the PNG")
    arguments = parser.parse_args()
    if (arguments.run is None) == (arguments.table is None):
        parser.error("give either a results folder or --table")
    if arguments.table and not arguments.output:
        parser.error("--table needs --output")

    if arguments.run:
        rows = choose(summarise(arguments.run), arguments.countries)
        write_table(rows, arguments.run / "crossings_by_country.csv")
    else:
        rows = choose(read_table(arguments.table), arguments.countries)
    output = arguments.output or arguments.run / "crossings_by_country.png"
    plot(rows, output)
    if arguments.html:
        plot_html(rows, output.with_suffix(".html"))


if __name__ == "__main__":
    main()
