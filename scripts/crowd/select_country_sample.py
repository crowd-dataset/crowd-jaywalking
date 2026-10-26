"""Pick a random daytime, car-mounted sample of CROWD segments for a list of countries.

Writes the CSV that ``crowd_selection`` reads (video_id, start_second, end_second, country).
The hours are split equally over the countries; segments are drawn at random (fixed seed)
until a country has its share of hours.

With ``--capitals-first`` a country is filled from its capital first (random source videos, all of
their daytime car segments), and only when the capital has too few hours from the next biggest cities
by population, again in random source videos.

    uv run python scripts/crowd/select_country_sample.py results/six_countries_selection.csv \
        --hours 100 --countries China Japan Bangladesh France Sweden Denmark
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.crowd.crowd_source import load_crowd_mapping  # noqa: E402

DAY = "0"
CAR = 0
# The mapping's ``locality`` of each capital.
CAPITALS = {
    "China": "Beijing",
    "Japan": "Tokyo",
    "Bangladesh": "Dhaka",
    "France": "Paris",
    "Sweden": "Stockholm",
    "Denmark": "København",
    "South Korea": "Seoul",
    "Vietnam": "Hanoi",
}


def _hours(segment: object) -> float:
    return (segment.end_second - segment.start_second) / 3600  # type: ignore[attr-defined]


def _population(segment: object) -> float:
    try:
        return float(segment.metadata.get("population_locality") or 0)  # type: ignore[attr-defined]
    except ValueError:
        return 0.0


def _city_order(pool: list, country: str, capitals_first: bool) -> list[list]:
    """The segments of each city, capital first and then by population, or all together."""

    if not capitals_first:
        return [pool]
    cities: dict[str, list] = defaultdict(list)
    for segment in pool:
        cities[segment.metadata.get("locality", "")].append(segment)
    capital = CAPITALS.get(country)
    if capital not in cities:
        raise SystemExit(f"No daytime car segments for the capital of {country} ({capital})")
    others = sorted((name for name in cities if name != capital), key=lambda n: -_population(cities[n][0]))
    return [cities[capital]] + [cities[name] for name in others]


def select(
    mapping: Path, countries: list[str], total_hours: float, seed: int, capitals_first: bool = False
) -> list[dict[str, object]]:
    segments = load_crowd_mapping(mapping)
    per_country = total_hours / len(countries)
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for country in countries:
        pool = [
            s for s in segments
            if s.metadata.get("country") == country and s.time_of_day == DAY and s.vehicle_type == CAR
        ]
        if not pool:
            raise SystemExit(f"No daytime car segments for {country}")
        hours = 0.0
        per_city: list[str] = []
        for city_segments in _city_order(pool, country, capitals_first):
            if capitals_first:
                # Whole source videos in random order, so videos are chosen independently of each other.
                by_video: dict[str, list] = defaultdict(list)
                for segment in city_segments:
                    by_video[segment.video_id].append(segment)
                units = list(by_video.values())
                rng.shuffle(units)
            else:
                units = [[segment] for segment in city_segments]
                rng.shuffle(units)
            before = hours
            for unit in units:
                if hours >= per_country:
                    break
                for segment in unit:
                    hours += _hours(segment)
                    rows.append(
                        {
                            "video_id": segment.video_id,
                            "start_second": segment.start_second,
                            "end_second": segment.end_second,
                            "country": country,
                            "city": segment.metadata.get("locality", ""),
                        }
                    )
            if hours > before:
                per_city.append(f"{city_segments[0].metadata.get('locality', '')} {hours - before:.1f} h")
            if hours >= per_country:
                break
        count = sum(r["country"] == country for r in rows)
        print(f"{country}: {count} segments, {hours:.1f} h ({', '.join(per_city)})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mapping", type=Path, default=Path("mapping.csv"))
    parser.add_argument("--countries", nargs="+", required=True)
    parser.add_argument("--hours", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capitals-first", action="store_true", help="fill each country from its capital first")
    arguments = parser.parse_args()
    rows = select(arguments.mapping, arguments.countries, arguments.hours, arguments.seed, arguments.capitals_first)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "start_second", "end_second", "country", "city"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} segments -> {arguments.output}")


if __name__ == "__main__":
    main()
