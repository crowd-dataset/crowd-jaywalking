"""Pick a random daytime, car-mounted sample of CROWD segments for a list of countries.

Writes the CSV that ``crowd_selection`` reads (video_id, start_second, end_second, country).
The hours are split equally over the countries; segments are drawn at random (fixed seed)
until a country has its share of hours.

    uv run python scripts/crowd/select_country_sample.py results/six_countries_selection.csv \
        --hours 100 --countries China Japan Bangladesh France Sweden Denmark
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.crowd.crowd_source import load_crowd_mapping  # noqa: E402

DAY = "0"
CAR = 0


def select(mapping: Path, countries: list[str], total_hours: float, seed: int) -> list[dict[str, object]]:
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
        rng.shuffle(pool)
        hours = 0.0
        for segment in pool:
            if hours >= per_country:
                break
            hours += (segment.end_second - segment.start_second) / 3600
            rows.append(
                {
                    "video_id": segment.video_id,
                    "start_second": segment.start_second,
                    "end_second": segment.end_second,
                    "country": country,
                }
            )
        print(f"{country}: {sum(r['country'] == country for r in rows)} segments, {hours:.1f} h")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mapping", type=Path, default=Path("mapping.csv"))
    parser.add_argument("--countries", nargs="+", required=True)
    parser.add_argument("--hours", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    rows = select(arguments.mapping, arguments.countries, arguments.hours, arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "start_second", "end_second", "country"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} segments -> {arguments.output}")


if __name__ == "__main__":
    main()
