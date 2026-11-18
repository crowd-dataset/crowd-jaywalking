"""Tests for merging the run summaries of several machines into the country tables."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.crowd.country_tables import merge, summarise, wilson, write_tables
from scripts.crowd.summarise_runs import BINS, new_hist, segment_key


def _segment(key: str, country: str, jaywalking: int, compliant: int, hours: float = 0.5) -> dict:
    return {"video_key": key, "country": country, "locality": "", "hours": hours, "tracked": 0,
            "detected": jaywalking + compliant + 1, "rejected": 1, "compliant": compliant, "jaywalking": jaywalking,
            "uncertain": 0, "zebra": 0, "light": 0, "sign": 0}


def _summary(segments: list[dict], planned: dict[str, list[str]], people: int) -> dict:
    hist = new_hist()
    for metric in hist:
        hist[metric]["n"] = people
        hist[metric]["sum"] = 0.8 * people
        hist[metric]["sumsq"] = 0.64 * people
    return {"segments": segments, "planned_keys": planned, "bins": BINS,
            "movement": {"Sweden|JAYWALKING": hist},
            "flags": {"Sweden|JAYWALKING": {"stopped": 1, "reversed": 0, "in_group": 2, "n": people}}}


class CountryTablesTests(unittest.TestCase):
    def test_segment_key_matches_crowd_source(self) -> None:
        self.assertEqual(segment_key("a b/c", 5, 305), "crowd_a_b_c_s000005_e000305")

    def test_merge_counts_each_segment_once_and_planned_as_a_union(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.json", Path(folder) / "b.json"
            # the second machine redid s1, which the first machine had planned and finished too
            first.write_text(json.dumps(_summary(
                [_segment("s1", "Sweden", 1, 2), _segment("s2", "Sweden", 0, 3)],
                {"Sweden": ["s1", "s2", "s3"]}, people=10)))
            second.write_text(json.dumps(_summary(
                [_segment("s1", "Sweden", 1, 2), _segment("s4", "Sweden", 2, 0)],
                {"Sweden": ["s1", "s4"]}, people=30)))
            data = merge([first, second])
            summary = summarise(data)

            self.assertEqual(summary["Sweden"]["segments"], 3)
            self.assertEqual(summary["Sweden"]["planned"], 4)  # s3 failed
            self.assertEqual(summary["Sweden"]["jaywalking"], 3)
            self.assertEqual(data["flags"]["Sweden|JAYWALKING"]["n"], 40)
            self.assertEqual(data["movement"]["Sweden|JAYWALKING"]["lateral_speed"]["n"], 40)

            write_tables(data, summary, Path(folder))
            with (Path(folder) / "crossing_movement.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["compared"], "True")
            self.assertEqual(rows[0]["people"], "40")
            with (Path(folder) / "crossings_by_country.csv").open(encoding="utf-8") as handle:
                self.assertEqual(next(csv.DictReader(handle))["jaywalking_per_hour"], "2.00")

    def test_summary_without_planned_keys_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / "old.json"
            old.write_text(json.dumps({"segments": [], "planned": {}}))
            with self.assertRaises(SystemExit):
                merge([old])

    def test_wilson_interval_contains_the_share(self) -> None:
        share, low, high = wilson(3, 10)
        self.assertAlmostEqual(share, 0.3)
        self.assertLess(low, share)
        self.assertGreater(high, share)


if __name__ == "__main__":
    unittest.main()
