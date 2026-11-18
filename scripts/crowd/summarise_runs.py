"""Summarise finished CROWD runs where they are, for the country tables (reads only, standard library only).

Writes one small JSON with mergeable aggregates: per-segment counts (no people), the segments each run planned
and finished, per country and label histograms and sums of the movement measures of the people who crossed,
and, with --validation, the reviewer comparison. No boxes, frames or video leave the machine, so the script can
be copied to a GPU machine and run there:

    python3 -I summarise_runs.py summary_wsl5090.json --root results \
        --run six_countries_100h_16bit=six_capitals_selection.csv --run korea_vietnam_16bit=korea_vietnam_selection.csv

`scripts/crowd/country_tables.py` merges the JSONs of several machines into the country tables. A segment that
finished in more than one run is counted once (the first run given). Movement is measured in the image of a
moving camera, in body heights (the median box height of the person) per second, assuming a 16:9 frame.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ASPECT = 16 / 9
VEHICLES = {"2", "3", "5", "7"}  # car, motorcycle, bus, truck
STILL = 0.15      # body heights per second: standing
MOVING = 0.30     # body heights per second: walking
HOLD = 0.5        # seconds a stop or a reversal must last
LABELS = ("JAYWALKING", "COMPLIANT")

BINS = {
    "lateral_speed": (0.0, 3.0, 0.05),
    "peak_speed": (0.0, 6.0, 0.1),
    "speed_cv": (0.0, 3.0, 0.05),
    "visible_seconds": (0.0, 30.0, 0.5),
    "vehicles": (0.0, 30.0, 0.5),
    "people": (0.0, 40.0, 0.5),
}
FLAGS = ("stopped", "reversed", "in_group")


def segment_key(video_id, start_second, end_second):
    """The video_key of a segment, as CrowdSegment.video_key in crowd_source.py makes it."""

    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", video_id).strip("_") or "video"
    return f"crowd_{safe_id}_s{int(start_second):06d}_e{int(end_second):06d}"


def metadata(segment, name):
    lowered = {str(k).lower(): v for k, v in segment.get("metadata", {}).items()}
    return str(lowered.get(name, ""))


def runs_of(values, threshold_test, min_len):
    """Start and end indices of runs where threshold_test holds for at least min_len items."""
    out, start = [], None
    for i, v in enumerate(values + [None]):
        ok = v is not None and threshold_test(v)
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def movement(boxes, fps):
    """Movement measures of one track: boxes is a frame-sorted list of (frame, x, h)."""
    if len(boxes) < max(5, int(fps / 2)):
        return None
    heights = sorted(b[2] for b in boxes)
    body = heights[len(heights) // 2]
    if body <= 0:
        return None
    k = max(1, round(0.2 * fps))
    velocity = []
    for i in range(len(boxes)):
        a, b = boxes[max(0, i - k)], boxes[min(len(boxes) - 1, i + k)]
        if b[0] == a[0]:
            continue
        velocity.append((b[1] - a[1]) * ASPECT / body / ((b[0] - a[0]) / fps))
    if len(velocity) < 3:
        return None
    speed = sorted(abs(v) for v in velocity)
    mean = sum(speed) / len(speed)
    sd = math.sqrt(sum((s - mean) ** 2 for s in speed) / len(speed))
    hold = max(2, round(HOLD * fps))
    moving_idx = [i for i, v in enumerate(velocity) if abs(v) > MOVING]
    stopped = False
    reversed_ = False
    if moving_idx:
        first, last = moving_idx[0], moving_idx[-1]
        inner = [abs(v) for v in velocity[first:last + 1]]
        stopped = bool(runs_of(inner, lambda s: s < STILL, hold))
        direction = 1 if (boxes[-1][1] - boxes[0][1]) >= 0 else -1
        reversed_ = bool(runs_of(velocity, lambda v: v * direction < -MOVING, hold))
    return {
        "lateral_speed": speed[len(speed) // 2],
        "peak_speed": speed[int(0.9 * (len(speed) - 1))],
        "speed_cv": sd / mean if mean > 0 else 0.0,
        "visible_seconds": (boxes[-1][0] - boxes[0][0]) / fps,
        "stopped": stopped,
        "reversed": reversed_,
    }


def new_hist():
    return {name: {"n": 0, "sum": 0.0, "sumsq": 0.0,
                   "counts": [0] * int(round((hi - lo) / step) + 1)}
            for name, (lo, hi, step) in BINS.items()}


def add(hist, name, value):
    lo, hi, step = BINS[name]
    h = hist[name]
    h["n"] += 1
    h["sum"] += value
    h["sumsq"] += value * value
    h["counts"][min(len(h["counts"]) - 1, max(0, int((value - lo) / step)))] += 1


def segment_summary(payload):
    """Counts of one segment, and the people whose movement is to be measured."""
    result = payload["result"]
    segment = payload["segment"]
    trim = payload.get("trim", {})
    hours = (float(trim.get("effective_end_second", segment["end_second"])) - float(segment["start_second"])) / 3600
    decisions = {d["person_id"]: d for d in result["person_decisions"]}
    row = {
        "video_key": payload["video_key"], "country": metadata(segment, "country"),
        "locality": metadata(segment, "locality"), "hours": round(hours, 5),
        "tracked": len(result.get("crossing_classifications", [])), "detected": 0, "rejected": 0,
        "compliant": 0, "jaywalking": 0, "uncertain": 0, "zebra": 0, "light": 0, "sign": 0,
    }
    crossers = []
    for item in result.get("crossing_classifications", []):
        if not item["predicted_crossing"]:
            continue
        row["detected"] += 1
        decision = decisions.get(item["person_id"])
        label = decision["label"] if decision else "NOT_EVALUATED"
        key = {"NOT_EVALUATED": "rejected", "COMPLIANT": "compliant", "JAYWALKING": "jaywalking",
               "UNCERTAIN": "uncertain"}.get(label)
        if key:
            row[key] += 1
        if label == "COMPLIANT":
            context = decision.get("context", {})
            row["zebra"] += context.get("marked_crosswalk") == "YES"
            row["light"] += (context.get("traffic_light") == "YES"
                             or context.get("permissive_pedestrian_signal") == "YES")
            row["sign"] += context.get("authorised_crossing_sign") == "YES"
        if decision and label in ("JAYWALKING", "COMPLIANT"):
            crossers.append((label, decision["event"]))
    return row, crossers


def read_tracks(path, wanted, windows):
    """Boxes of the wanted people and per-frame counts of road users inside the windows."""
    lo = min([w[0] for w in windows] + [s for s, _ in wanted.values()])
    hi = max([w[1] for w in windows] + [e for _, e in wanted.values()])
    boxes = defaultdict(list)
    vehicles = defaultdict(int)
    people = defaultdict(int)
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame = int(float(row["frame-count"])) - 1
            except (TypeError, ValueError):
                continue
            if frame < lo or frame > hi:
                continue
            cls = row["yolo-id"]
            if cls == "0":
                people[frame] += 1
                try:
                    pid = int(float(row["unique-id"]))
                except (TypeError, ValueError):
                    pid = -1
                if pid in wanted and wanted[pid][0] <= frame <= wanted[pid][1]:
                    boxes[pid].append((frame, float(row["x-center"]), float(row["height"])))
            elif cls in VEHICLES:
                vehicles[frame] += 1
    return boxes, vehicles, people


def window_mean(counts, a, b):
    return sum(counts.get(f, 0) for f in range(a, b + 1)) / (b - a + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="JSON to write")
    parser.add_argument("--root", type=Path, required=True, help="folder with the results folders and selections")
    parser.add_argument("--run", action="append", default=[], help="NAME=SELECTION.csv, in order of preference")
    parser.add_argument("--validation", type=Path, help="reviewer comparison CSV")
    args = parser.parse_args()

    segments, planned, failed = [], defaultdict(int), defaultdict(int)
    planned_keys = defaultdict(set)
    hist = defaultdict(new_hist)
    flags = defaultdict(lambda: {f: 0 for f in FLAGS} | {"n": 0})
    skipped = 0
    done = set()
    for spec in args.run:
        name, selection = spec.split("=", 1)
        run = args.root / name
        with open(args.root / selection, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                planned[row.get("country", "?")] += 1
                planned_keys[row.get("country", "?")].add(
                    segment_key(row["video_id"], row["start_second"], row["end_second"]))
        if (run / "errors.csv").exists():
            with open(run / "errors.csv", encoding="utf-8", newline="") as handle:
                failed_keys = {r["video_key"] for r in csv.DictReader(handle)}
        else:
            failed_keys = set()
        run_done = set()
        for details in sorted((run / "details").glob("*.json")):
            payload = json.loads(details.read_text(encoding="utf-8"))
            if "segment" not in payload:
                continue
            run_done.add(payload["video_key"])
            if payload["video_key"] in done:  # finished in an earlier run too: count it once
                continue
            done.add(payload["video_key"])
            row, crossers = segment_summary(payload)
            segments.append(row)
            if not crossers:
                continue
            fps = float(payload["fps"])
            track_file = payload.get("precomputed_tracks")
            if not track_file or not Path(track_file).exists():
                skipped += len(crossers)
                continue
            margin = round(fps)
            wanted = {int(e["person_id"]): (int(e["start_frame"]), int(e["end_frame"])) for _, e in crossers}
            windows = [(int(e["transition_start_frame"]) - margin, int(e["transition_end_frame"]) + margin)
                       for _, e in crossers]
            boxes, vehicles, people = read_tracks(track_file, wanted, windows)
            for index, ((label, event), (a, b)) in enumerate(zip(crossers, windows)):
                pid = int(event["person_id"])
                measures = movement(sorted(boxes.get(pid, [])), fps)
                if measures is None:
                    skipped += 1
                    continue
                measures["vehicles"] = window_mean(vehicles, a, b)
                measures["people"] = max(0.0, window_mean(people, a, b) - 1)
                measures["in_group"] = any(
                    j != index and windows[j][0] <= b and a <= windows[j][1] for j in range(len(windows)))
                key = f"{row['country']}|{label}"
                for metric in BINS:
                    add(hist[key], metric, measures[metric])
                flags[key]["n"] += 1
                for flag in FLAGS:
                    flags[key][flag] += bool(measures[flag])
        failed[name] += len(failed_keys - run_done)

    out = {"segments": segments, "planned": planned, "failed_by_run": failed, "skipped_people": skipped,
           "planned_keys": {country: sorted(keys) for country, keys in planned_keys.items()},
           "bins": BINS, "movement": hist, "flags": flags}
    if args.validation:
        with open(args.validation, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        out["validation"] = [{k: r[k] for k in ("country", "reviewer", "stored", "control", "whole_track")}
                             for r in rows]
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {args.out}: {len(segments)} segments, {sum(planned.values())} planned")


if __name__ == "__main__":
    main()
