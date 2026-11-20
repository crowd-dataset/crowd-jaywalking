"""Pedestrians in view per analysed segment, the denominators for per-pedestrian crossing rates.

For every segment with a result (``<run>/details/*.json``) this reads the segment's CROWD track file
(``precomputed_tracks``) and counts, within the analysed part of the segment:

- ``pedestrians``: person tracks (YOLO class 0, confidence >= 0.7 as in stage 1) seen for at least one
  second, so short detection fragments do not count as people;
- ``pedestrian_hours``: person-frames at that confidence divided by the frame rate, i.e. hours of pedestrians
  in view (a person seen for 10 s adds 10 s); and
- ``candidates``: the people stage 1 judged as possible crossers (``crossing_classifications``), the people
  who came onto the road strip, as a count of crossing opportunities.

Standard library only, so it runs next to the results on any machine. One CSV row per segment; a segment
already counted by an earlier --run is skipped, as in summarise_runs.py.

    python3 -I scripts/crowd/count_pedestrians.py pedestrians.csv --root results --run india_a_16bit
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PERSON = "0"
MIN_CONFIDENCE = 0.7
FIELDS = ("video_key", "run", "country", "hours", "pedestrians", "pedestrian_hours", "candidates", "track_file")


def count(track_file: Path, last_frame: int, fps: float) -> tuple[int, float]:
    """Person tracks seen for at least a second, and person-hours, up to ``last_frame`` (0-based)."""
    frames: dict[str, int] = defaultdict(int)
    person_frames = 0
    with track_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["yolo-id"] != PERSON:
                continue
            try:
                frame = int(float(row["frame-count"])) - 1
                confidence = float(row["confidence"])
            except (TypeError, ValueError):
                continue
            if frame > last_frame or confidence < MIN_CONFIDENCE:
                continue
            person_frames += 1
            if row["unique-id"] not in ("", "-1"):
                frames[row["unique-id"]] += 1
    minimum = max(1, round(fps))
    return sum(n >= minimum for n in frames.values()), person_frames / fps / 3600


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="CSV to write")
    parser.add_argument("--root", type=Path, required=True, help="folder with the results folders")
    parser.add_argument("--run", action="append", default=[], help="results folder name, in order of preference")
    arguments = parser.parse_args()

    done: set[str] = set()
    missing = 0
    with arguments.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for name in arguments.run:
            rows = 0
            for details in sorted((arguments.root / name / "details").glob("*.json")):
                payload = json.loads(details.read_text(encoding="utf-8"))
                if "segment" not in payload or payload["video_key"] in done:
                    continue
                done.add(payload["video_key"])
                segment = payload["segment"]
                trim = payload.get("trim", {})
                start = float(segment["start_second"])
                hours = (float(trim.get("effective_end_second", segment["end_second"])) - start) / 3600
                fps = float(payload["fps"])
                track_file = Path(payload.get("precomputed_tracks") or "")
                if not track_file.is_file():
                    missing += 1
                    continue
                pedestrians, pedestrian_hours = count(track_file, int(hours * 3600 * fps), fps)
                metadata = {str(k).lower(): v for k, v in segment.get("metadata", {}).items()}
                writer.writerow({
                    "video_key": payload["video_key"], "run": name, "country": metadata.get("country", ""),
                    "hours": round(hours, 5), "pedestrians": pedestrians,
                    "pedestrian_hours": round(pedestrian_hours, 5),
                    "candidates": len(payload["result"].get("crossing_classifications", [])),
                    "track_file": track_file.name,
                })
                rows += 1
            print(f"{name}: {rows} segments", flush=True)
    print(f"wrote {arguments.out}; {missing} segments without a track file")


if __name__ == "__main__":
    main()
