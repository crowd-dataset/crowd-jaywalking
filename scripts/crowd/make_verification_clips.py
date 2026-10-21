"""Cut short clips of the crossings of a CROWD run, for checking them by eye.

Per country it takes a random sample from each group of stage 1 crossings:
  unmarked     the VLM says: crosses, no zebra crossing, traffic light or sign (label JAYWALKING)
  marked       crosses at a zebra crossing or traffic light (label COMPLIANT)
  unconfirmed  stage 1 found a crossing but the VLM check did not confirm it (NOT_EVALUATED)
The clip covers the track plus a little before and after, with the person's box drawn on it and a
caption. ``index.csv`` lists every clip with a link to the same moment on YouTube.

    uv run python scripts/crowd/make_verification_clips.py results/six_countries \
        --unmarked 8 --marked 4 --unconfirmed 6
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.core.config import ProjectConfig  # noqa: E402
from scripts.crowd.crowd_analysis import CrowdAnalysisRunner  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or ""
GROUPS = {"JAYWALKING": "unmarked", "COMPLIANT": "marked", "NOT_EVALUATED": "unconfirmed"}
PAD_SECONDS = 2.0
MAX_SECONDS = 14.0
COLOURS = {"unmarked": (60, 60, 255), "marked": (80, 200, 80), "unconfirmed": (0, 200, 255)}


def pick(rows: list[dict[str, str]], counts: dict[str, int], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_group: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["predicted_crossing"] == "True" and row["jaywalking_label"] in GROUPS:
            by_group[(row["country"], GROUPS[row["jaywalking_label"]])].append(row)
    chosen = []
    for (country, group), items in sorted(by_group.items()):
        rng.shuffle(items)
        chosen.extend(dict(item, group=group) for item in items[: counts[group]])
    return chosen


def track_boxes(bbox_dirs: list[Path], row: dict[str, str]) -> dict[int, tuple[float, float, float, float]]:
    """Box (x, y, w, h as fractions) of the person per segment frame, from the CROWD track CSV."""

    matches = [
        path for folder in bbox_dirs
        for path in sorted(folder.glob(f"{row['video_id']}_{row['start_second']}_*.csv"))
    ]
    if not matches:
        return {}
    boxes = {}
    with matches[0].open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            if item["yolo-id"] == "0" and str(int(float(item["unique-id"]))) == row["person_id"]:
                boxes[int(item["frame-count"])] = tuple(float(item[key]) for key in ("x-center", "y-center", "width", "height"))
    return boxes


def write_clip(source: Path, offset_seconds: float, row: dict[str, str], output: Path, bbox_dirs: list[Path]) -> float:
    """Cut the crossing out of ``source`` (whose time 0 is ``offset_seconds`` into the segment)."""

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(3)), int(capture.get(4))
    first = max(1, int(row["track_start_frame"]))
    last = int(row["track_end_frame"])
    start_frame = max(1, int(first - PAD_SECONDS * fps))
    end_frame = min(last + int(PAD_SECONDS * fps), start_frame + int(MAX_SECONDS * fps))
    boxes = track_boxes(bbox_dirs, row)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(offset_seconds * fps)) + start_frame - 1)
    colour = COLOURS[row["group"]]
    caption = f"{row['country']} | {row['group']} | person {row['person_id']}"
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame_number in range(start_frame, end_frame + 1):
            ok, image = capture.read()
            if not ok:
                break
            box = boxes.get(frame_number)
            if box:
                x, y, w, h = box
                cv2.rectangle(image, (int((x - w / 2) * width), int((y - h / 2) * height)),
                              (int((x + w / 2) * width), int((y + h / 2) * height)), colour, 3)
            cv2.rectangle(image, (0, 0), (width, 38), (0, 0, 0), -1)
            cv2.putText(image, caption, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            writer.write(image)
        writer.release()
        capture.release()
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-an", str(output)], check=True)
    return (int(row["start_second"]) * fps + start_frame - 1) / fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--unmarked", type=int, default=8)
    parser.add_argument("--marked", type=int, default=4)
    parser.add_argument("--unconfirmed", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="clip folder (default: <run>/verification_clips)")
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    rows = list(csv.DictReader((run / "per_person_results.csv").open(encoding="utf-8", newline="")))
    chosen = pick(rows, {"unmarked": arguments.unmarked, "marked": arguments.marked, "unconfirmed": arguments.unconfirmed}, arguments.seed)
    chosen.sort(key=lambda row: (row["video_id"], row["start_second"]))
    if not FFMPEG:
        raise SystemExit("ffmpeg is needed to write the clips: install it and put it on the PATH.")
    runner = CrowdAnalysisRunner(ProjectConfig.load())
    bbox_dirs = [Path(runner.settings["bbox_download_dir"]), *map(Path, runner.settings["bbox_dirs"])]
    output = arguments.output or run / "verification_clips"
    index = []
    cached_id, cached_path, downloaded = None, None, False
    for number, row in enumerate(chosen, start=1):
        name = "/".join((row["country"], row["group"], f"{row['video_key']}_person{row['person_id']}.mp4"))
        segment = Path(row["video_path"])
        if segment.is_file():
            source, offset = segment, 0.0
        else:
            if cached_id != row["video_id"]:
                if downloaded and cached_path:
                    cached_path.unlink(missing_ok=True)
                prepared = runner._downloader().download(row["video_id"], runner.settings["download_dir"])
                cached_id, cached_path, downloaded = row["video_id"], prepared.path, prepared.downloaded_this_run
            source, offset = cached_path, float(row["start_second"])
        try:
            seconds = write_clip(source, offset, row, output / name, bbox_dirs)
        except Exception as error:  # keep going; list the failure
            print(f"[{number}/{len(chosen)}] FAILED {name}: {type(error).__name__}: {error}")
            continue
        index.append({"country": row["country"], "group": row["group"], "clip": name, "video_id": row["video_id"],
                      "source_time_seconds": f"{seconds:.1f}", "youtube": f"https://youtu.be/{row['video_id']}?t={int(seconds)}",
                      "person_id": row["person_id"], "vlm_reason": row["decision_reason"][:120]})
        print(f"[{number}/{len(chosen)}] {name}")
    if downloaded and cached_path:
        cached_path.unlink(missing_ok=True)
    with (output / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    print(f"{len(index)} clips in {output}")


if __name__ == "__main__":
    main()
