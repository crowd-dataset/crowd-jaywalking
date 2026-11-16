"""Body keypoints of the people who crossed without a marking, for the analysis of how they behave.

For every person of one or more CROWD runs with the chosen label (default JAYWALKING: crossed with no
zebra crossing, traffic light or crossing sign in view), this reads the person's boxes from the CROWD
track file, opens the source video, crops each frame around the box, runs a YOLO11 pose model on the
crops, and keeps the pose that matches the tracked box best. COMPLIANT people (crossed at a marking)
can be added with ``--labels`` for a comparison in the same footage.

Output below ``--output``:

  keypoints/<video_key>_person<id>.csv   one row per frame of the track: time, the tracked box, the 17 COCO
                                          keypoints (x, y in pixels of the full frame, confidence) and the
                                          per-frame head and body measures below
  people.csv                              one row per person: run, country, label, frames, measures
  errors.csv                              people that could not be done, with the reason

A person whose keypoint CSV exists is skipped on a rerun. Videos are taken from ``--video-dir`` when the
file ``<video_id>.mp4`` is there, else downloaded from the CROWD file server as the runs do, and deleted
again after their people are done (``--keep-videos`` keeps them).

Per-frame measures (a keypoint counts when its confidence is at least ``--min-keypoint-conf``):
  face_to_camera   nose and both eyes visible: the face is turned to the camera, which is in the car
                   approaching the person, so the person looks towards the car
  side_on          shoulder span below ``SIDE_ON_RATIO`` of the torso length: the body faces across the
                   road, as when walking across it
  glance           face_to_camera while side_on: the head is turned to the car while the body crosses
  head_yaw         nose minus the middle of the ears (or eyes), over the shoulder span; sign follows the
                   image x axis

Per-person measures: frames with a pose, the share of them with face_to_camera, side_on and glance, the
number of separate glances (at least ``--min-glance-seconds`` long), whether the person glanced at the
car before stepping onto the road, and how long before.

    uv run python scripts/crowd/run_keypoints.py results/six_countries_100h_16bit results/korea_vietnam_16bit \
        --output results/keypoints_16bit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
INDEX = {name: number for number, name in enumerate(KEYPOINTS)}
SIDE_ON_RATIO = 0.45  # shoulder span over torso length below which the body is seen from the side
CROP_MARGIN = 0.35    # margin added around the tracked box on every side, as a share of its size
MIN_CROP_HEIGHT = 256  # crops are scaled up to at least this height before the pose model sees them

FRAME_FIELDS = (
    ["frame", "seconds", "box_x1", "box_y1", "box_x2", "box_y2", "pose_found", "pose_iou", "pose_conf"]
    + [f"{name}_{part}" for name in KEYPOINTS for part in ("x", "y", "conf")]
    + ["face_to_camera", "side_on", "glance", "head_yaw", "on_road"]
)
PERSON_FIELDS = (
    "run", "video_key", "video_id", "person_id", "country", "locality", "label", "fps",
    "track_frames", "pose_frames", "pose_share", "face_to_camera_share", "side_on_share", "glance_share",
    "glances", "glanced_before_road", "first_glance_seconds_before_road", "head_yaw_abs_mean",
)
ERROR_FIELDS = ("run", "video_key", "person_id", "error")


# ----------------------------------------------------------------------------------------------- people


@dataclass
class Person:
    """One person of a run to measure, with where their video and boxes are."""

    run: str
    video_key: str
    video_id: str
    start_second: int
    person_id: int
    label: str
    country: str
    locality: str
    fps: float
    start_frame: int
    end_frame: int
    road_start_frame: int
    road_end_frame: int
    track_file: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.video_key}_person{self.person_id}"


def _metadata(segment: dict[str, Any], name: str) -> str:
    lowered = {str(key).lower(): value for key, value in segment.get("metadata", {}).items()}
    return str(lowered.get(name, ""))


def people_of_run(run: Path, labels: set[str]) -> list[Person]:
    """The people of one run whose stage 2 label is in ``labels``, read from the details files."""

    people = []
    for details in sorted((run / "details").glob("*.json")):
        payload = json.loads(details.read_text(encoding="utf-8"))
        if "segment" not in payload:
            continue
        segment = payload["segment"]
        fps = float(payload.get("trim", {}).get("source_fps") or payload["fps"])
        for decision in payload["result"]["person_decisions"]:
            if decision["label"] not in labels:
                continue
            event = decision["event"]
            people.append(
                Person(
                    run=run.name,
                    video_key=payload["video_key"],
                    video_id=segment["video_id"],
                    start_second=int(segment["start_second"]),
                    person_id=int(decision["person_id"]),
                    label=decision["label"],
                    country=_metadata(segment, "country"),
                    locality=_metadata(segment, "locality"),
                    fps=fps,
                    start_frame=int(event["start_frame"]),
                    end_frame=int(event["end_frame"]),
                    road_start_frame=int(event["transition_start_frame"]),
                    road_end_frame=int(event["transition_end_frame"]),
                    track_file=payload.get("precomputed_tracks"),
                    metadata=segment,
                )
            )
    return people


def person_boxes(track_file: Path, person: Person) -> dict[int, tuple[float, float, float, float]]:
    """The person's tracked box per frame index (x1, y1, x2, y2 as fractions of the frame).

    CROWD's frame-count starts at 1; the frame index of the details starts at 0.
    """

    boxes = {}
    with track_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["yolo-id"] != "0" or row["unique-id"] in ("", None, "nan"):
                continue
            if int(float(row["unique-id"])) != person.person_id:
                continue
            frame = int(float(row["frame-count"])) - 1
            if person.start_frame <= frame <= person.end_frame:
                x, y = float(row["x-center"]), float(row["y-center"])
                w, h = float(row["width"]), float(row["height"])
                boxes[frame] = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    return boxes


# ------------------------------------------------------------------------------------------- geometry


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""

    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop_window(
    box: tuple[float, ...], width: int, height: int, margin: float = CROP_MARGIN
) -> tuple[int, int, int, int]:
    """Pixel window around a fractional box, with ``margin`` of its size added on every side."""

    x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(width, int(x2 + mx) + 1), min(height, int(y2 + my) + 1))


def best_pose(
    target: tuple[float, ...],
    boxes: Iterable[tuple[float, ...]],
    min_iou: float,
) -> tuple[int, float] | None:
    """Index and IoU of the pose box that overlaps the target box most, if at least ``min_iou``."""

    best = None
    for number, box in enumerate(boxes):
        overlap = iou(target, box)
        if overlap >= min_iou and (best is None or overlap > best[1]):
            best = (number, overlap)
    return best


# ------------------------------------------------------------------------------------------- measures


def frame_measures(points: list[tuple[float, float, float]], min_conf: float) -> dict[str, Any]:
    """Head and body measures of one pose; ``points`` are 17 (x, y, conf) in pixels."""

    def seen(name: str) -> tuple[float, float] | None:
        x, y, conf = points[INDEX[name]]
        return (x, y) if conf >= min_conf else None

    nose, left_eye, right_eye = seen("nose"), seen("left_eye"), seen("right_eye")
    left_ear, right_ear = seen("left_ear"), seen("right_ear")
    left_sh, right_sh = seen("left_shoulder"), seen("right_shoulder")
    left_hip, right_hip = seen("left_hip"), seen("right_hip")
    face = bool(nose and left_eye and right_eye)
    side_on = None
    span = None
    if left_sh and right_sh:
        span = abs(left_sh[0] - right_sh[0])
        hips = [p for p in (left_hip, right_hip) if p]
        if hips:
            shoulder_y = (left_sh[1] + right_sh[1]) / 2
            hip_y = sum(p[1] for p in hips) / len(hips)
            torso = abs(hip_y - shoulder_y)
            if torso > 0:
                side_on = span / torso < SIDE_ON_RATIO
    head_yaw = None
    pair = (left_ear, right_ear) if left_ear and right_ear else (left_eye, right_eye)
    if nose and pair[0] and pair[1] and span:
        centre = (pair[0][0] + pair[1][0]) / 2
        head_yaw = (nose[0] - centre) / span
    return {
        "face_to_camera": face,
        "side_on": side_on,
        "glance": bool(face and side_on),
        "head_yaw": head_yaw,
    }


def runs_of(flags: list[bool], min_length: int) -> list[tuple[int, int]]:
    """(first, last) positions of the runs of True at least ``min_length`` long."""

    found, start = [], None
    for position, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            if position - start >= min_length:
                found.append((start, position - 1))
            start = None
    return found


def person_summary(person: Person, rows: list[dict[str, Any]], min_glance_seconds: float) -> dict[str, Any]:
    """Per-person measures from the per-frame rows (only frames with a pose count)."""

    posed = [row for row in rows if row["pose_found"]]
    n = len(posed)

    def share(key: str) -> str:
        values = [row[key] for row in posed if row[key] is not None and row[key] != ""]
        return f"{sum(bool(v) for v in values) / len(values):.3f}" if values else ""

    # glances counted over consecutive frames; a frame without a pose breaks a glance
    min_frames = max(1, round(min_glance_seconds * person.fps))
    flags = [bool(row["pose_found"] and row["glance"]) for row in rows]
    glances = runs_of(flags, min_frames)
    before = [(rows[a]["frame"], rows[b]["frame"]) for a, b in glances if rows[a]["frame"] < person.road_start_frame]
    first_before = (
        f"{(person.road_start_frame - before[0][0]) / person.fps:.2f}" if before else ""
    )
    yaws = [abs(row["head_yaw"]) for row in posed if row["head_yaw"] not in (None, "")]
    return {
        "run": person.run,
        "video_key": person.video_key,
        "video_id": person.video_id,
        "person_id": person.person_id,
        "country": person.country,
        "locality": person.locality,
        "label": person.label,
        "fps": person.fps,
        "track_frames": len(rows),
        "pose_frames": n,
        "pose_share": f"{n / len(rows):.3f}" if rows else "",
        "face_to_camera_share": share("face_to_camera"),
        "side_on_share": share("side_on"),
        "glance_share": share("glance"),
        "glances": len(glances),
        "glanced_before_road": bool(before) if n else "",
        "first_glance_seconds_before_road": first_before,
        "head_yaw_abs_mean": f"{sum(yaws) / len(yaws):.3f}" if yaws else "",
    }


# ----------------------------------------------------------------------------------------------- pose


class PoseModel:
    """A YOLO pose model run on crops around the tracked person."""

    def __init__(self, weights: str, device: str | None, batch: int, min_iou: float) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.batch = batch
        self.min_iou = min_iou

    def poses(self, crops: list[Any]) -> list[Any]:
        results = []
        for start in range(0, len(crops), self.batch):
            results += self.model.predict(crops[start:start + self.batch], device=self.device, verbose=False)
        return results


def measure_person(
    video: Path,
    person: Person,
    boxes: dict[int, tuple[float, float, float, float]],
    model: PoseModel,
    min_conf: float,
    stride: int,
) -> list[dict[str, Any]]:
    """Per-frame rows of one person: tracked box, matched pose keypoints and measures."""

    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video.name}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = sorted(frame for frame in boxes if (frame - person.start_frame) % stride == 0)
    if not frames:
        capture.release()
        return []
    first_source_frame = round(person.start_second * person.fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_source_frame + frames[0])
    crops, windows, kept = [], [], []
    position = frames[0]
    wanted = set(frames)
    while position <= frames[-1]:
        ok, image = capture.read()
        if not ok:
            break
        if position in wanted:
            x1, y1, x2, y2 = crop_window(boxes[position], width, height)
            crop = image[y1:y2, x1:x2]
            scale = max(1.0, MIN_CROP_HEIGHT / max(1, crop.shape[0]))
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            crops.append(crop)
            windows.append((x1, y1, scale))
            kept.append(position)
        position += 1
    capture.release()

    rows = []
    for frame, (x1, y1, scale), result in zip(kept, windows, model.poses(crops)):
        box = boxes[frame]
        target = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
        row: dict[str, Any] = {
            "frame": frame,
            "seconds": f"{person.start_second + frame / person.fps:.3f}",
            "box_x1": f"{target[0]:.1f}", "box_y1": f"{target[1]:.1f}",
            "box_x2": f"{target[2]:.1f}", "box_y2": f"{target[3]:.1f}",
            "pose_found": False, "pose_iou": "", "pose_conf": "",
            "face_to_camera": "", "side_on": "", "glance": "", "head_yaw": "",
            "on_road": person.road_start_frame <= frame <= person.road_end_frame,
        }
        pose_boxes = []
        if result.boxes is not None and len(result.boxes):
            for bx in result.boxes.xyxy.tolist():
                pose_boxes.append((bx[0] / scale + x1, bx[1] / scale + y1, bx[2] / scale + x1, bx[3] / scale + y1))
        match = best_pose(target, pose_boxes, model.min_iou)
        if match is not None:
            number, overlap = match
            xy = result.keypoints.xy[number].tolist()
            conf = result.keypoints.conf[number].tolist() if result.keypoints.conf is not None else [1.0] * len(xy)
            points = [(px / scale + x1, py / scale + y1, c) for (px, py), c in zip(xy, conf)]
            row.update({"pose_found": True, "pose_iou": f"{overlap:.3f}",
                        "pose_conf": f"{float(result.boxes.conf[number]):.3f}"})
            for name, (px, py, c) in zip(KEYPOINTS, points):
                row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_conf"] = f"{px:.1f}", f"{py:.1f}", f"{c:.3f}"
            row.update(frame_measures(points, min_conf))
            if row["head_yaw"] is not None:
                row["head_yaw"] = round(row["head_yaw"], 3)
            else:
                row["head_yaw"] = ""
            if row["side_on"] is None:
                row["side_on"] = ""
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------------------------- files


def _write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_frames(path: Path) -> list[dict[str, Any]]:
    """Per-frame rows of a saved keypoint CSV, with the fields person_summary needs typed again."""

    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "pose_found": row["pose_found"] == "True",
                    "face_to_camera": {"True": True, "False": False}.get(row["face_to_camera"], ""),
                    "side_on": {"True": True, "False": False}.get(row["side_on"], ""),
                    "glance": row["glance"] == "True",
                    "head_yaw": float(row["head_yaw"]) if row["head_yaw"] else "",
                }
            )
    return rows


# ------------------------------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="results folders of CROWD runs (with details/)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["JAYWALKING"], help="stage 2 labels to measure")
    parser.add_argument("--countries", nargs="*", help="only these countries")
    parser.add_argument("--max-people", type=int, help="stop after this many people (for a test)")
    parser.add_argument("--video-dir", type=Path, action="append", default=[],
                        help="folder with <video_id>.mp4 files to use before downloading (repeatable)")
    parser.add_argument("--keep-videos", action="store_true", help="keep downloaded source videos")
    parser.add_argument("--pose-model", default="yolo11x-pose.pt")
    parser.add_argument("--device", default=None, help="cuda, mps, cpu (default: Ultralytics' choice)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1, help="use every n-th frame of the track")
    parser.add_argument("--min-iou", type=float, default=0.3, help="pose box against tracked box")
    parser.add_argument("--min-keypoint-conf", type=float, default=0.5)
    parser.add_argument("--min-glance-seconds", type=float, default=0.2)
    arguments = parser.parse_args()

    labels = set(arguments.labels)
    people = [person for run in arguments.runs for person in people_of_run(run.resolve(), labels)]
    if arguments.countries:
        people = [person for person in people if person.country in set(arguments.countries)]
    people.sort(key=lambda person: (person.video_id, person.start_second, person.person_id))
    if arguments.max_people:
        people = people[: arguments.max_people]
    output = arguments.output
    keypoint_dir = output / "keypoints"
    todo = [person for person in people if not (keypoint_dir / f"{person.name}.csv").exists()]
    print(f"{len(people)} people with labels {sorted(labels)}; "
          f"{len(people) - len(todo)} done before, {len(todo)} to do")

    runner = None
    model = None
    errors: list[dict[str, Any]] = []
    by_video: dict[str, list[Person]] = defaultdict(list)
    for person in todo:
        by_video[person.video_id].append(person)
    done = 0
    for video_id, group in by_video.items():
        if model is None:
            model = PoseModel(arguments.pose_model, arguments.device, arguments.batch, arguments.min_iou)
        local = next((folder / f"{video_id}.mp4" for folder in arguments.video_dir
                      if (folder / f"{video_id}.mp4").is_file()), None)
        downloaded = None
        try:
            if local is None:
                if runner is None:
                    from scripts.core.config import ProjectConfig
                    from scripts.crowd.crowd_analysis import CrowdAnalysisRunner

                    runner = CrowdAnalysisRunner(ProjectConfig.load())
                prepared = runner._downloader().download(video_id, Path(runner.settings["download_dir"]))
                local = prepared.path
                downloaded = prepared.path if prepared.downloaded_this_run else None
        except Exception as error:  # the people of this video fail, the others go on
            errors += [{"run": p.run, "video_key": p.video_key, "person_id": p.person_id,
                        "error": f"video: {type(error).__name__}: {error}"} for p in group]
            continue
        for person in group:
            try:
                track_file = Path(person.track_file) if person.track_file else None
                if track_file is None or not track_file.is_file():
                    from scripts.crowd.crowd_tracks import find_local_track_file

                    if runner is None:
                        from scripts.core.config import ProjectConfig
                        from scripts.crowd.crowd_analysis import CrowdAnalysisRunner

                        runner = CrowdAnalysisRunner(ProjectConfig.load())
                    folders = [Path(runner.settings["bbox_download_dir"]), *map(Path, runner.settings["bbox_dirs"])]
                    track_file = find_local_track_file(person.video_id, person.start_second, folders, person.fps)
                    if track_file is None:
                        raise FileNotFoundError("no CROWD track file for this segment")
                boxes = person_boxes(track_file, person)
                rows = measure_person(local, person, boxes, model, arguments.min_keypoint_conf, arguments.stride)
                if not rows:
                    raise RuntimeError("no frames of the person could be read")
                _write_csv(keypoint_dir / f"{person.name}.csv", FRAME_FIELDS, rows)
                done += 1
                posed = sum(bool(row["pose_found"]) for row in rows)
                print(f"[{done}/{len(todo)}] {person.country} {person.name}: pose in {posed} of {len(rows)} frames")
            except Exception as error:
                errors.append({"run": person.run, "video_key": person.video_key, "person_id": person.person_id,
                               "error": f"{type(error).__name__}: {error}"})
                print(f"FAILED {person.name}: {type(error).__name__}: {error}")
        if downloaded is not None and not arguments.keep_videos:
            downloaded.unlink(missing_ok=True)

    summaries = []
    for person in people:
        path = keypoint_dir / f"{person.name}.csv"
        if path.exists():
            summaries.append(person_summary(person, _read_frames(path), arguments.min_glance_seconds))
    _write_csv(output / "people.csv", PERSON_FIELDS, summaries)
    _write_csv(output / "errors.csv", ERROR_FIELDS, errors)
    print(f"{len(summaries)} people in {output / 'people.csv'}; {len(errors)} errors")


if __name__ == "__main__":
    main()
