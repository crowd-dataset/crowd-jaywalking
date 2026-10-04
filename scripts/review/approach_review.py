"""Review flag: is a zebra crossing on the road ahead in the seconds before a claimed crossing?

The infrastructure check of stage 2 sees only a second around the crossing. A zebra the camera
car is stopped at, or has just approached, can be at the bottom edge of the frame or seen edge-on
there, and the claim is wrong although the check answered NO. This step runs after a run, over
the saved claims (people labelled JAYWALKING in ``<root>/**/details``): it takes frames from a
few seconds before each crossing and asks the VLM whether a pedestrian crossing is on the road
ahead. It only flags claims for a manual look; no claim or decision is changed.

A flag is not a verdict. On JAAD, 22 of 23 claims were not flagged; the VLM can read a painted
kerb as stripes, so a flagged claim has to be looked at, not dropped.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from custom_logger import CustomLogger
from scripts.core.models import EvidenceImage

logger = CustomLogger(__name__)  # use custom logger

PROMPT = """You are inspecting road infrastructure in chronological images from a vehicle dashcam, taken in the seconds before a pedestrian crossed in front of the car.
FULL SCENE views show the whole image. ROAD AHEAD views show the whole road between the camera and the horizon at full width.

Task: decide whether painted pedestrian crossing markings are on the road ahead of the car, within about 20 metres in front of it, in any of the images. They can be at the bottom edge of the image directly in front of the car, or seen from the side as a row of short stripes. Ignore side streets far away. Ignore the camera car's own bonnet, dashboard, and windscreen reflections at the bottom of the images: blurred bright patches and streaks there are not road markings.

Return one JSON object with exactly these keys:
{
  "zebra_crossing": "YES|NO|UNCERTAIN",
  "evidence_summary": "one short sentence about the road markings only"
}

zebra_crossing:
- YES when zebra stripes, a ladder pattern, or two parallel painted lines bounding a crosswalk are on the road ahead within about 20 metres, even if faded, partly covered, seen from a low angle, or small in the distance.
- NO only when the road surface ahead is clearly visible in at least one image and has no such markings. Lane lines, centre lines, stop lines, give-way lines, arrows, text, parking bays, kerbs or kerb stones painted white and black, road edge lines, barriers, and shadows are not crossings. A crossing is a set of parallel bars, or two parallel lines, that run across the whole carriageway at right angles to the direction of traffic.
- UNCERTAIN when faint, worn, or partly hidden stripes might be a crossing, or when the road surface is hidden, dark, or blurred.

Output JSON only."""  # noqa: E501

# Flag names written to flags.csv; NO_VIDEO means the saved segment video is gone.
ZEBRA_AHEAD = "ZEBRA_AHEAD"
UNCLEAR = "UNCLEAR"
CLEAR = "CLEAR"
NO_VIDEO = "NO_VIDEO"
ANSWER_FLAGS = {"YES": ZEBRA_AHEAD, "UNCERTAIN": UNCLEAR, "NO": CLEAR}
FIELDS = (
    "key", "source", "video_id", "person_id", "video_time_s", "window_s", "frames",
    "flag", "answer", "evidence_summary",
)


@dataclass(frozen=True)
class ClaimRef:
    """One claimed person and where to find the video of the claim."""

    source: str
    video_id: str
    person_id: int
    video_path: Path
    # Seconds the video starts into its source video: the CROWD segment start, 0 for JAAD.
    start_second: float
    fps: float | None
    transition_start_frame: int

    @property
    def key(self) -> str:
        return f"{self.source}/{self.video_id}/{self.person_id}"


def approach_frame_indices(
    transition_start_frame: int, fps: float, seconds: float, end_seconds: float, count: int
) -> list[int]:
    """Frames from ``seconds`` before the crossing to ``end_seconds`` before it, never before frame 0."""

    first = max(0, transition_start_frame - int(round(seconds * fps)))
    last = max(first, transition_start_frame - int(round(end_seconds * fps)))
    if count == 1:
        return [last]
    return sorted({int(round(first + (last - first) * step / (count - 1))) for step in range(count)})


def find_claims(results_root: str | Path, video_dirs: Iterable[Path] = ()) -> Iterator[ClaimRef]:
    """Every claimed person under a results folder; a missing video is looked for by name in ``video_dirs``."""

    root = Path(results_root)
    for details in sorted(root.glob("**/details")):
        if not details.is_dir() or {"jaywalking_law", "approach_review"} & set(details.parts):
            continue
        run_dir = details.parent
        source = str(run_dir.relative_to(root)).replace("\\", "/") or "."
        for path in sorted(details.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = payload["result"]
            video_path = Path(result["video_path"])
            if not video_path.is_file():
                for folder in (*video_dirs, run_dir / "segments"):
                    if (Path(folder) / video_path.name).is_file():
                        video_path = Path(folder) / video_path.name
                        break
            for decision in result["person_decisions"]:
                if decision["label"] != "JAYWALKING":
                    continue
                yield ClaimRef(
                    source=source,
                    video_id=str(payload.get("video_key") or payload.get("video_id") or path.stem),
                    person_id=int(decision["person_id"]),
                    video_path=video_path,
                    start_second=float(payload.get("segment", {}).get("start_second", 0)),
                    fps=float(payload["fps"]) if payload.get("fps") else None,
                    transition_start_frame=int(decision["event"]["transition_start_frame"]),
                )


def approach_evidence(claim: ClaimRef, settings: dict[str, Any], folder: Path) -> tuple[list[EvidenceImage], float]:
    """Save full scene and road ahead views of the approach; return them and the window length in seconds."""

    import cv2

    capture = cv2.VideoCapture(str(claim.video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open the video of {claim.key}: {claim.video_path}")
    try:
        fps = claim.fps or float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        indices = approach_frame_indices(
            claim.transition_start_frame, fps, settings["seconds"], settings["end_seconds"], settings["frames"]
        )
        folder.mkdir(parents=True, exist_ok=True)
        stem = claim.key.replace("/", "_")
        evidence: list[EvidenceImage] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            # The bottom strip of a dashcam frame can show the car's own bonnet and its reflections.
            height = frame.shape[0]
            frame = frame[: int(round((1.0 - settings["crop_bottom"]) * height))]
            height, width = frame.shape[:2]
            scale = min(1.0, 1280 / width)
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            road_top = int(0.40 * height)
            full = cv2.resize(frame, size)
            road = cv2.resize(frame[road_top:], (size[0], max(1, int((height - road_top) * scale))))
            full_path, road_path = folder / f"{stem}_{index}_full.jpg", folder / f"{stem}_{index}_road.jpg"
            if not (cv2.imwrite(str(full_path), full) and cv2.imwrite(str(road_path), road)):
                raise RuntimeError(f"Could not save approach frames for {claim.key}")
            evidence.append(
                EvidenceImage(
                    frame_index=index, context_path=full_path, focus_path=full_path, lower_road_path=road_path
                )
            )
        return evidence, (claim.transition_start_frame - indices[0]) / fps
    finally:
        capture.release()


def run_approach_review(
    vlm: Any,
    results_root: str | Path,
    settings: dict[str, Any],
    video_dirs: Iterable[Path] = (),
) -> dict[str, Any]:
    """Review every claim and write ``<root>/approach_review/``; claims already reviewed are skipped."""

    output = Path(results_root) / "approach_review"
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "flags.jsonl"
    done = set()
    if records_path.exists():
        done = {json.loads(line)["key"] for line in records_path.read_text(encoding="utf-8").splitlines()}
    with records_path.open("a", encoding="utf-8") as handle:
        for claim in find_claims(results_root, video_dirs):
            if claim.key in done:
                continue
            record: dict[str, Any] = {
                "key": claim.key,
                "source": claim.source,
                "video_id": claim.video_id,
                "person_id": claim.person_id,
                "video_time_s": None,
                "window_s": None,
                "frames": 0,
                "flag": NO_VIDEO,
                "answer": "",
                "evidence_summary": "",
            }
            try:
                evidence, window = approach_evidence(claim, settings, output / "frames")
            except FileNotFoundError:
                evidence, window = [], None
            if evidence:
                payload = vlm.evaluate_approach(evidence, PROMPT)
                answer = str(payload.get("zebra_crossing", "")).strip().upper()
                if answer not in ANSWER_FLAGS:
                    raise ValueError(f"Unexpected approach review answer for {claim.key}: {payload}")
                fps = claim.fps or 30.0
                record.update(
                    video_time_s=round(claim.start_second + claim.transition_start_frame / fps, 1),
                    window_s=round(window, 1),
                    frames=len(evidence),
                    flag=ANSWER_FLAGS[answer],
                    answer=answer,
                    evidence_summary=str(payload.get("evidence_summary", "")).strip(),
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            logger.info("{}: {} {}", claim.key, record["flag"], record["evidence_summary"][:100])

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    with (output / "flags.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "claims": len(records),
        "flags": dict(Counter(record["flag"] for record in records)),
        "to_review": [record["key"] for record in records if record["flag"] in (ZEBRA_AHEAD, UNCLEAR)],
        "settings": settings,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
