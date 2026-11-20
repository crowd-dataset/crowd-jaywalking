"""Rebuild the evidence images of every claimed person, so stage 3 can run on runs that kept none.

Runs made with ``crowd_keep_evidence: false`` delete the evidence images after stage 2, but stage 3
(``run_jaywalking_law.py``) judges each claim on them. The images are drawn again exactly as stage 2
drew them (``EvidenceBuilder`` with the configured evidence settings), from what a run keeps:

- the claim clip ``<run>/segments/<video_key>.mp4`` (``crowd_keep_claim_segment_videos``),
- the CROWD track file named in the details (``precomputed_tracks``), and
- the crossing event saved with the claim in ``<run>/details/<video_key>.json``.

Claims whose images are already there are skipped; a claim without its clip or track file is
counted and reported, never guessed.

    uv run python scripts/law/rebuild_evidence.py results/india_a_16bit results/vietnam_16bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.core.config import ProjectConfig  # noqa: E402
from scripts.core.evidence import EvidenceBuilder  # noqa: E402
from scripts.core.models import CrossingEvent, CrossingFeatures, RejectionReason  # noqa: E402
from scripts.crowd.crowd_tracks import load_crowd_tracks  # noqa: E402


def crossing_event(event: dict) -> CrossingEvent:
    return CrossingEvent(
        person_id=int(event["person_id"]),
        start_frame=int(event["start_frame"]),
        end_frame=int(event["end_frame"]),
        transition_start_frame=int(event["transition_start_frame"]),
        transition_end_frame=int(event["transition_end_frame"]),
        valid=bool(event["valid"]),
        rejection_reason=RejectionReason(event["rejection_reason"]),
        features=CrossingFeatures(**event["features"]),
    )


def joined(observations: list, fps: float) -> list:
    """The observations with stage 1's ids: crowd-city joins broken tracks of one person under a new id."""

    from scripts.crossing.crowd_city_crossing import CrowdCityCrossingDetector

    detector = CrowdCityCrossingDetector()
    detector._joined([o for o in observations if o.confidence >= detector.min_confidence], fps)
    return detector.relabel(observations)


def frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def rebuild(run: Path, builder: EvidenceBuilder) -> dict[str, int]:
    counts = {"claims": 0, "built": 0, "already": 0, "no_clip": 0, "no_tracks": 0, "failed": 0}
    for path in sorted((run / "details").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["result"]
        claims = [d for d in result["person_decisions"] if d["label"] == "JAYWALKING"]
        if not claims:
            continue
        stem = Path(result["video_path"]).stem
        clip = run / "segments" / f"{stem}.mp4"
        tracks = Path(payload.get("precomputed_tracks") or "")
        observations = None
        for decision in claims:
            counts["claims"] += 1
            event = decision["event"]
            folder = run / "evidence" / stem / (
                f"person_{decision['person_id']}_transition_"
                f"{event['transition_start_frame']}_{event['transition_end_frame']}"
            )
            if any(folder.glob("frame_*_context.jpg")):
                counts["already"] += 1
                continue
            if not clip.is_file():
                counts["no_clip"] += 1
                print(f"  no clip: {run.name}/{stem} person {decision['person_id']}")
                continue
            if not tracks.is_file():
                counts["no_tracks"] += 1
                print(f"  no track file: {run.name}/{stem} ({tracks})")
                continue
            try:
                if observations is None:
                    # Stage 1 joins broken tracks before it judges anyone, and the event's person id and
                    # frames refer to the joined track, so the evidence is drawn from the joined tracks too.
                    raw = load_crowd_tracks(tracks, max_frames=frame_count(clip))
                    observations = joined(raw, float(payload["fps"]))
                builder.build(clip, crossing_event(event), observations, run / "evidence", float(payload["fps"]))
                counts["built"] += 1
            except Exception as error:  # report and go on; stage 3 reports the claim as missing
                counts["failed"] += 1
                print(f"  failed: {run.name}/{stem} person {decision['person_id']}: {type(error).__name__}: {error}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="results folders of finished runs")
    arguments = parser.parse_args()
    builder = EvidenceBuilder(ProjectConfig.load().evidence_settings())
    for run in arguments.runs:
        print(run.name, rebuild(run, builder), flush=True)


if __name__ == "__main__":
    main()
