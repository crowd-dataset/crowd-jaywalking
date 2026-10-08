"""Score crowd-city's crossing decision (stage 1 alone, no VLM) against JAAD's labels.

For every JAAD video of the three splits, both crowd-city rules are run on the saved YOLO11x
tracks: ``road_crossing`` (the default, needs the clip for SegFormer) and ``detector`` (the
older box-only rule). Each accepted track is matched to a JAAD pedestrian by box overlap:

* precision: the share of accepted tracks whose pedestrian JAAD labels as crossing (tracks
  that match a bystander without a behaviour label or no pedestrian count as wrong);
* recall: the share of JAAD crossers that an accepted track matches, and the same for the
  crossers with no zebra crossing and no traffic light on their crossing frames.

    uv run python -u .\scripts\crossing\evaluate_crowd_city_crossing.py

The output is ``<results>/crowd_city_stage1/summary.json``.
"""

from __future__ import annotations

import json
from collections import defaultdict

import common
from custom_logger import CustomLogger
from logmod import logs
from scripts.core.config import ProjectConfig
from scripts.core.tracking import load_observations_csv
from scripts.crossing.crowd_city_crossing import MIN_CONFIDENCE, CrowdCityCrossingDetector
from scripts.crossing.crowd_city_road_rule import CrowdCityRoadCrossingDetector
from scripts.jaad.jaad_person_audit import JAADPersonAudit, eligible_crossers, match_pedestrian

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    config = ProjectConfig.load()
    rules = {
        "road_crossing": CrowdCityRoadCrossingDetector(config.crowd_city_settings()),
        "detector": CrowdCityCrossingDetector(),
    }
    stats: dict[str, dict[str, int]] = {name: defaultdict(int) for name in rules}
    videos = 0
    for split in ("train", "val", "test"):
        audit = JAADPersonAudit(config, split)
        for path in sorted(audit.tracks_dir.glob("*.csv")):
            video_id = path.stem
            if video_id in audit.excluded:
                continue
            videos += 1
            annotations = audit.dataset.load_video(video_id)
            observations = load_observations_csv(path)
            crossers = [p for p in annotations.behaviour_tracks if p.is_crossing]
            eligible = {p.pedestrian_id for p in eligible_crossers(annotations)}
            for name, detector in rules.items():
                if name == "road_crossing":
                    result = detector.detect(observations, 30.0, audit.dataset.clip_path(video_id))
                else:
                    result = detector.detect(observations, 30.0)
                joined = detector._joined([o for o in observations if o.confidence >= MIN_CONFIDENCE], 30.0)
                boxes: dict[int, dict] = defaultdict(dict)
                for item in joined:
                    if item.class_id == 0:
                        boxes[item.track_id][item.frame_index] = item.box
                counts, found = stats[name], set()
                for event in result.valid_events:
                    track = {f: b for f, b in boxes[event.person_id].items() if event.start_frame <= f <= event.end_frame}
                    pedestrian = match_pedestrian(annotations, track, audit.iou, audit.minimum_frames)
                    counts["accepted"] += 1
                    if pedestrian is None:
                        counts["unmatched"] += 1
                    elif not pedestrian.behaviour_annotated:
                        counts["bystander"] += 1
                    elif not pedestrian.is_crossing:
                        counts["not_crossing"] += 1
                    else:
                        counts["crossing"] += 1
                        found.add(pedestrian.pedestrian_id)
                counts["jaad_crossers"] += len(crossers)
                counts["jaad_eligible"] += len(eligible)
                counts["crossers_found"] += len(found)
                counts["eligible_found"] += len(found & eligible)
            logger.info("{} ({} videos): {}", video_id, videos, {k: dict(v) for k, v in stats.items()})

    summary = {"videos": videos}
    for name, counts in stats.items():
        counts = dict(counts)
        summary[name] = {
            **counts,
            "precision": counts.get("crossing", 0) / max(counts.get("accepted", 0), 1),
            "recall": counts.get("crossers_found", 0) / max(counts.get("jaad_crossers", 0), 1),
            "eligible_recall": counts.get("eligible_found", 0) / max(counts.get("jaad_eligible", 0), 1),
        }
    output = config.path("results") / "crowd_city_stage1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    logger.info("Saved: {}", output / "summary.json")


if __name__ == "__main__":
    main()
