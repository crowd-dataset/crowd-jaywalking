"""Compute BoT-SORT camera motion (GMC) for every JAAD video with saved benchmark tracks.

The camera motion aware crossing gate (crossing_gate_camera_motion) is trained on these
files, and run_jaad_person_audit.py reuses them. Existing files are kept, so the run
can be interrupted and resumed.

    uv run python -u .\compute_jaad_camera_motion.py
"""

from __future__ import annotations

import os

from crowd_jaywalking.camera_motion import estimate_camera_motion, save_camera_motion_csv
from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.jaad import JAADDataset


def main() -> None:
    config = ProjectConfig.load(os.environ.get("CROWD_JAYWALKING_CONFIG"))
    benchmark = config.path("jaad_benchmark_results")
    dataset = JAADDataset(config.path("jaad_root"))
    output = benchmark / "camera_motion"
    output.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        {path.stem for split in ("train", "val", "test") for path in (benchmark / split / "tracks").glob("*.csv")}
    )
    for index, video_id in enumerate(videos, start=1):
        target = output / f"{video_id}.csv"
        if target.exists():
            continue
        save_camera_motion_csv(target, estimate_camera_motion(dataset.clip_path(video_id)))
        print(f"[{index:03d}/{len(videos):03d}] {video_id}", flush=True)
    print(f"Camera motion for {len(videos)} videos in {output}")


if __name__ == "__main__":
    main()
