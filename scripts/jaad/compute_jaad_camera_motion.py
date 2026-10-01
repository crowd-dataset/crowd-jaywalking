"""Compute BoT-SORT camera motion (GMC) for every JAAD video with saved benchmark tracks.

The camera motion aware crossing gate (crossing_gate_camera_motion) is trained on these
files, and scripts/jaad/run_jaad_person_audit.py reuses them. Existing files are kept, so the run
can be interrupted and resumed.

    uv run python -u .\scripts\jaad\compute_jaad_camera_motion.py
"""

from __future__ import annotations


from scripts.crossing.camera_motion import estimate_camera_motion, save_camera_motion_csv
from scripts.core.config import ProjectConfig
from scripts.jaad.jaad import JAADDataset
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    config = ProjectConfig.load()
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
        logger.info("[{:03d}/{:03d}] {}", index, len(videos), video_id)
    logger.info("Camera motion for {} videos in {}", len(videos), output)


if __name__ == "__main__":
    main()
