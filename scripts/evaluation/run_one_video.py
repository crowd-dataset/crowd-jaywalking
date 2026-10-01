"""Smoke test one video before starting the complete evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.core.config import ProjectConfig
from scripts.core.models import to_jsonable
from scripts.core.pipeline import JaywalkingPipeline
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    config = ProjectConfig.load()
    video_value = str(config.get("smoke_test_video") or "").strip()
    if not video_value:
        raise RuntimeError("Set smoke_test_video in config to a video path before running this smoke test.")
    video_path = Path(video_value).resolve()
    smoke_dir = config.path("results") / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    pipeline = JaywalkingPipeline(config)
    result = pipeline.process_video(video_path, smoke_dir / "evidence")
    output_path = smoke_dir / f"{video_path.stem}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(result), handle, indent=2)

    logger.info("Prediction: {}", result.prediction.value)
    logger.info("Person decisions: {}", len(result.person_decisions))
    logger.info("Rejected crossing candidates: {}", len(result.rejected_candidates))
    logger.info("Saved: {}", output_path)


if __name__ == "__main__":
    main()
