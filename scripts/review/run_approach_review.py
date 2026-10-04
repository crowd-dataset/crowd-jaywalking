"""Approach review: flag claims for a manual look after a run has finished.

Run on the results folder of a finished stage 1 and 2 run (CROWD or JAAD), before or after stage 3:

    uv run python scripts/review/run_approach_review.py results/crowd_jaywalking

For every claimed person it looks at frames from a few seconds before the crossing and flags a
zebra crossing on the road ahead. Nothing is dropped or changed; the list is in
``<folder>/approach_review/flags.csv``. The VLM is ``vlm_model`` and the window comes from the
``approach_review_*`` settings. Reruns skip claims already reviewed.
"""

from __future__ import annotations

import argparse
import json

from scripts.core.config import ProjectConfig
from scripts.core.vlm import HuggingFaceContextClassifier
from scripts.review.approach_review import run_approach_review
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="results folder of a finished stage 1 and 2 run")
    args = parser.parse_args()

    config = ProjectConfig.load()
    video_dirs = [config.path("jaad_root") / "JAAD_clips", *config.paths("videos")]
    summary = run_approach_review(
        HuggingFaceContextClassifier(config.vlm_settings()),
        args.results,
        config.approach_review_settings(),
        video_dirs,
    )
    logger.info("{}", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
