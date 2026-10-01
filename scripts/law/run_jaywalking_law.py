"""Stage 3: decide whether the claimed people are jaywalking under their country's law.

Run after stages 1 and 2 have finished, on their results folder:

    # CROWD: every segment's country, state, and city come from mapping.csv
    uv run python scripts/law/run_jaywalking_law.py results/crowd_jaywalking_v2

    # JAAD has no per video country, so give one or more
    uv run python scripts/law/run_jaywalking_law.py results/jaad_crowd_city_vlm_v3 --country CAN --country UKR

Options: ``--evidence-from <folder>`` when the evidence images were saved by another run
of the same videos. The VLM is ``jaywalking_law_model`` from the configuration
(default: ``vlm_model``) and the rules are ``jaywalking_law_rules`` (default:
configs/jaywalking_rules.json). Results go to ``<folder>/jaywalking_law/``.
"""

from __future__ import annotations

import argparse
import json

from scripts.core.config import ProjectConfig
from scripts.law.jaywalking_law import JaywalkingLawJudge, LawLocation
from scripts.law.law_stage import run_law_stage
from scripts.core.vlm import HuggingFaceContextClassifier
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="results folder of a finished stage 1 and 2 run")
    parser.add_argument("--country", action="append", help="country for claims without a location (repeatable)")
    parser.add_argument("--evidence-from", help="run folder that holds the evidence images")
    args = parser.parse_args()

    config = ProjectConfig.load()
    settings = config.jaywalking_law_settings()
    default = (
        LawLocation(settings["country"], settings["state"], settings["locality"])
        if settings["country"]
        else None
    )
    vlm = HuggingFaceContextClassifier(settings["vlm"])
    summary = run_law_stage(
        JaywalkingLawJudge(settings["rules"]),
        vlm,
        settings["vlm"]["model_id"],
        args.results,
        countries=args.country,
        default_location=default,
        evidence_from=args.evidence_from,
    )
    logger.info("{}", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
