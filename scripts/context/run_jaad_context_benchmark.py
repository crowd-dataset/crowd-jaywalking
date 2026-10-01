"""Evaluate the local VLM against completed JAAD context annotations."""

from __future__ import annotations


from scripts.core.config import ProjectConfig
from scripts.context.jaad_context_evaluation import JAADContextBenchmark
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    config = ProjectConfig.load()
    JAADContextBenchmark(config).run()


if __name__ == "__main__":
    main()
