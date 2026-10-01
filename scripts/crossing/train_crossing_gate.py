"""Train the high precision crossing gate on saved JAAD train and val tracks and check it on JAAD test."""

from __future__ import annotations


from scripts.core.config import ProjectConfig
from scripts.crossing.crossing_gate import CrossingGateTrainer
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    config = ProjectConfig.load()
    CrossingGateTrainer(config).run()


if __name__ == "__main__":
    main()
