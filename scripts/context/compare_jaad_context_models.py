"""Compare configured Hugging Face VLMs on labelled JAAD context evidence."""

from __future__ import annotations


from scripts.core.config import ProjectConfig
from scripts.context.vlm_comparison import VLMModelComparison
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    VLMModelComparison(ProjectConfig.load()).run()


if __name__ == "__main__":
    main()
