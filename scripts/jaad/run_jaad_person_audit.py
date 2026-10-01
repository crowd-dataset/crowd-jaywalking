"""Run the configured pipeline over a saved JAAD split and audit every claimed person.

The split is jaad_audit_split in config (train, val, or test):

    uv run python -u .\scripts\jaad\run_jaad_person_audit.py
"""

from __future__ import annotations

from scripts.core.config import ProjectConfig
from scripts.jaad.jaad_person_audit import JAADPersonAudit
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


def main() -> None:
    config = ProjectConfig.load()
    split = str(config.get("jaad_audit_split")).strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("jaad_audit_split must be train, val, or test")
    JAADPersonAudit(config, split).run()


if __name__ == "__main__":
    main()
