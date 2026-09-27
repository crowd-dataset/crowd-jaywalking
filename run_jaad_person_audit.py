"""Run the configured pipeline over a saved JAAD split and audit every claimed person.

Use with the strict profile to reproduce the paper precision figure:

    $env:CROWD_JAYWALKING_CONFIG = ".\strict.config"
    $env:CROWD_JAYWALKING_JAAD_SPLIT = "test"
    uv run python -u .\run_jaad_person_audit.py
"""

from __future__ import annotations

import os

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.jaad_person_audit import JAADPersonAudit


def main() -> None:
    config = ProjectConfig.load(os.environ.get("CROWD_JAYWALKING_CONFIG"))
    split = os.environ.get("CROWD_JAYWALKING_JAAD_SPLIT", "test").strip().lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("CROWD_JAYWALKING_JAAD_SPLIT must be train, val, or test")
    JAADPersonAudit(config, split).run()


if __name__ == "__main__":
    main()
