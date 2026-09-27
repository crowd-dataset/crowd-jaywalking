"""Train the high precision crossing gate on saved JAAD train and val tracks and check it on JAAD test."""

from __future__ import annotations

import os

from crowd_jaywalking.config import ProjectConfig
from crowd_jaywalking.crossing_gate import CrossingGateTrainer


def main() -> None:
    config_path = os.environ.get("CROWD_JAYWALKING_CONFIG")
    config = ProjectConfig.load(config_path)
    CrossingGateTrainer(config).run()


if __name__ == "__main__":
    main()
