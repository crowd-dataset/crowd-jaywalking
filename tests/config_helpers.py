"""Load a ProjectConfig from a temporary project folder through common.get_configs."""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import common
from scripts.core.config import ProjectConfig

REPOSITORY = Path(__file__).resolve().parents[1]


def template() -> dict[str, Any]:
    return json.loads((REPOSITORY / "default.config").read_text(encoding="utf-8"))


@contextlib.contextmanager
def project_root(
    default: dict[str, Any] | None = None,
    active: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Iterator[Path]:
    """A folder with default.config and config, used as common.root_dir."""

    with contextlib.ExitStack() as stack:
        folder = root or Path(stack.enter_context(tempfile.TemporaryDirectory()))
        default = template() if default is None else default
        (folder / "default.config").write_text(json.dumps(default), encoding="utf-8")
        (folder / "config").write_text(json.dumps(default if active is None else active), encoding="utf-8")
        stack.enter_context(mock.patch.object(common, "root_dir", str(folder)))
        yield folder


def load_config(raw: dict[str, Any] | None = None, root: Path | None = None) -> ProjectConfig:
    """ProjectConfig for these settings; a missing key falls back to the code defaults."""

    with project_root(raw, root=root):
        return ProjectConfig.load()
