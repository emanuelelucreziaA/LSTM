"""Shared runtime/bootstrap helpers for command-line entry points."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def initialize_environment(*, ensure_data_dir: bool = False) -> tuple[Path, Path | None]:
    """Load environment variables and resolve common project paths."""
    load_dotenv()

    project_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()

    data_dir = None
    if ensure_data_dir:
        data_dir = Path(os.getenv("DATA_DIR", project_root / "data"))
        data_dir.mkdir(parents=True, exist_ok=True)

    return project_root, data_dir
