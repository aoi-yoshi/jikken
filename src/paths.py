from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("THESIS_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
