from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from .paths import project_root


def load_yaml(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merged_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = load_yaml(project_root() / "config" / "default.yaml")
    if overrides:
        cfg = deep_merge(cfg, overrides)
    root = project_root()
    art = cfg.setdefault("artifacts", {})
    art["root"] = str(root / art.get("root", "artifacts"))
    for k in ("checkpoints", "logs", "metrics", "runs"):
        if k in art and not Path(art[k]).is_absolute():
            art[k] = str(root / art[k])
    data = cfg.setdefault("data", {})
    for key in ("synthetic_dir", "manifest_path"):
        if key in data and not Path(data[key]).is_absolute():
            data[key] = str(root / data[key])
    if "nexar_video_glob" in data and not Path(data["nexar_video_glob"]).is_absolute():
        data["nexar_video_glob"] = str(root / data["nexar_video_glob"])
    return cfg


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out
