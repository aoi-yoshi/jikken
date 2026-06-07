"""実験 run の共通初期化（step04b / step05 規約）。

各 step の artifacts/runs/<step_dir>/<run_id>/ に:
  used_config.yaml, run_meta.json, summary.json, *.jsonl
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from .logging_utils import RunLogger
from .paths import ensure_dirs
from .resource_metrics import build_environment_block


def make_run_id(suffix: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"run_{ts}"
    return f"{base}{suffix}" if suffix else base


def save_used_config(run_dir: Path, cfg: Mapping[str, Any]) -> Path:
    path = run_dir / "used_config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)
    return path


def init_run(
    cfg: Mapping[str, Any],
    *,
    step: str,
    step_dir: str,
    log_name: str = "events",
    run_id: Optional[str] = None,
    run_suffix: str = "",
    cli: Optional[Mapping[str, Any]] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Path, RunLogger, Dict[str, Any]]:
    run_id = run_id or make_run_id(run_suffix)
    run_dir = Path(cfg["artifacts"]["runs"]) / step_dir / run_id
    ensure_dirs(run_dir)
    save_used_config(run_dir, cfg)
    env = build_environment_block(dict(cfg))
    log = RunLogger(run_dir, name=log_name)
    meta: Dict[str, Any] = {
        "step": step,
        "run_id": run_id,
        "status": "started",
        "environment": env,
    }
    if cli is not None:
        meta["cli"] = dict(cli)
    if extra_meta:
        meta.update(dict(extra_meta))
    log.log_meta(meta)
    return run_id, run_dir, log, env
