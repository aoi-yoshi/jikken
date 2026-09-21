"""Step 7 Flower — CLI 上書きと checkpoint パス。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.fl_data import apply_fl_cli_overrides


def apply_step07_cli_overrides(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Step 7 Flower / round_loop 共通の CLI 上書き。"""
    cfg = apply_fl_cli_overrides(cfg, args)
    scfg = cfg.setdefault("step07", {})
    if getattr(args, "repertoire", None):
        scfg["repertoire"] = str(args.repertoire)
    if getattr(args, "stage_profile", None):
        scfg["stage_profile"] = str(args.stage_profile)
    if getattr(args, "stage", None) is not None:
        scfg["_cli_stage"] = int(args.stage)
    if getattr(args, "mode", None):
        scfg["_cli_mode"] = str(args.mode)
    if getattr(args, "teacher_source", None):
        scfg["_cli_teacher_source"] = str(args.teacher_source)
    if getattr(args, "aggregation", None):
        scfg.setdefault("aggregation", {})["method"] = str(args.aggregation)
    if getattr(args, "distribution", None):
        scfg["distribution"] = str(args.distribution)
    if getattr(args, "distill_steps", None) is not None:
        scfg["distill_steps"] = int(args.distill_steps)
    if getattr(args, "num_clients", None) is not None:
        scfg["num_clients"] = int(args.num_clients)
    if getattr(args, "client_train_per_class", None) is not None:
        scfg["client_train_per_class"] = int(args.client_train_per_class)
    if getattr(args, "client_eval_per_class", None) is not None:
        scfg["client_eval_per_class"] = int(args.client_eval_per_class)
    if getattr(args, "server_train_per_class", None) is not None:
        scfg["server_train_per_class"] = int(args.server_train_per_class)
    if getattr(args, "server_eval_per_class", None) is not None:
        scfg["server_eval_per_class"] = int(args.server_eval_per_class)
    if getattr(args, "max_client_steps", None) is not None:
        scfg["max_client_steps"] = int(args.max_client_steps)
    if getattr(args, "max_server_steps", None) is not None:
        scfg["max_server_steps"] = int(args.max_server_steps)
    return cfg


def step07_run_dir_name(repertoire_name: str) -> str:
    return f"step07_fl_{repertoire_name}"


def step07_fl_checkpoint_path(cfg: Dict[str, Any], run_id: str, root: Path) -> Path:
    art = dict(cfg.get("artifacts", {}))
    scfg = dict(cfg.get("step07", {}))
    base = Path(str(art.get("checkpoints", "artifacts/checkpoints")))
    sub = str(scfg.get("checkpoint_subdir", "step07_fl"))
    name = str(scfg.get("checkpoint_name", "step07_fedavg_reference.pt"))
    path = base / sub / run_id / name
    if not path.is_absolute():
        path = root / path
    return path
