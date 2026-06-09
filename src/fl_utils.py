from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn


def fl_checkpoint_dir(cfg: Mapping[str, Any], root: Path) -> Path:
    """artifacts/checkpoints/step05_fl/ 等（run_id 別サブフォルダの親）。"""
    art = dict(cfg.get("artifacts", {}))
    fcfg = dict(cfg.get("fl", {}))
    base = Path(str(art.get("checkpoints", "artifacts/checkpoints")))
    sub = str(fcfg.get("checkpoint_subdir", "step05_fl"))
    path = base / sub
    if not path.is_absolute():
        path = root / path
    return path


def fl_checkpoint_path(cfg: Mapping[str, Any], run_id: str, root: Path) -> Path:
    """run ごと: artifacts/checkpoints/step05_fl/<run_id>/step05_fedavg_global.pt"""
    fcfg = dict(cfg.get("fl", {}))
    name = str(fcfg.get("checkpoint_name", "step05_fedavg_global.pt"))
    return fl_checkpoint_dir(cfg, root) / run_id / name


def write_latest_checkpoint_pointer(checkpoint_path: Path, cfg: Mapping[str, Any], root: Path) -> Path:
    ptr = fl_checkpoint_dir(cfg, root) / "LATEST_CHECKPOINT"
    ptr.parent.mkdir(parents=True, exist_ok=True)
    ptr.write_text(str(Path(checkpoint_path).resolve()), encoding="utf-8")
    return ptr


def resolve_latest_checkpoint(cfg: Mapping[str, Any], root: Path) -> Path | None:
    ptr = fl_checkpoint_dir(cfg, root) / "LATEST_CHECKPOINT"
    if ptr.is_file():
        p = Path(ptr.read_text(encoding="utf-8").strip())
        if p.is_file():
            return p
    art = dict(cfg.get("artifacts", {}))
    fcfg = dict(cfg.get("fl", {}))
    legacy = Path(str(art.get("checkpoints", "artifacts/checkpoints"))) / str(
        fcfg.get("checkpoint_name", "step05_fedavg_global.pt")
    )
    if not legacy.is_absolute():
        legacy = root / legacy
    return legacy if legacy.is_file() else None


def trainable_state_vector(model: nn.Module) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    tensors: List[torch.Tensor] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".surrogate." in n and "lora_" in n:
            continue
        names.append(n)
        tensors.append(p.detach().cpu().float().reshape(-1))
    if not tensors:
        return [], np.zeros((), dtype=np.float32)
    vec = torch.cat(tensors, dim=0).numpy()
    return names, vec


def flower_parameters_to_vector(
    parameters: Any,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    """Flower Parameters / FitRes を学習可能ベクトル（1 本の float32）に戻す。"""
    from flwr.common import parameters_to_ndarrays

    params = parameters.parameters if hasattr(parameters, "parameters") else parameters
    ndarrays = parameters_to_ndarrays(params)
    if not ndarrays:
        raise ValueError("empty Flower parameters")
    if len(ndarrays) == 1:
        vec = np.asarray(ndarrays[0], dtype=np.float32).reshape(-1)
    else:
        vec = np.concatenate(
            [np.asarray(a, dtype=np.float32).reshape(-1) for a in ndarrays],
            axis=0,
        )
    if expected_size is not None and vec.size != expected_size:
        raise ValueError(
            f"checkpoint vector size mismatch: got {vec.size}, expected {expected_size}"
        )
    return vec


def vector_to_trainable_state(model: nn.Module, names: List[str], vec: np.ndarray) -> None:
    if not names:
        return
    torch_vec = torch.tensor(vec, dtype=torch.float32)
    offset = 0
    name_to_param = {n: p for n, p in model.named_parameters()}
    for n in names:
        p = name_to_param[n]
        numel = p.numel()
        chunk = torch_vec[offset : offset + numel].view_as(p).to(device=p.device, dtype=p.dtype)
        p.data.copy_(chunk)
        offset += numel


def fedavg_weights(weights: List[Tuple[List[str], np.ndarray]]) -> Tuple[List[str], np.ndarray]:
    if not weights:
        raise ValueError("empty weights")
    names = weights[0][0]
    stacked = np.stack([w for _, w in weights], axis=0)
    avg = np.mean(stacked, axis=0)
    return names, avg


def save_fl_checkpoint(
    path: Path,
    *,
    names: List[str],
    vector: np.ndarray,
    meta: Dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "names": names,
        "vector": vector,
        "meta": meta or {},
    }
    torch.save(payload, path)
    names_path = path.with_suffix(".names.json")
    with names_path.open("w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    return path


def load_fl_checkpoint(path: Path) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    names = list(payload["names"])
    vec = np.asarray(payload["vector"], dtype=np.float32)
    meta = dict(payload.get("meta", {}))
    return names, vec, meta


def build_initial_fl_state(cfg: Dict[str, Any], device: torch.device) -> Tuple[List[str], np.ndarray]:
    """共有初期パラメータ（classifier + client LoRA）を生成する。"""
    from src.metrics import set_seed
    from src.peft_setup import attach_dual_lora
    from src.vl_model import build_model, unfreeze_backbone

    set_seed(int(cfg["train"]["seed"]))
    model, _ = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    names, vec = trainable_state_vector(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return names, vec
