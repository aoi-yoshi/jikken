"""Step 7 送信物 — 勾配ベクトル / 1-step 適応後 logits の export・集約・通信コスト。

logits の export・集約は src/fl_logits.py（既存）。
本ファイルは Lgrad / Lpost 用の追加送信物を扱う:
  - grad_vectors: server_train 上の L_task をクライアント LoRA で微分した flat ベクトル
    （サーバは同じ定義で基盤 LoRA の勾配を計算し MSE / cosine で整合 → Lgrad）
  - post_logits: server_train 各サンプルで 1-step SGD 適応した直後の logits
    （サーバは transmitted post_logits を teacher に KL → Lpost）
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.amp import autocast

from src.adaptation_losses import copy_params, sgd_step_inplace, snapshot_params
from src.dataset_manifest import load_sample_row
from src.vl_model import logits_loss


def export_client_grad_vector(
    model,
    processor,
    rows: List[dict],
    dcfg: dict,
    *,
    adapter: str,
    lora_params: List[nn.Parameter],
    prompt: str,
    max_length: int,
    device: torch.device,
    max_samples: int | None = None,
) -> torch.Tensor:
    """server_train 上の L_task 勾配（クライアント LoRA について）をサンプル平均した flat ベクトル。

    サーバ側 Lgrad は同じ ℓ = L_task を基盤 LoRA で微分して比較する（定義を揃える）。
    """
    model.train()
    model.backbone.set_adapter(adapter)
    for p in lora_params:
        p.requires_grad = True
    acc: torch.Tensor | None = None
    n = 0
    for r in rows:
        if max_samples is not None and n >= max_samples:
            break
        sample = load_sample_row(r, dcfg)
        y = torch.tensor([sample.label], device=device, dtype=torch.long)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits, _ = model.forward_batch(
                processor, sample.frames, prompt, max_length, output_hidden_states=True
            )
            loss = logits_loss(logits.float(), y)
        grads = torch.autograd.grad(loss, lora_params, allow_unused=True, retain_graph=False)
        parts = []
        for p, g in zip(lora_params, grads):
            parts.append(
                g.detach().float().reshape(-1) if g is not None
                else torch.zeros(p.numel(), device=device, dtype=torch.float32)
            )
        vec = torch.cat(parts, dim=0)
        acc = vec if acc is None else acc + vec
        n += 1
    if acc is None:
        raise RuntimeError("export_client_grad_vector: no samples")
    return (acc / n).cpu()


@torch.no_grad()
def _forward_logits(model, processor, frames, prompt, max_length, device) -> torch.Tensor:
    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        logits, _ = model.forward_batch(
            processor, frames, prompt, max_length, output_hidden_states=True
        )
    return logits.detach().float()


def export_client_post_logits(
    model,
    processor,
    rows: List[dict],
    dcfg: dict,
    *,
    adapter: str,
    lora_params: List[nn.Parameter],
    inner_lr: float,
    prompt: str,
    max_length: int,
    device: torch.device,
    max_samples: int | None = None,
) -> Dict[str, List[float]]:
    """server_train 各サンプルで 1-step SGD 適応 → forward → 復元。{sample_id: logits} を返す。"""
    model.train()
    model.backbone.set_adapter(adapter)
    for p in lora_params:
        p.requires_grad = True
    out: Dict[str, List[float]] = {}
    n = 0
    for r in rows:
        if max_samples is not None and n >= max_samples:
            break
        sample = load_sample_row(r, dcfg)
        y = torch.tensor([sample.label], device=device, dtype=torch.long)
        snap = snapshot_params(lora_params)
        with torch.enable_grad():
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits0, _ = model.forward_batch(
                    processor, sample.frames, prompt, max_length, output_hidden_states=True
                )
                loss = logits_loss(logits0.float(), y)
            grads = torch.autograd.grad(loss, lora_params, allow_unused=True, retain_graph=False)
        with torch.no_grad():
            sgd_step_inplace(lora_params, list(grads), inner_lr)
            logits1 = _forward_logits(model, processor, sample.frames, prompt, max_length, device)
            copy_params(lora_params, snap)
        vec = logits1.cpu().tolist()
        out[str(r["id"])] = vec[0] if isinstance(vec[0], list) else vec
        n += 1
    return out


def save_grad_vector_sidecar(path: Path, round_num: int, client_id: int, vec: torch.Tensor) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "round": round_num,
            "client_id": client_id,
            "numel": int(vec.numel()),
            "vec": vec.to(torch.float16),
        },
        path,
    )
    return path


def load_grad_vector_sidecar(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["vec"].float()


def aggregate_grad_vectors(
    vecs: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """クライアント勾配ベクトルの（重み付き）平均。"""
    if not vecs:
        raise ValueError("empty grad vectors")
    n = len(vecs)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(float(w) for w in weights)
        weights = [float(w) / total for w in weights]
    acc = vecs[0].float() * weights[0]
    for w, v in zip(weights[1:], vecs[1:]):
        if v.numel() != acc.numel():
            raise ValueError(f"grad vector numel mismatch: {v.numel()} vs {acc.numel()}")
        acc = acc + v.float() * w
    return acc


def sidecar_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0
