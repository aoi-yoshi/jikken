"""LoRA / 出力 / 勾配の差分計測ユーティリティ（基盤確認・診断用）。"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptation_losses import grad_vec
from .vl_model import kl_distillation, logits_loss


def lora_inventory(
    peft_model: nn.Module,
    adapter: str,
) -> Dict[str, Any]:
    """アダプタごとの LoRA テンソル名・形状・パラメータ数を列挙。"""
    needle = f".{adapter}."
    tensors: List[Dict[str, Any]] = []
    total = 0
    for name, p in peft_model.named_parameters():
        if "lora_" not in name or needle not in name:
            continue
        n = int(p.numel())
        total += n
        tensors.append(
            {
                "name": name,
                "shape": list(p.shape),
                "numel": n,
                "norm": float(p.detach().float().norm().cpu()),
            }
        )
    return {
        "adapter": adapter,
        "num_tensors": len(tensors),
        "total_params": total,
        "tensors": tensors,
    }


def per_tensor_lora_delta(
    peft_model: nn.Module,
    adapter: str,
    before: Sequence[torch.Tensor],
    *,
    top_k: int = 10,
) -> Tuple[List[Dict[str, Any]], float]:
    """テンソルごとの ||Δ|| を計算し、変化の大きい順に top_k 件返す。"""
    params = []
    names = []
    needle = f".{adapter}."
    for name, p in peft_model.named_parameters():
        if "lora_" in name and needle in name:
            names.append(name)
            params.append(p)
    deltas: List[Dict[str, Any]] = []
    total_sq = 0.0
    for name, p, snap in zip(names, params, before):
        d = (p.detach().float().cpu() - snap.float()).norm().item()
        total_sq += d * d
        deltas.append({"name": name, "delta_norm": d})
    deltas.sort(key=lambda x: x["delta_norm"], reverse=True)
    return deltas[:top_k], total_sq**0.5


def snapshot_lora(params: Sequence[nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().float().cpu().clone() for p in params]


def lora_norm(params: Sequence[nn.Parameter]) -> float:
    sq = 0.0
    for p in params:
        sq += float(p.detach().float().pow(2).sum().cpu())
    return sq**0.5


def lora_delta_norm(
    before: Sequence[torch.Tensor],
    after: Sequence[torch.Tensor],
) -> float:
    sq = 0.0
    for b, a in zip(before, after):
        d = (a - b).float()
        sq += float(d.pow(2).sum())
    return sq**0.5


@torch.no_grad()
def probe_logits(logits: torch.Tensor) -> Dict[str, Any]:
    probs = F.softmax(logits.float(), dim=-1)
    pred = int(torch.argmax(probs, dim=-1).item())
    return {
        "logits": logits.detach().float().cpu().tolist(),
        "probs": probs.detach().cpu().tolist(),
        "pred": pred,
    }


def logits_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    af = a.detach().float()
    bf = b.detach().float()
    l2 = float(torch.norm(af - bf).cpu())
    kl = float(kl_distillation(af, bf, temperature=1.0).cpu())
    mse_prob = float(F.mse_loss(F.softmax(af, -1), F.softmax(bf, -1)).cpu())
    return {"logits_l2": l2, "kl_teacher_a_student_b": kl, "prob_mse": mse_prob}


def grad_probe(
    model,
    processor,
    pil_images,
    label: torch.Tensor,
    *,
    prompt: str,
    max_length: int,
    client_adapter: str,
    surrogate_adapter: str,
    client_params: List[nn.Parameter],
    surrogate_params: List[nn.Parameter],
) -> Dict[str, float]:
    """Client / Surrogate それぞれの LoRA 勾配（L_task）とその類似度。"""
    model.train()
    for p in client_params + surrogate_params:
        p.requires_grad_(True)

    model.backbone.set_adapter(client_adapter)
    logits_c, _ = model.forward_batch(
        processor, pil_images, prompt, max_length, output_hidden_states=True
    )
    loss_c = logits_loss(logits_c.float(), label)
    g_c = grad_vec(loss_c, client_params).detach().float()

    model.backbone.set_adapter(surrogate_adapter)
    logits_s, _ = model.forward_batch(
        processor, pil_images, prompt, max_length, output_hidden_states=True
    )
    loss_s = logits_loss(logits_s.float(), label)
    g_s = grad_vec(loss_s, surrogate_params).detach().float()

    model.zero_grad(set_to_none=True)

    cos = float(
        F.cosine_similarity(g_c.unsqueeze(0), g_s.unsqueeze(0), dim=1).cpu().item()
    )
    return {
        "loss_task_client": float(loss_c.detach().cpu()),
        "loss_task_surrogate": float(loss_s.detach().cpu()),
        "grad_norm_client": float(g_c.norm().cpu()),
        "grad_norm_surrogate": float(g_s.norm().cpu()),
        "grad_l2_diff": float((g_c - g_s).norm().cpu()),
        "grad_cosine": cos,
    }
