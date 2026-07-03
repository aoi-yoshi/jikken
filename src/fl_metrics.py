"""Flower FL 用の評価指標（step04b の surrogate=基盤 F に合わせる）。"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from .diff_probe import lora_delta_norm, snapshot_lora
from .fl_utils import trainable_state_vector
from .metrics import evaluate_classifier
from .peft_setup import lora_params_for_adapter


def client_trainable_vector(model, client_lora_params: Sequence[nn.Parameter]) -> np.ndarray:
    """分類頭 + client LoRA を明示的に連結（PEFT 初期化直後も次元が一致する）。"""
    tensors = [
        p.detach().cpu().float().reshape(-1)
        for p in list(model.classifier.parameters()) + list(client_lora_params)
    ]
    if not tensors:
        return np.zeros((), dtype=np.float32)
    return torch.cat(tensors, dim=0).numpy()


def state_vector_l2_delta(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"vector size mismatch: {a.size} vs {b.size}")
    return float(np.linalg.norm(a - b))


def restore_lora_snapshot(params: Sequence[nn.Parameter], snapshot: Sequence[torch.Tensor]) -> None:
    for p, s in zip(params, snapshot):
        p.data.copy_(s.to(device=p.device, dtype=p.dtype))


@torch.no_grad()
def evaluate_foundation_generalization(
    model,
    processor,
    eval_rows: List[dict],
    *,
    device: torch.device,
    prompt: str,
    max_length: int,
    batch_size: int,
    foundation_lora_snap: Sequence[torch.Tensor],
    dcfg: dict,
) -> Dict[str, float]:
    """凍結した surrogate（基盤 F）LoRA で eval セット精度を測る。"""
    surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
    restore_lora_snapshot(surrogate_lora, foundation_lora_snap)
    return evaluate_classifier(
        model,
        processor,
        eval_rows,
        device=device,
        prompt=prompt,
        max_length=max_length,
        batch_size=batch_size,
        adapter="surrogate",
        dcfg=dcfg,
    )


def compute_weight_divergence(
    *,
    model,
    client_lora_params: Sequence[nn.Parameter],
    foundation_vec: np.ndarray,
    foundation_lora_snap: Sequence[torch.Tensor],
) -> Dict[str, float]:
    """基盤（初期 F）とローカル（client 適応後）の重み乖離。"""
    local_vec = client_trainable_vector(model, client_lora_params)
    client_snap = snapshot_lora(client_lora_params)
    return {
        "weight_divergence_trainable": state_vector_l2_delta(local_vec, foundation_vec),
        "weight_divergence_lora": lora_delta_norm(foundation_lora_snap, client_snap),
    }


def proximal_penalty(model, global_vec: np.ndarray, mu: float, device: torch.device) -> torch.Tensor:
    """FedProx: (mu/2) * ||w - w_global||^2"""
    _, current = trainable_state_vector(model)
    g = torch.tensor(global_vec, device=device, dtype=torch.float32)
    c = torch.tensor(current, device=device, dtype=torch.float32)
    return (float(mu) / 2.0) * torch.sum((c - g) ** 2)


def build_round_timing(
    *,
    apply_global_params_sec: float,
    post_redistribution_eval_sec: float,
    fit_duration_sec: float,
    local_adaptation_eval_sec: float,
    foundation_generalization_eval_sec: float,
    weight_divergence_sec: float,
    round_duration_sec: float,
) -> Dict[str, float]:
    """1 ラウンドの所要時間をフェーズ別に分解。"""
    parts = {
        "apply_global_params_sec": float(apply_global_params_sec),
        "post_redistribution_eval_sec": float(post_redistribution_eval_sec),
        "fit_duration_sec": float(fit_duration_sec),
        "local_adaptation_eval_sec": float(local_adaptation_eval_sec),
        "foundation_generalization_eval_sec": float(foundation_generalization_eval_sec),
        "weight_divergence_sec": float(weight_divergence_sec),
        "round_duration_sec": float(round_duration_sec),
    }
    accounted = sum(v for k, v in parts.items() if k != "round_duration_sec")
    parts["round_overhead_sec"] = float(max(0.0, round_duration_sec - accounted))
    return parts


def build_round_metric_record(
    *,
    server_round: int,
    cid: int,
    post_redistribution: Dict[str, float],
    local_adaptation: Dict[str, float],
    foundation_generalization: Dict[str, float],
    weight_divergence: Dict[str, float],
    round_timing: Dict[str, float],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "event": "fl_round_metrics",
        "server_round": server_round,
        "cid": cid,
        "post_redistribution_accuracy": float(post_redistribution["accuracy"]),
        "post_redistribution_f1_macro": float(post_redistribution["f1_macro"]),
        "local_adaptation_accuracy": float(local_adaptation["accuracy"]),
        "local_adaptation_f1_macro": float(local_adaptation["f1_macro"]),
        "foundation_generalization_accuracy": float(foundation_generalization["accuracy"]),
        "foundation_generalization_f1_macro": float(foundation_generalization["f1_macro"]),
        **weight_divergence,
        "round_timing": dict(round_timing),
        **round_timing,
    }
    if extra:
        row.update(extra)
    return row


def flower_fit_metrics_from_record(record: Dict[str, Any]) -> Dict[str, float]:
    """Flower fit() の第3戻り値（サーバ集約用）。"""
    keys = (
        "post_redistribution_accuracy",
        "local_adaptation_accuracy",
        "foundation_generalization_accuracy",
        "weight_divergence_trainable",
        "weight_divergence_lora",
        "round_duration_sec",
        "apply_global_params_sec",
        "post_redistribution_eval_sec",
        "fit_duration_sec",
        "local_adaptation_eval_sec",
        "foundation_generalization_eval_sec",
        "weight_divergence_sec",
        "round_overhead_sec",
        "avg_train_loss",
        "avg_loss_task",
        "avg_loss_pred",
        "avg_loss_grad",
        "avg_loss_post",
    )
    return {k: float(record[k]) for k in keys if k in record}
