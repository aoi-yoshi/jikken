"""Step 7 ⑤ 配布 — 更新した基盤 LoRA を各クライアントへ渡す。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.fl_logits import (
    distill_adapter_from_logits,
    export_client_logits,
    save_client_logits_artifact,
)

from .client_state import load_client_state, snapshot_client_state


def distribute_foundation_to_clients(
    *,
    method: str,
    client_model,
    client_processor,
    server_model,
    server_processor,
    client_lora: List[nn.Parameter],
    surrogate_lora: List[nn.Parameter],
    client_states: List[Dict[str, Any]],
    fedavg_state: Dict[str, Any],
    server_train: List[dict],
    dcfg: dict,
    prompt: str,
    max_length: int,
    device: torch.device,
    distill_lr: float,
    distill_steps: int | None,
    temperature: float,
    run_dir: Path,
    rnd: int,
) -> Dict[str, Any]:
    """更新した基盤 LoRA を各クライアント状態へ配布する。

    copy:    基盤 LoRA の重み + サーバ側分類ヘッドをクライアント LoRA + ヘッドに書き込む（同種のみ）。
    distill: server_train 上の基盤 LoRA logits を teacher に KL 蒸留（FedMD Digest / FedDF 型・7B/3B 対応）。
    """
    info: Dict[str, Any] = {"method": method}
    if method == "copy":
        surrogate_tensors = [p.detach().clone() for p in surrogate_lora]
        if len(surrogate_tensors) != len(client_states[0]["lora"]) or any(
            s.shape != c.shape for s, c in zip(surrogate_tensors, client_states[0]["lora"])
        ):
            raise RuntimeError(
                "copy 配布は基盤 LoRA とクライアント LoRA が同形状の場合のみ。"
                "異種モデル（7B/3B）では distribution=distill。"
            )
        server_head = {k: v.detach().clone() for k, v in server_model.classifier.state_dict().items()}
        for k in range(len(client_states)):
            client_states[k] = {
                "lora": [t.clone() for t in surrogate_tensors],
                "classifier": {n: v.clone() for n, v in server_head.items()},
            }
        return info

    if method != "distill":
        raise ValueError(f"unknown distribution method: {method!r}")

    server_logits = export_client_logits(
        server_model,
        server_processor,
        server_train,
        dcfg,
        adapter="surrogate",
        prompt=prompt,
        max_length=max_length,
        device=device,
    )
    path = save_client_logits_artifact(run_dir, rnd, server_logits, name="server_logits")
    info["server_logits_path"] = str(path)

    load_client_state(client_model, client_lora, fedavg_state)
    stats = distill_adapter_from_logits(
        client_model,
        client_processor,
        server_train,
        dcfg,
        teacher_logits_map=server_logits,
        adapter="client",
        trainable_params=list(client_model.classifier.parameters()) + client_lora,
        prompt=prompt,
        max_length=max_length,
        device=device,
        lr=distill_lr,
        max_steps=distill_steps,
        temperature=temperature,
    )
    distilled = snapshot_client_state(client_model, client_lora)
    for k in range(len(client_states)):
        client_states[k] = {
            "lora": [t.clone() for t in distilled["lora"]],
            "classifier": {n: v.clone() for n, v in distilled["classifier"].items()},
        }
    info["distill_stats"] = stats
    return info
