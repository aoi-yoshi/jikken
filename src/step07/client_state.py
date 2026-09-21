"""クライアント LoRA + 分類ヘッドの退避・復元（1 GPU 直列シミュレーション用）。"""
from __future__ import annotations

from typing import Any, Dict, List

import torch


def snapshot_client_state(model, client_lora: List[torch.nn.Parameter]) -> Dict[str, Any]:
    return {
        "lora": [p.detach().clone() for p in client_lora],
        "classifier": {k: v.detach().clone() for k, v in model.classifier.state_dict().items()},
    }


def load_client_state(model, client_lora: List[torch.nn.Parameter], state: Dict[str, Any]) -> None:
    with torch.no_grad():
        for p, t in zip(client_lora, state["lora"]):
            p.data.copy_(t)
    model.classifier.load_state_dict(state["classifier"])


def fedavg_client_states(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """等重み FedAvg: クライアント LoRA + 分類ヘッドの要素ごと平均。基盤 LoRA は含めない。"""
    n = len(states)
    if n == 1:
        return {
            "lora": [t.clone() for t in states[0]["lora"]],
            "classifier": {k: v.clone() for k, v in states[0]["classifier"].items()},
        }
    avg_lora = []
    for i in range(len(states[0]["lora"])):
        acc = states[0]["lora"][i].float().clone()
        for s in states[1:]:
            acc += s["lora"][i].float()
        avg_lora.append((acc / n).to(states[0]["lora"][i].dtype))
    avg_cls: Dict[str, torch.Tensor] = {}
    for k in states[0]["classifier"]:
        acc = states[0]["classifier"][k].float().clone()
        for s in states[1:]:
            acc += s["classifier"][k].float()
        avg_cls[k] = (acc / n).to(states[0]["classifier"][k].dtype)
    return {"lora": avg_lora, "classifier": avg_cls}
