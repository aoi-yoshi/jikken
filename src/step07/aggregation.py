"""Step 7 ③ 集約 — FedAvg 後の重みを作る方式（FedOpt / FedACG 等はここに追加）。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .client_state import fedavg_client_states

AGGREGATION_METHODS = ("fedavg", "fedopt", "fedacg", "fedomg")


def list_aggregation_methods() -> List[str]:
    return list(AGGREGATION_METHODS)


def aggregate_client_states(
    method: str,
    states: List[Dict[str, Any]],
    *,
    client_weights: Optional[Sequence[float]] = None,
    server_momentum: Optional[Dict[str, Any]] = None,
    round_num: int = 1,
) -> Dict[str, Any]:
    """複数クライアント状態 → FedAvg 後の重み（参照 adapter）。

    method:
      fedavg  — 等重み（または client_weights）平均。現行デフォルト。
      fedopt  — Reddi et al. FedAdam 型 pseudo-gradient 集約（未実装）。
      fedacg  — Kim et al. 加速 client gradient 集約（未実装）。
      fedomg  — Nguyen et al. on-server matching gradient 前提の集約（未実装）。

    server_momentum: FedOpt / FedACG 用のサーバ状態（将来）。
    """
    key = str(method).lower().strip()
    if key == "fedavg":
        if client_weights is not None and len(client_weights) == len(states):
            return _weighted_average_states(states, client_weights)
        return fedavg_client_states(states)
    if key == "fedopt":
        raise NotImplementedError(
            "aggregation.method=fedopt は未実装。"
            "Reddi et al. ICLR 2021 のサーバ側 FedAdam を src/step07/aggregation.py に追加する。"
        )
    if key == "fedacg":
        raise NotImplementedError(
            "aggregation.method=fedacg は未実装。"
            "Kim et al. CVPR 2024 のクライアント更新方向集約を src/step07/aggregation.py に追加する。"
        )
    if key == "fedomg":
        raise NotImplementedError(
            "aggregation.method=fedomg は未実装。"
            "FedOMG 型はクライアントモデル差分からサーバが pseudo-gradient を構築する。"
        )
    raise ValueError(f"Unknown aggregation.method={method!r}. Choose from {AGGREGATION_METHODS}")


def _weighted_average_states(
    states: List[Dict[str, Any]],
    weights: Sequence[float],
) -> Dict[str, Any]:
    total_w = sum(float(w) for w in weights)
    if total_w <= 0:
        raise ValueError("client_weights must sum to positive")
    norm = [float(w) / total_w for w in weights]
    out_lora = [states[0]["lora"][i].float().clone() * norm[0] for i in range(len(states[0]["lora"]))]
    for j, s in enumerate(states[1:], start=1):
        for i in range(len(out_lora)):
            out_lora[i] += s["lora"][i].float() * norm[j]
    out_lora = [t.to(states[0]["lora"][i].dtype) for i, t in enumerate(out_lora)]
    out_cls: Dict[str, Any] = {}
    for k in states[0]["classifier"]:
        acc = states[0]["classifier"][k].float().clone() * norm[0]
        for j, s in enumerate(states[1:], start=1):
            acc += s["classifier"][k].float() * norm[j]
        out_cls[k] = acc.to(states[0]["classifier"][k].dtype)
    return {"lora": out_lora, "classifier": out_cls}
