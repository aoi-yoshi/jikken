"""Step 7 パイプライン判定 — ローカル 3B/3B でも GCP 7B/3B と同じ経路を使う。"""
from __future__ import annotations

from typing import Any, Dict, Tuple


def resolve_step07_model_ids(cfg: Dict[str, Any]) -> Tuple[str, str]:
    """基盤モデル（サーバ）と surrogate モデル（クライアント）の HF model id。"""
    mcfg = cfg.get("model", {})
    base = str(mcfg.get("id", ""))
    server_id = str(mcfg.get("server_id") or base)
    client_id = str(mcfg.get("client_id") or base)
    return server_id, client_id


def model_ids_differ(server_model_id: str, client_model_id: str) -> bool:
    return server_model_id != client_model_id


def distill_pipeline_mode(cfg: Dict[str, Any]) -> bool:
    """True = FedAvg → 整合（送信 logits 等）→ 基盤 LoRA 更新 → distill 配布の統一経路。

    step07.distill_pipeline が未指定のときは従来どおり model id の不一致で判定。
    既定 True により、ローカル 3B/3B も GCP 7B/3B と同じ設定で動かす。
    """
    scfg = cfg.get("step07", {})
    if "distill_pipeline" in scfg:
        return bool(scfg["distill_pipeline"])
    server_id, client_id = resolve_step07_model_ids(cfg)
    return model_ids_differ(server_id, client_id)
