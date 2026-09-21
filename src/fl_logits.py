"""Step 7 Stage 1+: logits の export・集約・蒸留配布（FedMD / FedDF 型）。

- export_client_logits: クライアント→サーバの logits 送信（Stage 1 の Lpred teacher）
- distill_adapter_from_logits: サーバ→クライアントの蒸留配布
  （更新した基盤 LoRA の logits を teacher に、クライアント LoRA + 分類ヘッドを KL で更新。
   LoRA 形状に依存しないため 7B 基盤モデル → 3B クライアントでも動く）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.amp import autocast

from .dataset_manifest import load_sample_row
from .vl_model import kl_distillation


@torch.no_grad()
def export_client_logits(
    model,
    processor,
    rows: List[dict],
    dcfg: dict,
    *,
    adapter: str,
    prompt: str,
    max_length: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """整合用サンプル上で eval forward し {sample_id: logits[ num_classes ]} を返す。"""
    model.eval()
    model.backbone.set_adapter(adapter)
    out: Dict[str, List[float]] = {}
    for r in rows:
        sample = load_sample_row(r, dcfg)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits, _ = model.forward_batch(
                processor,
                sample.frames,
                prompt,
                max_length,
                output_hidden_states=True,
            )
        vec = logits.detach().float().cpu().tolist()
        if isinstance(vec[0], list):
            out[str(r["id"])] = vec[0]
        else:
            out[str(r["id"])] = vec
    model.train()
    return out


def save_client_logits_artifact(
    run_dir: Path,
    round_num: int,
    logits_map: Dict[str, List[float]],
    *,
    name: str = "client_logits",
) -> Path:
    sample_ids = list(logits_map.keys())
    payload = {
        "round": round_num,
        "sample_ids": sample_ids,
        "logits": [logits_map[sid] for sid in sample_ids],
    }
    path = run_dir / f"{name}_r{round_num}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_client_logits_artifact(path: Path) -> Dict[str, List[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        sid: logits
        for sid, logits in zip(payload["sample_ids"], payload["logits"])
    }


def aggregate_client_logits(
    client_maps: Sequence[Dict[str, List[float]]],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, List[float]]:
    """sample_id ごとに softmax 確率を重み付き平均し、log 領域の pseudo-logits として返す。"""
    if not client_maps:
        return {}
    n = len(client_maps)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

    all_ids = set()
    for m in client_maps:
        all_ids.update(m.keys())

    out: Dict[str, List[float]] = {}
    for sid in sorted(all_ids):
        probs_acc: Optional[torch.Tensor] = None
        for w, m in zip(weights, client_maps):
            if sid not in m:
                continue
            p = F.softmax(torch.tensor(m[sid], dtype=torch.float32), dim=-1)
            probs_acc = p * w if probs_acc is None else probs_acc + p * w
        if probs_acc is None:
            continue
        probs_acc = probs_acc.clamp(min=1e-8)
        pseudo = (probs_acc.log()).tolist()
        out[sid] = pseudo
    return out


def logits_map_to_tensor(
    logits_map: Dict[str, List[float]],
    sample_id: str,
    device: torch.device,
) -> torch.Tensor:
    """1 サンプル分の logits を (1, num_classes) テンソルに。"""
    vec = logits_map[sample_id]
    return torch.tensor([vec], device=device, dtype=torch.float32)


def distill_adapter_from_logits(
    model,
    processor,
    rows: List[dict],
    dcfg: dict,
    *,
    teacher_logits_map: Dict[str, List[float]],
    adapter: str,
    trainable_params: List[torch.nn.Parameter],
    prompt: str,
    max_length: int,
    device: torch.device,
    lr: float,
    max_steps: int | None = None,
    temperature: float = 2.0,
) -> Dict[str, float]:
    """蒸留配布: teacher logits（更新済み基盤 LoRA の出力）へ KL で adapter + ヘッドを寄せる。

    FedMD の Digest / FedDF のサーバ蒸留と同型。重みを直接コピーしないため、
    基盤モデル（7B）とクライアントモデル（3B）で LoRA 形状が違ってもエラーにならない。
    rows は公開整合セット（server_train）。teacher_logits_map に無い ID はスキップ。
    """
    model.train()
    model.backbone.set_adapter(adapter)
    for p in trainable_params:
        p.requires_grad = True
    opt = torch.optim.AdamW(trainable_params, lr=lr)
    steps = 0
    last_kl = 0.0
    for r in rows:
        if max_steps is not None and steps >= max_steps:
            break
        sid = str(r["id"])
        if sid not in teacher_logits_map:
            continue
        sample = load_sample_row(r, dcfg)
        teacher = logits_map_to_tensor(teacher_logits_map, sid, device)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits, _ = model.forward_batch(
                processor,
                sample.frames,
                prompt,
                max_length,
                output_hidden_states=True,
            )
            loss = kl_distillation(logits.float(), teacher, temperature)
        loss.backward()
        opt.step()
        last_kl = float(loss.detach().cpu())
        steps += 1
    return {"distill_steps": float(steps), "distill_kl_last": last_kl}
