from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def attach_dual_lora(backbone: nn.Module, cfg: dict, *, client: str, surrogate: str) -> nn.Module:
    lcfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["lora_alpha"]),
        lora_dropout=float(cfg["lora"]["lora_dropout"]),
        target_modules=list(cfg["lora"]["target_modules"]),
        bias=str(cfg["lora"].get("bias", "none")),
    )
    peft_model = get_peft_model(backbone, lcfg, adapter_name=client)
    peft_model.add_adapter(surrogate, lcfg)
    peft_model.set_adapter(client)
    return peft_model


def ensure_client_trainable(model: nn.Module, peft_model: nn.Module, *, client: str = "client") -> None:
    """PEFT の set_adapter 後も client LoRA + 分類頭が学習対象になるよう requires_grad を立てる。"""
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    for p in lora_params_for_adapter(peft_model, client):
        p.requires_grad_(True)


def lora_params_for_adapter(peft_model: nn.Module, adapter: str) -> List[nn.Parameter]:
    out: List[nn.Parameter] = []
    needle = f".{adapter}."
    for name, p in peft_model.named_parameters():
        if "lora_" in name and needle in name:
            out.append(p)
    if not out:
        raise RuntimeError(
            f"No LoRA tensors matched for adapter={adapter}. "
            "Check adapter names passed to attach_dual_lora()."
        )
    return out


def sync_surrogate_from_client(peft_model: nn.Module, *, client: str, surrogate: str) -> None:
    c = lora_params_for_adapter(peft_model, client)
    s = lora_params_for_adapter(peft_model, surrogate)
    if len(c) != len(s):
        raise RuntimeError("Adapter parameter count mismatch.")
    with torch.no_grad():
        for pc, ps in zip(c, s):
            ps.copy_(pc)


@torch.no_grad()
def ema_surrogate_from_client(
    client_params: List[nn.Parameter],
    surrogate_params: List[nn.Parameter],
    alpha: float,
) -> None:
    """surrogate <- alpha * surrogate + (1 - alpha) * client.

    alpha=1.0 で静止（更新しない）、alpha<1 で client の方へ少し動く。
    Ablation の Lpred / Lgrad を非自明にするため毎ステップ呼ぶ想定。
    """
    if alpha >= 1.0:
        return
    a = float(alpha)
    one_minus_a = 1.0 - a
    for pc, ps in zip(client_params, surrogate_params):
        ps.data.mul_(a).add_(pc.data, alpha=one_minus_a)
