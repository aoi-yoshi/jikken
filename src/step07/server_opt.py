"""Step 7 サーバ optimizer — 基盤 LoRA 更新の optimizer をレパートリーから選択。

adamw   — 既定（本プロジェクトの標準）
fedadam — FedOpt [1]（Reddi et al., ICLR 2021）の FedAdam をサーバ整合の optimizer に再解釈。
          論文値: β1=0.9, β2=0.99, τ=1e-3（Adam の eps に対応）
momentum — FedACG [5]（Kim et al., CVPR 2024）の momentum 加速（λ=0.85）を
           SGD + Nesterov momentum として再解釈
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

SERVER_OPTIMIZERS = ("adamw", "fedadam", "momentum")


def build_server_optimizer(
    name: str,
    params: List[nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    key = str(name).lower().strip()
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "fedadam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99), eps=1e-3, weight_decay=0.0)
    if key == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.85, nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"Unknown server_optimizer={name!r}. Choose from {SERVER_OPTIMIZERS}")
