from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def trainable_state_vector(model: nn.Module) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    tensors: List[torch.Tensor] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".surrogate." in n and "lora_" in n:
            continue
        names.append(n)
        tensors.append(p.detach().cpu().float().reshape(-1))
    if not tensors:
        return [], np.zeros((), dtype=np.float32)
    vec = torch.cat(tensors, dim=0).numpy()
    return names, vec


def vector_to_trainable_state(model: nn.Module, names: List[str], vec: np.ndarray) -> None:
    if not names:
        return
    torch_vec = torch.tensor(vec, dtype=torch.float32)
    offset = 0
    name_to_param = {n: p for n, p in model.named_parameters()}
    for n in names:
        p = name_to_param[n]
        numel = p.numel()
        chunk = torch_vec[offset : offset + numel].view_as(p).to(device=p.device, dtype=p.dtype)
        p.data.copy_(chunk)
        offset += numel


def fedavg_weights(weights: List[Tuple[List[str], np.ndarray]]) -> Tuple[List[str], np.ndarray]:
    if not weights:
        raise ValueError("empty weights")
    names = weights[0][0]
    stacked = np.stack([w for _, w in weights], axis=0)
    avg = np.mean(stacked, axis=0)
    return names, avg
