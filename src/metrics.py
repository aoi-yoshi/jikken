from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .dataset_manifest import load_sample_row
from .train_common import iter_row_chunks


@torch.no_grad()
def evaluate_classifier(
    model,
    processor,
    rows: List[dict],
    *,
    device: torch.device,
    prompt: str,
    max_length: int,
    batch_size: int,
    adapter: str | None = "client",
    dcfg: dict | None = None,
) -> Dict[str, float]:
    model.eval()
    if hasattr(model.backbone, "set_adapter") and adapter:
        model.backbone.set_adapter(adapter)
    dcfg = dcfg or {}
    ys: List[int] = []
    preds: List[int] = []
    bs = max(1, int(batch_size))
    for chunk in iter_row_chunks(rows, bs):
        samples = [load_sample_row(r, dcfg) for r in chunk]
        frames_batch = [s.frames for s in samples]
        logits, _ = model.forward_samples(
            processor,
            frames_batch,
            prompt,
            max_length,
            output_hidden_states=True,
        )
        pred = torch.argmax(logits, dim=-1)
        ys.extend(int(s.label) for s in samples)
        preds.extend(int(x) for x in pred.detach().cpu().tolist())
    acc = float(accuracy_score(ys, preds))
    f1 = float(f1_score(ys, preds, average="macro", zero_division=0))
    return {"accuracy": acc, "f1_macro": f1}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
