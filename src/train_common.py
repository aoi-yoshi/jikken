from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

from .dataset_manifest import Sample, load_sample_row


def ensure_sys_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def device_or_auto(prefer_cuda: bool = True) -> str:
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def iter_row_chunks(rows: Sequence[dict], batch_size: int) -> Iterator[List[dict]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(rows), bs):
        yield list(rows[i : i + bs])


def rows_for_step(
    rows: Sequence[dict],
    *,
    step: int,
    batch_size: int,
) -> List[dict]:
    """step 番目の optimizer update に使う batch_size 行を返す（末尾は先頭へ wrap）。"""
    if not rows:
        raise RuntimeError("rows is empty")
    bs = max(1, int(batch_size))
    n = len(rows)
    start = (int(step) * bs) % n
    out: List[dict] = []
    for j in range(bs):
        out.append(rows[(start + j) % n])
    return out


def load_samples(rows: Sequence[dict], dcfg: Dict[str, Any]) -> List[Sample]:
    return [load_sample_row(r, dcfg) for r in rows]
