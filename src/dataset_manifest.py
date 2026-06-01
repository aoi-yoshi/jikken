from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class Sample:
    sample_id: str
    frames: List[Image.Image]
    label: int
    source: str


def write_manifest(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_manifest(rows: Sequence[Dict[str, Any]], dcfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """manifest 行を data 設定（weather 等）で絞り込む。"""
    weather = str(dcfg.get("weather_filter", "") or "").strip()
    if not weather:
        return list(rows)
    return [
        r
        for r in rows
        if str(r.get("weather", "") or "").strip() == weather
    ]


def read_manifest_filtered(path: Path, dcfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = read_manifest(path)
    return filter_manifest(rows, dcfg)


def frame_paths_for_model(row: Mapping[str, Any], dcfg: Mapping[str, Any]) -> List[str]:
    """manifest の frame_paths からモデル入力用に末尾/先頭 N 枚を選ぶ。"""
    paths = list(row["frame_paths"])
    use_n = int(dcfg.get("use_num_frames") or 0)
    if use_n <= 0 or use_n >= len(paths):
        return paths
    mode = str(dcfg.get("use_frame_slice", "last")).strip().lower()
    if mode == "last":
        return paths[-use_n:]
    if mode == "first":
        return paths[:use_n]
    raise ValueError(f"Unknown use_frame_slice: {mode!r} (expected 'last' or 'first')")


def split_clients(
    rows: Sequence[Dict[str, Any]], seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    mid = len(idx) // 2
    a = [rows[i] for i in idx[:mid]]
    b = [rows[i] for i in idx[mid:]]
    return a, b


def load_sample_row(row: Dict[str, Any], dcfg: Mapping[str, Any] | None = None) -> Sample:
    dcfg = dcfg or {}
    paths = frame_paths_for_model(row, dcfg)
    frames = [Image.open(p).convert("RGB") for p in paths]
    return Sample(
        sample_id=str(row["id"]),
        frames=frames,
        label=int(row["label"]),
        source=str(row.get("source", "")),
    )


def stratified_subset(
    rows: List[Dict[str, Any]],
    max_per_class: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if max_per_class is None:
        return rows
    rng = random.Random(seed)
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(int(r["label"]), []).append(r)
    out: List[Dict[str, Any]] = []
    for _, items in sorted(buckets.items()):
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    rng.shuffle(out)
    return out
