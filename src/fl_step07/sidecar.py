"""Step 7 Flower — logits 等の sidecar ファイルパス。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def sidecar_root(run_dir: Path) -> Path:
    return run_dir / "sidecar"


def client_logits_path(run_dir: Path, server_round: int, client_id: int) -> Path:
    return sidecar_root(run_dir) / "client_logits" / f"r{server_round}_c{client_id}.json"


def server_logits_path(run_dir: Path, server_round: int) -> Path:
    return sidecar_root(run_dir) / "server_logits" / f"r{server_round}.json"


def aggregated_client_logits_path(run_dir: Path, server_round: int) -> Path:
    return sidecar_root(run_dir) / "aggregated_client_logits" / f"r{server_round}.json"


def client_grads_path(run_dir: Path, server_round: int, client_id: int) -> Path:
    return sidecar_root(run_dir) / "client_grads" / f"r{server_round}_c{client_id}.pt"


def client_post_logits_path(run_dir: Path, server_round: int, client_id: int) -> Path:
    return sidecar_root(run_dir) / "client_post_logits" / f"r{server_round}_c{client_id}.json"


def foundation_checkpoint_path(checkpoint_dir: Path, run_id: str, server_round: int) -> Path:
    return checkpoint_dir / run_id / f"foundation_r{server_round}.pt"


def write_logits_sidecar(path: Path, round_num: int, logits_map: Dict[str, List[float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = list(logits_map.keys())
    payload = {
        "round": round_num,
        "sample_ids": sample_ids,
        "logits": [logits_map[sid] for sid in sample_ids],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
