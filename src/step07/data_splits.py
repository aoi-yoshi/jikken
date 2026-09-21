"""Step 7 データ分割 — クライアント train / eval と server_train / server_eval。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.dataset_manifest import read_manifest_filtered


def split_step07_data_pools(
    rows: List[dict],
    *,
    num_clients: int,
    client_train_per_class: int,
    client_eval_per_class: int,
    server_train_per_class: int,
    server_eval_per_class: int,
    seed: int,
) -> Tuple[List[List[dict]], List[dict], List[dict], List[dict]]:
    """クラス別に (クライアント別 train × N, client_eval, server_train, server_eval) へ分割（ID 重複なし）。"""
    rng = random.Random(seed)
    buckets: Dict[int, List[dict]] = {}
    for r in rows:
        buckets.setdefault(int(r["label"]), []).append(r)

    client_trains: List[List[dict]] = [[] for _ in range(num_clients)]
    client_eval: List[dict] = []
    server_train: List[dict] = []
    server_eval: List[dict] = []

    for label in sorted(buckets.keys()):
        items = list(buckets[label])
        rng.shuffle(items)
        need = (
            num_clients * client_train_per_class
            + client_eval_per_class
            + server_train_per_class
            + server_eval_per_class
        )
        if len(items) < need:
            raise RuntimeError(
                f"class {label} has {len(items)} samples; need {need} "
                f"(num_clients={num_clients}, client train/eval={client_train_per_class}/{client_eval_per_class}, "
                f"server train/eval={server_train_per_class}/{server_eval_per_class})"
            )
        i = 0
        for k in range(num_clients):
            client_trains[k].extend(items[i : i + client_train_per_class])
            i += client_train_per_class
        client_eval.extend(items[i : i + client_eval_per_class])
        i += client_eval_per_class
        server_train.extend(items[i : i + server_train_per_class])
        i += server_train_per_class
        server_eval.extend(items[i : i + server_eval_per_class])

    for pool in client_trains:
        rng.shuffle(pool)
    for pool in (client_eval, server_train, server_eval):
        rng.shuffle(pool)
    return client_trains, client_eval, server_train, server_eval


def split_paired_step07_data_pools(
    rows: List[dict],
    *,
    num_clients: int,
    client_train_pairs: int,
    client_eval_pairs: int,
    server_train_pairs: int,
    server_eval_pairs: int,
    seed: int,
) -> Tuple[List[List[dict]], List[dict], List[dict], List[dict]]:
    """risk/normal pair を壊さずに各poolへ割り当てる。各pairはlabel 0/1を1件ずつ含む。"""
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError("paired split requires pair_id on every row")
        grouped.setdefault(pair_id, []).append(row)
    pairs: List[List[dict]] = []
    for pair_id, members in grouped.items():
        labels = sorted(int(row["label"]) for row in members)
        if labels != [0, 1]:
            raise ValueError(f"pair {pair_id} must contain one normal and one risk row, got {labels}")
        times = {str(row.get("frame_times_sec", "")) for row in members}
        if len(times) != 1:
            raise ValueError(f"pair {pair_id} does not share identical absolute frame times")
        pairs.append(sorted(members, key=lambda row: int(row["label"])))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    need = (
        num_clients * client_train_pairs
        + client_eval_pairs
        + server_train_pairs
        + server_eval_pairs
    )
    if len(pairs) < need:
        raise RuntimeError(f"paired manifest has {len(pairs)} pairs; need {need}")

    cursor = 0
    client_trains: List[List[dict]] = []
    for _ in range(num_clients):
        selected = pairs[cursor : cursor + client_train_pairs]
        cursor += client_train_pairs
        client_trains.append([row for pair in selected for row in pair])
    client_eval = [
        row for pair in pairs[cursor : cursor + client_eval_pairs] for row in pair
    ]
    cursor += client_eval_pairs
    server_train = [
        row for pair in pairs[cursor : cursor + server_train_pairs] for row in pair
    ]
    cursor += server_train_pairs
    server_eval = [
        row for pair in pairs[cursor : cursor + server_eval_pairs] for row in pair
    ]
    for pool in client_trains:
        rng.shuffle(pool)
    for pool in (client_eval, server_train, server_eval):
        rng.shuffle(pool)
    return client_trains, client_eval, server_train, server_eval


def save_data_splits_artifact(
    run_dir: Path,
    *,
    client_trains: List[List[dict]],
    client_eval: List[dict],
    server_train: List[dict],
    server_eval: List[dict],
    seed: int,
) -> Path:
    payload = {
        "seed": seed,
        "client_train_ids": {
            f"client_{k}": [str(r["id"]) for r in rows] for k, rows in enumerate(client_trains)
        },
        "client_eval_ids": [str(r["id"]) for r in client_eval],
        "server_train_ids": [str(r["id"]) for r in server_train],
        "server_eval_ids": [str(r["id"]) for r in server_eval],
        "counts": {
            "num_clients": len(client_trains),
            "client_train_per_client": [len(rows) for rows in client_trains],
            "client_eval": len(client_eval),
            "server_train": len(server_train),
            "server_eval": len(server_eval),
        },
    }
    path = run_dir / "data_splits.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_data_splits_artifact(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_rows_by_ids(all_rows: List[dict], ids: List[str]) -> List[dict]:
    by_id = {str(r["id"]): r for r in all_rows}
    missing = [sid for sid in ids if sid not in by_id]
    if missing:
        raise KeyError(f"sample ids not in manifest: {missing[:5]}")
    return [by_id[sid] for sid in ids]


def load_step07_pools_from_artifact(
    artifact_path: Path,
    *,
    manifest_path: Path,
    dcfg: dict,
    client_id: int | None = None,
) -> Tuple[List[dict] | List[List[dict]], List[dict], List[dict], List[dict], dict]:
    """data_splits.json と manifest から行リストを復元。client_id 指定時はそのクライアント train のみ。"""
    payload = load_data_splits_artifact(artifact_path)
    rows = read_manifest_filtered(manifest_path, dcfg)
    client_trains = [
        resolve_rows_by_ids(rows, payload["client_train_ids"][f"client_{k}"])
        for k in range(payload["counts"]["num_clients"])
    ]
    client_eval = resolve_rows_by_ids(rows, payload["client_eval_ids"])
    server_train = resolve_rows_by_ids(rows, payload["server_train_ids"])
    server_eval = resolve_rows_by_ids(rows, payload["server_eval_ids"])
    if client_id is not None:
        return client_trains[client_id], client_eval, server_train, server_eval, payload
    return client_trains, client_eval, server_train, server_eval, payload
