"""Flower 用データプール構築とクライアント分割（IID / Non-IID）。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dataset_manifest import read_manifest_filtered, split_train_eval_rows, stratified_subset


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[int, int]:
    c: Counter[int] = Counter()
    for r in rows:
        c[int(r["label"])] += 1
    return dict(sorted(c.items()))


def _key_counts(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        c[str(r.get(key, "") or "unknown")] += 1
    return dict(c.most_common())


def split_clients_iid(rows: Sequence[Dict[str, Any]], num_clients: int, seed: int) -> List[List[Dict[str, Any]]]:
    """ランダムシャッフル後にほぼ均等分割（従来の IID）。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    splits = np.array_split(idx, num_clients)
    return [[rows[int(i)] for i in part] for part in splits]


def split_clients_label_skew_dirichlet(
    rows: Sequence[Dict[str, Any]],
    num_clients: int,
    seed: int,
    alpha: float,
) -> List[List[Dict[str, Any]]]:
    """クラスごとに Dirichlet 比率でクライアントへ割当（alpha が小さいほど Non-IID）。"""
    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[int(r["label"])].append(i)

    client_idx: List[List[int]] = [[] for _ in range(num_clients)]
    for _label, indices in sorted(buckets.items()):
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        if n == 0:
            continue
        props = rng.dirichlet([max(alpha, 1e-3)] * num_clients)
        counts = (props * n).astype(int)
        diff = n - int(counts.sum())
        for j in range(abs(diff)):
            counts[j % num_clients] += 1 if diff > 0 else -1
        counts = np.maximum(counts, 0)
        start = 0
        for cid in range(num_clients):
            end = start + int(counts[cid])
            client_idx[cid].extend(indices[start:end])
            start = end

    return [[rows[i] for i in sorted(part)] for part in client_idx]


def split_clients_label_skew_extreme(
    rows: Sequence[Dict[str, Any]],
    num_clients: int,
) -> List[List[Dict[str, Any]]]:
    """各クライアントが主に 1 クラスを持つ極端な偏り（binary 想定: client i → label i%2）。"""
    parts: List[List[Dict[str, Any]]] = [[] for _ in range(num_clients)]
    for r in rows:
        label = int(r["label"])
        cid = label % num_clients
        parts[cid].append(r)
    return parts


def split_clients_key_skew(
    rows: Sequence[Dict[str, Any]],
    num_clients: int,
    key: str,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    """manifest のキー（scene 等）単位でまとめて 1 クライアントに割当（キー偏り Non-IID）。"""
    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(key, "") or "unknown")].append(r)

    keys = sorted(buckets.keys(), key=lambda k: len(buckets[k]), reverse=True)
    rng.shuffle(keys)
    parts: List[List[Dict[str, Any]]] = [[] for _ in range(num_clients)]
    for i, k in enumerate(keys):
        parts[i % num_clients].extend(buckets[k])
    return parts


def partition_fl_clients(
    rows: Sequence[Dict[str, Any]],
    fcfg: Mapping[str, Any],
    *,
    seed: int,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """fl.partition 設定に従いクライアント分割。"""
    pcfg = dict(fcfg.get("partition") or {})
    mode = str(pcfg.get("mode", "iid_shuffle")).strip().lower()
    num_clients = max(1, int(fcfg.get("num_clients", 2)))
    part_seed = int(pcfg.get("seed", seed))
    alpha = float(pcfg.get("label_skew_alpha", 0.3))
    key_field = str(pcfg.get("key_field", "scene"))

    if mode == "iid_shuffle":
        parts = split_clients_iid(rows, num_clients, part_seed)
    elif mode == "label_skew_dirichlet":
        parts = split_clients_label_skew_dirichlet(rows, num_clients, part_seed, alpha)
    elif mode == "label_skew_extreme":
        parts = split_clients_label_skew_extreme(rows, num_clients)
    elif mode == "key_skew":
        parts = split_clients_key_skew(rows, num_clients, key_field, part_seed)
    else:
        raise ValueError(
            f"Unknown fl.partition.mode={mode!r}. "
            "Use iid_shuffle | label_skew_dirichlet | label_skew_extreme | key_skew"
        )

    max_per = pcfg.get("max_samples_per_client")
    if max_per is not None:
        cap = int(max_per)
        for i in range(len(parts)):
            if len(parts[i]) > cap:
                rng = np.random.default_rng(part_seed + i)
                sel = rng.choice(len(parts[i]), size=cap, replace=False)
                parts[i] = [parts[i][int(j)] for j in sorted(sel)]

    stats: Dict[str, Any] = {
        "partition_mode": mode,
        "num_clients": num_clients,
        "partition_seed": part_seed,
        "label_skew_alpha": alpha if "label" in mode else None,
        "key_field": key_field if mode == "key_skew" else None,
        "pool_size": len(rows),
        "clients": [],
    }
    for cid, part in enumerate(parts):
        stats["clients"].append(
            {
                "client_id": cid,
                "num_samples": len(part),
                "label_counts": _label_counts(part),
                "scene_counts": _key_counts(part, "scene") if part else {},
            }
        )
    return parts, stats


def build_fl_dataset(
    cfg: Mapping[str, Any],
    *,
    root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[List[Dict[str, Any]]], Dict[str, Any]]:
    """FL 用 train pool / eval / クライアント分割を構築。"""
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    fcfg = cfg["fl"]

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = root / manifest

    # step04b_lora_diff_probe.py と同じデータ前処理
    rows = read_manifest_filtered(manifest, dcfg)
    seed = int(tcfg["seed"])
    rows = stratified_subset(rows, int(tcfg.get("max_train_samples", 200)), seed)

    eval_max = int(tcfg.get("eval_max", 20))
    pool, eval_rows = split_train_eval_rows(
        rows,
        train_ratio=float(dcfg.get("train_ratio", 0.8)),
        eval_max=eval_max,
    )

    client_parts, part_stats = partition_fl_clients(pool, fcfg, seed=seed)
    meta = {
        "manifest": str(manifest),
        "total_after_subset": len(rows),
        "train_pool_size": len(pool),
        "eval_size": len(eval_rows),
        "eval_label_counts": _label_counts(eval_rows),
        "pool_label_counts": _label_counts(pool),
        **part_stats,
    }
    return pool, eval_rows, client_parts, meta


def save_partition_artifact(run_dir: Path, meta: Mapping[str, Any], client_parts: Sequence[Sequence[Dict[str, Any]]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "fl_partition.json"
    payload = {
        **dict(meta),
        "client_sample_ids": [[str(r["id"]) for r in part] for part in client_parts],
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def apply_fl_cli_overrides(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """step04b と同様、CLI で上書き（別 YAML は使わない）。"""
    fcfg = cfg.setdefault("fl", {})
    pcfg = fcfg.setdefault("partition", {})
    if getattr(args, "num_rounds", None) is not None:
        fcfg["num_rounds"] = int(args.num_rounds)
    if getattr(args, "partition_mode", None):
        pcfg["mode"] = str(args.partition_mode)
    if getattr(args, "label_skew_alpha", None) is not None:
        pcfg["label_skew_alpha"] = float(args.label_skew_alpha)
    if getattr(args, "eval_max", None) is not None:
        cfg.setdefault("train", {})["eval_max"] = int(args.eval_max)
    if getattr(args, "max_train_samples", None) is not None:
        cfg.setdefault("train", {})["max_train_samples"] = int(args.max_train_samples)
    if getattr(args, "train_ratio", None) is not None:
        cfg.setdefault("data", {})["train_ratio"] = float(args.train_ratio)
    if getattr(args, "max_samples_per_client", None) is not None:
        pcfg["max_samples_per_client"] = int(args.max_samples_per_client)
    if getattr(args, "num_clients", None) is not None:
        fcfg["num_clients"] = int(args.num_clients)
        fcfg["min_fit_clients"] = int(args.num_clients)
        fcfg["min_available_clients"] = int(args.num_clients)
    if getattr(args, "strategy", None):
        fcfg["strategy"] = str(args.strategy)
    if getattr(args, "fedprox_mu", None) is not None:
        fcfg["fedprox_mu"] = float(args.fedprox_mu)
    return cfg


def get_client_train_rows(
    client_parts: Sequence[Sequence[Dict[str, Any]]],
    client_id: int,
) -> List[Dict[str, Any]]:
    if client_id < 0 or client_id >= len(client_parts):
        raise IndexError(f"client_id={client_id} out of range (num_clients={len(client_parts)})")
    return list(client_parts[client_id])


def build_fl_flower_report(
    cfg: Mapping[str, Any],
    *,
    server_host: str = "127.0.0.1",
    server_port: str = "8080",
    trainable_vector_dim: int | None = None,
) -> Dict[str, Any]:
    """後方互換。完全版は build_fl_flower_settings_full。"""
    from .fl_flower_config import build_fl_flower_settings_full

    return build_fl_flower_settings_full(
        cfg,
        role="report",
        server_host=server_host,
        server_port=server_port,
        trainable_vector_dim=trainable_vector_dim,
    )

