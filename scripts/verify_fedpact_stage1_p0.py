"""FedPACT Stage I P0: paired input, fixed split, privacy, and state contracts."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.fedpact_stage1 import (  # noqa: E402
    audit_split_video_overlap,
    audit_server_payload,
    canonical_tensor_digest,
    sha256_file,
    weighted_average_vectors,
)
from src.config_loader import load_yaml, merged_config  # noqa: E402
from src.dataset_manifest import read_manifest_filtered  # noqa: E402
from src.fl_utils import load_fl_checkpoint  # noqa: E402
from src.step07.data_splits import (  # noqa: E402
    save_data_splits_artifact,
    split_paired_step07_data_pools,
)


def _parse_times(row: dict[str, Any]) -> list[float]:
    return [float(value) for value in str(row["frame_times_sec"]).split(",")]


def _assert_content_disjoint(pools: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    for pool_name, rows in pools.items():
        hashes = [str(row["source_video_sha256"]) for row in rows]
        counts[pool_name] = len(hashes)
        if len(hashes) != len(set(hashes)):
            raise AssertionError(f"duplicate video content inside {pool_name}")
        for digest in hashes:
            if digest in seen:
                raise AssertionError(
                    f"video content overlaps across pools: {seen[digest]} and {pool_name}"
                )
            seen[digest] = pool_name
    return {"status": "pass", "pool_counts": counts, "unique_content_count": len(seen)}


def _audit_paired_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("pair_id", "")), []).append(row)
    if "" in grouped:
        raise AssertionError("manifest row without pair_id")
    label_counts = {label: sum(int(row["label"]) == label for row in rows) for label in (0, 1)}
    if label_counts != {0: 60, 1: 60} or len(grouped) != 60:
        raise AssertionError(f"expected 60 pairs, got labels={label_counts}, pairs={len(grouped)}")

    checked_files = 0
    for pair_id, pair in grouped.items():
        pair = sorted(pair, key=lambda row: int(row["label"]))
        if [int(row["label"]) for row in pair] != [0, 1]:
            raise AssertionError(f"pair {pair_id} does not contain normal/risk")
        normal, risk = pair
        normal_times = _parse_times(normal)
        risk_times = _parse_times(risk)
        if not np.allclose(normal_times, risk_times, atol=5e-7, rtol=0.0):
            raise AssertionError(f"pair {pair_id} does not share absolute seconds")
        alert = float(risk["time_of_alert"])
        event = float(risk["time_of_event"])
        expected = [alert + (event - alert) * fraction for fraction in (0, 1 / 3, 2 / 3, 1)]
        if not np.allclose(risk_times, expected, atol=1e-5, rtol=0.0):
            raise AssertionError(f"pair {pair_id} risk quartiles mismatch")
        for row in pair:
            if row.get("frame_sampling") != "paired_alert_event_quartiles":
                raise AssertionError(f"pair {pair_id} has mixed sampling")
            if len(row.get("frame_paths", [])) != 4 or len(row.get("frame_indices", [])) != 4:
                raise AssertionError(f"pair {pair_id} is not four-frame")
            total = int(row["source_video_frame_count"])
            fps = float(row["source_video_fps"])
            expected_indices = [max(0, min(total - 1, int(round(value * fps)))) for value in risk_times]
            if [int(value) for value in row["frame_indices"]] != expected_indices:
                raise AssertionError(f"pair {pair_id} frame index mismatch")
            source_path = Path(str(row["video_path"]))
            if sha256_file(source_path) != str(row["source_video_sha256"]):
                raise AssertionError(f"source video hash mismatch: {row['id']}")
            frame_hashes = list(row.get("frame_sha256", []))
            for frame_path, expected_hash in zip(row["frame_paths"], frame_hashes):
                if sha256_file(Path(frame_path)) != str(expected_hash):
                    raise AssertionError(f"frame hash mismatch: {frame_path}")
                checked_files += 1
    hashes = [str(row["source_video_sha256"]) for row in rows]
    if len(hashes) != len(set(hashes)):
        raise AssertionError("duplicate source-video content in manifest")
    return {
        "status": "pass",
        "pair_count": len(grouped),
        "label_counts": label_counts,
        "frame_file_count_verified": checked_files,
        "unique_source_video_content": len(set(hashes)),
    }


def _make_contact_sheets(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), []).append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    pair_items = sorted(grouped.items())
    tile_w, tile_h, label_w, row_h = 120, 68, 155, 96
    for page, start in enumerate(range(0, len(pair_items), 12), start=1):
        subset = pair_items[start : start + 12]
        canvas = Image.new("RGB", (label_w + tile_w * 8, row_h * len(subset)), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, (pair_id, members) in enumerate(subset):
            members = sorted(members, key=lambda item: int(item["label"]))
            y = row_index * row_h
            draw.text((4, y + 4), f"{pair_id}\nnormal | risk", fill="black")
            paths = [path for member in members for path in member["frame_paths"]]
            for frame_index, frame_path in enumerate(paths):
                with Image.open(frame_path) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail((tile_w, tile_h))
                    x = label_w + frame_index * tile_w
                    canvas.paste(thumb, (x, y))
            draw.text((label_w, y + tile_h + 2), str(members[0]["frame_times_sec"]), fill="black")
        path = output_dir / f"paired_frames_{page:02d}.jpg"
        canvas.save(path, quality=88)
        outputs.append(str(path.relative_to(output_dir.parent)))
    return outputs


def _pool_map(pools: tuple[list[list[dict]], list[dict], list[dict], list[dict]]) -> dict[str, list[dict]]:
    clients, validation, server_seed, final_eval = pools
    return {
        "client-0-private": clients[0],
        "client-1-private": clients[1],
        "validation": validation,
        "server-seed": server_seed,
        "final-evaluation": final_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_id = args.run_id or datetime.now().strftime("p0_%Y%m%d_%H%M%S")
    run_dir = _ROOT / "artifacts" / "runs" / "fedpact_stage1" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    names = ["lora.a", "head.weight"]
    client_0 = np.asarray([1.0, 3.0, -2.0], dtype=np.float32)
    client_1 = np.asarray([5.0, -1.0, 2.0], dtype=np.float32)
    out_names, averaged, weights = weighted_average_vectors(
        [(names, client_0), (names, client_1)], [1, 3]
    )
    expected = (client_0 + 3.0 * client_1) / 4.0
    if out_names != names or not np.array_equal(averaged, expected.astype(np.float32)):
        raise AssertionError("weighted FedAvg mismatch")

    tensor_a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    digest_1 = canonical_tensor_digest([("state", tensor_a)])
    if digest_1 != canonical_tensor_digest([("state", tensor_a.clone())]):
        raise AssertionError("canonical tensor digest instability")
    if digest_1 == canonical_tensor_digest([("state", tensor_a + 1.0)]):
        raise AssertionError("canonical tensor digest missed a change")

    artifact = {
        "artifact": "update.npz",
        "tensor_count": 1,
        "parameter_count": 2,
        "l2_norm": 0.5,
        "sha256": "0" * 64,
    }
    valid_payload = {
        "schema": "fedpact.client_payload.v2",
        "client_id": "client-0",
        "round": 1,
        "local_train_sample_count": 4,
        "trainable_update": {
            "client_lora": artifact,
            "classification_head": artifact,
        },
        "prototypes": [],
        "privacy": {
            "private_raw_data_included": False,
            "private_sample_id_included": False,
            "sample_level_output_included": False,
            "q_pre_sent_independently": False,
        },
    }
    audit_server_payload(valid_payload, forbidden_values=["private-123"])
    invalid_payload = dict(valid_payload)
    invalid_payload["sample_ids"] = ["private-123"]
    try:
        audit_server_payload(invalid_payload, forbidden_values=["private-123"])
    except ValueError:
        pass
    else:
        raise AssertionError("allowlist negative control was not detected")

    config_path = _ROOT / "config" / "fedpact_stage1.yaml"
    cfg = merged_config(load_yaml(config_path))
    scfg = cfg["fedpact_stage1"]
    dcfg = cfg["data"]
    if dcfg["frame_sampling"] != "paired_alert_event_quartiles" or int(dcfg["num_frames"]) != 4:
        raise AssertionError("Stage I frame sampling contract mismatch")
    manifest_path = Path(dcfg["manifest_path"])
    rows = read_manifest_filtered(manifest_path, dcfg)
    manifest_audit = _audit_paired_manifest(rows)
    contact_sheets = _make_contact_sheets(rows, run_dir / "visual_audit")

    seed = int(cfg["train"]["seed"])
    p1a = split_paired_step07_data_pools(
        rows,
        num_clients=2,
        client_train_pairs=int(scfg["client_train_per_class_p1a"]),
        client_eval_pairs=int(scfg["validation_per_class"]),
        server_train_pairs=int(scfg["server_seed_per_class_p1b"]),
        server_eval_pairs=int(scfg["final_evaluation_per_class"]),
        seed=seed,
    )
    p1d = split_paired_step07_data_pools(
        rows,
        num_clients=2,
        client_train_pairs=int(scfg["client_train_per_class_p1d"]),
        client_eval_pairs=int(scfg["validation_per_class"]),
        server_train_pairs=int(scfg["server_seed_per_class_p1d"]),
        server_eval_pairs=int(scfg["final_evaluation_per_class"]),
        seed=seed,
    )
    split_checks: dict[str, Any] = {}
    for name, split in (("p1a", p1a), ("p1d", p1d)):
        pool_map = _pool_map(split)
        split_checks[name] = {
            "path_identity": audit_split_video_overlap(pool_map),
            "content_identity": _assert_content_disjoint(pool_map),
        }
        (run_dir / name).mkdir(parents=True, exist_ok=True)
        save_data_splits_artifact(
            run_dir / name,
            client_trains=split[0],
            client_eval=split[1],
            server_train=split[2],
            server_eval=split[3],
            seed=seed,
        )

    checkpoint_path = Path(str(scfg["initial_global_checkpoint"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = _ROOT / checkpoint_path
    checkpoint_names, checkpoint_vector, checkpoint_meta = load_fl_checkpoint(checkpoint_path)
    if checkpoint_meta.get("strategy") != "FedAvg" or not checkpoint_names or checkpoint_vector.size == 0:
        raise AssertionError("M_G^0 is not a non-empty explicit FedAvg checkpoint")

    summary = {
        "run_id": run_id,
        "gate": "P0",
        "status": "pass",
        "checks": {
            "weighted_fedavg_unequal_counts": {"status": "pass", "weights": weights},
            "canonical_tensor_digest_stability": "pass",
            "payload_exact_allowlist_positive": "pass",
            "payload_unexpected_field_negative_control": "pass",
            "paired_manifest": manifest_audit,
            "fixed_splits": split_checks,
            "explicit_fedavg_initial_checkpoint": "pass",
        },
        "config": {"path": str(config_path.relative_to(_ROOT)), "sha256": sha256_file(config_path)},
        "code_sha256": {
            "p0_runner": sha256_file(Path(__file__)),
            "manifest_builder": sha256_file(_ROOT / "scripts" / "step02_nexar_or_synthetic.py"),
            "fedpact_stage1": sha256_file(_ROOT / "src" / "fedpact_stage1.py"),
            "data_splits": sha256_file(_ROOT / "src" / "step07" / "data_splits.py"),
        },
        "manifest": {
            "path": str(manifest_path.relative_to(_ROOT)),
            "sha256": sha256_file(manifest_path),
            "sample_count": len(rows),
        },
        "visual_audit_contact_sheets": contact_sheets,
        "initial_global_checkpoint": {
            "path": str(checkpoint_path.relative_to(_ROOT)),
            "sha256": sha256_file(checkpoint_path),
            "source_run_id": checkpoint_meta.get("run_id"),
            "strategy": checkpoint_meta.get("strategy"),
            "tensor_count": len(checkpoint_names),
            "parameter_count": int(checkpoint_vector.size),
        },
        "claim_limit": "Input/state/privacy contract checks only; this is not a performance result.",
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
