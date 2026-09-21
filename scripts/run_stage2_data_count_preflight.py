"""Preflight the amount of client data needed for same-task local adaptation.

The experiment keeps the fixed B0 state, the untouched confirmation set, two
local epochs, and the video4 clip-classification contract.  Only the number of
independent matched video pairs changes (2/4/8/16).  This is intentionally a
one-draw preflight; any apparent endpoint must be repeated with more subset
draws and local seeds before it is treated as evidence.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fedpact_stage2_3b_screening import predict_logits, restore_state
from scripts.run_stage2_local_adaptation_confirmation import (
    evaluate_logits,
    metric_gains,
)
from scripts.run_stage2_local_adaptation_screening import paired_rows, train_paired
from src.config_loader import load_yaml, merged_config
from src.dataset_manifest import read_manifest
from src.metrics import set_seed
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


PAIR_COUNTS = (2, 4, 8, 16)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ids(rows: list[dict]) -> set[str]:
    return {str(row["pair_id"]) for row in rows}


def rows_for_pairs(rows: list[dict], pair_ids: list[str]) -> list[dict]:
    wanted = set(pair_ids)
    result = [row for row in rows if str(row["pair_id"]) in wanted]
    if len(paired_rows(result)) != len(wanted):
        raise RuntimeError("a selected pair is incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage2_phase_a.yaml")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/fedpact_stage2_coverage_v2_20260920"),
    )
    parser.add_argument(
        "--b0-run-dir",
        type=Path,
        default=Path(
            "artifacts/runs/fedpact_stage2/"
            "stage2_local_adaptation_confirmation_20260921_01"
        ),
    )
    parser.add_argument("--run-id", default="stage2_data_count_preflight_20260921_01")
    parser.add_argument("--subset-seed", type=int, default=20260921)
    parser.add_argument("--local-seed", type=int, default=20260921)
    parser.add_argument(
        "--selection-mode",
        choices=("anchor_original8", "random"),
        default="anchor_original8",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    data_dir = (args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir).resolve()
    b0_run_dir = (args.b0_run_dir if args.b0_run_dir.is_absolute() else ROOT / args.b0_run_dir).resolve()
    run_dir = ROOT / "artifacts" / "runs" / "fedpact_stage2" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "status.json", {"status": "running", "run_id": args.run_id})

    master = read_manifest(data_dir / "manifest_master.jsonl")
    original_train = read_manifest(data_dir / "manifest_client_train.jsonl")
    confirmation = read_manifest(data_dir / "manifest_seed_a_target.jsonl")
    retention = read_manifest(data_dir / "manifest_retention_test.jsonl")
    server = read_manifest(data_dir / "manifest_d_server_lab.jsonl")

    excluded = ids(confirmation) | ids(retention) | ids(server)
    target_rows = [
        row
        for row in master
        if str(row["pair_id"]) not in excluded
        and str(row.get("light_conditions")) == "Normal"
        and str(row.get("weather")) == "Cloudy"
        and str(row.get("scene")) == "Highway"
    ]
    candidate_pairs = sorted({str(row["pair_id"]) for row in target_rows})
    original_pairs = sorted(ids(original_train))
    if not set(original_pairs).issubset(candidate_pairs):
        raise RuntimeError("original client training pairs are not in the eligible pool")
    extra_pairs = sorted(set(candidate_pairs) - set(original_pairs))
    rng = random.Random(args.subset_seed)
    if args.selection_mode == "anchor_original8":
        rng.shuffle(original_pairs)
        rng.shuffle(extra_pairs)
        ordered_pairs = original_pairs + extra_pairs
    else:
        ordered_pairs = list(candidate_pairs)
        rng.shuffle(ordered_pairs)
    if len(ordered_pairs) < max(PAIR_COUNTS):
        raise RuntimeError("not enough eligible target-condition pairs")
    selected_by_count = {str(count): ordered_pairs[:count] for count in PAIR_COUNTS}
    save_json(
        run_dir / "split.json",
        {
            "sampling_unit": "matched source-video pair",
            "nested": True,
            "subset_seed": args.subset_seed,
            "selection_mode": args.selection_mode,
            "eligible_pair_count": len(candidate_pairs),
            "excluded_from_training": ["B0 server data", "confirmation", "retention"],
            "pair_ids_by_count": selected_by_count,
        },
    )

    state = np.load(b0_run_dir / "states.npz", allow_pickle=False)
    b0_names = [str(value) for value in state["names"].tolist()]
    b0_vector = np.asarray(state["b0"], dtype=np.float32)
    prior_summary = json.loads((b0_run_dir / "summary.json").read_text(encoding="utf-8"))
    baseline_metrics = prior_summary["baseline_metrics"]

    cfg = merged_config(load_yaml(args.config))
    dcfg = dict(cfg["data"])
    dcfg.update(
        {
            "input_mode": "video",
            "use_num_frames": 4,
            "use_frame_slice": "last",
            "weather_filter": "",
        }
    )
    device = torch.device(device_or_auto(True) if args.device == "auto" else args.device)
    set_seed(args.local_seed)
    model, processor = build_model(cfg, device, model_id=str(cfg["model"]["client_id"]))
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    lora_params = lora_params_for_adapter(model.backbone, "client")
    restore_state(model, b0_names, b0_vector)

    results = []
    for count in PAIR_COUNTS:
        print(f"data-count condition: {count} matched pairs", flush=True)
        restore_state(model, b0_names, b0_vector)
        set_seed(args.local_seed)
        train_rows = rows_for_pairs(target_rows, selected_by_count[str(count)])
        trace = train_paired(
            model,
            processor,
            train_rows,
            dcfg,
            lora_params,
            device,
            scope="head_only",
            lora_lr=0.0,
            head_lr=1e-4,
            epochs=2,
            seed=args.local_seed,
        )
        split_results = {}
        for split, rows in (("confirmation", confirmation), ("retention", retention)):
            logits = predict_logits(model, processor, rows, dcfg, batch_size=1)
            metrics = evaluate_logits(rows, logits)
            split_results[split] = {
                "metrics": metrics,
                "gains": metric_gains(baseline_metrics[split], metrics),
            }
        item = {
            "pair_count": count,
            "video_count": 2 * count,
            "local_epochs": 2,
            "optimizer_steps": trace["optimizer_steps"],
            "training": trace,
            "splits": split_results,
        }
        results.append(item)
        save_json(run_dir / "summary.partial.json", {"results": results})

    summary = {
        "run_id": args.run_id,
        "status": "completed",
        "claim_scope": "one-subset-draw, one-local-seed data-count preflight; not confirmatory",
        "task": "alert-interval clip classification, not early prediction",
        "fixed": {
            "input": "video4",
            "scope": "head_only",
            "head_lr": 1e-4,
            "local_epochs": 2,
            "confirmation_videos": len(confirmation),
            "retention_videos": len(retention),
        },
        "baseline_metrics": baseline_metrics,
        "results": results,
        "elapsed_sec": time.perf_counter() - started,
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "status.json", {"status": "completed", "run_id": args.run_id})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
