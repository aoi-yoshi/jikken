"""Compare LoRA-only and LoRA+head on 16 matched pairs with video16."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fedpact_stage2_3b_screening import (
    predict_logits,
    restore_state,
    snapshot_state,
    train_supervised,
)
from scripts.run_stage2_local_adaptation_confirmation import evaluate_logits, metric_gains
from scripts.run_stage2_local_adaptation_screening import train_paired
from src.config_loader import load_yaml, merged_config
from src.dataset_manifest import read_manifest
from src.fedpact_stage1 import apply_named_vector
from src.fl_utils import load_fl_checkpoint
from src.metrics import set_seed
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


CONDITIONS = (
    {"name": "lora_only", "scope": "lora_only", "lora_lr": 1e-5, "head_lr": 0.0},
    {"name": "lora_head", "scope": "lora_head", "lora_lr": 1e-5, "head_lr": 1e-4},
)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage2_phase_a.yaml")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/fedpact_stage2_video16_lora_20260921"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/step05_fl/run_20260607_200402/step05_fedavg_global.pt"),
    )
    parser.add_argument("--run-id", default="stage2_video16_lora_compare_20260921_01")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--b0-steps", type=int, default=60)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    run_dir = ROOT / "artifacts" / "runs" / "fedpact_stage2" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "status.json", {"status": "running", "run_id": args.run_id})
    data_dir = (args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir).resolve()
    checkpoint = (args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint).resolve()
    rows = {
        name: read_manifest(data_dir / f"manifest_{name}.jsonl")
        for name in ("client_train", "d_server_lab", "confirmation", "retention")
    }

    cfg = merged_config(load_yaml(args.config))
    cfg["train"]["min_pixels"] = 50176
    cfg["train"]["max_pixels"] = 50176
    dcfg = dict(cfg["data"])
    dcfg.update(
        {
            "input_mode": "video",
            "use_num_frames": 16,
            "use_frame_slice": "last",
            "weather_filter": "",
        }
    )
    device = torch.device(device_or_auto(True) if args.device == "auto" else args.device)
    set_seed(args.seed)
    model, processor = build_model(cfg, device, model_id=str(cfg["model"]["client_id"]))
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    lora_params = lora_params_for_adapter(model.backbone, "client")
    checkpoint_names, checkpoint_vector, checkpoint_meta = load_fl_checkpoint(checkpoint)
    apply_named_vector(model, checkpoint_names, checkpoint_vector)

    print("training fixed video16 B0", flush=True)
    b0_trace = train_supervised(
        model,
        processor,
        rows["d_server_lab"],
        dcfg,
        lora_params,
        device,
        steps=args.b0_steps,
        seed=args.seed,
        lr=1e-4,
        update_head=True,
    )
    b0_names, b0_vector = snapshot_state(model, lora_params)
    baseline_logits = {
        split: predict_logits(model, processor, rows[split], dcfg, batch_size=1)
        for split in ("confirmation", "retention")
    }
    baseline_metrics = {
        split: evaluate_logits(rows[split], baseline_logits[split])
        for split in baseline_logits
    }

    results = []
    state_vectors = {"names": np.asarray(b0_names), "b0": b0_vector}
    for condition in CONDITIONS:
        print(f"condition: {condition['name']}", flush=True)
        restore_state(model, b0_names, b0_vector)
        set_seed(args.seed)
        trace = train_paired(
            model,
            processor,
            rows["client_train"],
            dcfg,
            lora_params,
            device,
            scope=str(condition["scope"]),
            lora_lr=float(condition["lora_lr"]),
            head_lr=float(condition["head_lr"]),
            epochs=2,
            seed=args.seed,
        )
        names, vector = snapshot_state(model, lora_params)
        if names != b0_names:
            raise RuntimeError("state names changed")
        state_vectors[str(condition["name"])] = vector
        split_results = {}
        for split in ("confirmation", "retention"):
            logits = predict_logits(model, processor, rows[split], dcfg, batch_size=1)
            metrics = evaluate_logits(rows[split], logits)
            split_results[split] = {
                "metrics": metrics,
                "gains": metric_gains(baseline_metrics[split], metrics),
            }
        item = {
            "condition": condition,
            "training": trace,
            "state_update_l2": float(np.linalg.norm(vector - b0_vector)),
            "splits": split_results,
        }
        results.append(item)
        save_json(run_dir / "summary.partial.json", {"results": results})

    np.savez_compressed(run_dir / "states.npz", **state_vectors)
    summary = {
        "run_id": args.run_id,
        "status": "completed",
        "claim_scope": "one-seed video16 comparison on 16 matched client pairs",
        "task": "alert-interval clip classification, not early prediction",
        "fixed": {
            "client_pairs": 16,
            "client_videos": 32,
            "local_epochs": 2,
            "num_frames": 16,
            "model_pixels_per_frame": 50176,
        },
        "checkpoint_meta": checkpoint_meta,
        "b0_training": b0_trace,
        "baseline_metrics": baseline_metrics,
        "results": results,
        "elapsed_sec": time.perf_counter() - started,
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "status.json", {"status": "completed", "run_id": args.run_id})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
