"""Screen image/video input representation before resuming FedPACT Layer 2.

The task remains the current alert-interval clip normal/risk diagnostic.  It is
explicitly not an early-anticipation experiment because the interval includes
the annotated event endpoint.
"""
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
    balanced_subset,
    metrics_from_logits,
    predict_logits,
    restore_state,
    snapshot_state,
    train_supervised,
)
from src.config_loader import load_yaml, merged_config
from src.dataset_manifest import read_manifest
from src.fedpact_stage1 import apply_named_vector
from src.fl_utils import load_fl_checkpoint
from src.metrics import set_seed
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.step07.fedpact_layer2 import softmax_vector
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(model, processor, rows: list[dict], dcfg: dict) -> dict[str, Any]:
    return metrics_from_logits(rows, predict_logits(model, processor, rows, dcfg, batch_size=1))


def class_delta_summary(
    rows: list[dict],
    before: dict[str, list[float]],
    after: dict[str, list[float]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in (0, 1):
        deltas = []
        for row in rows:
            if int(row["label"]) != label:
                continue
            sid = str(row["id"])
            deltas.append(softmax_vector(after[sid]) - softmax_vector(before[sid]))
        matrix = np.stack(deltas)
        result[str(label)] = {
            "mean_delta": matrix.mean(axis=0).tolist(),
            "std_delta": matrix.std(axis=0).tolist(),
            "risk_direction_agreement": float((np.sign(matrix[:, 1]) == np.sign(matrix[:, 1].mean())).mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage1.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/step05_fl/run_20260607_200402/step05_fedavg_global.pt"),
    )
    parser.add_argument("--run-id", default="stage2_temporal_input_screen_20260921_01")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--b0-steps", type=int, default=60)
    parser.add_argument("--b0-lr", type=float, default=1e-4)
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / "runs" / "fedpact_stage2" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "status.json", {"status": "running", "run_id": args.run_id})

    v2 = ROOT / "artifacts" / "datasets" / "fedpact_stage2_coverage_v2_20260920"
    v3 = ROOT / "artifacts" / "datasets" / "fedpact_stage2_temporal_v3_20260921"
    conditions = [
        {"id": "image4", "dataset_dir": v2, "input_mode": "images", "num_frames": 4},
        {"id": "video4", "dataset_dir": v2, "input_mode": "video", "num_frames": 4},
        {"id": "video8", "dataset_dir": v3, "input_mode": "video", "num_frames": 8},
    ]
    learning_rates = (1e-5, 3e-5, 1e-4)
    step_counts = (0, 4, 8, 16)
    if args.quick_smoke:
        learning_rates = (1e-5,)
        step_counts = (0, 1)

    cfg = merged_config(load_yaml(args.config))
    cfg["train"]["seed"] = args.seed
    device = torch.device(device_or_auto(True) if args.device == "auto" else args.device)
    set_seed(args.seed)
    model, processor = build_model(cfg, device, model_id=str(cfg["model"]["client_id"]))
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    lora_params = lora_params_for_adapter(model.backbone, "client")
    checkpoint = (args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint).resolve()
    checkpoint_names, checkpoint_vector, checkpoint_meta = load_fl_checkpoint(checkpoint)
    apply_named_vector(model, checkpoint_names, checkpoint_vector)
    initial_names, initial_vector = snapshot_state(model, lora_params)
    if initial_names != checkpoint_names:
        raise RuntimeError("initial checkpoint names do not match 3B model")

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "status": "running",
        "claim_scope": "single-seed input screening on alert-interval clip classification; not early anticipation",
        "checkpoint_meta": checkpoint_meta,
        "grid": {"learning_rates": learning_rates, "steps": step_counts},
        "conditions": {},
    }
    for condition_index, condition in enumerate(conditions):
        condition_started = time.perf_counter()
        condition_id = str(condition["id"])
        print(f"condition {condition_index + 1}/{len(conditions)}: {condition_id}", flush=True)
        data_dir = Path(condition["dataset_dir"])
        manifests = {
            name: read_manifest(data_dir / f"manifest_{name}.jsonl")
            for name in ("client_train", "gate_validation", "d_server_lab", "retention_test")
        }
        if args.quick_smoke:
            manifests = {
                "client_train": balanced_subset(manifests["client_train"], 1),
                "gate_validation": balanced_subset(manifests["gate_validation"], 1),
                "d_server_lab": balanced_subset(manifests["d_server_lab"], 2),
                "retention_test": balanced_subset(manifests["retention_test"], 1),
            }
        dcfg = dict(cfg["data"])
        dcfg.update(
            {
                "weather_filter": "",
                "use_num_frames": int(condition["num_frames"]),
                "use_frame_slice": "last",
                "input_mode": str(condition["input_mode"]),
            }
        )
        restore_state(model, initial_names, initial_vector)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        b0_trace = train_supervised(
            model,
            processor,
            manifests["d_server_lab"],
            dcfg,
            lora_params,
            device,
            steps=2 if args.quick_smoke else int(args.b0_steps),
            seed=args.seed,
            lr=float(args.b0_lr),
            update_head=True,
        )
        b0_names, b0_vector = snapshot_state(model, lora_params)
        b0_validation = evaluate(model, processor, manifests["gate_validation"], dcfg)
        b0_retention = evaluate(model, processor, manifests["retention_test"], dcfg)
        client_before = predict_logits(
            model, processor, manifests["client_train"], dcfg, batch_size=1
        )
        trials: list[dict[str, Any]] = []
        for lr in learning_rates:
            for steps in step_counts:
                restore_state(model, b0_names, b0_vector)
                trace = None
                if steps:
                    trace = train_supervised(
                        model,
                        processor,
                        manifests["client_train"],
                        dcfg,
                        lora_params,
                        device,
                        steps=steps,
                        seed=args.seed + 100 + steps,
                        lr=lr,
                        update_head=True,
                    )
                validation = evaluate(model, processor, manifests["gate_validation"], dcfg)
                client_after = predict_logits(
                    model, processor, manifests["client_train"], dcfg, batch_size=1
                )
                post_names, post_vector = snapshot_state(model, lora_params)
                if post_names != b0_names:
                    raise RuntimeError("state names changed during local update")
                trial = {
                    "lr": lr,
                    "steps": steps,
                    "training": trace,
                    "validation": validation,
                    "validation_gain_macro_f1": validation["macro_f1"] - b0_validation["macro_f1"],
                    "validation_gain_auc": validation["auc"] - b0_validation["auc"],
                    "state_update_l2": float(np.linalg.norm(post_vector - b0_vector)),
                    "client_delta": class_delta_summary(
                        manifests["client_train"], client_before, client_after
                    ),
                }
                trials.append(trial)
                print(
                    f"  lr={lr:g} steps={steps}: F1={validation['macro_f1']:.3f} "
                    f"gain={trial['validation_gain_macro_f1']:+.3f} AUC={validation['auc']:.3f}",
                    flush=True,
                )
        best = max(
            trials,
            key=lambda item: (
                item["validation"]["macro_f1"],
                item["validation"]["auc"],
                -item["steps"],
                -item["lr"],
            ),
        )
        restore_state(model, b0_names, b0_vector)
        if best["steps"]:
            train_supervised(
                model,
                processor,
                manifests["client_train"],
                dcfg,
                lora_params,
                device,
                steps=int(best["steps"]),
                seed=args.seed + 100 + int(best["steps"]),
                lr=float(best["lr"]),
                update_head=True,
            )
        best_retention = evaluate(model, processor, manifests["retention_test"], dcfg)
        elapsed = time.perf_counter() - condition_started
        peak_gb = (
            float(torch.cuda.max_memory_allocated(device) / (1024**3))
            if device.type == "cuda"
            else None
        )
        summary["conditions"][condition_id] = {
            "dataset_dir": str(data_dir),
            "input_mode": condition["input_mode"],
            "num_frames": condition["num_frames"],
            "b0_training": b0_trace,
            "b0_validation": b0_validation,
            "b0_retention": b0_retention,
            "trials": trials,
            "best_validation_trial": best,
            "best_retention": best_retention,
            "elapsed_sec": elapsed,
            "peak_cuda_allocated_gb": peak_gb,
        }
        save_json(run_dir / "summary.partial.json", summary)

    summary["status"] = "completed"
    summary["elapsed_sec"] = sum(
        value["elapsed_sec"] for value in summary["conditions"].values()
    )
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "status.json", {"status": "completed", "run_id": args.run_id})
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
