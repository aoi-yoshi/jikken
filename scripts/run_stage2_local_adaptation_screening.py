"""Establish local adaptation on the clip-classification subtask.

The experiment removes image-mode comparisons and tests video4/video8.  Each
optimizer step accumulates one matched normal/risk pair before updating, so a
single-class sample cannot move the classifier by itself.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import autocast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fedpact_stage2_3b_screening import (
    forward_sample,
    predict_logits,
    restore_state,
    snapshot_state,
    train_supervised,
)
from src.config_loader import load_yaml, merged_config
from src.dataset_manifest import load_sample_row, read_manifest
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


def paired_rows(rows: Sequence[dict]) -> list[tuple[dict, dict]]:
    groups: dict[str, dict[int, dict]] = {}
    for row in rows:
        groups.setdefault(str(row["pair_id"]), {})[int(row["label"])] = row
    output: list[tuple[dict, dict]] = []
    for pair_id, items in sorted(groups.items()):
        if set(items) != {0, 1}:
            raise RuntimeError(f"pair {pair_id} does not contain both labels")
        output.append((items[0], items[1]))
    return output


def configure_scope(
    model,
    lora_params: Sequence[torch.nn.Parameter],
    scope: str,
    lora_lr: float,
    head_lr: float,
) -> tuple[list[dict[str, Any]], list[torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups: list[dict[str, Any]] = []
    trainable: list[torch.nn.Parameter] = []
    if scope in {"lora_only", "lora_head"}:
        for parameter in lora_params:
            parameter.requires_grad_(True)
        groups.append({"params": list(lora_params), "lr": float(lora_lr)})
        trainable.extend(lora_params)
    if scope in {"head_only", "lora_head"}:
        head_params = list(model.classifier.parameters())
        for parameter in head_params:
            parameter.requires_grad_(True)
        groups.append({"params": head_params, "lr": float(head_lr)})
        trainable.extend(head_params)
    if not groups:
        raise ValueError(f"unknown scope: {scope}")
    return groups, trainable


def train_paired(
    model,
    processor,
    rows: Sequence[dict],
    dcfg: dict,
    lora_params: Sequence[torch.nn.Parameter],
    device: torch.device,
    *,
    scope: str,
    lora_lr: float,
    head_lr: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    groups, trainable = configure_scope(model, lora_params, scope, lora_lr, head_lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    prompt = str(dcfg["image_prompt"])
    pairs = paired_rows(rows)
    rng = random.Random(seed)
    losses: list[float] = []
    losses_by_class: dict[str, list[float]] = {"0": [], "1": []}
    grad_norms: list[float] = []
    pair_orders: list[list[str]] = []
    model.train()
    for _ in range(int(epochs)):
        order = list(pairs)
        rng.shuffle(order)
        pair_orders.append([str(normal["pair_id"]) for normal, _ in order])
        for normal_row, risk_row in order:
            optimizer.zero_grad(set_to_none=True)
            pair_loss = 0.0
            for row in (normal_row, risk_row):
                sample = load_sample_row(row, dcfg)
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits, _ = forward_sample(model, processor, sample, prompt, dcfg)
                    target = torch.tensor([sample.label], device=device, dtype=torch.long)
                    loss = F.cross_entropy(logits.float(), target) / 2.0
                loss.backward()
                pair_loss += float(loss.detach().cpu())
                losses_by_class[str(sample.label)].append(float(loss.detach().cpu()) * 2.0)
                for frame in sample.frames:
                    frame.close()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            losses.append(pair_loss)
            grad_norms.append(float(grad_norm.detach().cpu()))
    return {
        "scope": scope,
        "lora_lr": lora_lr,
        "head_lr": head_lr,
        "epochs": epochs,
        "optimizer_steps": len(losses),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": float(np.mean(losses)),
        "loss_mean_by_class": {
            label: float(np.mean(values)) for label, values in losses_by_class.items()
        },
        "grad_norm_mean_before_clip": float(np.mean(grad_norms)),
        "pair_orders": pair_orders,
    }


def probabilities(rows: Sequence[dict], logits: dict[str, list[float]]) -> np.ndarray:
    return np.asarray([softmax_vector(logits[str(row["id"])])[1] for row in rows])


def classification_metrics(rows: Sequence[dict], logits: dict[str, list[float]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probs = probabilities(rows, logits)
    margins = np.asarray(
        [float(logits[str(row["id"])][1] - logits[str(row["id"])][0]) for row in rows]
    )
    preds = (probs >= 0.5).astype(np.int64)
    means = {str(label): float(probs[labels == label].mean()) for label in (0, 1)}
    margin_means = {str(label): float(margins[labels == label].mean()) for label in (0, 1)}
    return {
        "n": len(rows),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision_risk": float(precision_score(labels, preds, zero_division=0)),
        "recall_risk": float(recall_score(labels, preds, zero_division=0)),
        "predicted_positive_rate": float(preds.mean()),
        "brier": float(brier_score_loss(labels, probs)),
        "mean_risk_probability_by_class": means,
        "class_separation": means["1"] - means["0"],
        "mean_risk_margin_by_class": margin_means,
        "margin_class_separation": margin_means["1"] - margin_means["0"],
    }


def score_change(
    rows: Sequence[dict],
    before: dict[str, list[float]],
    after: dict[str, list[float]],
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    delta = probabilities(rows, after) - probabilities(rows, before)
    before_margin = np.asarray(
        [float(before[str(row["id"])][1] - before[str(row["id"])][0]) for row in rows]
    )
    after_margin = np.asarray(
        [float(after[str(row["id"])][1] - after[str(row["id"])][0]) for row in rows]
    )
    margin_delta = after_margin - before_margin
    by_class = {str(label): float(delta[labels == label].mean()) for label in (0, 1)}
    margin_by_class = {
        str(label): float(margin_delta[labels == label].mean()) for label in (0, 1)
    }
    return {
        "mean_by_class": by_class,
        "opposite_direction": bool(by_class["0"] < 0 < by_class["1"]),
        "separation_delta": by_class["1"] - by_class["0"],
        "mean_margin_delta_by_class": margin_by_class,
        "margin_separation_delta": margin_by_class["1"] - margin_by_class["0"],
        "sample_deltas": [
            {"id": str(row["id"]), "label": int(row["label"]), "risk_probability_delta": float(value)}
            for row, value in zip(rows, delta)
        ],
    }


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {"scope": "lora_only", "lora_lr": 1e-5, "head_lr": 0.0, "epochs": 2},
        {"scope": "lora_only", "lora_lr": 3e-5, "head_lr": 0.0, "epochs": 2},
        {"scope": "head_only", "lora_lr": 0.0, "head_lr": 1e-4, "epochs": 2},
        {"scope": "head_only", "lora_lr": 0.0, "head_lr": 3e-4, "epochs": 2},
        {"scope": "lora_head", "lora_lr": 1e-5, "head_lr": 1e-4, "epochs": 2},
        {"scope": "lora_head", "lora_lr": 3e-5, "head_lr": 1e-4, "epochs": 2},
    ]


def gate(baseline: dict[str, Any], post: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    auc_gain = post["auc"] - baseline["auc"]
    auprc_gain = post["auprc"] - baseline["auprc"]
    checks = {
        "auc_nondecrease": auc_gain >= 0.0,
        "auprc_nondecrease": auprc_gain >= 0.0,
        "one_ranking_gain_ge_0_02": max(auc_gain, auprc_gain) >= 0.02,
        "margin_class_separation_gain": (
            post["margin_class_separation"] > baseline["margin_class_separation"]
        ),
        "macro_f1_noninferior": post["macro_f1"] - baseline["macro_f1"] >= -0.01,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "directional_diagnostic": bool(change["opposite_direction"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage2_phase_a.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/step05_fl/run_20260607_200402/step05_fedavg_global.pt"),
    )
    parser.add_argument("--run-id", default="stage2_local_adaptation_screen_20260921_01")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--b0-steps", type=int, default=60)
    parser.add_argument("--quick-smoke", action="store_true")
    parser.add_argument("--include-video8", action="store_true")
    parser.add_argument(
        "--grid-profile",
        choices=("stage_a", "head_epoch", "frozen_head2"),
        default="stage_a",
    )
    parser.add_argument("--only-condition", choices=("video4", "video8"), default=None)
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / "runs" / "fedpact_stage2" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "status.json", {"status": "running", "run_id": args.run_id})
    conditions = [
        {
            "id": "video4",
            "dataset_dir": ROOT / "artifacts" / "datasets" / "fedpact_stage2_coverage_v2_20260920",
            "num_frames": 4,
        },
    ]
    if args.include_video8:
        conditions.append(
            {
                "id": "video8",
                "dataset_dir": ROOT / "artifacts" / "datasets" / "fedpact_stage2_temporal_v3_20260921",
                "num_frames": 8,
            }
        )
    if args.only_condition is not None:
        conditions = [condition for condition in conditions if condition["id"] == args.only_condition]
        if not conditions and args.only_condition == "video8":
            conditions = [
                {
                    "id": "video8",
                    "dataset_dir": ROOT / "artifacts" / "datasets" / "fedpact_stage2_temporal_v3_20260921",
                    "num_frames": 8,
                }
            ]
    if args.grid_profile == "stage_a":
        grid = candidate_grid()
    elif args.grid_profile == "head_epoch":
        grid = [
            {"scope": "head_only", "lora_lr": 0.0, "head_lr": 1e-4, "epochs": epochs}
            for epochs in (1, 4)
        ]
    else:
        grid = [
            {"scope": "head_only", "lora_lr": 0.0, "head_lr": 1e-4, "epochs": 2}
        ]
    if args.quick_smoke:
        grid = grid[:1]

    cfg = merged_config(load_yaml(args.config))
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
        raise RuntimeError("initial checkpoint names do not match model")

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "status": "running",
        "task_contract": "alert_interval_clip_classification_not_early_anticipation",
        "checkpoint_meta": checkpoint_meta,
        "success_gate": {
            "auc_and_auprc": "both non-decreasing; at least one gain >= 0.02",
            "risk_margin_class_separation_gain": "positive",
            "macro_f1_gain_min": -0.01,
            "mean_delta_direction": "diagnostic only; class0 risk down and class1 risk up is preferred",
        },
        "conditions": {},
    }

    for condition in conditions:
        started = time.perf_counter()
        condition_id = str(condition["id"])
        print(f"condition {condition_id}", flush=True)
        data_dir = Path(condition["dataset_dir"])
        train_rows = read_manifest(data_dir / "manifest_client_train.jsonl")
        validation_rows = read_manifest(data_dir / "manifest_gate_validation.jsonl")
        server_rows = read_manifest(data_dir / "manifest_d_server_lab.jsonl")
        if args.quick_smoke:
            first_pair_id = str(train_rows[0]["pair_id"])
            train_rows = [row for row in train_rows if str(row["pair_id"]) == first_pair_id]
            first_validation_pair = str(validation_rows[0]["pair_id"])
            validation_rows = [
                row for row in validation_rows if str(row["pair_id"]) == first_validation_pair
            ]
            server_pair_ids = list(dict.fromkeys(str(row["pair_id"]) for row in server_rows))[:2]
            server_rows = [row for row in server_rows if str(row["pair_id"]) in server_pair_ids]
        dcfg = dict(cfg["data"])
        dcfg.update(
            {
                "input_mode": "video",
                "use_num_frames": int(condition["num_frames"]),
                "use_frame_slice": "last",
                "weather_filter": "",
            }
        )
        restore_state(model, initial_names, initial_vector)
        b0_trace = train_supervised(
            model,
            processor,
            server_rows,
            dcfg,
            lora_params,
            device,
            steps=2 if args.quick_smoke else int(args.b0_steps),
            seed=args.seed,
            lr=1e-4,
            update_head=True,
        )
        b0_names, b0_vector = snapshot_state(model, lora_params)
        b0_head_bias = model.classifier.bias.detach().float().cpu().numpy().copy()
        baseline_logits = predict_logits(model, processor, validation_rows, dcfg, batch_size=1)
        baseline_metrics = classification_metrics(validation_rows, baseline_logits)
        trials: list[dict[str, Any]] = []
        for trial_index, candidate in enumerate(grid):
            restore_state(model, b0_names, b0_vector)
            trace = train_paired(
                model,
                processor,
                train_rows,
                dcfg,
                lora_params,
                device,
                scope=str(candidate["scope"]),
                lora_lr=float(candidate["lora_lr"]),
                head_lr=float(candidate["head_lr"]),
                epochs=int(candidate["epochs"]),
                seed=args.seed + 1000 + trial_index,
            )
            post_logits = predict_logits(model, processor, validation_rows, dcfg, batch_size=1)
            post_metrics = classification_metrics(validation_rows, post_logits)
            change = score_change(validation_rows, baseline_logits, post_logits)
            gate_result = gate(baseline_metrics, post_metrics, change)
            names, vector = snapshot_state(model, lora_params)
            head_bias = model.classifier.bias.detach().float().cpu().numpy().copy()
            if names != b0_names:
                raise RuntimeError("state names changed")
            trial = {
                "candidate": candidate,
                "training": trace,
                "metrics": post_metrics,
                "gains": {
                    key: post_metrics[key] - baseline_metrics[key]
                    for key in (
                        "macro_f1",
                        "auc",
                        "auprc",
                        "class_separation",
                        "margin_class_separation",
                    )
                },
                "score_change": change,
                "gate": gate_result,
                "state_update_l2": float(np.linalg.norm(vector - b0_vector)),
                "head_bias_delta": (head_bias - b0_head_bias).tolist(),
            }
            trials.append(trial)
            print(
                f"  {candidate['scope']} lora={candidate['lora_lr']:g} head={candidate['head_lr']:g} "
                f"epochs={candidate['epochs']}: F1 {post_metrics['macro_f1']:.3f} "
                f"AUC {post_metrics['auc']:.3f} AUPRC {post_metrics['auprc']:.3f} "
                f"sep {post_metrics['class_separation']:.3f} gate={gate_result['pass']}",
                flush=True,
            )
        passing = [trial for trial in trials if trial["gate"]["pass"]]
        ranked = passing or trials
        best = max(
            ranked,
            key=lambda trial: (
                trial["gate"]["pass"],
                trial["metrics"]["auc"],
                trial["metrics"]["auprc"],
                trial["metrics"]["margin_class_separation"],
                trial["metrics"]["macro_f1"],
            ),
        )
        summary["conditions"][condition_id] = {
            "dataset_dir": str(data_dir),
            "num_frames": condition["num_frames"],
            "b0_training": b0_trace,
            "baseline_metrics": baseline_metrics,
            "trials": trials,
            "passing_trial_count": len(passing),
            "best_trial": best,
            "elapsed_sec": time.perf_counter() - started,
        }
        save_json(run_dir / "summary.partial.json", summary)

    all_best = [
        {"condition": name, **value["best_trial"]}
        for name, value in summary["conditions"].items()
    ]
    summary["best_overall"] = max(
        all_best,
        key=lambda trial: (
            trial["gate"]["pass"],
            trial["metrics"]["auc"],
            trial["metrics"]["auprc"],
            trial["metrics"]["margin_class_separation"],
            trial["metrics"]["macro_f1"],
        ),
    )
    summary["status"] = "completed"
    summary["elapsed_sec"] = sum(value["elapsed_sec"] for value in summary["conditions"].values())
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "status.json", {"status": "completed", "run_id": args.run_id})
    print(json.dumps({"status": "completed", "best": summary["best_overall"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
