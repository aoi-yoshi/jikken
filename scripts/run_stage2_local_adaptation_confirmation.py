"""Confirm the frozen video4 local-adaptation condition on untouched pairs.

This is a clip-classification confirmation, not early accident prediction and
not a Layer-2/foundation-transfer experiment.  The condition is frozen from
screening: video4, head-only, LR=1e-4, two pair-balanced epochs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fedpact_stage2_3b_screening import (
    predict_logits,
    restore_state,
    snapshot_state,
    train_supervised,
)
from scripts.run_stage2_local_adaptation_screening import (
    paired_rows,
    score_change,
    train_paired,
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


LOCAL_SEEDS = (20260921, 20260922, 20260923)
FROZEN_LOCAL_CONDITION = {
    "input_mode": "video",
    "num_frames": 4,
    "scope": "head_only",
    "lora_lr": 0.0,
    "head_lr": 1e-4,
    "epochs": 2,
    "decision_threshold": 0.5,
}
CONFIRMATION_GATE = {
    "mean_auc_gain_min": 0.0,
    "mean_auprc_gain_min": 0.0,
    "one_mean_ranking_gain_min": 0.02,
    "mean_margin_separation_gain_strictly_positive": True,
    "mean_macro_f1_gain_min": -0.01,
    "seeds_with_auc_and_auprc_nondecrease_min": 2,
    "per_seed_auc_or_auprc_major_drop_floor": -0.02,
    "retention_mean_auc_gain_min": -0.02,
    "retention_mean_auprc_gain_min": -0.02,
    "retention_mean_macro_f1_gain_min": -0.02,
    "confirmed_requires_ranking_composite_bootstrap_ci_lower_gt_zero": True,
}


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_frame_hashes(rows: Sequence[dict]) -> set[str]:
    return {str(value) for row in rows for value in row.get("frame_sha256", [])}


def overlap_audit(data_dir: Path, holdout_rows: Sequence[dict]) -> dict[str, Any]:
    comparisons = {}
    holdout_ids = {str(row["id"]) for row in holdout_rows}
    holdout_sources = {str(row["source_video_sha256"]) for row in holdout_rows}
    holdout_frames = flatten_frame_hashes(holdout_rows)
    for name in (
        "client_train",
        "gate_validation",
        "d_server_lab",
        "s_seed_c_factor_separated",
        "target_test",
        "retention_test",
    ):
        rows = read_manifest(data_dir / f"manifest_{name}.jsonl")
        comparisons[name] = {
            "id_overlap": len(holdout_ids & {str(row["id"]) for row in rows}),
            "source_video_sha256_overlap": len(
                holdout_sources & {str(row["source_video_sha256"]) for row in rows}
            ),
            "frame_sha256_overlap": len(holdout_frames & flatten_frame_hashes(rows)),
        }
    passed = all(all(value == 0 for value in item.values()) for item in comparisons.values())
    return {"pass": passed, "comparisons": comparisons}


def arrays_from_logits(
    rows: Sequence[dict], logits: dict[str, list[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probs = np.asarray([softmax_vector(logits[str(row["id"])])[1] for row in rows])
    margins = np.asarray(
        [float(logits[str(row["id"])][1] - logits[str(row["id"])][0]) for row in rows]
    )
    return labels, probs, margins


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probs >= edges[index]) & (probs <= edges[index + 1])
        else:
            mask = (probs >= edges[index]) & (probs < edges[index + 1])
        if not np.any(mask):
            continue
        accuracy = float(labels[mask].mean())
        confidence = float(probs[mask].mean())
        value += float(mask.mean()) * abs(accuracy - confidence)
    return float(value)


def metrics_from_arrays(labels: np.ndarray, probs: np.ndarray, margins: np.ndarray) -> dict[str, Any]:
    preds = (probs >= FROZEN_LOCAL_CONDITION["decision_threshold"]).astype(np.int64)
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    prob_means = {str(label): float(probs[labels == label].mean()) for label in (0, 1)}
    margin_means = {str(label): float(margins[labels == label].mean()) for label in (0, 1)}
    return {
        "n": int(labels.size),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision_risk": float(precision_score(labels, preds, zero_division=0)),
        "recall_risk": float(recall_score(labels, preds, zero_division=0)),
        "predicted_positive_rate": float(preds.mean()),
        "brier": float(brier_score_loss(labels, probs)),
        "ece_10bin": expected_calibration_error(labels, probs),
        "confusion_matrix_tn_fp_fn_tp": [
            int(matrix[0, 0]),
            int(matrix[0, 1]),
            int(matrix[1, 0]),
            int(matrix[1, 1]),
        ],
        "mean_risk_probability_by_class": prob_means,
        "class_separation": prob_means["1"] - prob_means["0"],
        "mean_risk_margin_by_class": margin_means,
        "margin_class_separation": margin_means["1"] - margin_means["0"],
    }


def evaluate_logits(rows: Sequence[dict], logits: dict[str, list[float]]) -> dict[str, Any]:
    return metrics_from_arrays(*arrays_from_logits(rows, logits))


def metric_gains(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    keys = (
        "macro_f1",
        "auc",
        "auprc",
        "balanced_accuracy",
        "predicted_positive_rate",
        "brier",
        "ece_10bin",
        "class_separation",
        "margin_class_separation",
    )
    return {key: float(after[key] - before[key]) for key in keys}


def pair_indices(rows: Sequence[dict]) -> list[np.ndarray]:
    pairs = paired_rows(rows)
    index_by_id = {str(row["id"]): index for index, row in enumerate(rows)}
    return [
        np.asarray([index_by_id[str(normal["id"])], index_by_id[str(risk["id"])]])
        for normal, risk in pairs
    ]


def paired_bootstrap(
    rows: Sequence[dict],
    baseline_logits: dict[str, list[float]],
    post_logits_by_seed: Sequence[dict[str, list[float]]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    labels, base_probs, base_margins = arrays_from_logits(rows, baseline_logits)
    post_arrays = [arrays_from_logits(rows, logits) for logits in post_logits_by_seed]
    groups = pair_indices(rows)
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("auc", "auprc", "macro_f1", "margin_class_separation", "ranking_composite")}
    for _ in range(repetitions):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in chosen])
        base = metrics_from_arrays(labels[indices], base_probs[indices], base_margins[indices])
        gains = []
        for post_labels, post_probs, post_margins in post_arrays:
            post = metrics_from_arrays(
                post_labels[indices], post_probs[indices], post_margins[indices]
            )
            gains.append(metric_gains(base, post))
        mean_gain = {
            key: float(np.mean([gain[key] for gain in gains]))
            for key in ("auc", "auprc", "macro_f1", "margin_class_separation")
        }
        mean_gain["ranking_composite"] = (mean_gain["auc"] + mean_gain["auprc"]) / 2.0
        for key in samples:
            samples[key].append(mean_gain[key])
    output = {}
    for key, values in samples.items():
        array = np.asarray(values)
        output[key] = {
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
            "probability_gain_gt_zero": float(np.mean(array > 0.0)),
        }
    return {
        "sampling_unit": "matched_pair",
        "conditioned_on_local_seeds": list(LOCAL_SEEDS),
        "repetitions": repetitions,
        "metrics": output,
    }


def judge(
    confirmation_gains: Sequence[dict[str, float]],
    retention_gains: Sequence[dict[str, float]],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    mean = {
        key: float(np.mean([gain[key] for gain in confirmation_gains]))
        for key in confirmation_gains[0]
    }
    retention_mean = {
        key: float(np.mean([gain[key] for gain in retention_gains]))
        for key in retention_gains[0]
    }
    stable = sum(gain["auc"] >= 0.0 and gain["auprc"] >= 0.0 for gain in confirmation_gains)
    no_major_drop = all(
        gain["auc"] >= CONFIRMATION_GATE["per_seed_auc_or_auprc_major_drop_floor"]
        and gain["auprc"] >= CONFIRMATION_GATE["per_seed_auc_or_auprc_major_drop_floor"]
        for gain in confirmation_gains
    )
    checks = {
        "mean_auc_nondecrease": mean["auc"] >= CONFIRMATION_GATE["mean_auc_gain_min"],
        "mean_auprc_nondecrease": mean["auprc"] >= CONFIRMATION_GATE["mean_auprc_gain_min"],
        "one_mean_ranking_gain_ge_0_02": max(mean["auc"], mean["auprc"])
        >= CONFIRMATION_GATE["one_mean_ranking_gain_min"],
        "mean_margin_separation_gain": mean["margin_class_separation"] > 0.0,
        "mean_macro_f1_noninferior": mean["macro_f1"]
        >= CONFIRMATION_GATE["mean_macro_f1_gain_min"],
        "at_least_two_of_three_seeds_same_ranking_direction": stable
        >= CONFIRMATION_GATE["seeds_with_auc_and_auprc_nondecrease_min"],
        "no_seed_major_ranking_drop": no_major_drop,
        "retention_auc_within_tolerance": retention_mean["auc"]
        >= CONFIRMATION_GATE["retention_mean_auc_gain_min"],
        "retention_auprc_within_tolerance": retention_mean["auprc"]
        >= CONFIRMATION_GATE["retention_mean_auprc_gain_min"],
        "retention_macro_f1_within_tolerance": retention_mean["macro_f1"]
        >= CONFIRMATION_GATE["retention_mean_macro_f1_gain_min"],
    }
    point_gate = all(checks.values())
    ci_lower = bootstrap["metrics"]["ranking_composite"]["ci95"][0]
    if point_gate and ci_lower > 0.0:
        verdict = "confirmed_conditionally"
    elif point_gate:
        verdict = "promising_but_inconclusive"
    else:
        verdict = "failed"
    return {
        "verdict": verdict,
        "point_gate_pass": point_gate,
        "checks": checks,
        "confirmation_mean_gains": mean,
        "retention_mean_gains": retention_mean,
        "stable_seed_count": stable,
        "ranking_composite_bootstrap_ci95": bootstrap["metrics"]["ranking_composite"]["ci95"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage2_phase_a.yaml")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/fedpact_stage2_coverage_v2_20260920"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/step05_fl/run_20260607_200402/step05_fedavg_global.pt"),
    )
    parser.add_argument("--run-id", default="stage2_local_adaptation_confirmation_20260921_01")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--b0-steps", type=int, default=60)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()

    started = time.perf_counter()
    run_dir = ROOT / "artifacts" / "runs" / "fedpact_stage2" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "status.json", {"status": "running", "run_id": args.run_id})

    data_dir = (args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir).resolve()
    checkpoint = (args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint).resolve()
    paths = {
        "client_train": data_dir / "manifest_client_train.jsonl",
        "confirmation": data_dir / "manifest_seed_a_target.jsonl",
        "retention": data_dir / "manifest_retention_test.jsonl",
        "server": data_dir / "manifest_d_server_lab.jsonl",
    }
    rows = {name: read_manifest(path) for name, path in paths.items()}
    audit = overlap_audit(data_dir, rows["confirmation"])
    if not audit["pass"]:
        raise RuntimeError("confirmation manifest overlaps a prior experiment manifest")
    if len(paired_rows(rows["confirmation"])) != 15:
        raise RuntimeError("confirmation must contain exactly 15 matched pairs")

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
    set_seed(20260921)
    model, processor = build_model(cfg, device, model_id=str(cfg["model"]["client_id"]))
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    lora_params = lora_params_for_adapter(model.backbone, "client")
    checkpoint_names, checkpoint_vector, checkpoint_meta = load_fl_checkpoint(checkpoint)
    apply_named_vector(model, checkpoint_names, checkpoint_vector)
    initial_names, initial_vector = snapshot_state(model, lora_params)
    if initial_names != checkpoint_names:
        raise RuntimeError("initial checkpoint names do not match model")

    print("training fixed B0 server-only state", flush=True)
    b0_trace = train_supervised(
        model,
        processor,
        rows["server"],
        dcfg,
        lora_params,
        device,
        steps=args.b0_steps,
        seed=20260921,
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
        for split in ("confirmation", "retention")
    }
    np.savez_compressed(run_dir / "states.npz", names=np.asarray(b0_names), b0=b0_vector)

    seed_results = []
    post_logits_by_seed = {"confirmation": [], "retention": []}
    state_vectors = {"names": np.asarray(b0_names), "b0": b0_vector}
    for local_seed in LOCAL_SEEDS:
        print(f"local seed {local_seed}", flush=True)
        restore_state(model, b0_names, b0_vector)
        set_seed(local_seed)
        trace = train_paired(
            model,
            processor,
            rows["client_train"],
            dcfg,
            lora_params,
            device,
            scope="head_only",
            lora_lr=0.0,
            head_lr=1e-4,
            epochs=2,
            seed=local_seed,
        )
        state_names, state_vector = snapshot_state(model, lora_params)
        if state_names != b0_names:
            raise RuntimeError("state names changed")
        state_vectors[f"local_{local_seed}"] = state_vector
        split_results = {}
        for split in ("confirmation", "retention"):
            logits = predict_logits(model, processor, rows[split], dcfg, batch_size=1)
            post_logits_by_seed[split].append(logits)
            post_metrics = evaluate_logits(rows[split], logits)
            split_results[split] = {
                "metrics": post_metrics,
                "gains": metric_gains(baseline_metrics[split], post_metrics),
                "score_change": score_change(rows[split], baseline_logits[split], logits),
            }
        seed_results.append(
            {
                "local_seed": local_seed,
                "training": trace,
                "state_update_l2": float(np.linalg.norm(state_vector - b0_vector)),
                "splits": split_results,
            }
        )
        save_json(run_dir / "summary.partial.json", {"seed_results": seed_results})

    np.savez_compressed(run_dir / "states.npz", **state_vectors)
    save_json(
        run_dir / "logits.json",
        {"baseline": baseline_logits, "post_by_seed": post_logits_by_seed},
    )
    bootstrap = paired_bootstrap(
        rows["confirmation"],
        baseline_logits["confirmation"],
        post_logits_by_seed["confirmation"],
        repetitions=args.bootstrap_repetitions,
        seed=20260921,
    )
    verdict = judge(
        [result["splits"]["confirmation"]["gains"] for result in seed_results],
        [result["splits"]["retention"]["gains"] for result in seed_results],
        bootstrap,
    )
    summary = {
        "run_id": args.run_id,
        "status": "completed",
        "claim_scope": (
            "frozen head-only local adaptation from one fixed B0 on untouched same-task "
            "alert-interval clip-classification pairs"
        ),
        "not_claimed": [
            "early accident anticipation",
            "video8 inferiority",
            "LoRA local adaptation",
            "FedPACT Layer-2 or foundation transfer",
            "generalization across B0 training seeds or domains",
        ],
        "frozen_local_condition": FROZEN_LOCAL_CONDITION,
        "confirmation_gate_preregistered_in_script": CONFIRMATION_GATE,
        "local_seeds": list(LOCAL_SEEDS),
        "dataset": {
            "directory": str(data_dir),
            "manifest_sha256": {name: sha256_file(path) for name, path in paths.items()},
            "counts": {
                name: {"rows": len(value), "pairs": len(paired_rows(value))}
                for name, value in rows.items()
            },
            "overlap_audit": audit,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint), "meta": checkpoint_meta},
        "b0_training": b0_trace,
        "baseline_metrics": baseline_metrics,
        "seed_results": seed_results,
        "pair_bootstrap": bootstrap,
        "judgement": verdict,
        "elapsed_sec": time.perf_counter() - started,
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "status.json", {"status": "completed", "run_id": args.run_id, "verdict": verdict["verdict"]})
    print(json.dumps({"run_id": args.run_id, "judgement": verdict}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
