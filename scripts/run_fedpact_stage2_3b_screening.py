"""One-seed 3B screening for the core FedPACT Stage-II question.

This intentionally tests information value before heterogeneous-model transfer:
Server Only, matched server replay, direct-private labels, private-input teacher,
Post Only proxy selection, and Post+Delta proxy selection.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.amp import autocast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_yaml, merged_config
from src.dataset_manifest import load_sample_row, read_manifest
from src.fedpact_stage1 import apply_named_vector, flatten_named_tensors, state_named_tensors
from src.fl_utils import load_fl_checkpoint
from src.metrics import set_seed
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.step07.fedpact_layer2 import build_classwise_prototypes, kl_divergence, softmax_vector
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def announce(message: str) -> None:
    print(message, flush=True)


def snapshot_state(model, lora_params: Sequence[torch.nn.Parameter]) -> tuple[list[str], np.ndarray]:
    groups = state_named_tensors(model, lora_params)
    lora_names, lora_vector = flatten_named_tensors(groups["global_surrogate_lora"])
    head_names, head_vector = flatten_named_tensors(groups["classification_head"])
    return [*lora_names, *head_names], np.concatenate([lora_vector, head_vector]).astype(np.float32)


def restore_state(model, names: Sequence[str], vector: np.ndarray) -> None:
    apply_named_vector(model, list(names), vector)
    model.backbone.set_adapter("client")


def configure_trainable(model, lora_params: Sequence[torch.nn.Parameter], *, head: bool) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in lora_params:
        parameter.requires_grad_(True)
    if head:
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def shuffled_cycle(rows: Sequence[dict], steps: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    output: list[dict] = []
    while len(output) < steps:
        epoch = list(rows)
        rng.shuffle(epoch)
        output.extend(epoch)
    return output[:steps]


def forward_sample(model, processor, sample, prompt: str, dcfg: dict):
    return model.forward_batch(
        processor,
        sample.frames,
        prompt,
        256,
        True,
        input_mode=str(dcfg.get("input_mode", "images")),
        video_fps=sample.temporal_fps,
    )


def train_supervised(
    model,
    processor,
    rows: Sequence[dict],
    dcfg: dict,
    lora_params: Sequence[torch.nn.Parameter],
    device: torch.device,
    *,
    steps: int,
    seed: int,
    lr: float,
    update_head: bool,
) -> dict[str, Any]:
    params = configure_trainable(model, lora_params, head=update_head)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    prompt = str(dcfg["image_prompt"])
    losses: list[float] = []
    model.train()
    for row in shuffled_cycle(rows, steps, seed):
        sample = load_sample_row(row, dcfg)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = forward_sample(model, processor, sample, prompt, dcfg)
            target = torch.tensor([sample.label], device=device, dtype=torch.long)
            loss = F.cross_entropy(logits.float(), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for frame in sample.frames:
            frame.close()
    return {"steps": steps, "loss_first": losses[0], "loss_last": losses[-1], "loss_mean": float(np.mean(losses))}


def train_distill(
    model,
    processor,
    examples: Sequence[dict],
    dcfg: dict,
    lora_params: Sequence[torch.nn.Parameter],
    device: torch.device,
    *,
    steps: int,
    seed: int,
    lr: float,
) -> dict[str, Any]:
    params = configure_trainable(model, lora_params, head=False)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    prompt = str(dcfg["image_prompt"])
    losses: list[float] = []
    model.train()
    for example in shuffled_cycle(examples, steps, seed):
        sample = load_sample_row(example["row"], dcfg)
        teacher = torch.tensor(example["teacher_probability"], device=device, dtype=torch.float32).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = forward_sample(model, processor, sample, prompt, dcfg)
            loss = F.kl_div(F.log_softmax(logits.float(), dim=-1), teacher, reduction="batchmean")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for frame in sample.frames:
            frame.close()
    return {"steps": steps, "loss_first": losses[0], "loss_last": losses[-1], "loss_mean": float(np.mean(losses))}


@torch.no_grad()
def predict_logits(
    model, processor, rows: Sequence[dict], dcfg: dict, batch_size: int = 2
) -> dict[str, list[float]]:
    prompt = str(dcfg["image_prompt"])
    model.eval()
    model.backbone.set_adapter("client")
    output: dict[str, list[float]] = {}
    for start in range(0, len(rows), batch_size):
        chunk = list(rows[start : start + batch_size])
        samples = [load_sample_row(row, dcfg) for row in chunk]
        logits, _ = model.forward_samples(
            processor,
            [sample.frames for sample in samples],
            prompt,
            256,
            True,
            input_mode=str(dcfg.get("input_mode", "images")),
            video_fps=[sample.temporal_fps for sample in samples],
        )
        for row, item_logits in zip(chunk, logits):
            output[str(row["id"])] = item_logits.detach().float().cpu().tolist()
        for sample in samples:
            for frame in sample.frames:
                frame.close()
    return output


def metrics_from_logits(rows: Sequence[dict], logits: dict[str, list[float]]) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.stack([softmax_vector(logits[str(row["id"])]) for row in rows])
    predictions = (probabilities[:, 1] >= 0.5).astype(np.int64)
    recalls = recall_score(labels, predictions, labels=[0, 1], average=None, zero_division=0)
    return {
        "n": len(rows),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities[:, 1])),
        "accuracy": float((labels == predictions).mean()),
        "recall_normal": float(recalls[0]),
        "recall_risk": float(recalls[1]),
        "mean_risk_probability": float(probabilities[:, 1].mean()),
    }


def evaluate(model, processor, rows: Sequence[dict], dcfg: dict) -> dict[str, Any]:
    return metrics_from_logits(rows, predict_logits(model, processor, rows, dcfg))


def balanced_subset(rows: Sequence[dict], per_class: int) -> list[dict]:
    output: list[dict] = []
    for label in (0, 1):
        selected = [row for row in rows if int(row["label"]) == label][:per_class]
        if len(selected) != per_class:
            raise RuntimeError(f"class {label} has {len(selected)}/{per_class} rows")
        output.extend(selected)
    return output


def select_proxies(
    prototypes: Sequence[dict],
    seed_rows: Sequence[dict],
    before: dict[str, list[float]],
    after: dict[str, list[float]],
) -> dict[str, Any]:
    methods: dict[str, list[dict]] = {"m1_post_only": [], "m2_post_delta": []}
    diagnostics: list[dict] = []
    for prototype in prototypes:
        label = int(prototype["class_label"])
        candidates: list[dict] = []
        target_delta = np.asarray(prototype["delta_mu_local"], dtype=np.float64)
        for row in seed_rows:
            if int(row["label"]) != label:
                continue
            sid = str(row["id"])
            pre = softmax_vector(before[sid])
            post = softmax_vector(after[sid])
            post_kl = kl_divergence(prototype["mu_post"], post)
            delta_mse = float(np.mean(np.square(target_delta - (post - pre))))
            candidates.append({"row": row, "post_kl": post_kl, "delta_mse": delta_mse})
        post_scale = float(np.median([item["post_kl"] for item in candidates])) + 1e-12
        delta_scale = float(np.median([item["delta_mse"] for item in candidates])) + 1e-12
        for item in candidates:
            item["normalized_joint"] = item["post_kl"] / post_scale + item["delta_mse"] / delta_scale
        m1 = sorted(candidates, key=lambda item: (item["post_kl"], str(item["row"]["id"])))[:4]
        m2 = sorted(candidates, key=lambda item: (item["normalized_joint"], str(item["row"]["id"])))[:4]
        teacher = list(prototype["mu_post"])
        for method, chosen in (("m1_post_only", m1), ("m2_post_delta", m2)):
            methods[method].extend(
                {"row": item["row"], "teacher_probability": teacher} for item in chosen
            )
        diagnostics.append(
            {
                "class_label": label,
                "post_scale": post_scale,
                "delta_scale": delta_scale,
                "m1": [
                    {"id": item["row"]["id"], "post_kl": item["post_kl"], "delta_mse": item["delta_mse"]}
                    for item in m1
                ],
                "m2": [
                    {
                        "id": item["row"]["id"],
                        "post_kl": item["post_kl"],
                        "delta_mse": item["delta_mse"],
                        "normalized_joint": item["normalized_joint"],
                    }
                    for item in m2
                ],
            }
        )
    return {"examples": methods, "diagnostics": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage2_phase_a.yaml")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/datasets/fedpact_stage2_coverage_v2_20260920"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/step05_fl/run_20260607_200402/step05_fedavg_global.pt"))
    parser.add_argument("--run-id", default="stage2_3b_screen_20260920_01")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--b0-steps", type=int, default=60)
    parser.add_argument("--method-steps", type=int, default=8)
    parser.add_argument("--supervised-lr", type=float, default=1e-4)
    parser.add_argument("--foundation-lr", type=float, default=1e-6)
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    run_dir = ROOT / "artifacts" / "runs" / "fedpact_stage2" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "status.json", {"status": "running", "run_id": args.run_id})
    cfg = merged_config(load_yaml(args.config))
    cfg["train"]["seed"] = args.seed
    dcfg = dict(cfg["data"])
    dcfg.update({"use_num_frames": 4, "use_frame_slice": "last", "weather_filter": ""})
    set_seed(args.seed)
    data_dir = (args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir).resolve()
    checkpoint = (args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint).resolve()
    manifests = {
        name: read_manifest(data_dir / f"manifest_{name}.jsonl")
        for name in ("client_train", "gate_validation", "target_test", "retention_test", "d_server_lab", "s_seed_c_factor_separated")
    }
    if args.quick_smoke:
        per_class = {
            "client_train": 2,
            "gate_validation": 1,
            "target_test": 1,
            "retention_test": 1,
            "d_server_lab": 2,
            "s_seed_c_factor_separated": 2,
        }
        manifests = {name: balanced_subset(rows, per_class[name]) for name, rows in manifests.items()}

    device = torch.device(device_or_auto(True) if args.device == "auto" else args.device)
    announce(f"loading 3B model on {device}")
    model, processor = build_model(cfg, device, model_id=str(cfg["model"]["client_id"]))
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    lora_params = lora_params_for_adapter(model.backbone, "client")
    checkpoint_names, checkpoint_vector, checkpoint_meta = load_fl_checkpoint(checkpoint)
    apply_named_vector(model, checkpoint_names, checkpoint_vector)
    initial_names, initial_vector = snapshot_state(model, lora_params)
    if initial_names != checkpoint_names:
        raise RuntimeError("initial checkpoint names do not match 3B model")

    traces: dict[str, Any] = {"initial_checkpoint_meta": checkpoint_meta}
    announce("phase 1/6: train B0 server-only")
    traces["b0_training"] = train_supervised(
        model, processor, manifests["d_server_lab"], dcfg, lora_params, device,
        steps=args.b0_steps, seed=args.seed, lr=args.supervised_lr, update_head=True,
    )
    b0_names, b0_vector = snapshot_state(model, lora_params)
    results: dict[str, Any] = {"validation": {}, "target_test": {}, "retention_test": {}}
    results["validation"]["b0_server_only"] = evaluate(model, processor, manifests["gate_validation"], dcfg)

    update_rows = balanced_subset(manifests["client_train"], min(4, len(manifests["client_train"]) // 2))
    announce("phase 2/6: direct-private and local adaptation")
    restore_state(model, b0_names, b0_vector)
    traces["b1_training"] = train_supervised(
        model, processor, update_rows, dcfg, lora_params, device,
        steps=args.method_steps, seed=args.seed + 1, lr=args.foundation_lr, update_head=False,
    )
    b1_names, b1_vector = snapshot_state(model, lora_params)
    results["validation"]["b1_direct_private"] = evaluate(model, processor, manifests["gate_validation"], dcfg)

    announce("phase 3/6: score factor-separated server seeds")
    restore_state(model, b0_names, b0_vector)
    client_pre_logits = predict_logits(model, processor, manifests["client_train"], dcfg)
    validation_pre = evaluate(model, processor, manifests["gate_validation"], dcfg)
    traces["local_training"] = train_supervised(
        model, processor, manifests["client_train"], dcfg, lora_params, device,
        steps=len(manifests["client_train"]), seed=args.seed + 2, lr=args.supervised_lr, update_head=True,
    )
    local_names, local_vector = snapshot_state(model, lora_params)
    client_post_logits = predict_logits(model, processor, manifests["client_train"], dcfg)
    validation_post = evaluate(model, processor, manifests["gate_validation"], dcfg)
    results["validation"]["local_pre"] = validation_pre
    results["validation"]["local_post"] = validation_post
    records = [
        {
            "label": int(row["label"]),
            "pre_logits": client_pre_logits[str(row["id"])],
            "post_logits": client_post_logits[str(row["id"])],
        }
        for row in manifests["client_train"]
    ]
    prototypes = build_classwise_prototypes(records, client_id="client-0", server_round=1)

    restore_state(model, b0_names, b0_vector)
    seed_before = predict_logits(model, processor, manifests["s_seed_c_factor_separated"], dcfg)
    restore_state(model, local_names, local_vector)
    seed_after = predict_logits(model, processor, manifests["s_seed_c_factor_separated"], dcfg)
    selections = select_proxies(prototypes, manifests["s_seed_c_factor_separated"], seed_before, seed_after)

    qpost_examples = [
        {"row": row, "teacher_probability": softmax_vector(client_post_logits[str(row["id"])]).tolist()}
        for row in update_rows
    ]
    method_states: dict[str, tuple[list[str], np.ndarray]] = {
        "b0_server_only": (b0_names, b0_vector),
        "b1_direct_private": (b1_names, b1_vector),
        "local_surrogate_post": (local_names, local_vector),
    }
    announce("phase 4/6: matched replay, private-teacher, M1 and M2 updates")
    restore_state(model, b0_names, b0_vector)
    traces["b0_replay_training"] = train_supervised(
        model, processor, balanced_subset(manifests["d_server_lab"], min(4, len(manifests["d_server_lab"]) // 2)), dcfg, lora_params, device,
        steps=args.method_steps, seed=args.seed + 3, lr=args.foundation_lr, update_head=False,
    )
    method_states["b0_replay"] = snapshot_state(model, lora_params)

    restore_state(model, b0_names, b0_vector)
    traces["b2_training"] = train_distill(
        model, processor, qpost_examples, dcfg, lora_params, device,
        steps=args.method_steps, seed=args.seed + 4, lr=args.foundation_lr,
    )
    method_states["b2_private_teacher"] = snapshot_state(model, lora_params)

    for offset, name in enumerate(("m1_post_only", "m2_post_delta"), start=5):
        restore_state(model, b0_names, b0_vector)
        traces[name + "_training"] = train_distill(
            model, processor, selections["examples"][name], dcfg, lora_params, device,
            steps=args.method_steps, seed=args.seed + offset, lr=args.foundation_lr,
        )
        method_states[name] = snapshot_state(model, lora_params)

    announce("phase 5/6: frozen target and retention evaluation")
    for method, (names, vector) in method_states.items():
        restore_state(model, names, vector)
        if method not in results["validation"]:
            results["validation"][method] = evaluate(model, processor, manifests["gate_validation"], dcfg)
        results["target_test"][method] = evaluate(model, processor, manifests["target_test"], dcfg)
        results["retention_test"][method] = evaluate(model, processor, manifests["retention_test"], dcfg)

    base = results["target_test"]["b0_server_only"]["macro_f1"]
    replay = results["target_test"]["b0_replay"]["macro_f1"]
    summary = {
        "run_id": args.run_id,
        "status": "completed",
        "claim_scope": "single-seed 3B-to-3B fixed-split screening; not confirmation",
        "configuration": {
            "seed": args.seed,
            "coverage": "c_factor_separated",
            "b0_steps": args.b0_steps,
            "method_steps": args.method_steps,
            "supervised_lr": args.supervised_lr,
            "foundation_lr": args.foundation_lr,
            "proxy_budget_per_class": 4,
            "proxy_transform": "identity",
            "joint_score": "post_kl/median(post_kl) + delta_mse/median(delta_mse)",
        },
        "gates_validation": {
            "headroom_b1_minus_b0": results["validation"]["b1_direct_private"]["macro_f1"] - results["validation"]["b0_server_only"]["macro_f1"],
            "local_gain_post_minus_pre": validation_post["macro_f1"] - validation_pre["macro_f1"],
            "private_teacher_b2_minus_b0": results["validation"]["b2_private_teacher"]["macro_f1"] - results["validation"]["b0_server_only"]["macro_f1"],
        },
        "target_differences": {
            "b1_minus_b0": results["target_test"]["b1_direct_private"]["macro_f1"] - base,
            "b2_minus_b0": results["target_test"]["b2_private_teacher"]["macro_f1"] - base,
            "m1_minus_b0_replay": results["target_test"]["m1_post_only"]["macro_f1"] - replay,
            "m2_minus_b0_replay": results["target_test"]["m2_post_delta"]["macro_f1"] - replay,
            "m2_minus_m1": results["target_test"]["m2_post_delta"]["macro_f1"] - results["target_test"]["m1_post_only"]["macro_f1"],
            "local_surrogate_post_minus_pre": results["target_test"]["local_surrogate_post"]["macro_f1"] - base,
        },
        "results": results,
        "prototypes": prototypes,
        "proxy_selection": selections["diagnostics"],
        "communication_bytes": len(json.dumps(prototypes, separators=(",", ":")).encode("utf-8")),
        "training_traces": traces,
        "elapsed_sec": time.perf_counter() - started,
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "status.json", {"status": "completed", "run_id": args.run_id})
    announce("phase 6/6: completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
