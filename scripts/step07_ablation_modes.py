"""
Step 7: Ablation（Lpred / Lpred+Lgrad / Lpred+Lpost）。
各実行は artifacts/runs/step07_ablation_<mode>/<run_id>/ にメトリクスを保存する。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.amp import autocast
from tqdm import tqdm

from src.adaptation_losses import AdaptationLosses
from src.config_loader import merged_config
from src.dataset_manifest import load_sample_row, read_manifest_filtered, split_train_eval_rows, stratified_subset
from src.diff_probe import lora_delta_norm, lora_inventory, snapshot_lora
from src.experiment_record import save_config_snapshot, write_step07_experiment_report
from src.paths import ensure_dirs
from src.peft_setup import (
    attach_dual_lora,
    ema_surrogate_from_client,
    lora_params_for_adapter,
    sync_surrogate_from_client,
)
from src.resource_metrics import (
    adapter_size_mb,
    build_training_hardware_snapshot,
    GpuUtilTracker,
    profile_one_step,
    reset_peak_memory_stats,
    trainable_param_count,
)
from src.run_context import init_run
from src.metrics import evaluate_classifier, set_seed
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    tcfg = cfg.setdefault("train", {})
    ccfg = cfg.setdefault("consistency", {})
    if args.max_train_samples is not None:
        tcfg["max_train_samples"] = int(args.max_train_samples)
    if args.eval_max is not None:
        tcfg["eval_max"] = int(args.eval_max)
    if args.epochs is not None:
        tcfg["num_epochs"] = int(args.epochs)
    if args.w_pred is not None:
        ccfg["w_pred"] = float(args.w_pred)
    if args.w_grad is not None:
        ccfg["w_grad"] = float(args.w_grad)
    if args.w_post is not None:
        ccfg["w_post"] = float(args.w_post)
    return cfg


def _build_experiment_design(
    cfg: dict,
    *,
    mode: str,
    train_samples: int,
    eval_samples: int,
    total_steps: int,
    adapter_size_mb: float,
    trainable_params: int,
) -> Dict[str, Any]:
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    lcfg = cfg["lora"]
    ccfg = cfg["consistency"]
    return {
        "step": 7,
        "mode": mode,
        "dataset": "NEXAR Collision Prediction (binary)",
        "model_id": cfg["model"]["id"],
        "lora_r": lcfg.get("r"),
        "lora_alpha": lcfg.get("lora_alpha"),
        "lora_target_modules": lcfg.get("target_modules"),
        "w_pred": ccfg.get("w_pred"),
        "w_grad": ccfg.get("w_grad"),
        "w_post": ccfg.get("w_post"),
        "surrogate_ema": ccfg.get("surrogate_ema"),
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "total_steps": total_steps,
        "lr": tcfg.get("lr"),
        "weight_decay": tcfg.get("weight_decay"),
        "batch_size": tcfg.get("batch_size"),
        "max_length": tcfg.get("max_length"),
        "seed": tcfg.get("seed"),
        "adapter_size_mb": adapter_size_mb,
        "trainable_parameters": trainable_params,
        "manifest_path": dcfg.get("manifest_path"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--mode", choices=["lpred", "lpred_grad", "lpred_post"], required=True)
    ap.add_argument("--epochs", type=int, default=None, help="学習 epoch 数（未指定時は train.num_epochs または 1）")
    ap.add_argument("--max-steps", type=int, default=None, help="総学習ステップ上限（スモーク用）")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--eval-max", type=int, default=None)
    ap.add_argument("--run-suffix", default="", help="run_id 接尾辞")
    ap.add_argument("--w-pred", type=float, default=None)
    ap.add_argument("--w-grad", type=float, default=None)
    ap.add_argument("--w-post", type=float, default=None)
    args = ap.parse_args()

    cfg = merged_config()
    cfg = _apply_cli_overrides(cfg, args)
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    ccfg = cfg["consistency"]
    art = cfg["artifacts"]

    run_id, run_dir, log, env = init_run(
        cfg,
        step="7",
        step_dir=f"step07_ablation_{args.mode}",
        log_name="train",
        run_suffix=args.run_suffix,
        cli=vars(args),
        extra_meta={"mode": args.mode, "consistency_mode": args.mode},
    )
    ensure_dirs(Path(art["checkpoints"]))

    latest_ptr = Path(art["runs"]) / f"step07_ablation_{args.mode}" / "LATEST_RUN_ID"
    latest_ptr.parent.mkdir(parents=True, exist_ok=True)
    latest_ptr.write_text(run_id, encoding="utf-8")

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    rows = stratified_subset(rows, int(tcfg.get("max_train_samples", 200)), int(cfg["train"]["seed"]))
    eval_max = int(tcfg.get("eval_max", 20))
    if eval_max >= len(rows):
        eval_max = max(1, len(rows) // 5)
    train_rows, eval_rows = split_train_eval_rows(
        rows, train_ratio=float(dcfg.get("train_ratio", 0.8)), eval_max=eval_max
    )
    if not train_rows:
        raise RuntimeError(
            f"train_rows is empty (manifest subset={len(rows)}, eval_max={eval_max}). "
            "Increase --max-train-samples or lower --eval-max."
        )

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")

    client_lora = lora_params_for_adapter(model.backbone, "client")
    surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
    lora_start = snapshot_lora(client_lora)
    adapter_mb = adapter_size_mb(client_lora)
    trainable_n = trainable_param_count(model)
    inv_client = lora_inventory(model.backbone, "client")
    inv_surrogate = lora_inventory(model.backbone, "surrogate")

    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))

    losses = AdaptationLosses(
        model,
        processor,
        w_pred=float(ccfg.get("w_pred", 0.5)),
        w_grad=float(ccfg.get("w_grad", 0.1)),
        w_post=float(ccfg.get("w_post", 0.5)),
        inner_lr=float(ccfg.get("inner_lr", 1e-4)),
        max_length=int(tcfg.get("max_length", 256)),
        prompt=str(dcfg.get("image_prompt", "")),
        client_adapter="client",
        surrogate_adapter="surrogate",
        client_lora_params=client_lora,
        surrogate_lora_params=surrogate_lora,
        temperature=float(ccfg.get("temperature", 2.0)),
    )

    prompt = str(dcfg.get("image_prompt", ""))
    bs = int(tcfg.get("batch_size", 1))
    epochs = int(args.epochs if args.epochs is not None else tcfg.get("num_epochs", 1))
    max_length = int(tcfg.get("max_length", 256))
    max_steps = args.max_steps
    sync_at_epoch = bool(ccfg.get("sync_surrogate_at_epoch", False))

    save_config_snapshot(
        run_dir,
        cfg=cfg,
        cli_args=vars(args),
        extra={
            "run_id": run_id,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "adapter_size_mb": adapter_mb,
        },
    )

    epoch_history_path = run_dir / "epoch_history.jsonl"
    history: List[Dict[str, Any]] = []
    global_step = 0
    last_stats: Dict[str, float] = {}
    profiler_result = None
    eval_before = evaluate_classifier(
        model, processor, eval_rows, device=device, prompt=prompt,
        max_length=max_length, batch_size=bs, adapter="client", dcfg=dcfg,
    )

    for ep in range(epochs):
        if sync_at_epoch:
            sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")
        model.train()
        reset_peak_memory_stats()
        epoch_start = time.perf_counter()
        gpu_tracker = GpuUtilTracker(device_index=0 if device.type == "cuda" else 0)
        epoch_loss_sum = 0.0
        epoch_steps = 0

        pbar = tqdm(train_rows, desc=f"{args.mode} ep{ep}")
        for r in pbar:
            if max_steps is not None and global_step >= max_steps:
                break

            sample = load_sample_row(r, dcfg)
            imgs = sample.frames
            y = torch.tensor([sample.label], device=device, dtype=torch.long)

            def _train_step() -> None:
                opt.zero_grad(set_to_none=True)
                model.backbone.set_adapter("client")
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    total, stats, extra = losses.compute(args.mode, imgs, y)
                    if extra.get("sym_post_logits_mse") is not None:
                        log.log({"sym_post_logits_mse": float(extra["sym_post_logits_mse"])})
                total.backward()
                opt.step()
                for p in surrogate_lora:
                    p.grad = None
                ema_surrogate_from_client(
                    client_lora, surrogate_lora, float(ccfg.get("surrogate_ema", 1.0))
                )
                nonlocal last_stats, epoch_loss_sum
                last_stats = dict(stats)
                epoch_loss_sum += stats["loss_total"]

            if global_step == 0 and device.type == "cuda":
                profiler_result = profile_one_step(_train_step, device=device)
            else:
                _train_step()

            global_step += 1
            epoch_steps += 1
            stats = dict(last_stats)
            stats["mode"] = args.mode
            stats["global_step"] = global_step
            stats["epoch"] = ep
            stats["sample_id"] = sample.sample_id
            log.log(stats)
            pbar.set_postfix(total=stats["loss_total"])
            gpu_tracker.maybe_sample()

        if max_steps is not None and global_step >= max_steps:
            break

        train_time = time.perf_counter() - epoch_start
        avg_loss = epoch_loss_sum / max(1, epoch_steps)
        eval_metrics = evaluate_classifier(
            model, processor, eval_rows, device=device, prompt=prompt,
            max_length=max_length, batch_size=bs, adapter="client", dcfg=dcfg,
        )
        client_delta = lora_delta_norm(lora_start, snapshot_lora(client_lora))
        hw = build_training_hardware_snapshot(
            elapsed_sec=train_time,
            steps=max(1, epoch_steps),
            gpu_tracker=gpu_tracker,
            device=device,
            profiler=profiler_result,
            phase="train",
        )
        ep_record = {
            "epoch": ep,
            "loss": {
                "loss_total": avg_loss,
                "loss_task": last_stats.get("loss_task"),
                "loss_pred": last_stats.get("loss_pred"),
                "loss_grad": last_stats.get("loss_grad"),
                "loss_post": last_stats.get("loss_post"),
            },
            "eval": eval_metrics,
            "client_delta": client_delta,
            "hardware_metrics": hw,
        }
        history.append(ep_record)
        with epoch_history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep_record, ensure_ascii=False) + "\n")
        log.log({"event": "epoch_end", **ep_record})

    eval_after = evaluate_classifier(
        model, processor, eval_rows, device=device, prompt=prompt,
        max_length=max_length, batch_size=bs, adapter="client", dcfg=dcfg,
    )

    ckpt = Path(art["checkpoints"]) / f"step07_{args.mode}_{run_id}.pt"
    torch.save(
        {
            "classifier": model.classifier.state_dict(),
            "backbone_peft": model.backbone.state_dict(),
            "meta": {"mode": args.mode, "run_id": run_id, "manifest": str(manifest)},
        },
        ckpt,
    )

    experiment_design = _build_experiment_design(
        cfg,
        mode=args.mode,
        train_samples=len(train_rows),
        eval_samples=len(eval_rows),
        total_steps=global_step,
        adapter_size_mb=adapter_mb,
        trainable_params=trainable_n,
    )
    hw_profile = history[-1]["hardware_metrics"] if history else {}
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": args.mode,
        "manifest": str(manifest),
        "environment": env,
        "experiment_design": experiment_design,
        "lora_inventory": {"client": inv_client, "surrogate": inv_surrogate},
        "history": history,
        "hardware_profile": hw_profile,
        "loss_summary": last_stats,
        "eval": {
            "accuracy_before": eval_before.get("accuracy"),
            "accuracy_after": eval_after.get("accuracy"),
            "f1_before": eval_before.get("f1_macro"),
            "f1_after": eval_after.get("f1_macro"),
            "n_eval": eval_after.get("n_eval"),
        },
        "checkpoint": str(ckpt),
        "status": "completed",
    }
    log.save_summary(summary)
    write_step07_experiment_report(run_dir, summary)
    log.log_meta({"status": "completed"})
    print("Saved:", ckpt, "eval:", eval_after)


if __name__ == "__main__":
    main()
