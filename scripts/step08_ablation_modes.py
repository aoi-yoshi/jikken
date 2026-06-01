"""
Step 8: Ablation（Lpred / Lpred+Lgrad / Lpred+Lpost）。
各実行は artifacts/runs/step08_ablation_<mode>/ にメトリクスを保存する。
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import torch
from torch.amp import autocast
from tqdm import tqdm

from src.adaptation_losses import AdaptationLosses
from src.config_loader import merged_config
from src.dataset_manifest import load_sample_row, read_manifest_filtered, stratified_subset
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import (
    attach_dual_lora,
    ema_surrogate_from_client,
    lora_params_for_adapter,
    sync_surrogate_from_client,
)
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--mode", choices=["lpred", "lpred_grad", "lpred_post"], required=True)
    args = ap.parse_args()
    cfg = merged_config()
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    ccfg = cfg["consistency"]
    art = cfg["artifacts"]
    run_dir = Path(art["runs"]) / f"step08_ablation_{args.mode}"
    ensure_dirs(run_dir, Path(art["checkpoints"]))
    log = RunLogger(run_dir, name="train")
    log.log_meta({"step": 8, "mode": args.mode})

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    rows = stratified_subset(rows, int(tcfg.get("max_train_samples", 200)), int(cfg["train"]["seed"]))
    n_train = int(len(rows) * float(dcfg.get("train_ratio", 0.8)))
    train_rows = rows[:n_train]
    eval_rows = rows[n_train:]

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")

    client_lora = lora_params_for_adapter(model.backbone, "client")
    surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
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
    epochs = int(tcfg.get("num_epochs", 1))
    max_length = int(tcfg.get("max_length", 256))

    sync_at_epoch = bool(ccfg.get("sync_surrogate_at_epoch", False))
    global_step = 0
    for ep in range(epochs):
        if sync_at_epoch:
            sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")
        model.train()
        pbar = tqdm(train_rows, desc=f"{args.mode} ep{ep}")
        for r in pbar:
            sample = load_sample_row(r, dcfg)
            imgs = sample.frames
            y = torch.tensor([sample.label], device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            model.backbone.set_adapter("client")
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                if args.mode == "lpred":
                    total, stats = losses.lpred_only(imgs, y)
                elif args.mode == "lpred_grad":
                    total, stats = losses.lpred_lgrad(imgs, y)
                else:
                    total, stats, extra = losses.lpred_lpost(imgs, y)
                    log.log({"sym_post_logits_mse": float(extra.get("sym_post_logits_mse", 0.0))})
            total.backward()
            opt.step()
            for p in surrogate_lora:
                p.grad = None
            ema_surrogate_from_client(
                client_lora, surrogate_lora, float(ccfg.get("surrogate_ema", 1.0))
            )
            global_step += 1
            stats["mode"] = args.mode
            stats["global_step"] = global_step
            stats["epoch"] = ep
            log.log(stats)
            pbar.set_postfix(total=stats["loss_total"])

        metrics = evaluate_classifier(
            model,
            processor,
            eval_rows,
            device=device,
            prompt=prompt,
            max_length=max_length,
            batch_size=bs,
            adapter="client",
            dcfg=dcfg,
        )
        log.log({"epoch": ep, **{f"eval_{k}": v for k, v in metrics.items()}})

    ckpt = Path(art["checkpoints"]) / f"step08_{args.mode}.pt"
    torch.save(
        {
            "classifier": model.classifier.state_dict(),
            "backbone_peft": model.backbone.state_dict(),
            "meta": {"mode": args.mode, "manifest": str(manifest)},
        },
        ckpt,
    )
    log.save_summary({"checkpoint": str(ckpt), "eval": metrics, "mode": args.mode})
    print("Saved:", ckpt, "eval:", metrics)


if __name__ == "__main__":
    main()
