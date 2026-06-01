"""
Step 4: LoRA（q_proj/v_proj）+ 分類ヘッドの単体学習（Adaptation損失なし）。
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

from src.config_loader import merged_config
from src.dataset_manifest import load_sample_row, read_manifest_filtered, stratified_subset
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, lora_params_for_adapter, sync_surrogate_from_client
from src.train_common import device_or_auto, iter_row_chunks, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = merged_config()
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    art = cfg["artifacts"]
    run_dir = Path(art["runs"]) / "step04_lora"
    ensure_dirs(run_dir, Path(art["checkpoints"]))
    log = RunLogger(run_dir, name="train")
    log.log_meta({"step": 4, "mode": "lora_ce_only"})

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
    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))

    prompt = str(dcfg.get("image_prompt", ""))
    bs = int(tcfg.get("batch_size", 1))
    epochs = int(tcfg.get("num_epochs", 1))
    max_length = int(tcfg.get("max_length", 256))

    global_step = 0
    for ep in range(epochs):
        sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")
        model.train()
        pbar = tqdm(list(iter_row_chunks(train_rows, bs)), desc=f"epoch {ep}")
        for chunk in pbar:
            samples = load_samples(chunk, dcfg)
            y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
            frames_batch = [s.frames for s in samples]
            opt.zero_grad(set_to_none=True)
            model.backbone.set_adapter("client")
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, _ = model.forward_samples(
                    processor, frames_batch, prompt, max_length, output_hidden_states=True
                )
                loss = logits_loss(logits.float(), y)
            loss.backward()
            opt.step()
            global_step += 1
            log.log({"step": global_step, "loss": float(loss.detach().cpu()), "epoch": ep, "batch_size": len(samples)})
            pbar.set_postfix(loss=float(loss.detach().cpu()), bs=len(samples))

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

    ckpt = Path(art["checkpoints"]) / "step04_lora_ce.pt"
    torch.save(
        {
            "classifier": model.classifier.state_dict(),
            "backbone_peft": model.backbone.state_dict(),
            "meta": {"manifest": str(manifest), "rows": len(rows)},
        },
        ckpt,
    )
    log.save_summary({"checkpoint": str(ckpt), "eval": metrics})
    print("Saved:", ckpt, "eval:", metrics)


if __name__ == "__main__":
    main()
