"""
Step 3: 単体学習ベースライン（バックボーン凍結 + 線形ヘッドのみ学習）。
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
from src.dataset_manifest import read_manifest_filtered, split_train_eval_rows, stratified_subset
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.run_context import init_run
from src.train_common import device_or_auto, iter_row_chunks, load_samples
from src.vl_model import build_model, freeze_backbone, logits_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-suffix", default="")
    ap.add_argument("--eval-max", type=int, default=None)
    args = ap.parse_args()
    cfg = merged_config()
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    art = cfg["artifacts"]
    eval_max = args.eval_max if args.eval_max is not None else int(tcfg.get("eval_max", 20))

    run_id, run_dir, log, env = init_run(
        cfg, step="3", step_dir="step03_baseline", log_name="train", run_id=args.run_id, run_suffix=args.run_suffix, cli=vars(args)
    )
    ensure_dirs(Path(art["checkpoints"]))

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    rows = stratified_subset(rows, int(tcfg.get("max_train_samples", 200)), int(cfg["train"]["seed"]))
    train_rows, eval_rows = split_train_eval_rows(
        rows, train_ratio=float(dcfg.get("train_ratio", 0.8)), eval_max=eval_max
    )

    model, processor = build_model(cfg, device)
    freeze_backbone(model)
    for p in model.classifier.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.classifier.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))

    prompt = str(dcfg.get("image_prompt", ""))
    bs = int(tcfg.get("batch_size", 1))
    epochs = int(tcfg.get("num_epochs", 1))
    max_length = int(tcfg.get("max_length", 256))

    global_step = 0
    for ep in range(epochs):
        model.train()
        pbar = tqdm(list(iter_row_chunks(train_rows, bs)), desc=f"epoch {ep}")
        for chunk in pbar:
            samples = load_samples(chunk, dcfg)
            y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
            frames_batch = [s.frames for s in samples]
            opt.zero_grad(set_to_none=True)
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
            adapter=None,
            dcfg=dcfg,
        )
        log.log({"epoch": ep, **{f"eval_{k}": v for k, v in metrics.items()}})

    ckpt = Path(art["checkpoints"]) / "step03_linear_probe.pt"
    torch.save(
        {
            "classifier": model.classifier.state_dict(),
            "meta": {"manifest": str(manifest), "rows": len(rows), "run_id": run_id},
        },
        ckpt,
    )
    log.save_summary({"run_id": run_id, "run_dir": str(run_dir), "environment": env, "checkpoint": str(ckpt), "eval": metrics})
    log.log_meta({"status": "completed"})
    print("Saved:", run_dir, ckpt, "eval:", metrics)


if __name__ == "__main__":
    main()
