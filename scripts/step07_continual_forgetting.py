"""
Step 7: 擬似継続学習（タスク分割 + 直前タスク精度の変化＝forgetting ログ）。
Avalanche 本体は必須にせず、まずは同一パイプラインで計測可能にする。
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import random

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


def split_tasks(rows: list[dict], num_tasks: int, seed: int = 0) -> list[list[dict]]:
    """Round-robin split keeping class balance roughly equal per task."""
    rng = random.Random(seed)
    by_class: dict[int, list[dict]] = {}
    for r in rows:
        by_class.setdefault(int(r["label"]), []).append(r)
    for k in by_class:
        rng.shuffle(by_class[k])
    out: list[list[dict]] = [[] for _ in range(num_tasks)]
    for _, items in by_class.items():
        for i, r in enumerate(items):
            out[i % num_tasks].append(r)
    for t in out:
        rng.shuffle(t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--tasks", type=int, default=3)
    args = ap.parse_args()
    cfg = merged_config()
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    art = cfg["artifacts"]
    run_dir = Path(art["runs"]) / "step07_continual"
    ensure_dirs(run_dir)
    log = RunLogger(run_dir, name="continual")
    log.log_meta({"step": 7, "tasks": int(args.tasks)})

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    rows = stratified_subset(rows, int(tcfg.get("max_train_samples", 200)), int(cfg["train"]["seed"]))
    tasks = split_tasks(rows, int(args.tasks), seed=int(cfg["train"]["seed"]))
    for ti, t in enumerate(tasks):
        ys = [r["label"] for r in t]
        log.log({"task_meta": ti, "size": len(t), "pos": sum(1 for y in ys if y == 1), "neg": sum(1 for y in ys if y == 0)})

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")

    client_lora = lora_params_for_adapter(model.backbone, "client")
    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))

    task_eval_rows: list[list[dict]] = []
    for t in tasks:
        task_eval_rows.append(t[: max(1, len(t) // 5)])

    prev_metrics: dict[int, dict[str, float]] = {}
    bs = int(tcfg.get("batch_size", 1))
    for ti, task_rows in enumerate(tasks):
        if not task_rows:
            continue
        sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")
        model.train()
        for chunk in tqdm(list(iter_row_chunks(task_rows, bs)), desc=f"task {ti}"):
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

        for pj, eval_rows in enumerate(task_eval_rows):
            if not eval_rows:
                continue
            m = evaluate_classifier(
                model,
                processor,
                eval_rows,
                device=device,
                prompt=prompt,
                max_length=max_length,
                batch_size=int(tcfg.get("batch_size", 1)),
                adapter="client",
                dcfg=dcfg,
            )
            if pj in prev_metrics:
                drop = prev_metrics[pj]["accuracy"] - m["accuracy"]
                log.log(
                    {
                        "after_task": ti,
                        "eval_task": pj,
                        "accuracy": m["accuracy"],
                        "f1": m["f1_macro"],
                        "forgetting_acc_drop": float(drop),
                    }
                )
            else:
                log.log({"after_task": ti, "eval_task": pj, "accuracy": m["accuracy"], "f1": m["f1_macro"]})
            prev_metrics[pj] = m

    log.save_summary({"note": "forgetting_acc_drop>0 means accuracy decreased on earlier eval slice"})
    print("Done. See:", run_dir)


if __name__ == "__main__":
    main()
