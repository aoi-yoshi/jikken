"""
Step 7 Alternating Self-Distillation:
  Phase A: LoRA-1 をラベル損失のみで学習（teacher 役を仕込む）
  Phase B: LoRA-2 を整合損失付きで学習。teacher = LoRA-1
           --mode lpred       : L_task + w_pred * L_pred
           --mode lpred_grad  : L_task + w_pred * L_pred + w_grad * L_grad
  Phase C: LoRA-2 の重みを LoRA-1 へコピー
  プラン:  A -> B -> C -> A -> B   （2 サイクル、最後の C なし）

Phase A 終了時は LoRA-1 を、Phase B 終了時は LoRA-2 をそれぞれ eval する。
ログは artifacts/runs/step07_alt_<mode>/ に保存され、各行に phase / cycle /
epoch_in_phase を付けて、フェーズ毎の loss と acc 推移が追えるようにする。
"""
from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from src.adaptation_losses import AdaptationLosses
from src.config_loader import merged_config
from src.dataset_manifest import load_sample_row, read_manifest_filtered
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import (
    attach_dual_lora,
    lora_params_for_adapter,
    sync_surrogate_from_client,
)
from src.train_common import device_or_auto, iter_row_chunks, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


# 学習データ・評価データを「クラスごと固定枚数」で分ける。
def stratified_train_eval_split(
    rows: List[dict],
    train_per_class: int,
    eval_per_class: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    buckets: Dict[int, List[dict]] = {}
    for r in rows:
        buckets.setdefault(int(r["label"]), []).append(r)
    train_rows: List[dict] = []
    eval_rows: List[dict] = []
    for label in sorted(buckets.keys()):
        items = list(buckets[label])
        rng.shuffle(items)
        need = train_per_class + eval_per_class
        if len(items) < need:
            raise RuntimeError(
                f"class {label} has only {len(items)} samples; need {need}"
            )
        train_rows.extend(items[:train_per_class])
        eval_rows.extend(items[train_per_class : train_per_class + eval_per_class])
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows, eval_rows


def _set_requires_grad(params: List[nn.Parameter], flag: bool) -> None:
    for p in params:
        p.requires_grad = flag


def train_phase_a(
    *,
    model,
    processor,
    train_rows: List[dict],
    dcfg: dict,
    batch_size: int,
    lora1_params: List[nn.Parameter],
    lora2_params: List[nn.Parameter],
    opt: torch.optim.Optimizer,
    prompt: str,
    max_length: int,
    device: torch.device,
    log: RunLogger,
    cycle: int,
    epochs: int,
    accum_steps: int,
    global_step_ref: List[int],
) -> None:
    """LoRA-1 をラベル損失のみで学習。LoRA-2 は触らない（requires_grad=False）。"""
    model.train()
    model.backbone.set_adapter("lora1")
    _set_requires_grad(lora1_params, True)
    _set_requires_grad(lora2_params, False)

    for ep in range(epochs):
        pbar = tqdm(list(iter_row_chunks(train_rows, batch_size)), desc=f"cycle{cycle} A ep{ep}")
        opt.zero_grad(set_to_none=True)
        accum_count = 0
        for chunk in pbar:
            samples = load_samples(chunk, dcfg)
            y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
            frames_batch = [s.frames for s in samples]
            with autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                logits, _ = model.forward_samples(
                    processor,
                    frames_batch,
                    prompt,
                    max_length,
                    output_hidden_states=True,
                )
                l_task = logits_loss(logits, y)
                loss = l_task / accum_steps
            loss.backward()
            for p in lora2_params:
                p.grad = None
            accum_count += 1
            if accum_count >= accum_steps:
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum_count = 0

            global_step_ref[0] += 1
            log.log(
                {
                    "phase": "A",
                    "cycle": cycle,
                    "epoch_in_phase": ep,
                    "global_step": global_step_ref[0],
                    "active_adapter": "lora1",
                    "batch_size": len(samples),
                    "loss_task": float(l_task.detach().cpu()),
                    "loss_pred": 0.0,
                    "loss_grad": 0.0,
                    "loss_post": 0.0,
                    "loss_total": float(l_task.detach().cpu()),
                }
            )
            pbar.set_postfix(task=float(l_task.detach().cpu()), bs=len(samples))

        if accum_count > 0:
            opt.step()
            opt.zero_grad(set_to_none=True)


def train_phase_b(
    *,
    model,
    processor,
    losses_b: AdaptationLosses,
    train_rows: List[dict],
    dcfg: dict,
    lora1_params: List[nn.Parameter],
    lora2_params: List[nn.Parameter],
    opt: torch.optim.Optimizer,
    mode: str,
    prompt: str,
    max_length: int,
    device: torch.device,
    log: RunLogger,
    cycle: int,
    epochs: int,
    accum_steps: int,
    global_step_ref: List[int],
) -> None:
    """LoRA-2 を整合損失付きで学習。teacher = LoRA-1（凍結扱い）。"""
    del prompt, max_length  # AdaptationLosses 内で保持
    model.train()
    # AdaptationLosses._set_adapter が両方 requires_grad=True に戻すので、
    # ここでは lora1 を learnable 扱いに残しつつ、optimizer に渡さない構成で
    # 「勾配計算は走るが更新はされない」状態を作る。各 backward 後に lora1 の
    # grad を捨てて、teacher 側に勾配が漏れないようにする。
    _set_requires_grad(lora1_params, True)
    _set_requires_grad(lora2_params, True)

    for ep in range(epochs):
        pbar = tqdm(train_rows, desc=f"cycle{cycle} B[{mode}] ep{ep}")
        opt.zero_grad(set_to_none=True)
        accum_count = 0
        for r in pbar:
            sample = load_sample_row(r, dcfg)
            y = torch.tensor([sample.label], device=device, dtype=torch.long)
            with autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                if mode == "lpred":
                    total, stats = losses_b.lpred_only(sample.frames, y)
                else:
                    total, stats = losses_b.lpred_lgrad(sample.frames, y)
                loss = total / accum_steps
            loss.backward()
            # teacher (lora1) への勾配は破棄して更新を抑止
            for p in lora1_params:
                p.grad = None
            accum_count += 1
            if accum_count >= accum_steps:
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum_count = 0

            global_step_ref[0] += 1
            row = {
                "phase": "B",
                "cycle": cycle,
                "epoch_in_phase": ep,
                "global_step": global_step_ref[0],
                "active_adapter": "lora2",
            }
            row.update(stats)
            log.log(row)
            pbar.set_postfix(total=stats["loss_total"])

        if accum_count > 0:
            opt.step()
            opt.zero_grad(set_to_none=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--mode", choices=["lpred", "lpred_grad"], required=True)
    ap.add_argument("--train-per-class", type=int, default=20)
    ap.add_argument("--eval-per-class", type=int, default=10)
    ap.add_argument("--phase-a-epochs", type=int, default=2)
    ap.add_argument("--phase-b-epochs", type=int, default=2)
    ap.add_argument("--cycles", type=int, default=2, help="A-B(-C) を何サイクル回すか。最終サイクルは C を省略")
    ap.add_argument("--accum-steps", type=int, default=2)
    ap.add_argument("--run-suffix", type=str, default="", help="出力フォルダ名に付ける追加サフィックス（短縮版を区別したいとき）")
    args = ap.parse_args()

    cfg = merged_config()
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    bs = max(1, int(tcfg.get("batch_size", 1)))
    ccfg = cfg["consistency"]
    art = cfg["artifacts"]

    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    run_dir = Path(art["runs"]) / f"step07_alt_self_distill_{args.mode}{suffix}"
    ensure_dirs(run_dir, Path(art["checkpoints"]))
    log = RunLogger(run_dir, name="train")
    cycles = max(1, int(args.cycles))
    plan_str = "-".join(
        [
            "A-B" + ("-C" if c < cycles else "")
            for c in range(1, cycles + 1)
        ]
    )
    log.log_meta(
        {
            "step": 7,
            "scheme": "alt_self_distill",
            "mode": args.mode,
            "train_per_class": args.train_per_class,
            "eval_per_class": args.eval_per_class,
            "phase_a_epochs": args.phase_a_epochs,
            "phase_b_epochs": args.phase_b_epochs,
            "cycles": cycles,
            "accum_steps": args.accum_steps,
            "plan": plan_str,
            "run_suffix": args.run_suffix,
        }
    )

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    train_rows, eval_rows = stratified_train_eval_split(
        rows,
        train_per_class=int(args.train_per_class),
        eval_per_class=int(args.eval_per_class),
        seed=int(cfg["train"]["seed"]),
    )
    print(f"[data] train={len(train_rows)} eval={len(eval_rows)}")

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(
        model.backbone, cfg, client="lora1", surrogate="lora2"
    )

    lora1_params = lora_params_for_adapter(model.backbone, "lora1")
    lora2_params = lora_params_for_adapter(model.backbone, "lora2")

    # Phase B 用: 学習対象 = lora2 (client 役), teacher = lora1 (surrogate 役)
    losses_b = AdaptationLosses(
        model,
        processor,
        w_pred=float(ccfg.get("w_pred", 0.5)),
        w_grad=float(ccfg.get("w_grad", 0.1)),
        w_post=float(ccfg.get("w_post", 0.5)),
        inner_lr=float(ccfg.get("inner_lr", 1e-4)),
        max_length=int(tcfg.get("max_length", 256)),
        prompt=str(dcfg.get("image_prompt", "")),
        client_adapter="lora2",
        surrogate_adapter="lora1",
        client_lora_params=lora2_params,
        surrogate_lora_params=lora1_params,
        temperature=float(ccfg.get("temperature", 2.0)),
    )

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    lr = float(tcfg["lr"])
    weight_decay = float(tcfg["weight_decay"])
    accum_steps = max(1, int(args.accum_steps))

    plan: List[tuple] = []
    for c in range(1, cycles + 1):
        plan.append(("A", int(args.phase_a_epochs), c))
        plan.append(("B", int(args.phase_b_epochs), c))
        if c < cycles:
            plan.append(("C", 0, c))

    global_step_ref = [0]
    eval_history: List[Dict] = []

    for stage, epochs, cycle in plan:
        if stage == "A":
            opt = torch.optim.AdamW(
                list(model.classifier.parameters()) + lora1_params,
                lr=lr,
                weight_decay=weight_decay,
            )
            train_phase_a(
                model=model,
                processor=processor,
                train_rows=train_rows,
                dcfg=dcfg,
                batch_size=bs,
                lora1_params=lora1_params,
                lora2_params=lora2_params,
                opt=opt,
                prompt=prompt,
                max_length=max_length,
                device=device,
                log=log,
                cycle=cycle,
                epochs=epochs,
                accum_steps=accum_steps,
                global_step_ref=global_step_ref,
            )
            metrics = evaluate_classifier(
                model,
                processor,
                eval_rows,
                device=device,
                prompt=prompt,
                max_length=max_length,
                batch_size=bs,
                adapter="lora1",
                dcfg=dcfg,
            )
            log.log(
                {
                    "event": "phase_eval",
                    "phase": "A",
                    "cycle": cycle,
                    "eval_adapter": "lora1",
                    "global_step": global_step_ref[0],
                    "eval_accuracy": metrics["accuracy"],
                    "eval_f1_macro": metrics["f1_macro"],
                }
            )
            eval_history.append(
                {
                    "stage": f"A{cycle}",
                    "adapter": "lora1",
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                }
            )
            print(f"[cycle {cycle} phase A] eval(lora1)={metrics}")

        elif stage == "B":
            opt = torch.optim.AdamW(
                list(model.classifier.parameters()) + lora2_params,
                lr=lr,
                weight_decay=weight_decay,
            )
            train_phase_b(
                model=model,
                processor=processor,
                losses_b=losses_b,
                train_rows=train_rows,
                dcfg=dcfg,
                lora1_params=lora1_params,
                lora2_params=lora2_params,
                opt=opt,
                mode=args.mode,
                prompt=prompt,
                max_length=max_length,
                device=device,
                log=log,
                cycle=cycle,
                epochs=epochs,
                accum_steps=accum_steps,
                global_step_ref=global_step_ref,
            )
            metrics = evaluate_classifier(
                model,
                processor,
                eval_rows,
                device=device,
                prompt=prompt,
                max_length=max_length,
                batch_size=bs,
                adapter="lora2",
                dcfg=dcfg,
            )
            log.log(
                {
                    "event": "phase_eval",
                    "phase": "B",
                    "cycle": cycle,
                    "eval_adapter": "lora2",
                    "global_step": global_step_ref[0],
                    "eval_accuracy": metrics["accuracy"],
                    "eval_f1_macro": metrics["f1_macro"],
                }
            )
            eval_history.append(
                {
                    "stage": f"B{cycle}",
                    "adapter": "lora2",
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                }
            )
            print(f"[cycle {cycle} phase B] eval(lora2)={metrics}")

        elif stage == "C":
            sync_surrogate_from_client(
                model.backbone, client="lora2", surrogate="lora1"
            )
            log.log(
                {
                    "event": "phase_copy",
                    "phase": "C",
                    "cycle": cycle,
                    "global_step": global_step_ref[0],
                    "note": "lora2 -> lora1",
                }
            )
            print(f"[cycle {cycle} phase C] copied lora2 -> lora1")

    ckpt = Path(art["checkpoints"]) / f"step07_alt_self_distill_{args.mode}.pt"
    torch.save(
        {
            "classifier": model.classifier.state_dict(),
            "backbone_peft": model.backbone.state_dict(),
            "meta": {
                "mode": args.mode,
                "scheme": "alt_self_distill",
                "manifest": str(manifest),
            },
        },
        ckpt,
    )
    log.save_summary(
        {
            "scheme": "alt_self_distill",
            "mode": args.mode,
            "checkpoint": str(ckpt),
            "eval_history": eval_history,
            "final_eval": eval_history[-1] if eval_history else {},
        }
    )
    print("Saved:", ckpt)
    print("eval_history:", eval_history)


if __name__ == "__main__":
    main()
