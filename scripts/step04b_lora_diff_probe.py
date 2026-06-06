"""
Step 4b: 学習前後の LoRA 差分・出力差分・勾配差分を記録する診断実験。

エポック単位で学習の推移（Loss、KL情報量、勾配差分、精度など）を細かくトラッキングし、
Ablation / FL の前に L_task による適応のダイナミクスを定量的に確認する。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.amp import autocast
from tqdm import tqdm

from src.config_loader import merged_config
from src.dataset_manifest import frame_paths_for_model, load_sample_row, read_manifest_filtered, stratified_subset
from src.diff_probe import (
    grad_probe,
    lora_delta_norm,
    lora_inventory,
    lora_norm,
    logits_diff,
    per_tensor_lora_delta,
    probe_logits,
    snapshot_lora,
)
from src.experiment_record import save_config_snapshot, write_experiment_report
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.resource_metrics import get_gpu_utilization, get_max_vram_gb, reset_peak_memory_stats
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.train_common import device_or_auto, load_samples, rows_for_step
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def _make_run_id(suffix: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}{suffix}"


def _pick_probe_and_train(
    rows: List[dict],
    *,
    probe_idx: int,
    train_count: int,
    seed: int,
) -> tuple[dict, List[dict]]:
    if not rows:
        raise RuntimeError("manifest is empty")
    probe_idx = probe_idx % len(rows)
    probe = rows[probe_idx]
    pool = [r for i, r in enumerate(rows) if i != probe_idx]
    
    if train_count <= 0:
        return probe, []
    
    if len(pool) < train_count:
        print(f"Warning: train_count ({train_count}) exceeds available pool ({len(pool)}). Clamping to pool size.", flush=True)
        train_count = len(pool)

    rng = random.Random(seed)
    rng.shuffle(pool)
    return probe, pool[:train_count]


def _measure_probe(
    model,
    processor,
    probe_row: dict,
    *,
    dcfg: dict,
    prompt: str,
    max_length: int,
    device: torch.device,
    client_adapter: str,
    surrogate_adapter: str,
    client_lora,
    surrogate_lora,
    tag: str,
) -> Dict[str, Any]:
    sample = load_sample_row(probe_row, dcfg)
    y = torch.tensor([sample.label], device=device, dtype=torch.long)

    model.backbone.set_adapter(client_adapter)
    with torch.no_grad():
        logits_c, _ = model.forward_batch(
            processor, sample.frames, prompt, max_length, output_hidden_states=True
        )
    model.backbone.set_adapter(surrogate_adapter)
    with torch.no_grad():
        logits_s, _ = model.forward_batch(
            processor, sample.frames, prompt, max_length, output_hidden_states=True
        )

    out_c = probe_logits(logits_c)
    out_s = probe_logits(logits_s)
    diff_cs = logits_diff(logits_c, logits_s)
    
    grad = grad_probe(
        model,
        processor,
        sample.frames,
        y,
        prompt=prompt,
        max_length=max_length,
        client_adapter=client_adapter,
        surrogate_adapter=surrogate_adapter,
        client_params=client_lora,
        surrogate_params=surrogate_lora,
    )

    model.zero_grad(set_to_none=True)

    return {
        "tag": tag,
        "probe_id": sample.sample_id,
        "probe_label": sample.label,
        "probe_source": sample.source,
        "probe_frame_paths": frame_paths_for_model(probe_row, dcfg),
        "lora_norm_client": lora_norm(client_lora),
        "lora_norm_surrogate": lora_norm(surrogate_lora),
        "client_pred": out_c["pred"],
        "surrogate_pred": out_s["pred"],
        "client_logits": out_c["logits"],
        "surrogate_logits": out_s["logits"],
        "client_probs": out_c["probs"],
        "surrogate_probs": out_s["probs"],
        **{f"client_vs_surrogate_{k}": v for k, v in diff_cs.items()},
        **grad,
    }


# [追加] 3. エラー分析用：評価データの予測生データを保存する関数
def _save_eval_predictions(
    model, processor, eval_rows: List[dict], filepath: Path, device: torch.device, prompt: str, max_length: int, bs: int, dcfg: dict
) -> None:
    model.eval()
    num_steps = (len(eval_rows) + bs - 1) // bs
    with filepath.open("w", encoding="utf-8") as f:
        for step in range(num_steps):
            batch_rows = rows_for_step(eval_rows, step=step, batch_size=bs)
            if not batch_rows: continue
            samples = load_samples(batch_rows, dcfg)
            frames_batch = [s.frames for s in samples]
            y = [s.label for s in samples]
            ids = [s.sample_id for s in samples]
            
            with torch.no_grad(), autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, _ = model.forward_samples(processor, frames_batch, prompt, max_length, output_hidden_states=True)
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy().tolist()
                preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            
            for i in range(len(samples)):
                rec = {
                    "sample_id": ids[i],
                    "true_label": y[i],
                    "pred_label": preds[i],
                    "probs": probs[i]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 85)
    print("Step 4b 診断サマリ（エポック推移・ハードウェアプロファイル版）")
    print(f"Run ID: {summary.get('run_id')}")
    print("=" * 85)
    ed = summary.get("experiment_design", {})
    print(f"  モデル: {ed.get('model_id')}")
    print(f"  LoRA: r={ed.get('lora_r')} alpha={ed.get('lora_alpha')} targets={ed.get('lora_target_modules')}")
    print(f"  Adapter物理サイズ: {ed.get('adapter_size_mb', 0.0):.2f} MB")
    print(f"  学習可能パラメータ数: {ed.get('trainable_parameters', 0):,}")
    print("-" * 85)
    print("  [学習ダイナミクス推移]")
    print("  Epoch | Train Loss | Eval Acc | KL(C vs S) | ||Δθ|| | Max VRAM | Train Time")
    print("  ---------------------------------------------------------------------------------")
    
    history = summary.get("history", [])
    for h in history:
        ep = h["epoch"]
        loss_val = h.get("train_loss")
        loss_str = f"{loss_val:.4f}" if loss_val is not None else "  --- "
        acc = f"{h['eval']['accuracy']:.2f}"
        kl = f"{h['probe']['client_vs_surrogate_kl_teacher_a_student_b']:.6f}"
        norm = f"{h.get('client_delta', 0.0):.4f}"
        
        hw = h.get("hardware_metrics", {})
        vram = f"{hw.get('max_vram_gb', 0.0):.1f}G"
        t_time = f"{hw.get('train_time_sec', 0.0):.1f}s"
        
        print(f"  {ep:5d} | {loss_str:>10} | {acc:>8} | {kl:>10} | {norm:>6} | {vram:>8} | {t_time:>10}")
    
    print("-" * 85)
    print(f"  レポート: {summary.get('run_dir')}/experiment_report.md")
    print(f"  ベストモデル: {summary.get('run_dir')}/client_adapter_best") # [追加] 
    print("=" * 85 + "\n")


def _build_experiment_design(cfg: dict, manifest_total: int, train_pool: int, train_ids: List[str], steps: int, num_epochs: int, train_count: int, probe_idx: int, eval_max: int, trainable_params: int, adapter_size_mb: float) -> Dict[str, Any]:
    lora = cfg["lora"]
    tcfg = cfg["train"]
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    r = int(lora["r"])
    alpha = int(lora["lora_alpha"])
    return {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "trainable_parameters": trainable_params,
        "adapter_size_mb": adapter_size_mb,
        "model_id": mcfg["id"],
        "attn_implementation": mcfg.get("attn_implementation", "sdpa"),
        "num_classes": int(dcfg["num_classes"]),
        "classifier_hidden": "from model.text_config.hidden_size at runtime",
        "lora_r": r,
        "lora_alpha": alpha,
        "lora_scale": alpha / r,
        "lora_dropout": float(lora["lora_dropout"]),
        "lora_target_modules": list(lora["target_modules"]),
        "lora_bias": str(lora.get("bias", "none")),
        "client_adapter": "client",
        "surrogate_adapter": "surrogate",
        "surrogate_update_in_this_run": False,
        "loss_used": "L_task (cross_entropy) only",
        "dataset": "NEXAR Collision Prediction (binary)",
        "num_frames": int(dcfg["num_frames"]),
        "use_num_frames": int(dcfg.get("use_num_frames") or 0),
        "use_frame_slice": str(dcfg.get("use_frame_slice", "last")),
        "weather_filter": str(dcfg.get("weather_filter", "") or ""),
        "frame_sampling": str(dcfg.get("frame_sampling", "")),
        "manifest_total": manifest_total,
        "train_pool_size": train_pool,
        "train_unique_samples": len(set(train_ids)),
        "train_sample_ids": train_ids,
        "lr": float(tcfg["lr"]),
        "weight_decay": float(tcfg["weight_decay"]),
        "batch_size": int(tcfg.get("batch_size", 1)),
        "max_length": int(tcfg.get("max_length", 256)),
        "min_pixels": int(tcfg.get("min_pixels", 50176)),
        "max_pixels": int(tcfg.get("max_pixels", 75264)),
        "seed": int(tcfg["seed"]),
        "steps_requested": steps,
        "num_epochs_calculated": num_epochs,
        "train_count": train_count,
        "probe_idx": probe_idx,
        "eval_max": eval_max,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4b: エポック毎のLoRA/出力/勾配差分の記録")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--train-count", type=int, default=None)
    ap.add_argument("--probe-idx", type=int, default=None)
    ap.add_argument("--eval-max", type=int, default=None)
    ap.add_argument("--run-suffix", default="", help="run_id suffix")
    args = ap.parse_args()

    cfg = merged_config()
    
    steps = args.steps if args.steps is not None else int(cfg["train"].get("steps", 10))
    train_count = args.train_count if args.train_count is not None else int(cfg["train"].get("train_count", 20))
    probe_idx = args.probe_idx if args.probe_idx is not None else int(cfg["train"].get("probe_idx", 0))
    eval_max = args.eval_max if args.eval_max is not None else int(cfg["train"].get("eval_max", 20))

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    art = cfg["artifacts"]

    run_id = _make_run_id(str(args.run_suffix))
    run_dir = Path(art["runs"]) / "step04b_diff_probe" / run_id
    ensure_dirs(run_dir)
    shutil.copy(args.config, run_dir / "used_config.yaml")

    log = RunLogger(run_dir, name="probe")
    train_log_path = run_dir / "epoch_history.jsonl"
    log.log_meta({"step": "4b", "run_id": run_id, "status": "started", "cli": vars(args)})

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    manifest_total = len(rows)
    rows = stratified_subset(rows, int(tcfg.get("max_train_samples", 200)), int(cfg["train"]["seed"]))
    
    n_train = int(len(rows) * float(dcfg.get("train_ratio", 0.8)))
    train_pool = rows[:n_train]
    eval_rows = rows[n_train : n_train + eval_max]

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    lr = float(tcfg["lr"])
    bs = max(1, int(tcfg.get("batch_size", 1)))

    probe_row, train_rows = _pick_probe_and_train(
        train_pool, probe_idx=probe_idx, train_count=train_count, seed=int(cfg["train"]["seed"])
    )
    
    steps_per_epoch = max(1, len(train_rows) // bs)
    num_epochs = max(1, steps // steps_per_epoch)
    
    print(f"[data] run_id={run_id} probe={probe_row['id']} data={len(train_rows)} bs={bs}", flush=True)
    print(f"[schedule] {steps_per_epoch} steps/epoch, total {num_epochs} epochs.", flush=True)

    model, processor = build_model(cfg, device)
    hidden = int(model.backbone.config.text_config.hidden_size)
    num_layers = int(model.backbone.config.text_config.num_hidden_layers)

    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    client_lora = lora_params_for_adapter(model.backbone, "client")
    surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
    inv_client = lora_inventory(model.backbone, "client")
    inv_surrogate = lora_inventory(model.backbone, "surrogate")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_size_mb = sum(p.numel() * p.element_size() for p in client_lora) / (1024 ** 2)
    print(f"[hardware] Trainable params: {trainable_params:,}, Adapter size: {adapter_size_mb:.2f} MB")

    train_ids: List[str] = []
    lora_start = snapshot_lora(client_lora)
    history_log: List[Dict[str, Any]] = []

    print("\n[Epoch 0] measuring BEFORE state...", flush=True)
    before_probe = _measure_probe(
        model, processor, probe_row, dcfg=dcfg, prompt=prompt, max_length=max_length,
        device=device, client_adapter="client", surrogate_adapter="surrogate",
        client_lora=client_lora, surrogate_lora=surrogate_lora, tag="epoch_0"
    )
    before_eval = evaluate_classifier(
        model, processor, eval_rows, device=device, prompt=prompt, max_length=max_length,
        batch_size=bs, adapter="client", dcfg=dcfg
    )
    
    ep0_record = {
        "epoch": 0,
        "train_loss": None,
        "probe": before_probe,
        "eval": before_eval,
        "client_delta": 0.0,
        "hardware_metrics": {}
    }
    history_log.append(ep0_record)
    with train_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ep0_record, ensure_ascii=False) + "\n")

    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(tcfg["weight_decay"]))

    # [追加] ベストチェックポイント追跡用の変数
    best_eval_acc = -1.0
    best_epoch = 0

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        if torch.cuda.is_available():
            reset_peak_memory_stats()
            
        model.train()
        epoch_loss_sum = 0.0
        
        epoch_start_time = time.time()
        gpu_utils = []

        print(f"\n[Epoch {epoch}/{num_epochs}] Training...")
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}")
        for _ in pbar:
            batch_rows = rows_for_step(train_rows, step=global_step, batch_size=bs)
            samples = load_samples(batch_rows, dcfg)
            if epoch == 1:
                train_ids.extend(s.sample_id for s in samples)
            y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
            frames_batch = [s.frames for s in samples]
            
            opt.zero_grad(set_to_none=True)
            model.backbone.set_adapter("client")
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, _ = model.forward_samples(processor, frames_batch, prompt, max_length, output_hidden_states=True)
                loss = logits_loss(logits.float(), y)
            loss.backward()
            opt.step()
            
            loss_val = float(loss.detach().cpu())
            epoch_loss_sum += loss_val
            global_step += 1
            pbar.set_postfix(loss=loss_val)
            
            if global_step % 5 == 0:
                gpu_utils.append(get_gpu_utilization())

        train_time = time.time() - epoch_start_time
        avg_step_time = train_time / steps_per_epoch
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else get_gpu_utilization()
        max_vram = get_max_vram_gb()

        avg_train_loss = epoch_loss_sum / steps_per_epoch
        
        print(f"[Epoch {epoch}] measuring probe and eval...", flush=True)
        eval_start_time = time.time()
        
        current_snap = snapshot_lora(client_lora)
        current_delta = lora_delta_norm(lora_start, current_snap)
        
        ep_probe = _measure_probe(
            model, processor, probe_row, dcfg=dcfg, prompt=prompt, max_length=max_length,
            device=device, client_adapter="client", surrogate_adapter="surrogate",
            client_lora=client_lora, surrogate_lora=surrogate_lora, tag=f"epoch_{epoch}"
        )
        ep_eval = evaluate_classifier(
            model, processor, eval_rows, device=device, prompt=prompt, max_length=max_length,
            batch_size=bs, adapter="client", dcfg=dcfg
        )
        
        eval_time = time.time() - eval_start_time
        
        ep_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "probe": ep_probe,
            "eval": ep_eval,
            "client_delta": current_delta,
            "hardware_metrics": {
                "train_time_sec": train_time,
                "avg_step_time_sec": avg_step_time,
                "eval_time_sec": eval_time,
                "max_vram_gb": max_vram,
                "avg_gpu_util_percent": avg_gpu_util
            }
        }
        history_log.append(ep_record)
        with train_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep_record, ensure_ascii=False) + "\n")

        # [追加] 1, 2, 3: ベストモデルの判定と保存処理
        curr_acc = ep_eval.get("accuracy", 0.0)
        if curr_acc >= best_eval_acc:
            best_eval_acc = curr_acc
            best_epoch = epoch
            
            best_dir = run_dir / "client_adapter_best"
            
            # 1. 精度が最も良かった時点のLoRA重みを保存
            model.backbone.save_pretrained(best_dir)
            
            # 2. プロセッサ情報の同梱
            processor.save_pretrained(best_dir)
            
            # 3. エラー分析用の生予測データをJSONL形式で出力
            pred_file = run_dir / "best_eval_predictions.jsonl"
            _save_eval_predictions(
                model, processor, eval_rows, pred_file,
                device, prompt, max_length, bs, dcfg
            )
            print(f"  -> Best model updated at Epoch {epoch} (Acc: {best_eval_acc:.4f})")

    # ---------------------------------------------------------
    # サマリ保存と最終モデルの出力
    # ---------------------------------------------------------
    experiment_design = _build_experiment_design(
        cfg, manifest_total=manifest_total, train_pool=len(train_pool), train_ids=train_ids,
        steps=steps, num_epochs=num_epochs, train_count=train_count, probe_idx=probe_idx, eval_max=eval_max,
        trainable_params=trainable_params, adapter_size_mb=adapter_size_mb
    )
    
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": str(manifest),
        "experiment_design": experiment_design,
        "lora_inventory": {"client": inv_client, "surrogate": inv_surrogate},
        "history": history_log,
    }
    
    log.save_summary(summary)
    write_experiment_report(run_dir, summary)
    _print_summary(summary)
    
    # [追加] 最終エポックのモデルも比較用に別途保存
    final_dir = run_dir / "client_adapter_final"
    model.backbone.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    
    print("Saved:")
    print(" ", run_dir / "summary.json")
    print(" ", run_dir / "epoch_history.jsonl")
    print(" ", run_dir / "used_config.yaml")
    print(" ", run_dir / "client_adapter_best", "(Best Model & Processor)")
    print(" ", run_dir / "best_eval_predictions.jsonl", "(Raw Predictions)")

if __name__ == "__main__":
    main()