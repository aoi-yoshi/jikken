"""
Step 7: 2段階ラウンドループ（クライアント L_task → サーバ整合 → 基盤 LoRA 配布）。

1ラウンド:
  ① 基盤 LoRA をクライアント側参照として保持
  ② クライアント: クライアント LoRA + ヘッドを L_task のみで更新（基盤 LoRA は触らない）
  ③ FedAvg（クライアント1台なら送信重みをそのまま FedAvg 後の重みとする）
  ④ サーバ: FedAvg 後の重み（クライアント LoRA）を固定参照とし、基盤 LoRA を整合損失で更新
  ⑤ 更新した基盤 LoRA を次ラウンド用に保持

ログ: artifacts/runs/step07_round_<mode>/<run_id>/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import random

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from src.adaptation_losses import AdaptationLosses
from src.config_loader import merged_config
from src.dataset_manifest import load_sample_row, read_manifest_filtered
from src.experiment_record import save_config_snapshot
from src.fl_logits import export_client_logits, logits_map_to_tensor, save_client_logits_artifact
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.run_context import init_run
from src.train_common import device_or_auto, iter_row_chunks, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def _set_requires_grad(params: List[nn.Parameter], flag: bool) -> None:
    for p in params:
        p.requires_grad = flag


def split_step07_data_pools(
    rows: List[dict],
    *,
    client_train_per_class: int,
    client_eval_per_class: int,
    server_train_per_class: int,
    server_eval_per_class: int,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """クラス別に4プールへ分割（ID 重複なし）。"""
    rng = random.Random(seed)
    buckets: Dict[int, List[dict]] = {}
    for r in rows:
        buckets.setdefault(int(r["label"]), []).append(r)

    client_train: List[dict] = []
    client_eval: List[dict] = []
    server_train: List[dict] = []
    server_eval: List[dict] = []

    for label in sorted(buckets.keys()):
        items = list(buckets[label])
        rng.shuffle(items)
        need = (
            client_train_per_class
            + client_eval_per_class
            + server_train_per_class
            + server_eval_per_class
        )
        if len(items) < need:
            raise RuntimeError(
                f"class {label} has {len(items)} samples; need {need} "
                f"(client train/eval={client_train_per_class}/{client_eval_per_class}, "
                f"server train/eval={server_train_per_class}/{server_eval_per_class})"
            )
        i = 0
        client_train.extend(items[i : i + client_train_per_class])
        i += client_train_per_class
        client_eval.extend(items[i : i + client_eval_per_class])
        i += client_eval_per_class
        server_train.extend(items[i : i + server_train_per_class])
        i += server_train_per_class
        server_eval.extend(items[i : i + server_eval_per_class])

    for pool in (client_train, client_eval, server_train, server_eval):
        rng.shuffle(pool)
    return client_train, client_eval, server_train, server_eval


def save_data_splits_artifact(
    run_dir: Path,
    *,
    client_train: List[dict],
    client_eval: List[dict],
    server_train: List[dict],
    server_eval: List[dict],
    seed: int,
) -> Path:
    payload = {
        "seed": seed,
        "client_train_ids": [str(r["id"]) for r in client_train],
        "client_eval_ids": [str(r["id"]) for r in client_eval],
        "server_train_ids": [str(r["id"]) for r in server_train],
        "server_eval_ids": [str(r["id"]) for r in server_eval],
        "counts": {
            "client_train": len(client_train),
            "client_eval": len(client_eval),
            "server_train": len(server_train),
            "server_eval": len(server_eval),
        },
    }
    path = run_dir / "data_splits.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def train_client_l_task(
    *,
    model,
    processor,
    train_rows: List[dict],
    dcfg: dict,
    client_lora: List[nn.Parameter],
    surrogate_lora: List[nn.Parameter],
    opt: torch.optim.Optimizer,
    prompt: str,
    max_length: int,
    device: torch.device,
    batch_size: int,
    max_steps: int | None,
) -> int:
    """第1の学習場所: クライアント LoRA + ヘッドを L_task のみで更新。"""
    model.train()
    _set_requires_grad(client_lora, True)
    _set_requires_grad(surrogate_lora, False)
    steps = 0
    for chunk in iter_row_chunks(train_rows, batch_size):
        if max_steps is not None and steps >= max_steps:
            break
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
        for p in surrogate_lora:
            p.grad = None
        steps += 1
    return steps


def train_server_consistency(
    *,
    model,
    processor,
    losses: AdaptationLosses,
    train_rows: List[dict],
    dcfg: dict,
    client_lora: List[nn.Parameter],
    surrogate_lora: List[nn.Parameter],
    opt: torch.optim.Optimizer,
    mode: str,
    device: torch.device,
    max_steps: int | None,
    teacher_source: str = "recompute",
    teacher_logits_map: Dict[str, List[float]] | None = None,
) -> Tuple[int, Dict[str, float]]:
    """第2の学習場所: 基盤 LoRA を整合損失で更新。

    teacher_source=recompute: FedAvg 後の重み（クライアント LoRA）をサーバ上で再 forward。
    teacher_source=transmitted: クライアント送信 logits を teacher として注入（Stage 1）。
    """
    model.train()
    _set_requires_grad(client_lora, False)
    _set_requires_grad(surrogate_lora, True)
    steps = 0
    last_stats: Dict[str, float] = {}
    use_transmitted = teacher_source == "transmitted" and mode == "lpred"
    if teacher_source == "transmitted" and mode != "lpred":
        use_transmitted = False
    if use_transmitted and not teacher_logits_map:
        raise RuntimeError("teacher_source=transmitted requires teacher_logits_map")

    for r in train_rows:
        if max_steps is not None and steps >= max_steps:
            break
        sample = load_sample_row(r, dcfg)
        y = torch.tensor([sample.label], device=device, dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        teacher_logits = None
        if use_transmitted:
            sid = str(r["id"])
            if sid not in teacher_logits_map:
                raise KeyError(f"sample_id {sid} missing from transmitted logits")
            teacher_logits = logits_map_to_tensor(teacher_logits_map, sid, device)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            total, stats, _extra = losses.compute(mode, sample.frames, y, teacher_logits=teacher_logits)
        total.backward()
        opt.step()
        for p in client_lora:
            p.grad = None
        last_stats = dict(stats)
        steps += 1
    return steps, last_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lpred", "lpred_grad", "lpred_post"], required=True)
    ap.add_argument("--stage", type=int, choices=[0, 1], default=0, help="0=再forward teacher, 1=logits送信")
    ap.add_argument(
        "--teacher-source",
        choices=["recompute", "transmitted"],
        default=None,
        help="未指定時: stage 0→recompute, stage 1→transmitted",
    )
    ap.add_argument("--num-rounds", type=int, default=5)
    ap.add_argument("--client-train-per-class", type=int, default=50)
    ap.add_argument("--client-eval-per-class", type=int, default=15)
    ap.add_argument("--server-train-per-class", type=int, default=25)
    ap.add_argument("--server-eval-per-class", type=int, default=15)
    ap.add_argument("--max-client-steps", type=int, default=None, help="1ラウンドのクライアント学習ステップ上限")
    ap.add_argument("--max-server-steps", type=int, default=None, help="1ラウンドのサーバ整合ステップ上限")
    ap.add_argument("--run-suffix", default="")
    args = ap.parse_args()
    teacher_source = args.teacher_source
    if teacher_source is None:
        teacher_source = "transmitted" if args.stage == 1 else "recompute"

    cfg = merged_config()
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    ccfg = cfg["consistency"]
    art = cfg["artifacts"]
    seed = int(tcfg["seed"])

    run_id, run_dir, log, env = init_run(
        cfg,
        step="7",
        step_dir=f"step07_round_{args.mode}",
        log_name="train",
        run_suffix=args.run_suffix,
        cli=vars(args),
        extra_meta={"mode": args.mode, "scheme": "round_loop", "stage": args.stage},
    )
    ensure_dirs(Path(art["checkpoints"]))
    latest_ptr = Path(art["runs"]) / f"step07_round_{args.mode}" / "LATEST_RUN_ID"
    latest_ptr.parent.mkdir(parents=True, exist_ok=True)
    latest_ptr.write_text(run_id, encoding="utf-8")

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    client_train, client_eval, server_train, server_eval = split_step07_data_pools(
        rows,
        client_train_per_class=args.client_train_per_class,
        client_eval_per_class=args.client_eval_per_class,
        server_train_per_class=args.server_train_per_class,
        server_eval_per_class=args.server_eval_per_class,
        seed=seed,
    )
    splits_path = save_data_splits_artifact(
        run_dir,
        client_train=client_train,
        client_eval=client_eval,
        server_train=server_train,
        server_eval=server_eval,
        seed=seed,
    )

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")

    client_lora = lora_params_for_adapter(model.backbone, "client")
    surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    bs = max(1, int(tcfg.get("batch_size", 1)))
    lr = float(tcfg["lr"])
    weight_decay = float(tcfg.get("weight_decay", 0.01))

    # サーバ整合: 基盤 LoRA を更新対象、FedAvg 後の重み（クライアント LoRA）を教師側として固定
    server_losses = AdaptationLosses(
        model,
        processor,
        w_pred=float(ccfg.get("w_pred", 0.5)),
        w_grad=float(ccfg.get("w_grad", 0.1)),
        w_post=float(ccfg.get("w_post", 0.5)),
        inner_lr=float(ccfg.get("inner_lr", 1e-4)),
        max_length=max_length,
        prompt=prompt,
        client_adapter="surrogate",
        surrogate_adapter="client",
        client_lora_params=surrogate_lora,
        surrogate_lora_params=client_lora,
        temperature=float(ccfg.get("temperature", 2.0)),
    )

    save_config_snapshot(
        run_dir,
        cfg=cfg,
        cli_args=vars(args),
        extra={
            "run_id": run_id,
            "data_splits": str(splits_path),
            "counts": {
                "client_train": len(client_train),
                "client_eval": len(client_eval),
                "server_train": len(server_train),
                "server_eval": len(server_eval),
            },
            "stage": args.stage,
            "teacher_source": teacher_source,
        },
    )

    round_history: List[Dict[str, Any]] = []
    client_opt = torch.optim.AdamW(
        list(model.classifier.parameters()) + client_lora, lr=lr, weight_decay=weight_decay
    )
    server_opt = torch.optim.AdamW(
        list(model.classifier.parameters()) + surrogate_lora, lr=lr, weight_decay=weight_decay
    )

    for rnd in range(1, int(args.num_rounds) + 1):
        round_t0 = time.perf_counter()
        log.log({"event": "round_start", "round": rnd})

        # ② クライアント: L_task のみ
        client_t0 = time.perf_counter()
        client_steps = train_client_l_task(
            model=model,
            processor=processor,
            train_rows=client_train,
            dcfg=dcfg,
            client_lora=client_lora,
            surrogate_lora=surrogate_lora,
            opt=client_opt,
            prompt=prompt,
            max_length=max_length,
            device=device,
            batch_size=bs,
            max_steps=args.max_client_steps,
        )
        client_train_sec = time.perf_counter() - client_t0

        client_eval_metrics = evaluate_classifier(
            model,
            processor,
            client_eval,
            device=device,
            prompt=prompt,
            max_length=max_length,
            batch_size=bs,
            adapter="client",
            dcfg=dcfg,
        )

        # ③ FedAvg 省略（1クライアント: クライアント LoRA がそのまま FedAvg 後の重み）
        log.log({"event": "fedavg_skipped", "round": rnd, "reason": "num_clients=1"})

        # ②b Stage 1: 整合用サンプル上でクライアント logits を送信（sidecar JSON）
        teacher_logits_map: Dict[str, List[float]] | None = None
        client_logits_path = None
        if teacher_source == "transmitted":
            teacher_logits_map = export_client_logits(
                model,
                processor,
                server_train,
                dcfg,
                adapter="client",
                prompt=prompt,
                max_length=max_length,
                device=device,
            )
            client_logits_path = save_client_logits_artifact(run_dir, rnd, teacher_logits_map)
            log.log(
                {
                    "event": "client_logits_exported",
                    "round": rnd,
                    "path": str(client_logits_path),
                    "num_samples": len(teacher_logits_map),
                }
            )

        # ④ サーバ: 整合損失で基盤 LoRA を更新
        server_t0 = time.perf_counter()
        server_steps, loss_stats = train_server_consistency(
            model=model,
            processor=processor,
            losses=server_losses,
            train_rows=server_train,
            dcfg=dcfg,
            client_lora=client_lora,
            surrogate_lora=surrogate_lora,
            opt=server_opt,
            mode=args.mode,
            device=device,
            max_steps=args.max_server_steps,
            teacher_source=teacher_source,
            teacher_logits_map=teacher_logits_map,
        )
        server_train_sec = time.perf_counter() - server_t0

        server_eval_metrics = evaluate_classifier(
            model,
            processor,
            server_eval,
            device=device,
            prompt=prompt,
            max_length=max_length,
            batch_size=bs,
            adapter="surrogate",
            dcfg=dcfg,
        )

        round_record = {
            "round": rnd,
            "stage": args.stage,
            "teacher_source": teacher_source,
            "client_steps": client_steps,
            "server_steps": server_steps,
            "client_train_sec": client_train_sec,
            "server_train_sec": server_train_sec,
            "round_sec": time.perf_counter() - round_t0,
            "client_eval": client_eval_metrics,
            "server_eval": server_eval_metrics,
            "server_loss": loss_stats,
            "client_logits_path": str(client_logits_path) if client_logits_path else None,
        }
        round_history.append(round_record)
        log.log({"event": "round_end", **round_record})

    ckpt = Path(art["checkpoints"]) / f"step07_round_{args.mode}_{run_id}.pt"
    torch.save(
        {
            "classifier": model.classifier.state_dict(),
            "backbone_peft": model.backbone.state_dict(),
            "meta": {
                "mode": args.mode,
                "run_id": run_id,
                "scheme": "round_loop",
                "stage": args.stage,
                "teacher_source": teacher_source,
            },
        },
        ckpt,
    )
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": args.mode,
        "stage": args.stage,
        "teacher_source": teacher_source,
        "scheme": "round_loop",
        "num_rounds": args.num_rounds,
        "environment": env,
        "data_splits": str(splits_path),
        "round_history": round_history,
        "checkpoint": str(ckpt),
        "status": "completed",
    }
    log.save_summary(summary)
    log.log_meta({"status": "completed"})
    print("Saved:", ckpt)
    print("rounds:", len(round_history))


if __name__ == "__main__":
    main()
