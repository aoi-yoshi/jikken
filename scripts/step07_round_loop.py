"""
Step 7: 2段階ラウンドループ（クライアント L_task → FedAvg → サーバ整合 → 基盤 LoRA 配布）。

1ラウンド:
  ② 各クライアント: クライアント LoRA + 分類ヘッドを L_task のみで更新（基盤 LoRA は触らない）
  ②b Stage 1: server_train（公開整合セット）上のクライアント logits をサーバへ送信
  ③ FedAvg: クライアント LoRA + 分類ヘッドを重み平均（num_clients=1 はスキップ）
  ④ サーバ: 整合損失のみ（consistency.w_task=0.0 で L_task を含めない）で基盤 LoRA を更新
  ⑤ 配布: 更新した基盤 LoRA を各クライアントへ渡して次ラウンドへ
     - copy:    基盤 LoRA の重み + サーバ側ヘッドをクライアント LoRA + ヘッドに書き込む（monolithic のみ・同種 3B/3B）
     - distill: server_train 上の基盤 LoRA logits を teacher に KL 蒸留（既定・Flower もこちら）

異種構成（7B サーバ + 3B クライアント）および step07.distill_pipeline=true（ローカル 3B/3B 既定）では
FedAvg 後の重みをサーバへ載せず、logits 送信 + distill 配布のみ（7B/3B と同経路）。

ログ: artifacts/runs/step07_round_<mode>/<run_id>/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from tqdm import tqdm

from src.adaptation_losses import AdaptationLosses
from src.config_loader import merged_config
from src.dataset_manifest import read_manifest_filtered
from src.experiment_record import save_config_snapshot
from src.step07 import (
    aggregate_client_states,
    distribute_foundation_to_clients,
    distill_pipeline_mode,
    load_client_state,
    resolve_stage_profile,
    resolve_step07_model_ids,
    snapshot_client_state,
    validate_profile_for_hetero,
)
from src.step07.data_splits import save_data_splits_artifact, split_step07_data_pools
from src.step07.stage_profile import list_stage_profiles
from src.step07.train_steps import train_client_l_task, train_server_consistency
from src.fl_logits import (
    aggregate_client_logits,
    export_client_logits,
    save_client_logits_artifact,
)
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.run_context import init_run
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lpred", "lpred_grad", "lpred_post"], required=True)
    ap.add_argument("--stage", type=int, choices=[0, 1], default=0, help="0=再forward teacher, 1=logits送信")
    ap.add_argument(
        "--stage-profile",
        default=None,
        help=f"段階レシピ（config step07.stage_profile より CLI 優先）。例: {list_stage_profiles()[:3]}...",
    )
    ap.add_argument(
        "--aggregation",
        choices=["fedavg", "fedopt", "fedacg", "fedomg"],
        default=None,
        help="③ 集約方式（未指定時は stage_profile / config step07.aggregation.method）",
    )
    ap.add_argument(
        "--teacher-source",
        choices=["recompute", "transmitted"],
        default=None,
        help="未指定時: stage 0→recompute, stage 1→transmitted",
    )
    ap.add_argument("--num-rounds", type=int, default=5)
    ap.add_argument("--num-clients", type=int, default=None, help="未指定時は config step07.num_clients（既定1）")
    ap.add_argument(
        "--distribution",
        choices=["copy", "distill"],
        default=None,
        help="⑤ 配布方式。未指定時は config step07.distribution（既定 distill）。Flower は distill のみ",
    )
    ap.add_argument("--distill-steps", type=int, default=None, help="distill 配布のステップ上限")
    ap.add_argument(
        "--client-train-per-class",
        type=int,
        default=20,
        help="1 client・1 class あたり train（2 client 本番: 20×2×2=80 合計）",
    )
    ap.add_argument(
        "--client-eval-per-class",
        type=int,
        default=15,
        help="共有 client eval（クラスあたり。2 class: 15×2=30 合計）",
    )
    ap.add_argument(
        "--server-train-per-class",
        type=int,
        default=20,
        help="server_train（クラスあたり。2 class: 20×2=40 合計）",
    )
    ap.add_argument(
        "--server-eval-per-class",
        type=int,
        default=15,
        help="server_eval（クラスあたり。2 class: 15×2=30 合計）",
    )
    ap.add_argument("--max-client-steps", type=int, default=None, help="1ラウンドのクライアント学習ステップ上限")
    ap.add_argument("--max-server-steps", type=int, default=None, help="1ラウンドのサーバ整合ステップ上限")
    ap.add_argument("--run-suffix", default="")
    args = ap.parse_args()

    cfg = merged_config()
    if args.stage_profile:
        cfg.setdefault("step07", {})["stage_profile"] = args.stage_profile

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    ccfg = cfg["consistency"]
    mcfg = cfg["model"]
    scfg = cfg.get("step07", {})
    art = cfg["artifacts"]
    seed = int(tcfg["seed"])

    profile = resolve_stage_profile(
        cfg,
        cli_stage=args.stage,
        cli_mode=args.mode,
        cli_teacher_source=args.teacher_source,
        cli_aggregation=args.aggregation,
        cli_distribution=args.distribution,
    )
    consistency_mode = profile.consistency_mode
    teacher_source = profile.teacher_source
    aggregation_method = profile.aggregation_method
    distribution = profile.distribution_method
    num_clients = args.num_clients if args.num_clients is not None else int(scfg.get("num_clients", 1))
    distill_steps = args.distill_steps if args.distill_steps is not None else scfg.get("distill_steps", 20)
    distill_steps = int(distill_steps) if distill_steps is not None else None
    distill_lr = float(scfg.get("distill_lr", tcfg["lr"]))

    server_model_id, client_model_id = resolve_step07_model_ids(cfg)
    distill_pipeline = distill_pipeline_mode(cfg)
    validate_profile_for_hetero(profile, distill_pipeline)

    # プロファイルが要求する送信内容のうち未実装のもの
    _implemented_transmits = {"logits"}
    missing = profile.client_transmits - _implemented_transmits
    if missing:
        raise SystemExit(
            f"stage_profile={profile.name} は client_transmits={sorted(profile.client_transmits)} を要求するが、"
            f"未実装: {sorted(missing)}。先に src/fl_transmission.py 等で送信・集約を実装する。"
        )

    run_id, run_dir, log, env = init_run(
        cfg,
        step="7",
        step_dir=f"step07_round_{consistency_mode}",
        log_name="train",
        run_suffix=args.run_suffix,
        cli=vars(args),
        extra_meta={
            "mode": consistency_mode,
            "stage_profile": profile.name,
            "scheme": "round_loop",
            "stage": args.stage,
            "num_clients": num_clients,
            "aggregation_method": aggregation_method,
            "distribution": distribution,
            "teacher_source": teacher_source,
            "server_model_id": server_model_id,
            "client_model_id": client_model_id,
            "distill_pipeline": distill_pipeline,
        },
    )
    ensure_dirs(Path(art["checkpoints"]))
    latest_ptr = Path(art["runs"]) / f"step07_round_{consistency_mode}" / "LATEST_RUN_ID"
    latest_ptr.parent.mkdir(parents=True, exist_ok=True)
    latest_ptr.write_text(run_id, encoding="utf-8")

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    client_trains, client_eval, server_train, server_eval = split_step07_data_pools(
        rows,
        num_clients=num_clients,
        client_train_per_class=args.client_train_per_class,
        client_eval_per_class=args.client_eval_per_class,
        server_train_per_class=args.server_train_per_class,
        server_eval_per_class=args.server_eval_per_class,
        seed=seed,
    )
    splits_path = save_data_splits_artifact(
        run_dir,
        client_trains=client_trains,
        client_eval=client_eval,
        server_train=server_train,
        server_eval=server_eval,
        seed=seed,
    )

    # distill パイプライン: サーバ用・クライアント用でモデルインスタンスを分離（7B/3B と同形）
    client_model, client_processor = build_model(cfg, device, model_id=client_model_id)
    unfreeze_backbone(client_model)
    client_model.backbone = attach_dual_lora(client_model.backbone, cfg, client="client", surrogate="surrogate")
    if distill_pipeline:
        server_model, server_processor = build_model(cfg, device, model_id=server_model_id)
        unfreeze_backbone(server_model)
        server_model.backbone = attach_dual_lora(server_model.backbone, cfg, client="client", surrogate="surrogate")
    else:
        server_model, server_processor = client_model, client_processor

    client_lora = lora_params_for_adapter(client_model.backbone, "client")
    client_side_surrogate_lora = lora_params_for_adapter(client_model.backbone, "surrogate")
    surrogate_lora = lora_params_for_adapter(server_model.backbone, "surrogate")
    server_side_client_lora = lora_params_for_adapter(server_model.backbone, "client")

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    bs = max(1, int(tcfg.get("batch_size", 1)))
    lr = float(tcfg["lr"])
    weight_decay = float(tcfg.get("weight_decay", 0.01))
    w_task = float(ccfg.get("w_task", profile.w_task))

    server_losses = AdaptationLosses(
        server_model,
        server_processor,
        w_task=w_task,
        w_pred=float(ccfg.get("w_pred", 0.5)),
        w_grad=float(ccfg.get("w_grad", 0.1)),
        w_post=float(ccfg.get("w_post", 0.5)),
        inner_lr=float(ccfg.get("inner_lr", 1e-4)),
        max_length=max_length,
        prompt=prompt,
        client_adapter="surrogate",
        surrogate_adapter="client",
        client_lora_params=surrogate_lora,
        surrogate_lora_params=server_side_client_lora,
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
                "num_clients": num_clients,
                "client_train_per_client": [len(r) for r in client_trains],
                "client_eval": len(client_eval),
                "server_train": len(server_train),
                "server_eval": len(server_eval),
            },
            "stage_profile": profile.name,
            "aggregation_method": aggregation_method,
            "teacher_source": teacher_source,
            "distribution": distribution,
            "w_task": w_task,
            "server_model_id": server_model_id,
            "client_model_id": client_model_id,
        },
    )

    # 全クライアントを同一初期状態から開始
    init_state = snapshot_client_state(client_model, client_lora)
    client_states: List[Dict[str, Any]] = [
        {
            "lora": [t.clone() for t in init_state["lora"]],
            "classifier": {k: v.clone() for k, v in init_state["classifier"].items()},
        }
        for _ in range(num_clients)
    ]

    server_opt = torch.optim.AdamW(
        list(server_model.classifier.parameters()) + surrogate_lora, lr=lr, weight_decay=weight_decay
    )

    round_history: List[Dict[str, Any]] = []

    for rnd in range(1, int(args.num_rounds) + 1):
        round_t0 = time.perf_counter()
        log.log({"event": "round_start", "round": rnd})

        # ② 各クライアント: L_task のみ（クライアントごとに fresh optimizer）
        client_t0 = time.perf_counter()
        client_eval_metrics_all: List[Dict[str, float]] = []
        client_loss_task_all: List[Dict[str, float]] = []
        client_steps_total = 0
        per_client_logits: List[Dict[str, List[float]]] = []
        for k in tqdm(range(num_clients), desc=f"round {rnd} clients"):
            load_client_state(client_model, client_lora, client_states[k])
            client_opt = torch.optim.AdamW(
                list(client_model.classifier.parameters()) + client_lora, lr=lr, weight_decay=weight_decay
            )
            steps, l_task_stats = train_client_l_task(
                model=client_model,
                processor=client_processor,
                train_rows=client_trains[k],
                dcfg=dcfg,
                client_lora=client_lora,
                frozen_lora=client_side_surrogate_lora,
                opt=client_opt,
                prompt=prompt,
                max_length=max_length,
                device=device,
                batch_size=bs,
                max_steps=args.max_client_steps,
            )
            client_steps_total += steps
            client_loss_task_all.append(l_task_stats)
            metrics = evaluate_classifier(
                client_model,
                client_processor,
                client_eval,
                device=device,
                prompt=prompt,
                max_length=max_length,
                batch_size=bs,
                adapter="client",
                dcfg=dcfg,
            )
            client_eval_metrics_all.append(metrics)
            log.log(
                {
                    "event": "client_fit",
                    "round": rnd,
                    "client_id": k,
                    "steps": steps,
                    **l_task_stats,
                    "eval_accuracy": metrics.get("accuracy"),
                    "eval_f1_macro": metrics.get("f1_macro"),
                }
            )

            # ②b Stage 1: 公開整合セット上の logits を送信（学習直後・eval mode）
            if "logits" in profile.client_transmits:
                per_client_logits.append(
                    export_client_logits(
                        client_model,
                        client_processor,
                        server_train,
                        dcfg,
                        adapter="client",
                        prompt=prompt,
                        max_length=max_length,
                        device=device,
                    )
                )
            client_states[k] = snapshot_client_state(client_model, client_lora)
        client_train_sec = time.perf_counter() - client_t0

        # ③ 集約 → FedAvg 後の重み（aggregation.method で方式切替）
        fedavg_state = aggregate_client_states(
            aggregation_method,
            client_states,
            round_num=rnd,
        )
        if num_clients == 1 and aggregation_method == "fedavg":
            log.log({"event": "fedavg_skipped", "round": rnd, "reason": "num_clients=1"})
        else:
            log.log(
                {
                    "event": "aggregation_done",
                    "round": rnd,
                    "method": aggregation_method,
                    "num_clients": num_clients,
                }
            )

        # distill パイプライン: FedAvg 後の重みはサーバへ載せない（形状非依存の transmitted teacher のみ）。
        # 同種専用（distill_pipeline=false）のみ FedAvg 後の重みを参照 adapter に書き込む。
        if not distill_pipeline:
            load_client_state(server_model, server_side_client_lora, fedavg_state)

        # ②b 集約: 複数クライアントの logits を softmax 平均（FedDF 型）
        teacher_logits_map: Dict[str, List[float]] | None = None
        client_logits_path = None
        if teacher_source == "transmitted" and "logits" in profile.client_transmits:
            teacher_logits_map = (
                per_client_logits[0] if num_clients == 1 else aggregate_client_logits(per_client_logits)
            )
            client_logits_path = save_client_logits_artifact(run_dir, rnd, teacher_logits_map)
            log.log(
                {
                    "event": "client_logits_exported",
                    "round": rnd,
                    "path": str(client_logits_path),
                    "num_samples": len(teacher_logits_map),
                    "num_clients": num_clients,
                }
            )

        # ④ サーバ: 整合損失のみで基盤 LoRA を更新（w_task=0 なら L_task 不参加）
        server_t0 = time.perf_counter()
        server_steps, loss_stats = train_server_consistency(
            model=server_model,
            processor=server_processor,
            losses=server_losses,
            train_rows=server_train,
            dcfg=dcfg,
            client_lora=server_side_client_lora,
            surrogate_lora=surrogate_lora,
            opt=server_opt,
            mode=consistency_mode,
            device=device,
            max_steps=args.max_server_steps,
            teacher_source=teacher_source,
            teacher_logits_map=teacher_logits_map,
        )
        server_train_sec = time.perf_counter() - server_t0

        server_eval_metrics = evaluate_classifier(
            server_model,
            server_processor,
            server_eval,
            device=device,
            prompt=prompt,
            max_length=max_length,
            batch_size=bs,
            adapter="surrogate",
            dcfg=dcfg,
        )

        # ⑤ 配布: 更新した基盤 LoRA を各クライアントへ（次ラウンドの②はこの状態から開始）
        dist_t0 = time.perf_counter()
        dist_info = distribute_foundation_to_clients(
            method=distribution,
            client_model=client_model,
            client_processor=client_processor,
            server_model=server_model,
            server_processor=server_processor,
            client_lora=client_lora,
            surrogate_lora=surrogate_lora,
            client_states=client_states,
            fedavg_state=fedavg_state,
            server_train=server_train,
            dcfg=dcfg,
            prompt=prompt,
            max_length=max_length,
            device=device,
            distill_lr=distill_lr,
            distill_steps=distill_steps,
            temperature=float(ccfg.get("temperature", 2.0)),
            run_dir=run_dir,
            rnd=rnd,
        )
        distribution_sec = time.perf_counter() - dist_t0
        log.log({"event": "distribution_done", "round": rnd, **{k: v for k, v in dist_info.items() if k != "distill_stats"}})

        n_eval = len(client_eval_metrics_all)
        client_eval_mean = {
            key: sum(m.get(key, 0.0) for m in client_eval_metrics_all) / n_eval
            for key in client_eval_metrics_all[0]
        }
        client_loss_task_mean = {
            key: sum(m.get(key, 0.0) for m in client_loss_task_all) / n_eval
            for key in client_loss_task_all[0]
        }
        round_record = {
            "round": rnd,
            "stage": args.stage,
            "teacher_source": teacher_source,
            "num_clients": num_clients,
            "distribution": dist_info,
            "client_steps": client_steps_total,
            "server_steps": server_steps,
            "client_train_sec": client_train_sec,
            "server_train_sec": server_train_sec,
            "distribution_sec": distribution_sec,
            "round_sec": time.perf_counter() - round_t0,
            "client_eval": client_eval_mean,
            "client_eval_per_client": client_eval_metrics_all,
            "client_loss_task": client_loss_task_mean,
            "client_loss_task_per_client": client_loss_task_all,
            "server_eval": server_eval_metrics,
            "server_loss": loss_stats,
            "client_logits_path": str(client_logits_path) if client_logits_path else None,
        }
        round_history.append(round_record)
        log.log({"event": "round_end", **round_record})

    ckpt = Path(art["checkpoints"]) / f"step07_round_{consistency_mode}_{run_id}.pt"
    torch.save(
        {
            "classifier": server_model.classifier.state_dict(),
            "backbone_peft": server_model.backbone.state_dict(),
            "fedavg_client": aggregate_client_states(aggregation_method, client_states),
            "meta": {
                "mode": consistency_mode,
                "stage_profile": profile.name,
                "run_id": run_id,
                "scheme": "round_loop",
                "stage": args.stage,
                "teacher_source": teacher_source,
                "aggregation_method": aggregation_method,
                "num_clients": num_clients,
                "distribution": distribution,
                "server_model_id": server_model_id,
                "client_model_id": client_model_id,
                "w_task": w_task,
            },
        },
        ckpt,
    )
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": consistency_mode,
        "stage_profile": profile.name,
        "stage": args.stage,
        "teacher_source": teacher_source,
        "aggregation_method": aggregation_method,
        "num_clients": num_clients,
        "distribution": distribution,
        "server_model_id": server_model_id,
        "client_model_id": client_model_id,
        "w_task": w_task,
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
