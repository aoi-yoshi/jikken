"""
Flower なしのローカル FedAvg シミュレーション（IID / Non-IID）。
gRPC 完走前に LoRA 集約 → checkpoint → eval パイプラインを検証する。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.amp import autocast

from src.config_loader import merged_config
from src.diff_probe import snapshot_lora
from src.fl_data import apply_fl_cli_overrides, build_fl_dataset, get_client_train_rows, save_partition_artifact
from src.fl_flower_config import build_fl_flower_settings_full
from src.fl_metrics import (
    build_round_metric_record,
    build_round_timing,
    client_trainable_vector,
    compute_weight_divergence,
    evaluate_foundation_generalization,
    proximal_penalty,
)
from src.fl_strategy import fl_strategy_name, is_fedprox
from src.fl_utils import fedavg_weights, save_fl_checkpoint, trainable_state_vector, vector_to_trainable_state
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, ensure_client_trainable, lora_params_for_adapter
from src.resource_metrics import (
    adapter_size_mb,
    build_environment_block,
    build_training_hardware_snapshot,
    GpuUtilTracker,
    profile_one_step,
    reset_peak_memory_stats,
)
from src.train_common import device_or_auto, iter_row_chunks, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def _client_round(
    model,
    processor,
    train_rows,
    eval_rows,
    *,
    cfg,
    device,
    cid: int,
    server_round: int,
    global_names,
    global_vec,
    foundation_vec,
    foundation_lora_snap,
) -> dict:
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    bs = max(1, int(tcfg.get("batch_size", 1)))
    local_epochs = int(cfg["fl"].get("local_epochs", 1))

    round_t0 = time.perf_counter()
    t0 = time.perf_counter()
    if global_names and global_vec is not None:
        vector_to_trainable_state(model, global_names, global_vec)
    apply_sec = time.perf_counter() - t0

    use_prox = is_fedprox(cfg) and global_vec is not None
    prox_mu = float(cfg["fl"].get("fedprox_mu", 0.01))

    model.eval()
    t0 = time.perf_counter()
    post_redist = evaluate_classifier(
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
    post_eval_sec = time.perf_counter() - t0

    client_lora = lora_params_for_adapter(model.backbone, "client")
    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))

    model.train()
    reset_peak_memory_stats()
    fit_t0 = time.perf_counter()
    steps = 0
    loss_sum = 0.0
    profiler_result = None
    gpu_tracker = GpuUtilTracker(device_index=0 if device.type == "cuda" else 0)
    for _ in range(local_epochs):
        for chunk in iter_row_chunks(train_rows, bs):
            samples = load_samples(chunk, dcfg)
            y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
            frames_batch = [s.frames for s in samples]

            def _train_step() -> None:
                opt.zero_grad(set_to_none=True)
                model.backbone.set_adapter("client")
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits, _ = model.forward_samples(
                        processor, frames_batch, prompt, max_length, output_hidden_states=True
                    )
                    step_loss = logits_loss(logits.float(), y)
                    if use_prox:
                        step_loss = step_loss + proximal_penalty(model, global_vec, prox_mu, device)
                step_loss.backward()
                opt.step()
                nonlocal loss_sum
                loss_sum += float(step_loss.detach().cpu())

            if steps == 0 and device.type == "cuda":
                profiler_result = profile_one_step(_train_step, device=device)
            else:
                _train_step()
            steps += 1
            gpu_tracker.maybe_sample()
    fit_elapsed = time.perf_counter() - fit_t0
    hw = build_training_hardware_snapshot(
        elapsed_sec=fit_elapsed,
        steps=steps,
        gpu_tracker=gpu_tracker,
        device=device,
        profiler=profiler_result,
        phase="fit",
    )

    model.eval()
    t0 = time.perf_counter()
    local_adapt = evaluate_classifier(
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
    local_eval_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    foundation_eval = evaluate_foundation_generalization(
        model,
        processor,
        eval_rows,
        device=device,
        prompt=prompt,
        max_length=max_length,
        batch_size=bs,
        foundation_lora_snap=foundation_lora_snap,
        dcfg=dcfg,
    )
    foundation_eval_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    names, vec = trainable_state_vector(model)
    w_div = compute_weight_divergence(
        model=model,
        client_lora_params=client_lora,
        foundation_vec=foundation_vec,
        foundation_lora_snap=foundation_lora_snap,
    )
    wdiv_sec = time.perf_counter() - t0
    round_elapsed = time.perf_counter() - round_t0

    round_timing = build_round_timing(
        apply_global_params_sec=apply_sec,
        post_redistribution_eval_sec=post_eval_sec,
        fit_duration_sec=fit_elapsed,
        local_adaptation_eval_sec=local_eval_sec,
        foundation_generalization_eval_sec=foundation_eval_sec,
        weight_divergence_sec=wdiv_sec,
        round_duration_sec=round_elapsed,
    )
    metric_row = build_round_metric_record(
        server_round=server_round,
        cid=cid,
        post_redistribution=post_redist,
        local_adaptation=local_adapt,
        foundation_generalization=foundation_eval,
        weight_divergence=w_div,
        round_timing=round_timing,
        extra={"avg_train_loss": loss_sum / max(1, steps), "steps": steps, **hw},
    )
    return {
        "names": names,
        "vector": vec,
        "num_examples": len(train_rows),
        "metrics": metric_row,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--cpu", action="store_true", help="Force CPU (slow but avoids GPU OOM)")
    ap.add_argument("--num-rounds", type=int, default=None)
    ap.add_argument("--partition-mode", default=None)
    ap.add_argument("--label-skew-alpha", type=float, default=None)
    ap.add_argument("--eval-max", type=int, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--train-ratio", type=float, default=None)
    ap.add_argument("--max-samples-per-client", type=int, default=None)
    ap.add_argument("--num-clients", type=int, default=None)
    ap.add_argument("--strategy", default=None, help="FedAvg | FedProx")
    ap.add_argument("--fedprox-mu", type=float, default=None)
    args = ap.parse_args()

    cfg = merged_config()
    cfg = apply_fl_cli_overrides(cfg, args)
    num_rounds = int(cfg["fl"].get("num_rounds", 1))
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    art = cfg["artifacts"]
    run_dir = Path(art["runs"]) / "step05_fl" / run_id
    ensure_dirs(run_dir, Path(art["checkpoints"]))
    shutil.copy(args.config, run_dir / "used_config.yaml")

    log = RunLogger(run_dir, name="fl_local_sim")
    strategy_name = fl_strategy_name(cfg)
    flower_settings = build_fl_flower_settings_full(
        cfg, role="local_fedavg_sim", strategy_class=strategy_name
    )
    log.log({"event": "experiment_record", "flower_settings": flower_settings})
    log.log_meta(
        {
            "mode": "local_fedavg_sim",
            "run_id": run_id,
            "environment": build_environment_block(cfg),
            "flower_settings": flower_settings,
        }
    )

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cpu" if args.cpu else device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]

    _pool, eval_rows, client_parts, part_meta = build_fl_dataset(cfg, root=_ROOT)
    save_partition_artifact(run_dir, part_meta, client_parts)
    num_clients = len(client_parts)

    global_names, global_vec = None, None
    all_round_metrics: list = []

    for server_round in range(1, num_rounds + 1):
        round_t0 = time.perf_counter()
        client_updates = []
        for cid in range(num_clients):
            train_rows = get_client_train_rows(client_parts, cid)
            model, processor = build_model(cfg, device)
            unfreeze_backbone(model)
            model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
            model.backbone.set_adapter("client")
            ensure_client_trainable(model, model.backbone)
            surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
            foundation_lora_snap = snapshot_lora(surrogate_lora)
            foundation_vec = client_trainable_vector(model, lora_params_for_adapter(model.backbone, "client"))
            if cid == 0 and global_names is None:
                global_names, global_vec = trainable_state_vector(model)

            out = _client_round(
                model,
                processor,
                train_rows,
                eval_rows,
                cfg=cfg,
                device=device,
                cid=cid,
                server_round=server_round,
                global_names=global_names,
                global_vec=global_vec,
                foundation_vec=foundation_vec,
                foundation_lora_snap=foundation_lora_snap,
            )
            client_updates.append((out["names"], out["vector"], out["num_examples"]))
            log.log(out["metrics"])
            all_round_metrics.append(out["metrics"])
            del model, processor
            if device.type == "cuda":
                torch.cuda.empty_cache()

        global_names, global_vec = fedavg_weights([(n, v) for n, v, _ in client_updates])
        server_round_sec = time.perf_counter() - round_t0
        log.log(
            {
                "event": "server_round_aggregate",
                "server_round": server_round,
                "server_round_duration_sec": server_round_sec,
                "num_clients": num_clients,
            }
        )

    ckpt_name = str(cfg["fl"].get("checkpoint_name", "step05_fedavg_global.pt"))
    ckpt = Path(art["checkpoints"]) / ckpt_name
    save_fl_checkpoint(
        ckpt,
        names=global_names,
        vector=global_vec,
        meta={
            "run_id": run_id,
            "mode": "local_fedavg_sim",
            "strategy": strategy_name,
            "partition_mode": part_meta.get("partition_mode"),
            "flower_settings": flower_settings,
        },
    )

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    vector_to_trainable_state(model, global_names, global_vec)
    global_metrics = evaluate_classifier(
        model,
        processor,
        eval_rows,
        device=device,
        prompt=str(dcfg.get("image_prompt", "")),
        max_length=int(tcfg.get("max_length", 256)),
        batch_size=max(1, int(tcfg.get("batch_size", 1))),
        adapter="client",
        dcfg=dcfg,
    )
    lora = lora_params_for_adapter(model.backbone, "client")
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": "local_fedavg_sim",
        "environment": build_environment_block(cfg),
        "flower_settings": flower_settings,
        "num_rounds": num_rounds,
        "strategy": strategy_name,
        "partition_mode": part_meta.get("partition_mode"),
        "global_checkpoint": str(ckpt),
        "adapter_size_mb": adapter_size_mb(lora),
        "global_eval": global_metrics,
        "round_metrics": all_round_metrics,
        "status": "completed",
    }
    log.save_summary(summary)
    log.log_meta({"status": "completed"})
    print("Local FedAvg sim OK:", summary)


if __name__ == "__main__":
    main()
