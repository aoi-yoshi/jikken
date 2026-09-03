"""
Step 5: Flower クライアント（ローカル学習→LoRA+ヘッドを送信）。
環境変数 THESIS_CLIENT_ID=0/1 でデータ分割を切り替える。
THESIS_FL_RUN_ID は step05_flower_server と同じ run_id を指定する。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.amp import autocast

from src.adaptation_losses import AdaptationLosses
from src.config_loader import merged_config
from src.fl_data import apply_fl_cli_overrides, build_fl_dataset, get_client_train_rows, save_partition_artifact
from src.fl_flower_config import build_fl_flower_settings_full
from src.fl_strategy import fl_strategy_name
from src.fl_metrics import (
    build_round_metric_record,
    build_round_timing,
    client_trainable_vector,
    compute_weight_divergence,
    evaluate_foundation_generalization,
    flower_fit_metrics_from_record,
    proximal_penalty,
)
from src.fl_strategy import is_fedprox
from src.fl_utils import trainable_state_vector, vector_to_trainable_state
from src.gpu_lock import gpu_file_lock
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.diff_probe import snapshot_lora
from src.peft_setup import attach_dual_lora, ema_surrogate_from_client, ensure_client_trainable, lora_params_for_adapter
from src.resource_metrics import (
    adapter_size_mb,
    build_environment_block,
    build_training_hardware_snapshot,
    GpuUtilTracker,
    profile_one_step,
    reset_peak_memory_stats,
    trainable_param_count,
)
from src.train_common import device_or_auto, iter_row_chunks, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def _resolve_run_dir(art: dict, run_id: str | None, cid: int) -> Path:
    base = Path(art["runs"]) / "step05_fl"
    if run_id:
        return base / run_id / f"client_{cid}"
    return Path(art["runs"]) / f"step05_fl_client_{cid}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--num-rounds", type=int, default=None)
    ap.add_argument("--partition-mode", default=None)
    ap.add_argument("--label-skew-alpha", type=float, default=None)
    ap.add_argument("--eval-max", type=int, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--train-ratio", type=float, default=None)
    ap.add_argument("--max-samples-per-client", type=int, default=None)
    ap.add_argument("--num-clients", type=int, default=None)
    ap.add_argument("--consistency-mode", default=None, help="none | lpred | lpred_grad | lpred_post")
    args = ap.parse_args()

    cfg = merged_config()
    cfg = apply_fl_cli_overrides(cfg, args)
    cid = int(os.environ.get("THESIS_CLIENT_ID", "0"))
    num_clients = int(cfg["fl"].get("num_clients", 2))

    run_id = os.environ.get("THESIS_FL_RUN_ID")
    if not run_id:
        latest_ptr = Path(cfg["artifacts"]["runs"]) / "step05_fl" / "LATEST_RUN_ID"
        if latest_ptr.is_file():
            run_id = latest_ptr.read_text(encoding="utf-8").strip()

    set_seed(int(cfg["train"]["seed"]))

    if os.environ.get("THESIS_FORCE_CPU") == "1":
        device = torch.device("cpu")
    else:
        device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    art = cfg["artifacts"]
    run_dir = _resolve_run_dir(art, run_id, cid)
    ensure_dirs(run_dir)

    log = RunLogger(run_dir, name="fl_client")
    env = build_environment_block(cfg)
    host, port = (args.server.rsplit(":", 1) + ["8080"])[:2] if ":" in args.server else (args.server, "8080")
    flower_settings = build_fl_flower_settings_full(
        cfg,
        role="client",
        server_host=host,
        server_port=port,
        strategy_class=fl_strategy_name(cfg),
        client_env={
            "THESIS_FL_RUN_ID": run_id,
            "THESIS_CLIENT_ID": str(cid),
            "THESIS_FORCE_CPU": os.environ.get("THESIS_FORCE_CPU"),
        },
        client_cli=vars(args),
    )
    log.log({"event": "experiment_record", "flower_settings": flower_settings})
    log.log_meta(
        {
            "step": "5-client",
            "client_id": cid,
            "num_clients": num_clients,
            "run_id": run_id,
            "status": "started",
            "cli": vars(args),
            "environment": env,
            "device": str(device),
            "flower_settings": flower_settings,
        }
    )

    _pool, eval_rows, client_parts, part_meta = build_fl_dataset(cfg, root=_ROOT)
    train_rows = get_client_train_rows(client_parts, cid)
    if cid == 0 and run_id:
        save_partition_artifact(run_dir.parent, part_meta, client_parts)
    log.log({"event": "partition", **part_meta, "client_id": cid, "train_samples": len(train_rows)})

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    ensure_client_trainable(model, model.backbone)

    client_lora = lora_params_for_adapter(model.backbone, "client")
    adapter_mb = adapter_size_mb(client_lora)
    trainable_n = trainable_param_count(model)
    log.log(
        {
            "event": "model_ready",
            "trainable_parameters": trainable_n,
            "adapter_size_mb": adapter_mb,
            "train_samples": len(train_rows),
            "eval_samples": len(eval_rows),
        }
    )

    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    bs = max(1, int(tcfg.get("batch_size", 1)))
    local_epochs = int(cfg["fl"].get("local_epochs", 1))

    import flwr as fl

    surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
    foundation_lora_snap = snapshot_lora(surrogate_lora)
    foundation_vec = client_trainable_vector(model, client_lora)
    foundation_gen = evaluate_foundation_generalization(
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
    log.log(
        {
            "event": "foundation_baseline",
            "cid": cid,
            "foundation_generalization_accuracy": foundation_gen["accuracy"],
            "foundation_generalization_f1_macro": foundation_gen["f1_macro"],
        }
    )

    fcfg = cfg["fl"]
    ccfg = cfg["consistency"]
    consistency_mode = str(fcfg.get("consistency_mode", "none")).lower()
    ac_losses: AdaptationLosses | None = None
    if consistency_mode != "none":
        ac_losses = AdaptationLosses(
            model,
            processor,
            w_pred=float(ccfg.get("w_pred", 0.5)),
            w_grad=float(ccfg.get("w_grad", 0.1)),
            w_post=float(ccfg.get("w_post", 0.5)),
            inner_lr=float(ccfg.get("inner_lr", 1e-4)),
            max_length=max_length,
            prompt=prompt,
            client_adapter="client",
            surrogate_adapter="surrogate",
            client_lora_params=client_lora,
            surrogate_lora_params=surrogate_lora,
            temperature=float(ccfg.get("temperature", 2.0)),
        )
    log.log({"event": "consistency_config", "consistency_mode": consistency_mode})

    gpu_serialize = bool(fcfg.get("gpu_serialize", False)) or os.environ.get("THESIS_GPU_SERIALIZE") == "1"
    gpu_lock_path = Path(art["root"]) / ".fl_gpu.lock"
    if not gpu_lock_path.is_absolute():
        gpu_lock_path = _ROOT / gpu_lock_path

    @contextmanager
    def gpu_round_lock():
        if device.type == "cuda" and gpu_serialize:
            print(f"[client {cid}] waiting GPU lock (single-GPU serialize)...", flush=True)
            with gpu_file_lock(gpu_lock_path, client_id=cid):
                print(f"[client {cid}] GPU lock acquired", flush=True)
                yield
        else:
            yield

    def export_fl_parameters() -> tuple[list[str], np.ndarray]:
        """foundation eval 後も client LoRA を含めて Flower に返す。"""
        model.backbone.set_adapter("client")
        ensure_client_trainable(model, model.backbone)
        return trainable_state_vector(model)

    class Client(fl.client.NumPyClient):
        def __init__(self) -> None:
            names, _ = export_fl_parameters()
            self._names = names
            self._round = 0

        def get_parameters(self, config):
            names, vec = export_fl_parameters()
            self._names = names
            return [vec]

        def set_parameters(self, parameters):
            if parameters is None:
                return
            if isinstance(parameters, list) and len(parameters) == 1:
                vec = parameters[0]
                if self._names:
                    vector_to_trainable_state(model, self._names, vec)

        def fit(self, parameters, config):
            with gpu_round_lock():
                return self._fit_body(parameters, config)

        def _fit_body(self, parameters, config):
            self._round += 1
            server_round = int(config.get("server_round", self._round))
            round_t0 = time.perf_counter()
            t0 = time.perf_counter()
            self.set_parameters(parameters)
            apply_sec = time.perf_counter() - t0

            global_vec = parameters[0] if parameters else None
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

            model.train()
            reset_peak_memory_stats()
            fit_t0 = time.perf_counter()
            steps = 0
            loss_sum = 0.0
            loss_task_sum = 0.0
            loss_pred_sum = 0.0
            loss_grad_sum = 0.0
            loss_post_sum = 0.0
            profiler_result = None
            gpu_tracker = GpuUtilTracker(device_index=0 if device.type == "cuda" else 0)
            surrogate_ema = float(ccfg.get("surrogate_ema", 1.0))
            for _ in range(local_epochs):
                for chunk in iter_row_chunks(train_rows, bs):
                    samples = load_samples(chunk, dcfg)

                    def _train_step_task_only() -> None:
                        y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
                        frames_batch = [s.frames for s in samples]
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
                        nonlocal loss_sum, loss_task_sum
                        loss_sum += float(step_loss.detach().cpu())
                        loss_task_sum += float(step_loss.detach().cpu())

                    def _train_step_ac() -> None:
                        nonlocal loss_sum, loss_task_sum, loss_pred_sum, loss_grad_sum, loss_post_sum
                        assert ac_losses is not None
                        for sample in samples:
                            y = torch.tensor([sample.label], device=device, dtype=torch.long)
                            opt.zero_grad(set_to_none=True)
                            model.backbone.set_adapter("client")
                            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                                total, stats, _extra = ac_losses.compute(consistency_mode, sample.frames, y)
                                if use_prox:
                                    total = total + proximal_penalty(model, global_vec, prox_mu, device)
                            total.backward()
                            opt.step()
                            for p in surrogate_lora:
                                p.grad = None
                            ema_surrogate_from_client(client_lora, surrogate_lora, surrogate_ema)
                            loss_sum += float(stats["loss_total"])
                            loss_task_sum += float(stats["loss_task"])
                            loss_pred_sum += float(stats["loss_pred"])
                            loss_grad_sum += float(stats["loss_grad"])
                            loss_post_sum += float(stats["loss_post"])

                    _train_step = _train_step_ac if ac_losses is not None else _train_step_task_only

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
            avg_loss = loss_sum / max(1, steps)
            avg_loss_task = loss_task_sum / max(1, steps)
            avg_loss_pred = loss_pred_sum / max(1, steps)
            avg_loss_grad = loss_grad_sum / max(1, steps)
            avg_loss_post = loss_post_sum / max(1, steps)

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
            names, vec = export_fl_parameters()
            self._names = names
            client_lora = lora_params_for_adapter(model.backbone, "client")
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
                extra={
                    "avg_train_loss": avg_loss,
                    "avg_loss_task": avg_loss_task,
                    "avg_loss_pred": avg_loss_pred,
                    "avg_loss_grad": avg_loss_grad,
                    "avg_loss_post": avg_loss_post,
                    "consistency_mode": consistency_mode,
                    "steps": steps,
                    "train_samples": len(train_rows),
                    **hw,
                },
            )
            log.log(metric_row)
            # Flower metrics はスカラーのみ（professor_links 等の dict は jsonl のみ）
            fit_return = {
                "avg_train_loss": avg_loss,
                "avg_loss_task": avg_loss_task,
                "avg_loss_pred": avg_loss_pred,
                "avg_loss_grad": avg_loss_grad,
                "avg_loss_post": avg_loss_post,
                **flower_fit_metrics_from_record(metric_row),
            }
            return [vec], len(train_rows), fit_return

        def evaluate(self, parameters, config):
            with gpu_round_lock():
                server_round = int(config.get("server_round", self._round))
                self.set_parameters(parameters)
                reset_peak_memory_stats()
                t0 = time.perf_counter()
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
                elapsed = time.perf_counter() - t0
                hw = build_training_hardware_snapshot(
                    elapsed_sec=elapsed, steps=max(1, len(eval_rows) // max(1, bs)), device=device, phase="evaluate"
                )
                eval_loss = 1.0 - float(metrics["accuracy"])
                log.log(
                    {
                        "event": "evaluate",
                        "cid": cid,
                        "server_round": server_round,
                        "eval_accuracy": metrics["accuracy"],
                        "eval_f1_macro": metrics["f1_macro"],
                        **hw,
                    }
                )
                return eval_loss, len(eval_rows), {
                    "accuracy": float(metrics["accuracy"]),
                    "f1_macro": float(metrics["f1_macro"]),
                    "eval_elapsed_sec": float(elapsed),
                }

    try:
        fl.client.start_numpy_client(server_address=args.server, client=Client())
    finally:
        log.save_summary(
            {
                "client_id": cid,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "environment": env,
                "device": str(device),
                "partition_mode": part_meta.get("partition_mode"),
                "train_samples": len(train_rows),
                "eval_samples": len(eval_rows),
                "adapter_size_mb": adapter_mb,
                "trainable_parameters": trainable_n,
                "flower_settings": flower_settings,
                "consistency_mode": consistency_mode,
                "status": "completed",
            }
        )
        log.log_meta({"status": "completed"})


if __name__ == "__main__":
    main()
