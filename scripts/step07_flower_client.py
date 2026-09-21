"""
Step 7: Flower クライアント — レパートリー対応。

fit_config（サーバから毎ラウンド受信）で動作が決まる:
  apply_global_weights=1 … baseline: FedAvg 後の重みを適用（Step 5 型配布）
  server_logits_path     … 提案系: ⑤ server logits から distill
  export_logits / export_grads / export_post_logits … ②b 送信物
  prox_mu > 0            … baseline_fedprox の proximal 項

環境変数:
  THESIS_CLIENT_ID=0/1
  THESIS_FL_RUN_ID=<server と同じ run_id>
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from src.config_loader import merged_config
from src.fl_flower_config import build_fl_flower_settings_full
from src.fl_logits import distill_adapter_from_logits, export_client_logits, load_client_logits_artifact
from src.fl_step07.config import apply_step07_cli_overrides, step07_run_dir_name
from src.fl_step07.sidecar import (
    client_grads_path,
    client_logits_path,
    client_post_logits_path,
    write_logits_sidecar,
)
from src.fl_utils import trainable_state_vector, vector_to_trainable_state
from src.gpu_lock import gpu_file_lock
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, ensure_client_trainable, lora_params_for_adapter
from src.resource_metrics import build_environment_block, reset_peak_memory_stats
from src.step07.data_splits import load_step07_pools_from_artifact
from src.step07.repertoire import repertoire_from_stage_profile, repertoire_record, resolve_repertoire
from src.step07.train_steps import train_client_l_task
from src.step07.transmission import (
    export_client_grad_vector,
    export_client_post_logits,
    save_grad_vector_sidecar,
    sidecar_size_bytes,
)
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def _resolve_run_dirs(art: dict, run_id: str | None, rep_name: str, cid: int) -> tuple[Path, Path]:
    base = Path(art["runs"]) / step07_run_dir_name(rep_name)
    if run_id:
        return base / run_id, base / run_id / f"client_{cid}"
    return base, base / f"client_{cid}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--num-rounds", type=int, default=None)
    ap.add_argument("--num-clients", type=int, default=None)
    ap.add_argument("--eval-max", type=int, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--repertoire", default=None)
    ap.add_argument("--stage-profile", default=None)
    ap.add_argument("--distill-steps", type=int, default=None)
    ap.add_argument("--max-client-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = merged_config()
    cfg = apply_step07_cli_overrides(cfg, args)
    scfg = cfg.setdefault("step07", {})
    cid = int(os.environ.get("THESIS_CLIENT_ID", "0"))

    cli_rep = args.repertoire
    if not cli_rep and args.stage_profile:
        mapped = repertoire_from_stage_profile(args.stage_profile)
        cli_rep = mapped.name if mapped else None
    rep = resolve_repertoire(cfg, cli_rep)

    run_id = os.environ.get("THESIS_FL_RUN_ID")
    if not run_id:
        latest_ptr = Path(cfg["artifacts"]["runs"]) / step07_run_dir_name(rep.name) / "LATEST_RUN_ID"
        if latest_ptr.is_file():
            run_id = latest_ptr.read_text(encoding="utf-8").strip()

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cpu" if os.environ.get("THESIS_FORCE_CPU") == "1" else device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    art = cfg["artifacts"]
    run_root, run_dir = _resolve_run_dirs(art, run_id, rep.name, cid)
    ensure_dirs(run_dir)

    log = RunLogger(run_dir, name="fl_client")
    env = build_environment_block(cfg)
    host, port = (args.server.rsplit(":", 1) + ["8080"])[:2] if ":" in args.server else (args.server, "8080")
    rep_record = repertoire_record(rep)
    flower_settings = build_fl_flower_settings_full(
        cfg,
        role="client",
        server_host=host,
        server_port=port,
        strategy_class=f"Step07Strategy[{rep.name}]",
        client_env={
            "THESIS_FL_RUN_ID": run_id,
            "THESIS_CLIENT_ID": str(cid),
            "THESIS_FORCE_CPU": os.environ.get("THESIS_FORCE_CPU"),
        },
        client_cli=vars(args),
    )
    log.log({"event": "experiment_record", "flower_settings": flower_settings, "repertoire": rep_record})
    log.log_meta(
        {
            "step": "7-fl-client",
            "scheme": "step07_flower",
            "client_id": cid,
            "run_id": run_id,
            "repertoire": rep_record,
            "status": "started",
            "cli": vars(args),
            "environment": env,
            "device": str(device),
            "flower_settings": flower_settings,
        }
    )

    splits_path = run_root / "data_splits.json"
    if not splits_path.is_file():
        raise FileNotFoundError(f"data_splits.json not found: {splits_path} (start step07_flower_server first)")
    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    train_rows, client_eval, server_train, _server_eval, splits_meta = load_step07_pools_from_artifact(
        splits_path, manifest_path=manifest, dcfg=dcfg, client_id=cid
    )
    log.log(
        {
            "event": "data_splits",
            "client_id": cid,
            "train_samples": len(train_rows),
            "client_eval_samples": len(client_eval),
            "server_train_samples": len(server_train),
            "counts": splits_meta.get("counts"),
        }
    )

    client_model_id = str(cfg["model"].get("client_id") or cfg["model"]["id"])
    model, processor = build_model(cfg, device, model_id=client_model_id)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    ensure_client_trainable(model, model.backbone)

    client_lora = lora_params_for_adapter(model.backbone, "client")
    frozen_surrogate = lora_params_for_adapter(model.backbone, "surrogate")
    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.01)))

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    bs = max(1, int(tcfg.get("batch_size", 1)))
    local_epochs = int(cfg["fl"].get("local_epochs", 1))
    max_client_steps = args.max_client_steps if args.max_client_steps is not None else scfg.get("max_client_steps")
    inner_lr = float(cfg["consistency"].get("inner_lr", 1e-4))
    temperature = float(cfg["consistency"].get("temperature", 2.0))

    fcfg = cfg["fl"]
    gpu_serialize = bool(fcfg.get("gpu_serialize", False)) or os.environ.get("THESIS_GPU_SERIALIZE") == "1"
    gpu_lock_path = Path(art["root"]) / ".fl_gpu.lock"
    if not gpu_lock_path.is_absolute():
        gpu_lock_path = _ROOT / gpu_lock_path

    @contextmanager
    def gpu_round_lock():
        if device.type == "cuda" and gpu_serialize:
            print(f"[step07-client {cid}] waiting GPU lock...", flush=True)
            with gpu_file_lock(gpu_lock_path, client_id=cid):
                print(f"[step07-client {cid}] GPU lock acquired", flush=True)
                yield
        else:
            yield

    def export_fl_parameters() -> tuple[list[str], np.ndarray]:
        model.backbone.set_adapter("client")
        ensure_client_trainable(model, model.backbone)
        return trainable_state_vector(model)

    import flwr as fl

    class Step07Client(fl.client.NumPyClient):
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
            if isinstance(parameters, list) and len(parameters) == 1 and self._names:
                vector_to_trainable_state(model, self._names, parameters[0])

        def _prox_loss_fn(self, global_vec: np.ndarray, mu: float) -> Callable[[], torch.Tensor]:
            """FedProx: (mu/2)*||w - w_global||^2（微分可能な形で構成）。"""
            gt = torch.tensor(np.asarray(global_vec), dtype=torch.float32)
            name_to_param = dict(model.named_parameters())

            def fn() -> torch.Tensor:
                total = torch.zeros((), device=device)
                offset = 0
                for n in self._names:
                    p = name_to_param[n]
                    numel = p.numel()
                    g = gt[offset : offset + numel].view_as(p).to(p.device)
                    total = total + ((p.float() - g) ** 2).sum()
                    offset += numel
                return (mu / 2.0) * total

            return fn

        def fit(self, parameters, config):
            with gpu_round_lock():
                return self._fit_body(parameters, config)

        def _fit_body(self, parameters, config):
            self._round += 1
            server_round = int(config.get("server_round", self._round))
            round_t0 = time.perf_counter()

            apply_global = int(config.get("apply_global_weights", 0)) == 1
            if apply_global:
                # baseline: FedAvg（+ feddf_pure 蒸留後）の重みを適用 = Step 5 型配布
                self.set_parameters(parameters)

            # ⑤ 提案系: 前ラウンド server logits から distill
            distill_stats: Dict[str, float] = {}
            server_logits_path_str = str(config.get("server_logits_path", "") or "")
            if server_logits_path_str and Path(server_logits_path_str).is_file():
                teacher_map = load_client_logits_artifact(Path(server_logits_path_str))
                distill_stats = distill_adapter_from_logits(
                    model,
                    processor,
                    server_train,
                    dcfg,
                    teacher_logits_map=teacher_map,
                    adapter="client",
                    trainable_params=list(model.classifier.parameters()) + client_lora,
                    prompt=prompt,
                    max_length=max_length,
                    device=device,
                    lr=float(config.get("distill_lr", tcfg["lr"])),
                    max_steps=int(config.get("distill_steps", 20)) or None,
                    temperature=temperature,
                )
                log.log(
                    {
                        "event": "distill_redistribution",
                        "server_round": server_round,
                        "cid": cid,
                        "server_logits_path": server_logits_path_str,
                        **distill_stats,
                    }
                )

            # ② L_task（baseline_fedprox は proximal 項を追加）
            prox_mu = float(config.get("prox_mu", 0.0))
            extra_loss_fn = None
            if prox_mu > 0.0 and parameters:
                extra_loss_fn = self._prox_loss_fn(parameters[0], prox_mu)

            model.train()
            reset_peak_memory_stats()
            fit_t0 = time.perf_counter()
            steps_total = 0
            loss_task_sum_total = 0.0
            loss_fit_sum_total = 0.0
            loss_task_last = 0.0
            loss_fit_last = 0.0
            for _ in range(local_epochs):
                epoch_steps, l_task_stats = train_client_l_task(
                    model=model,
                    processor=processor,
                    train_rows=train_rows,
                    dcfg=dcfg,
                    client_lora=client_lora,
                    frozen_lora=frozen_surrogate,
                    opt=opt,
                    prompt=prompt,
                    max_length=max_length,
                    device=device,
                    batch_size=bs,
                    max_steps=max_client_steps,
                    extra_loss_fn=extra_loss_fn,
                )
                steps_total += epoch_steps
                loss_task_sum_total += l_task_stats["avg_loss_task"] * epoch_steps
                loss_fit_sum_total += l_task_stats["avg_loss_fit"] * epoch_steps
                loss_task_last = l_task_stats["loss_task_last"]
                loss_fit_last = l_task_stats["loss_fit_last"]
            fit_elapsed = time.perf_counter() - fit_t0
            n_steps = max(1, steps_total)
            avg_loss_task = loss_task_sum_total / n_steps
            avg_loss_fit = loss_fit_sum_total / n_steps

            # ②b 送信物（sidecar）: logits / 勾配ベクトル / 1-step 適応後 logits
            comm: Dict[str, Any] = {}
            if int(config.get("export_logits", 0)) == 1:
                logits_map = export_client_logits(
                    model, processor, server_train, dcfg,
                    adapter="client", prompt=prompt, max_length=max_length, device=device,
                )
                out_path = client_logits_path(run_root, server_round, cid)
                write_logits_sidecar(out_path, server_round, logits_map)
                comm["client_logits_bytes"] = sidecar_size_bytes(out_path)
                log.log(
                    {
                        "event": "client_logits_exported",
                        "server_round": server_round,
                        "cid": cid,
                        "path": str(out_path),
                        "num_samples": len(logits_map),
                    }
                )

            if int(config.get("export_grads", 0)) == 1:
                grad_vec = export_client_grad_vector(
                    model, processor, server_train, dcfg,
                    adapter="client", lora_params=client_lora,
                    prompt=prompt, max_length=max_length, device=device,
                )
                out_path = client_grads_path(run_root, server_round, cid)
                save_grad_vector_sidecar(out_path, server_round, cid, grad_vec)
                comm["client_grads_bytes"] = sidecar_size_bytes(out_path)
                log.log(
                    {
                        "event": "client_grads_exported",
                        "server_round": server_round,
                        "cid": cid,
                        "path": str(out_path),
                        "numel": int(grad_vec.numel()),
                    }
                )

            if int(config.get("export_post_logits", 0)) == 1:
                post_map = export_client_post_logits(
                    model, processor, server_train, dcfg,
                    adapter="client", lora_params=client_lora, inner_lr=inner_lr,
                    prompt=prompt, max_length=max_length, device=device,
                )
                out_path = client_post_logits_path(run_root, server_round, cid)
                write_logits_sidecar(out_path, server_round, post_map)
                comm["client_post_logits_bytes"] = sidecar_size_bytes(out_path)
                log.log(
                    {
                        "event": "client_post_logits_exported",
                        "server_round": server_round,
                        "cid": cid,
                        "path": str(out_path),
                        "num_samples": len(post_map),
                    }
                )

            model.eval()
            metrics = evaluate_classifier(
                model, processor, client_eval,
                device=device, prompt=prompt, max_length=max_length, batch_size=bs,
                adapter="client", dcfg=dcfg,
            )
            names, vec = export_fl_parameters()
            self._names = names
            comm["weights_bytes"] = int(vec.nbytes)
            round_elapsed = time.perf_counter() - round_t0
            log.log(
                {
                    "event": "fit_round",
                    "server_round": server_round,
                    "cid": cid,
                    "repertoire": str(config.get("repertoire", rep.name)),
                    "apply_global_weights": apply_global,
                    "steps": steps_total,
                    "fit_duration_sec": fit_elapsed,
                    "round_duration_sec": round_elapsed,
                    "avg_loss_task": avg_loss_task,
                    "loss_task_last": loss_task_last,
                    "avg_loss_fit": avg_loss_fit,
                    "loss_fit_last": loss_fit_last,
                    "eval_accuracy": metrics.get("accuracy"),
                    "eval_f1_macro": metrics.get("f1_macro"),
                    "distill_kl_last": distill_stats.get("distill_kl_last"),
                    "communication_bytes": comm,
                }
            )
            fit_metrics = {
                "cid": float(cid),
                "post_redistribution_accuracy": float(metrics.get("accuracy", 0.0)),
                "local_adaptation_accuracy": float(metrics.get("accuracy", 0.0)),
                "fit_duration_sec": float(fit_elapsed),
                "round_duration_sec": float(round_elapsed),
                "distill_steps": float(distill_stats.get("distill_steps", 0.0)),
                "avg_loss_task": float(avg_loss_task),
                "loss_task_last": float(loss_task_last),
                "avg_loss_fit": float(avg_loss_fit),
                "loss_fit_last": float(loss_fit_last),
                **{k: float(v) for k, v in comm.items()},
            }
            return [vec], len(train_rows), fit_metrics

        def evaluate(self, parameters, config):
            with gpu_round_lock():
                server_round = int(config.get("server_round", self._round))
                if int(config.get("apply_global_weights", 0)) == 1:
                    self.set_parameters(parameters)
                metrics = evaluate_classifier(
                    model, processor, client_eval,
                    device=device, prompt=prompt, max_length=max_length, batch_size=bs,
                    adapter="client", dcfg=dcfg,
                )
                eval_loss = 1.0 - float(metrics["accuracy"])
                log.log(
                    {
                        "event": "evaluate",
                        "cid": cid,
                        "server_round": server_round,
                        "eval_accuracy": metrics["accuracy"],
                    }
                )
                return eval_loss, len(client_eval), {
                    "accuracy": float(metrics["accuracy"]),
                    "f1_macro": float(metrics["f1_macro"]),
                }

    try:
        fl.client.start_numpy_client(server_address=args.server, client=Step07Client())
    finally:
        log.save_summary(
            {
                "client_id": cid,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "scheme": "step07_flower",
                "repertoire": rep_record,
                "environment": env,
                "device": str(device),
                "train_samples": len(train_rows),
                "flower_settings": flower_settings,
                "status": "completed",
            }
        )
        log.log_meta({"status": "completed"})


if __name__ == "__main__":
    main()
