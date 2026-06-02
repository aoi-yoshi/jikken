"""
Step 6: Flower クライアント（ローカル学習→LoRA+ヘッドを送信）。
環境変数 THESIS_CLIENT_ID=0/1 でデータ分割を切り替える。
THESIS_FL_RUN_ID は step05 と同じ run_id を指定する。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.amp import autocast

from src.config_loader import deep_merge, load_yaml, merged_config
from src.dataset_manifest import read_manifest_filtered, split_clients, stratified_subset
from src.fl_utils import trainable_state_vector, vector_to_trainable_state
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, lora_params_for_adapter, sync_surrogate_from_client
from src.resource_metrics import (
    adapter_size_mb,
    build_environment_block,
    reset_peak_memory_stats,
    snapshot_hardware,
    trainable_param_count,
)
from src.train_common import device_or_auto, iter_row_chunks, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def _load_config(config_arg: str) -> dict:
    cfg = merged_config()
    if config_arg and config_arg != "config/default.yaml":
        override = load_yaml(_ROOT / config_arg if not Path(config_arg).is_absolute() else config_arg)
        cfg = deep_merge(cfg, override)
    return cfg


def _resolve_run_dir(art: dict, run_id: str | None, cid: int) -> Path:
    base = Path(art["runs"]) / "step06_fl"
    if run_id:
        return base / run_id / f"client_{cid}"
    return Path(art["runs"]) / f"step06_fl_client_{cid}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--server", default="127.0.0.1:8080")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    cid = int(os.environ.get("THESIS_CLIENT_ID", "0"))
    run_id = os.environ.get("THESIS_FL_RUN_ID")
    if not run_id:
        latest_ptr = Path(cfg["artifacts"]["runs"]) / "step06_fl" / "LATEST_RUN_ID"
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
    log.log_meta(
        {
            "step": 6,
            "client_id": cid,
            "run_id": run_id,
            "status": "started",
            "environment": build_environment_block(cfg),
        }
    )

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    max_samples = int(dcfg.get("max_train_samples", 300))
    rows = stratified_subset(rows, max_samples, int(cfg["train"]["seed"]))
    n_train = int(len(rows) * float(dcfg.get("train_ratio", 0.8)))
    pool = rows[:n_train]
    eval_max = int(tcfg.get("eval_max", 20))
    eval_rows = rows[n_train : n_train + eval_max]
    a, b = split_clients(pool, int(cfg["train"]["seed"]))
    train_rows = a if cid == 0 else b

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")

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
    bs = int(tcfg.get("batch_size", 1))
    local_epochs = int(cfg["fl"].get("local_epochs", int(tcfg.get("num_epochs", 1))))

    import flwr as fl

    class Client(fl.client.NumPyClient):
        def __init__(self) -> None:
            names, _ = trainable_state_vector(model)
            self._names = names
            self._round = 0

        def get_parameters(self, config):
            names, vec = trainable_state_vector(model)
            self._names = names
            return [vec]

        def set_parameters(self, parameters):
            if parameters is None:
                return
            if isinstance(parameters, list) and len(parameters) == 1:
                vec = parameters[0]
                if self._names:
                    vector_to_trainable_state(model, self._names, vec)
            sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")

        def fit(self, parameters, config):
            self._round += 1
            server_round = int(config.get("server_round", self._round))
            self.set_parameters(parameters)
            model.train()
            reset_peak_memory_stats()
            t0 = time.perf_counter()
            steps = 0
            loss_sum = 0.0
            for _ in range(local_epochs):
                for chunk in iter_row_chunks(train_rows, bs):
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
                    loss_sum += float(loss.detach().cpu())
                    steps += 1
            elapsed = time.perf_counter() - t0
            hw = snapshot_hardware(elapsed_sec=elapsed, steps=steps)
            avg_loss = loss_sum / max(1, steps)
            names, vec = trainable_state_vector(model)
            self._names = names
            row: Dict[str, Any] = {
                "event": "fit",
                "cid": cid,
                "server_round": server_round,
                "avg_train_loss": avg_loss,
                "steps": steps,
                "adapter_size_mb": adapter_mb,
                **hw,
            }
            log.log(row)
            return [vec], len(train_rows), {"avg_train_loss": avg_loss, **hw}

        def evaluate(self, parameters, config):
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
            hw = snapshot_hardware(elapsed_sec=elapsed, steps=max(1, len(eval_rows) // max(1, bs)))
            eval_loss = 1.0 - float(metrics["accuracy"])
            row = {
                "event": "evaluate",
                "cid": cid,
                "server_round": server_round,
                "eval_accuracy": metrics["accuracy"],
                "eval_f1_macro": metrics["f1_macro"],
                "eval_loss_proxy": eval_loss,
                **hw,
            }
            log.log(row)
            return eval_loss, len(eval_rows), {
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                **hw,
            }

    try:
        fl.client.start_numpy_client(server_address=args.server, client=Client())
    finally:
        log.save_summary(
            {
                "client_id": cid,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "train_samples": len(train_rows),
                "eval_samples": len(eval_rows),
                "adapter_size_mb": adapter_mb,
                "trainable_parameters": trainable_n,
                "status": "completed",
            }
        )
        log.log_meta({"status": "completed"})


if __name__ == "__main__":
    main()
