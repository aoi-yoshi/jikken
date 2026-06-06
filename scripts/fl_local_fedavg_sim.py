"""
Flower なしのローカル FedAvg シミュレーション（IID, 2 clients, 1 round）。
gRPC 完走前に LoRA 集約 → checkpoint → eval パイプラインを検証する。
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.amp import autocast

from src.config_loader import deep_merge, load_yaml, merged_config
from src.dataset_manifest import read_manifest_filtered, split_clients, stratified_subset
from src.fl_utils import fedavg_weights, save_fl_checkpoint, trainable_state_vector, vector_to_trainable_state
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, lora_params_for_adapter, sync_surrogate_from_client
from src.resource_metrics import adapter_size_mb, build_environment_block, reset_peak_memory_stats, snapshot_hardware
from src.train_common import device_or_auto, iter_row_chunks, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def _load_config(config_arg: str) -> dict:
    cfg = merged_config()
    if config_arg and config_arg != "config/default.yaml":
        override = load_yaml(_ROOT / config_arg if not Path(config_arg).is_absolute() else config_arg)
        cfg = deep_merge(cfg, override)
    return cfg


def _train_local(model, processor, train_rows, *, cfg, device, cid: int) -> dict:
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    bs = int(tcfg.get("batch_size", 1))
    local_epochs = int(cfg["fl"].get("local_epochs", 1))

    client_lora = lora_params_for_adapter(model.backbone, "client")
    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))

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
    hw = snapshot_hardware(elapsed_sec=time.perf_counter() - t0, steps=steps)
    names, vec = trainable_state_vector(model)
    return {
        "cid": cid,
        "names": names,
        "vector": vec,
        "num_examples": len(train_rows),
        "avg_train_loss": loss_sum / max(1, steps),
        **hw,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/fl_e2e_smoke.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--cpu", action="store_true", help="Force CPU (slow but avoids GPU OOM)")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    art = cfg["artifacts"]
    run_dir = Path(art["runs"]) / "step06_fl" / run_id
    ensure_dirs(run_dir, Path(art["checkpoints"]))

    log = RunLogger(run_dir, name="fl_local_sim")
    log.log_meta({"mode": "local_fedavg_sim", "run_id": run_id, "environment": build_environment_block(cfg)})

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cpu" if args.cpu else device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    rows = stratified_subset(rows, int(dcfg.get("max_train_samples", 300)), int(cfg["train"]["seed"]))
    n_train = int(len(rows) * float(dcfg.get("train_ratio", 0.8)))
    pool = rows[:n_train]
    eval_max = int(tcfg.get("eval_max", 20))
    eval_rows = rows[n_train : n_train + eval_max]
    a, b = split_clients(pool, int(cfg["train"]["seed"]))

    client_updates = []
    init_names = None
    for cid, train_rows in enumerate((a, b)):
        model, processor = build_model(cfg, device)
        unfreeze_backbone(model)
        model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
        model.backbone.set_adapter("client")
        if cid == 0:
            init_names, init_vec = trainable_state_vector(model)
        else:
            vector_to_trainable_state(model, init_names, init_vec)
            sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")

        stats = _train_local(model, processor, train_rows, cfg=cfg, device=device, cid=cid)
        client_updates.append((stats["names"], stats["vector"], stats["num_examples"]))
        log.log({"event": "local_fit", **stats})

        metrics = evaluate_classifier(
            model,
            processor,
            eval_rows,
            device=device,
            prompt=str(dcfg.get("image_prompt", "")),
            max_length=int(tcfg.get("max_length", 256)),
            batch_size=int(tcfg.get("batch_size", 1)),
            adapter="client",
            dcfg=dcfg,
        )
        log.log({"event": "local_eval", "cid": cid, **metrics})
        del model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    names, avg_vec = fedavg_weights([(n, v) for n, v, _ in client_updates])
    ckpt = Path(art["checkpoints"]) / "step06_fedavg_global.pt"
    save_fl_checkpoint(
        ckpt,
        names=names,
        vector=avg_vec,
        meta={"run_id": run_id, "mode": "local_fedavg_sim", "num_clients": 2},
    )

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    vector_to_trainable_state(model, names, avg_vec)
    sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")
    global_metrics = evaluate_classifier(
        model,
        processor,
        eval_rows,
        device=device,
        prompt=str(dcfg.get("image_prompt", "")),
        max_length=int(tcfg.get("max_length", 256)),
        batch_size=int(tcfg.get("batch_size", 1)),
        adapter="client",
        dcfg=dcfg,
    )
    lora = lora_params_for_adapter(model.backbone, "client")
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": "local_fedavg_sim",
        "global_checkpoint": str(ckpt),
        "adapter_size_mb": adapter_size_mb(lora),
        "global_eval": global_metrics,
        "client_updates": len(client_updates),
        "status": "completed",
    }
    log.save_summary(summary)
    log.log_meta({"status": "completed"})
    print("Local FedAvg sim OK:", summary)


if __name__ == "__main__":
    main()
