"""
FedAvg 後のグローバル LoRA checkpoint を eval セットで評価する。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from src.config_loader import deep_merge, load_yaml, merged_config
from src.dataset_manifest import read_manifest_filtered, stratified_subset
from src.fl_utils import load_fl_checkpoint, vector_to_trainable_state
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, sync_surrogate_from_client
from src.resource_metrics import build_environment_block
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def _load_config(config_arg: str) -> dict:
    cfg = merged_config()
    if config_arg and config_arg != "config/default.yaml":
        override = load_yaml(_ROOT / config_arg if not Path(config_arg).is_absolute() else config_arg)
        cfg = deep_merge(cfg, override)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/step06_fedavg_global.pt",
        help="step05 が保存した FedAvg グローバル LoRA",
    )
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    cfg = _load_config(args.config)
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    art = cfg["artifacts"]

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _ROOT / ckpt_path
    names, vec, meta = load_fl_checkpoint(ckpt_path)

    run_dir = Path(art["runs"]) / "step06_fl" / (args.run_id or meta.get("run_id", "post_eval"))
    ensure_dirs(run_dir)
    log = RunLogger(run_dir, name="fl_post_eval")
    log.log_meta({"step": "6-eval", "checkpoint": str(ckpt_path), "environment": build_environment_block(cfg)})

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    rows = stratified_subset(rows, int(dcfg.get("max_train_samples", 300)), int(cfg["train"]["seed"]))
    n_train = int(len(rows) * float(dcfg.get("train_ratio", 0.8)))
    eval_max = int(tcfg.get("eval_max", 20))
    eval_rows = rows[n_train : n_train + eval_max]

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    vector_to_trainable_state(model, names, vec)
    sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    bs = int(tcfg.get("batch_size", 1))

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
    summary = {
        "checkpoint": str(ckpt_path),
        "eval_rows": len(eval_rows),
        "metrics": metrics,
        "meta": meta,
    }
    log.log(summary)
    log.save_summary(summary)
    print("Global model eval:", metrics)


if __name__ == "__main__":
    main()
