"""
Step 5: FedAvg 後のグローバル LoRA checkpoint を eval セットで評価する。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from src.config_loader import merged_config
from src.fl_data import apply_fl_cli_overrides, build_fl_dataset
from src.fl_flower_config import build_fl_flower_settings_full
from src.fl_utils import load_fl_checkpoint, vector_to_trainable_state
from src.logging_utils import RunLogger
from src.metrics import evaluate_classifier, set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, sync_surrogate_from_client
from src.resource_metrics import build_environment_block, build_training_hardware_snapshot
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/step05_fedavg_global.pt",
        help="step05 が保存した FedAvg グローバル LoRA",
    )
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--eval-max", type=int, default=None)
    args = ap.parse_args()

    cfg = merged_config()
    cfg = apply_fl_cli_overrides(cfg, args)
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    art = cfg["artifacts"]

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _ROOT / ckpt_path
    names, vec, meta = load_fl_checkpoint(ckpt_path)

    _pool, eval_rows, _client_parts, part_meta = build_fl_dataset(cfg, root=_ROOT)

    run_dir = Path(art["runs"]) / "step05_fl" / (args.run_id or meta.get("run_id", "post_eval"))
    ensure_dirs(run_dir)
    log = RunLogger(run_dir, name="fl_post_eval")
    env = build_environment_block(cfg)
    flower_settings = build_fl_flower_settings_full(
        cfg,
        role="post_eval",
        trainable_vector_dim=int(vec.size),
        param_name_count=len(names),
    )
    log.log({"event": "experiment_record", "flower_settings": flower_settings, "checkpoint_meta": meta})
    log.log_meta(
        {
            "step": "5-post-eval",
            "checkpoint": str(ckpt_path),
            "environment": env,
            "flower_settings": flower_settings,
            "checkpoint_meta": meta,
        }
    )

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    vector_to_trainable_state(model, names, vec)
    sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    bs = int(tcfg.get("batch_size", 1))

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
    hw = build_training_hardware_snapshot(elapsed_sec=elapsed, device=device, phase="post_eval")

    summary = {
        "checkpoint": str(ckpt_path),
        "eval_rows": len(eval_rows),
        "metrics": metrics,
        "meta": meta,
        "partition_mode": part_meta.get("partition_mode"),
        "environment": env,
        "flower_settings": flower_settings,
        **hw,
    }
    log.log(summary)
    log.save_summary(summary)
    print("Global model eval:", metrics)


if __name__ == "__main__":
    main()
