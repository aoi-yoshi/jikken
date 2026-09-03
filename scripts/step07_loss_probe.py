"""
Step 7: 1 サンプルで Lpred / Lgrad / Lpost を同一バッチ上で計算できることを確認する診断スクリプト。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.amp import autocast

from src.adaptation_losses import AdaptationLosses
from src.config_loader import merged_config
from src.dataset_manifest import load_sample_row, read_manifest_filtered, stratified_subset
from src.metrics import set_seed
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.run_context import init_run
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-idx", type=int, default=0)
    args = ap.parse_args()

    cfg = merged_config()
    set_seed(int(cfg["train"]["seed"]))
    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    ccfg = cfg["consistency"]

    run_id, run_dir, log, env = init_run(
        cfg,
        step="7-probe",
        step_dir="step07_loss_probe",
        log_name="probe",
        cli=vars(args),
    )

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    rows = stratified_subset(rows, int(tcfg.get("max_train_samples", 200)), int(cfg["train"]["seed"]))
    idx = args.sample_idx % len(rows)
    row = rows[idx]

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")

    client_lora = lora_params_for_adapter(model.backbone, "client")
    surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")

    losses = AdaptationLosses(
        model,
        processor,
        w_pred=float(ccfg.get("w_pred", 0.5)),
        w_grad=float(ccfg.get("w_grad", 0.1)),
        w_post=float(ccfg.get("w_post", 0.5)),
        inner_lr=float(ccfg.get("inner_lr", 1e-4)),
        max_length=int(tcfg.get("max_length", 256)),
        prompt=str(dcfg.get("image_prompt", "")),
        client_adapter="client",
        surrogate_adapter="surrogate",
        client_lora_params=client_lora,
        surrogate_lora_params=surrogate_lora,
        temperature=float(ccfg.get("temperature", 2.0)),
    )

    sample = load_sample_row(row, dcfg)
    imgs = sample.frames
    y = torch.tensor([sample.label], device=device, dtype=torch.long)

    results = {}
    model.train()
    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        for mode in ("lpred", "lpred_grad", "lpred_post"):
            total, stats, extra = losses.compute(mode, imgs, y)
            row_out = dict(stats)
            if extra:
                row_out.update(extra)
            row_out["finite"] = bool(torch.isfinite(total).item())
            results[mode] = row_out
            log.log({"mode": mode, **row_out})

    out_path = run_dir / "loss_probe.json"
    payload = {
        "run_id": run_id,
        "sample_id": sample.sample_id,
        "sample_label": sample.label,
        "environment": env,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.save_summary(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
