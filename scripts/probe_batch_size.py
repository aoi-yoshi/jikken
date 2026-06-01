"""
GPU 上で batch_size の上限を試行し、推奨値を表示する。

例:
  python scripts/probe_batch_size.py
  python scripts/probe_batch_size.py --candidates 1 2 4 8
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from src.config_loader import merged_config
from src.dataset_manifest import load_sample_row, read_manifest_filtered
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.train_common import device_or_auto, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def try_batch_size(
    *,
    model,
    processor,
    rows,
    dcfg,
    prompt: str,
    max_length: int,
    device: torch.device,
    opt,
    batch_size: int,
) -> tuple[bool, float | None, str | None]:
    import torch

    torch.cuda.empty_cache()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        samples = load_samples(rows[:batch_size], dcfg)
        if len(samples) < batch_size:
            samples = [load_sample_row(rows[i % len(rows)], dcfg) for i in range(batch_size)]
        frames_batch = [s.frames for s in samples]
        y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        model.backbone.set_adapter("client")
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            logits, _ = model.forward_samples(
                processor,
                frames_batch,
                prompt,
                max_length,
                output_hidden_states=True,
            )
            loss = logits_loss(logits.float(), y)
        loss.backward()
        opt.step()
        peak_gb = None
        if device.type == "cuda":
            peak_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
        return True, peak_gb, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe max train batch_size on GPU")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 6, 8],
        help="batch sizes to try in ascending order",
    )
    ap.add_argument("--write-config", action="store_true", help="write recommended batch_size to artifacts")
    args = ap.parse_args()

    cfg = merged_config()
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    device = torch.device(device_or_auto(True))
    print(f"device: {device}")
    if device.type != "cuda":
        print("CUDA not available. Run this script on a GPU machine for meaningful results.")
        print(f"Current config batch_size={tcfg.get('batch_size', 1)}")

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    if not rows:
        raise RuntimeError("manifest is empty")

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    client_lora = lora_params_for_adapter(model.backbone, "client")
    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))

    results = []
    max_ok = 1
    for bs in sorted(set(max(1, b) for b in args.candidates)):
        ok, peak_gb, err = try_batch_size(
            model=model,
            processor=processor,
            rows=rows,
            dcfg=dcfg,
            prompt=prompt,
            max_length=max_length,
            device=device,
            opt=opt,
            batch_size=bs,
        )
        row = {"batch_size": bs, "ok": ok, "peak_gb": peak_gb, "error": err}
        results.append(row)
        if ok:
            max_ok = bs
            msg = f"batch_size={bs}: OK"
            if peak_gb is not None:
                msg += f"  peak_VRAM={peak_gb:.2f} GB"
            print(msg)
        else:
            print(f"batch_size={bs}: FAIL  {err}")
            break

    print(f"\nRecommended batch_size: {max_ok}")
    print("Set in config/default.yaml under train.batch_size")

    out_dir = Path(cfg["artifacts"]["runs"]) / "batch_size_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "recommended_batch_size": max_ok,
        "device": str(device),
        "use_num_frames": int(dcfg.get("use_num_frames") or 0),
        "results": results,
    }
    out_path = out_dir / "probe_result.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
