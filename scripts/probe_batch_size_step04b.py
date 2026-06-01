"""
Step 4b と同条件（Qwen2.5-VL + LoRA client + 4フレーム）で batch_size 上限を測定。
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
from src.dataset_manifest import read_manifest_filtered
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.train_common import device_or_auto, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def try_bs(
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
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        samples = load_samples(rows[:batch_size], dcfg)
        if len(samples) < batch_size:
            from src.dataset_manifest import load_sample_row

            samples = [load_sample_row(rows[i % len(rows)], dcfg) for i in range(batch_size)]
        frames_batch = [s.frames for s in samples]
        y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        model.backbone.set_adapter("client")
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=True,
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
        peak = float(torch.cuda.max_memory_allocated() / (1024**3))
        return True, peak, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe batch_size for step04b")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--candidates", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    cfg = merged_config()
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    device = torch.device(device_or_auto(True))
    print(f"device: {device}")
    if device.type != "cuda":
        print("CUDA not available — install PyTorch cu128 nightly for RTX 50xx.")
        raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"compute capability: sm_{cap[0]}{cap[1]}")

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    print(f"manifest rows (Clear): {len(rows)}, use_num_frames={dcfg.get('use_num_frames')}")

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    client_lora = lora_params_for_adapter(model.backbone, "client")
    opt = torch.optim.AdamW(
        list(model.classifier.parameters()) + client_lora,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))

    results = []
    max_ok = 1
    for bs in sorted(set(max(1, b) for b in args.candidates)):
        ok, peak_gb, err = try_bs(
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
        results.append({"batch_size": bs, "ok": ok, "peak_gb": peak_gb, "error": err})
        if ok:
            max_ok = bs
            print(f"batch_size={bs}: OK  peak_VRAM={peak_gb:.2f} GB")
        else:
            print(f"batch_size={bs}: FAIL  {err}")
            break

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\nGPU VRAM total: {total_gb:.1f} GB")
    print(f"Recommended for step04b (config train.batch_size): {max_ok}")

    out = Path(cfg["artifacts"]["runs"]) / "step04b_batch_probe" / "probe_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "step": "4b",
                "recommended_batch_size": max_ok,
                "gpu": torch.cuda.get_device_name(0),
                "vram_total_gb": total_gb,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
