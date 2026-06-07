"""
Step 1: Qwen2.5-VL-3B をロードし、1枚の画像から短い生成を行う（推論がGPUで動くか確認）。
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.config_loader import merged_config
from src.paths import ensure_dirs
from src.run_context import init_run
from src.train_common import device_or_auto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help="Optional image path; default random noise image")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-suffix", default="")
    args = ap.parse_args()
    cfg = merged_config()
    mcfg = cfg["model"]
    art = cfg["artifacts"]
    ensure_dirs(Path(art["runs"]))

    run_id, run_dir, log, env = init_run(
        cfg, step="1", step_dir="step01_infer", log_name="events", run_id=args.run_id, run_suffix=args.run_suffix, cli=vars(args)
    )

    device = torch.device(device_or_auto(True))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mcfg["id"],
        torch_dtype=dtype,
        attn_implementation=str(mcfg.get("attn_implementation", "sdpa")),
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(mcfg["id"], trust_remote_code=True)

    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        arr = (np.random.rand(512, 512, 3) * 255).astype("uint8")
        img = Image.fromarray(arr)

    demo_path = run_dir / "demo_input.png"
    img.save(demo_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe the image briefly in Japanese."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=64)

    trimmed = out_ids[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    print("Generated:\n", text)
    log.log({"device": str(device), "dtype": str(dtype), "output_chars": len(text)})
    log.save_summary(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "environment": env,
            "demo_input": str(demo_path),
            "output": text,
        }
    )
    log.log_meta({"status": "completed"})


if __name__ == "__main__":
    main()
