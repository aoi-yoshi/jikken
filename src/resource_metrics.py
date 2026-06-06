"""GPU / Adapter リソース計測（全ステップ共通スキーマ）。"""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List

import torch
import torch.nn as nn


def get_gpu_utilization(device_index: int = 0) -> float:
    try:
        res = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={device_index}",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        utils = [float(x.strip()) for x in res.strip().split("\n") if x.strip()]
        return utils[0] if utils else 0.0
    except Exception:
        return 0.0


def get_max_vram_gb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    return 0.0


def reset_peak_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def adapter_size_mb(lora_params: List[nn.Parameter]) -> float:
    return sum(p.numel() * p.element_size() for p in lora_params) / (1024**2)


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_environment_block(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    if cfg:
        dcfg = cfg.get("data", {})
        tcfg = cfg.get("train", {})
        lora = cfg.get("lora", {})
        block.update(
            {
                "batch_size": int(tcfg.get("batch_size", 1)),
                "num_frames": int(dcfg.get("use_num_frames") or dcfg.get("num_frames", 0)),
                "lora_r": int(lora.get("r", 0)),
                "lora_target_modules": list(lora.get("target_modules", [])),
            }
        )
    return block


def snapshot_hardware(*, elapsed_sec: float, steps: int = 1) -> Dict[str, Any]:
    return {
        "elapsed_sec": float(elapsed_sec),
        "avg_step_time_sec": float(elapsed_sec / max(1, steps)),
        "max_vram_gb": get_max_vram_gb(),
        "avg_gpu_util_percent": get_gpu_utilization(),
    }
