"""GPU / Adapter リソース計測（全ステップ共通）。

教授指定 4 リンク対応:
  1. PyTorch CUDA semantics (memory_allocated / reserved / peak / active)
  2. torch.profiler (fit 1 step あたり op 時間・CUDA memory)
  3. nvidia-smi (utilization, memory, power, temperature, clocks)
  4. torch.cuda.memory_stats (activation / segment / oom 内訳)

確認: python -m src.resource_metrics
"""
from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

_MEMORY_STATS_KEYS = (
    "allocated_bytes.all.current",
    "allocated_bytes.all.peak",
    "reserved_bytes.all.current",
    "reserved_bytes.all.peak",
    "active_bytes.all.current",
    "active_bytes.all.peak",
    "inactive_split_bytes.all.current",
    "inactive_split_bytes.all.peak",
    "requested_bytes.all.current",
    "requested_bytes.all.peak",
    "allocation.all.current",
    "segment.all.current",
    "num_alloc_retries",
    "num_ooms",
    "num_sync_all_streams",
    "num_device_alloc",
    "num_device_free",
)

_NVIDIA_SMI_FIELDS = (
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.free",
    "memory.total",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.max.sm",
    "clocks.max.memory",
    "driver_version",
    "cuda_version",
)


def _bytes_to_gb(n: int | float) -> float:
    return float(n) / (1024**3)


def get_memory_allocated_gb(device: torch.device | None = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    idx = device.index if device is not None and device.index is not None else None
    return _bytes_to_gb(torch.cuda.memory_allocated(idx))


def get_memory_reserved_gb(device: torch.device | None = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    idx = device.index if device is not None and device.index is not None else None
    return _bytes_to_gb(torch.cuda.memory_reserved(idx))


def get_max_memory_allocated_gb(device: torch.device | None = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    idx = device.index if device is not None and device.index is not None else None
    return _bytes_to_gb(torch.cuda.max_memory_allocated(idx))


def get_max_memory_reserved_gb(device: torch.device | None = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    idx = device.index if device is not None and device.index is not None else None
    return _bytes_to_gb(torch.cuda.max_memory_reserved(idx))


get_max_vram_gb = get_max_memory_allocated_gb


def reset_peak_memory_stats(device: torch.device | None = None) -> None:
    if torch.cuda.is_available():
        idx = device.index if device is not None and device.index is not None else None
        torch.cuda.reset_peak_memory_stats(idx)


def _nvidia_smi_query(device_index: int, fields: str) -> List[str]:
    try:
        res = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
                f"--id={device_index}",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        return [x.strip() for x in res.strip().split("\n") if x.strip()]
    except Exception:
        return []


def _parse_smi_float(val: str) -> Optional[float]:
    try:
        return float(val.replace(" W", "").replace(" %", "").strip())
    except (TypeError, ValueError):
        return None


def build_nvidia_smi_snapshot(device_index: int = 0) -> Dict[str, Any]:
    """Link 3: nvidia-smi 全項目（フィールドごとに取得して Windows CSV 崩れを回避）。"""
    out: Dict[str, Any] = {"device_index": device_index, "available": False}
    field_map = {
        "utilization.gpu": ("smi_utilization_gpu_percent", "percent"),
        "utilization.memory": ("smi_utilization_memory_percent", "percent"),
        "memory.used": ("smi_memory_used_gb", "mem_gb"),
        "memory.free": ("smi_memory_free_gb", "mem_gb"),
        "memory.total": ("smi_memory_total_gb", "mem_gb"),
        "power.draw": ("smi_power_draw_w", "float"),
        "power.limit": ("smi_power_limit_w", "float"),
        "temperature.gpu": ("smi_temperature_gpu_c", "float"),
        "clocks.current.sm": ("smi_clocks_current_sm_mhz", "float"),
        "clocks.current.memory": ("smi_clocks_current_memory_mhz", "float"),
        "clocks.max.sm": ("smi_clocks_max_sm_mhz", "float"),
        "clocks.max.memory": ("smi_clocks_max_memory_mhz", "float"),
        "driver_version": ("driver_version", "str"),
        "cuda_version": ("cuda_version", "str"),
    }
    got_any = False
    for smi_field, (out_key, kind) in field_map.items():
        vals = _nvidia_smi_query(device_index, smi_field)
        if not vals:
            continue
        raw = vals[0]
        got_any = True
        if kind == "str":
            out[out_key] = raw
        elif kind == "mem_gb":
            mb = _parse_smi_float(raw)
            out[out_key] = (mb / 1024.0) if mb is not None else None
        elif kind == "percent":
            out[out_key] = _parse_smi_float(raw)
        else:
            out[out_key] = _parse_smi_float(raw)
    out["available"] = got_any
    return out


def get_gpu_utilization(device_index: int = 0) -> float:
    vals = _nvidia_smi_query(device_index, "utilization.gpu")
    return float(vals[0]) if vals else 0.0


def get_nvidia_smi_used_memory_gb(device_index: int = 0) -> Optional[float]:
    vals = _nvidia_smi_query(device_index, "memory.used")
    mb = _parse_smi_float(vals[0]) if vals else None
    return (mb / 1024.0) if mb is not None else None


def get_nvidia_smi_total_memory_gb(device_index: int = 0) -> Optional[float]:
    vals = _nvidia_smi_query(device_index, "memory.total")
    mb = _parse_smi_float(vals[0]) if vals else None
    return (mb / 1024.0) if mb is not None else None


def build_pytorch_cuda_semantics(device: torch.device | None = None) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    idx = device.index if device is not None and device.index is not None else torch.cuda.current_device()
    dev = device or torch.device(f"cuda:{idx}")
    props = torch.cuda.get_device_properties(idx)
    return {
        "cuda_available": True,
        "device_index": idx,
        "device_name": props.name,
        "memory_allocated_gb": get_memory_allocated_gb(dev),
        "memory_reserved_gb": get_memory_reserved_gb(dev),
        "max_memory_allocated_gb": get_max_memory_allocated_gb(dev),
        "max_memory_reserved_gb": get_max_memory_reserved_gb(dev),
        "total_memory_gb": _bytes_to_gb(props.total_memory),
        "multi_processor_count": props.multi_processor_count,
        "cuda_capability": f"{props.major}.{props.minor}",
    }


def build_pytorch_memory_stats(device: torch.device | None = None) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    idx = device.index if device is not None and device.index is not None else torch.cuda.current_device()
    stats = torch.cuda.memory_stats(idx)
    raw = {k: int(stats[k]) for k in _MEMORY_STATS_KEYS if k in stats}
    gb = {k.replace("_bytes", "_gb").replace(".", "_"): _bytes_to_gb(v) for k, v in raw.items() if "bytes" in k}
    return {"memory_stats_raw": raw, "memory_stats_gb": gb}


class GpuUtilTracker:
    def __init__(self, *, device_index: int = 0, sample_every_steps: int = 1) -> None:
        self.device_index = device_index
        self.sample_every_steps = max(1, sample_every_steps)
        self._step = 0
        self.gpu_util_samples: List[float] = []
        self.mem_util_samples: List[float] = []
        self.mem_used_gb_samples: List[float] = []

    def maybe_sample(self) -> None:
        self._step += 1
        if self._step % self.sample_every_steps != 0:
            return
        snap = build_nvidia_smi_snapshot(self.device_index)
        if not snap.get("available"):
            return
        gu = snap.get("smi_utilization_gpu_percent")
        mu = snap.get("smi_utilization_memory_percent")
        mem = snap.get("smi_memory_used_gb")
        if gu is not None:
            self.gpu_util_samples.append(float(gu))
        if mu is not None:
            self.mem_util_samples.append(float(mu))
        if mem is not None:
            self.mem_used_gb_samples.append(float(mem))

    def _agg(self, xs: List[float]) -> Dict[str, Optional[float]]:
        if not xs:
            return {"min": None, "max": None, "avg": None, "n": 0}
        return {"min": float(min(xs)), "max": float(max(xs)), "avg": float(sum(xs) / len(xs)), "n": len(xs)}

    def summary(self) -> Dict[str, Any]:
        return {
            "gpu_util_percent": self._agg(self.gpu_util_samples),
            "memory_util_percent": self._agg(self.mem_util_samples),
            "nvidia_smi_used_memory_gb": self._agg(self.mem_used_gb_samples),
        }

    def average(self) -> float:
        if self.gpu_util_samples:
            return float(sum(self.gpu_util_samples) / len(self.gpu_util_samples))
        return get_gpu_utilization(self.device_index)


def profile_one_step(step_fn: Callable[[], None], device: torch.device | None = None) -> Dict[str, Any]:
    if not torch.cuda.is_available() or device is None or device.type != "cuda":
        return {"profiler_available": False, "reason": "cuda_not_used"}
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        return {"profiler_available": False, "reason": "profiler_import_failed"}

    out: Dict[str, Any] = {"profiler_available": True}
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            step_fn()
        cuda_us = 0.0
        cpu_us = 0.0
        top_ops: List[Dict[str, Any]] = []
        for evt in prof.key_averages():
            cuda_t = getattr(evt, "cuda_time_total", None) or getattr(evt, "device_time_total", 0)
            cpu_t = getattr(evt, "cpu_time_total", 0) or getattr(evt, "self_cpu_time_total", 0)
            cuda_us += int(cuda_t)
            cpu_us += int(cpu_t)
        for evt in prof.key_averages()[:15]:
            cuda_t = getattr(evt, "cuda_time_total", None) or getattr(evt, "device_time_total", 0)
            cpu_t = getattr(evt, "cpu_time_total", 0) or getattr(evt, "self_cpu_time_total", 0)
            mem = getattr(evt, "cuda_memory_usage", 0) or getattr(evt, "device_memory_usage", 0)
            top_ops.append(
                {
                    "name": evt.key,
                    "cuda_time_ms": round(int(cuda_t) / 1000.0, 3),
                    "cpu_time_ms": round(int(cpu_t) / 1000.0, 3),
                    "cuda_memory_mb": round(int(mem) / (1024**2), 3) if mem else 0.0,
                    "count": evt.count,
                }
            )
        out.update(
            {
                "cuda_time_total_ms": round(cuda_us / 1000.0, 3),
                "cpu_time_total_ms": round(cpu_us / 1000.0, 3),
                "top_ops_by_cuda_time": top_ops,
            }
        )
    except Exception as exc:
        out["profiler_available"] = False
        out["reason"] = str(exc)
    return out


def build_professor_links_record(
    *,
    device: torch.device | None = None,
    device_index: int = 0,
    gpu_tracker: GpuUtilTracker | None = None,
    profiler: Mapping[str, Any] | None = None,
    phase: str = "snapshot",
) -> Dict[str, Any]:
    idx = device_index
    if device is not None and device.type == "cuda" and device.index is not None:
        idx = device.index
    rec: Dict[str, Any] = {
        "phase": phase,
        "link1_pytorch_cuda_semantics": build_pytorch_cuda_semantics(device),
        "link3_nvidia_smi_snapshot": build_nvidia_smi_snapshot(idx),
        "link4_pytorch_memory_stats": build_pytorch_memory_stats(device),
    }
    if gpu_tracker is not None:
        rec["link3_nvidia_smi_timeseries"] = gpu_tracker.summary()
    if profiler is not None:
        rec["link2_torch_profiler"] = dict(profiler)
    return rec


def quantization_info(cfg: Dict[str, Any]) -> Dict[str, Any]:
    mcfg = cfg.get("model", {})
    tcfg = cfg.get("train", {})
    load_4bit = bool(mcfg.get("load_in_4bit", False))
    load_8bit = bool(mcfg.get("load_in_8bit", False))
    amp = bool(tcfg.get("use_amp", True))
    return {
        "quantization_enabled": load_4bit or load_8bit,
        "load_in_4bit": load_4bit,
        "load_in_8bit": load_8bit,
        "amp_enabled": amp,
        "amp_dtype": "bfloat16" if amp else "float32",
        "gradient_checkpointing": bool(tcfg.get("use_gradient_checkpointing", False)),
    }


def adapter_size_mb(lora_params: List[nn.Parameter]) -> float:
    return sum(p.numel() * p.element_size() for p in lora_params) / (1024**2)


def trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_environment_block(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    block: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "nvidia_smi_total_memory_gb": get_nvidia_smi_total_memory_gb(0),
    }
    if torch.cuda.is_available():
        block["professor_links_baseline"] = build_professor_links_record(
            device=torch.device("cuda:0"), device_index=0, phase="run_start"
        )
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
                **quantization_info(cfg),
            }
        )
    return block


def build_training_hardware_snapshot(
    *,
    elapsed_sec: float,
    steps: int = 1,
    gpu_tracker: GpuUtilTracker | None = None,
    device: torch.device | None = None,
    device_index: int = 0,
    eval_time_sec: float | None = None,
    profiler: Mapping[str, Any] | None = None,
    phase: str = "fit",
) -> Dict[str, Any]:
    idx = device_index
    if device is not None and device.type == "cuda" and device.index is not None:
        idx = device.index
    snap: Dict[str, Any] = {
        "phase": phase,
        "train_time_sec": float(elapsed_sec),
        "elapsed_sec": float(elapsed_sec),
        "avg_step_time_sec": float(elapsed_sec / max(1, steps)),
        "steps": int(steps),
        "memory_allocated_gb": get_memory_allocated_gb(device),
        "memory_reserved_gb": get_memory_reserved_gb(device),
        "max_memory_allocated_gb": get_max_memory_allocated_gb(device),
        "max_memory_reserved_gb": get_max_memory_reserved_gb(device),
        "avg_gpu_util_percent": gpu_tracker.average() if gpu_tracker else get_gpu_utilization(idx),
        "nvidia_smi_used_memory_gb": get_nvidia_smi_used_memory_gb(idx),
    }
    if eval_time_sec is not None:
        snap["eval_time_sec"] = float(eval_time_sec)
    snap["professor_links"] = build_professor_links_record(
        device=device,
        device_index=idx,
        gpu_tracker=gpu_tracker,
        profiler=profiler,
        phase=phase,
    )
    if gpu_tracker is not None:
        snap["nvidia_smi_timeseries"] = gpu_tracker.summary()
    return snap


def print_gpu_diagnostics(device_index: int = 0) -> None:
    print("=== GPU diagnostics (professor links) ===")
    for k, v in build_environment_block().items():
        if k != "professor_links_baseline":
            print(f"  {k}: {v}")
    rec = build_professor_links_record(
        device=torch.device(f"cuda:{device_index}") if torch.cuda.is_available() else None,
        device_index=device_index,
    )
    for section, data in rec.items():
        print(f"  [{section}]")
        if isinstance(data, dict):
            for kk, vv in data.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"    {data}")


if __name__ == "__main__":
    print_gpu_diagnostics()
