"""
Step 5 FL run の jsonl から進捗会・教授報告用グラフを生成する。

使用例:
  python scripts/plot_step05_fl_run.py --run-dir artifacts/runs/step05_fl/run_20260607_200402
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np


def _setup_japanese_font() -> None:
    for family in ("Meiryo", "Yu Gothic", "MS Gothic", "DejaVu Sans"):
        try:
            plt.rcParams["font.family"] = family
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _round_metrics(run_dir: Path) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """client_id -> server_round -> list of metric dicts (usually one fit row per round)."""
    out: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
    for cid in (0, 1):
        path = run_dir / f"client_{cid}" / "fl_client.jsonl"
        out[cid] = {}
        for row in _load_jsonl(path):
            if row.get("event") != "fl_round_metrics":
                continue
            rnd = int(row["server_round"])
            out[cid].setdefault(rnd, []).append(row)
    return out


def _post_eval_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    for row in reversed(_load_jsonl(run_dir / "fl_post_eval.jsonl")):
        if "metrics" in row:
            return row
    return None


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_post_eval(run_dir: Path, out_dir: Path) -> Optional[Path]:
    row = _post_eval_summary(run_dir)
    if not row:
        return None
    m = row["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"post_eval ({row.get('meta', {}).get('run_id', run_dir.name)})", fontsize=13)

    axes[0].bar(["accuracy", "F1 macro"], [m["accuracy"], m["f1_macro"]], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("分類精度")
    for i, v in enumerate([m["accuracy"], m["f1_macro"]]):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    mem_keys = ("memory_allocated_gb", "max_memory_allocated_gb", "nvidia_smi_used_memory_gb")
    mem_vals = [float(row.get(k, 0)) for k in mem_keys]
    axes[1].bar(["alloc", "peak alloc", "smi used"], mem_vals, color="#2ca02c")
    axes[1].set_ylabel("GB")
    axes[1].set_title("VRAM（post_eval 1回）")

    axes[2].bar(
        ["elapsed_sec", "avg_gpu_util%"],
        [float(row.get("elapsed_sec", 0)), float(row.get("avg_gpu_util_percent", 0))],
        color=["#9467bd", "#d62728"],
    )
    axes[2].set_title("時間・GPU利用率")
    axes[2].text(1, float(row.get("avg_gpu_util_percent", 0)) + 1, f"{row.get('eval_rows', '?')} samples", ha="center")

    fig.tight_layout()
    return _save(fig, out_dir, "01_post_eval_summary.png")


def plot_round_accuracies(
    rm: Dict[int, Dict[int, List[Dict[str, Any]]]], run_dir: Path, out_dir: Path, run_name: str
) -> Path:
    rounds = sorted({r for per in rm.values() for r in per})
    metrics = (
        ("post_redistribution_accuracy", "post_redist"),
        ("local_adaptation_accuracy", "local_adapt"),
        ("foundation_generalization_accuracy", "foundation"),
        ("eval_accuracy", "flower_eval"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    fig.suptitle(f"ラウンド別 accuracy ({run_name})", fontsize=13)
    colors = {0: "#1f77b4", 1: "#ff7f0e"}

    for ax, (key, title) in zip(axes.flat, metrics):
        for cid in sorted(rm):
            ys: List[float] = []
            for rnd in rounds:
                rows = rm[cid].get(rnd, [])
                if key == "eval_accuracy":
                    # evaluate event is separate; use last fl_round row + evaluate from jsonl
                    ys.append(float("nan"))
                else:
                    val = rows[0].get(key) if rows else None
                    ys.append(float(val) if val is not None else float("nan"))
            if key != "eval_accuracy":
                ax.plot(rounds, ys, marker="o", label=f"client {cid}", color=colors[cid])
        if key == "eval_accuracy":
            for cid in sorted(rm):
                eval_rows = _load_jsonl(run_dir / f"client_{cid}" / "fl_client.jsonl")
                by_r = {
                    int(r["server_round"]): float(r["eval_accuracy"])
                    for r in eval_rows
                    if r.get("event") == "evaluate"
                }
                ax.plot(rounds, [by_r.get(r, float("nan")) for r in rounds], marker="s", label=f"client {cid}", color=colors[cid])
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, out_dir, "02_round_accuracies.png")


def plot_train_loss_and_timing(rm: Dict[int, Dict[int, List[Dict[str, Any]]]], out_dir: Path, run_name: str) -> Path:
    rounds = sorted({r for per in rm.values() for r in per})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"学習 loss・時間 ({run_name})", fontsize=13)
    colors = {0: "#1f77b4", 1: "#ff7f0e"}

    for cid in sorted(rm):
        loss_y, fit_y, round_y = [], [], []
        for rnd in rounds:
            row = rm[cid].get(rnd, [{}])[0]
            loss_y.append(float(row.get("avg_train_loss", float("nan"))))
            rt = row.get("round_timing") or {}
            fit_y.append(float(rt.get("fit_duration_sec", float("nan"))) / 3600.0)
            round_y.append(float(rt.get("round_duration_sec", float("nan"))) / 3600.0)
        ax1.plot(rounds, loss_y, marker="o", label=f"client {cid}", color=colors[cid])
        ax2.plot(rounds, fit_y, marker="o", label=f"client {cid} fit", color=colors[cid])
        ax2.plot(rounds, round_y, linestyle="--", marker="o", label=f"client {cid} round", color=colors[cid], alpha=0.7)

    ax1.set_xlabel("server round")
    ax1.set_ylabel("avg_train_loss")
    ax1.set_title("ローカル CE loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("server round")
    ax2.set_ylabel("hours")
    ax2.set_title("fit / round 所要時間")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, out_dir, "03_train_loss_and_timing.png")


def plot_nvidia_smi_summary(rm: Dict[int, Dict[int, List[Dict[str, Any]]]], out_dir: Path, run_name: str) -> Path:
    """jsonl に保存されている smi 集計（min/avg/max）をラウンド別に可視化。"""
    rounds = sorted({r for per in rm.values() for r in per})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"nvidia-smi 集計 per fit ({run_name})\n※生の時系列は未保存（avg/min/max のみ）", fontsize=11)

    titles = ("GPU util %", "memory util %", "VRAM used GB")
    keys = ("gpu_util_percent", "memory_util_percent", "nvidia_smi_used_memory_gb")
    width = 0.35
    x = np.arange(len(rounds))

    for ax, title, key in zip(axes, titles, keys):
        for i, cid in enumerate(sorted(rm)):
            avg_vals, min_vals, max_vals = [], [], []
            for rnd in rounds:
                row = rm[cid].get(rnd, [{}])[0]
                ts = row.get("nvidia_smi_timeseries") or row.get("professor_links", {}).get(
                    "link3_nvidia_smi_timeseries", {}
                )
                block = ts.get(key, {})
                avg_vals.append(float(block.get("avg", float("nan"))))
                min_vals.append(float(block.get("min", float("nan"))))
                max_vals.append(float(block.get("max", float("nan"))))
            offset = (i - 0.5) * width
            ax.bar(x + offset, avg_vals, width=width, label=f"client {cid} avg", alpha=0.85)
            ax.errorbar(
                x + offset,
                avg_vals,
                yerr=[np.array(avg_vals) - np.array(min_vals), np.array(max_vals) - np.array(avg_vals)],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{r}" for r in rounds])
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return _save(fig, out_dir, "04_nvidia_smi_summary.png")


def plot_profiler_cpu_ops(rm: Dict[int, Dict[int, List[Dict[str, Any]]]], out_dir: Path, run_name: str, round_no: int = 1) -> Optional[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Link2 torch.profiler（round {round_no}・先頭1 step）\n"
        f"RTX5050: cuda_time=0 のため CPU 上位 op を表示 ({run_name})",
        fontsize=11,
    )

    any_data = False
    for ax, cid in zip(axes, sorted(rm)):
        rows = rm[cid].get(round_no, [])
        if not rows:
            ax.set_visible(False)
            continue
        pl = rows[0].get("professor_links", {}).get("link2_torch_profiler", {})
        ops = pl.get("top_ops_by_cuda_time") or []
        if not ops:
            ax.text(0.5, 0.5, "profiler データなし", ha="center", va="center")
            continue
        any_data = True
        # cuda が 0 なので cpu_time でソート
        ops = sorted(ops, key=lambda o: float(o.get("cpu_time_ms", 0)), reverse=True)[:12]
        names = [o["name"].split("#")[0][:40] for o in ops]
        cpu_ms = [float(o.get("cpu_time_ms", 0)) for o in ops]
        cuda_ms = [float(o.get("cuda_time_ms", 0)) for o in ops]
        y = np.arange(len(names))
        ax.barh(y, cpu_ms, color="#1f77b4", alpha=0.8, label="CPU ms")
        ax.barh(y, cuda_ms, left=cpu_ms, color="#ff7f0e", alpha=0.8, label="CUDA ms")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("ms")
        ax.set_title(f"client {cid}\nCPU total={pl.get('cpu_time_total_ms')}  CUDA total={pl.get('cuda_time_total_ms')}")
        ax.legend(fontsize=7)

    if not any_data:
        plt.close(fig)
        return None
    fig.tight_layout()
    return _save(fig, out_dir, "05_profiler_cpu_ops_round1.png")


def plot_memory_link1(rm: Dict[int, Dict[int, List[Dict[str, Any]]]], out_dir: Path, run_name: str) -> Path:
    rounds = sorted({r for per in rm.values() for r in per})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle(f"Link1 PyTorch VRAM peak ({run_name})", fontsize=13)
    colors = {0: "#1f77b4", 1: "#ff7f0e"}

    for cid in sorted(rm):
        peak_alloc, peak_reserved = [], []
        for rnd in rounds:
            row = rm[cid].get(rnd, [{}])[0]
            l1 = row.get("professor_links", {}).get("link1_pytorch_cuda_semantics", {})
            peak_alloc.append(float(l1.get("max_memory_allocated_gb", float("nan"))))
            peak_reserved.append(float(l1.get("max_memory_reserved_gb", float("nan"))))
        ax.plot(rounds, peak_alloc, marker="o", color=colors[cid], label=f"client {cid} peak allocated")
        ax.plot(rounds, peak_reserved, linestyle="--", marker="s", color=colors[cid], alpha=0.6, label=f"client {cid} peak reserved")

    ax.set_xlabel("server round")
    ax.set_ylabel("GB")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "06_link1_vram_peak.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="artifacts/runs/step05_fl/<run_id>")
    ap.add_argument("--out-dir", default=None, help="default: <run-dir>/plots")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _ROOT / run_dir
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    _setup_japanese_font()
    rm = _round_metrics(run_dir)
    saved: List[Path] = []

    for fn in (
        lambda: plot_post_eval(run_dir, out_dir),
        lambda: plot_round_accuracies(rm, run_dir, out_dir, run_dir.name),
        lambda: plot_train_loss_and_timing(rm, out_dir, run_dir.name),
        lambda: plot_nvidia_smi_summary(rm, out_dir, run_dir.name),
        lambda: plot_profiler_cpu_ops(rm, out_dir, run_dir.name),
        lambda: plot_memory_link1(rm, out_dir, run_dir.name),
    ):
        p = fn()
        if p:
            saved.append(p)

    print(f"Saved {len(saved)} plots to {out_dir}")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
