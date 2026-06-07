"""
Step 4b: 複数 run を重ねて比較（Loss / ||Δθ|| / Accuracy / KL）。

使用例:
  python scripts/plot_step04b_compare_runs.py \\
    --run-dir artifacts/runs/step04b_diff_probe/run_20260528_035028 \\
    --run-dir artifacts/runs/step04b_diff_probe/run_20260528_145742 \\
    --run-dir artifacts/runs/step04b_diff_probe/run_20260601_185826
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

import matplotlib.pyplot as plt
import yaml

_hist_mod_path = _ROOT / "scripts" / "plot_step04b_epoch_history.py"
_spec = importlib.util.spec_from_file_location("plot_step04b_epoch_history", _hist_mod_path)
assert _spec and _spec.loader
_ph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ph)

_epochs = _ph._epochs
_eval_series = _ph._eval_series
_probe_series = _ph._probe_series
_setup_japanese_font = _ph._setup_japanese_font
_train_loss_series = _ph._train_loss_series
load_epoch_history = _ph.load_epoch_history

_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

# スライド・印刷向け（凡例・軸・タイトル）
FS_SUPTITLE = 28
FS_PANEL_TITLE = 26
FS_AXIS_LABEL = 26
FS_TICK = 25
FS_LEGEND = 28


def _resolve_run_dir(path: Path) -> Path:
    p = path if path.is_absolute() else _ROOT / path
    if not p.is_dir():
        raise FileNotFoundError(f"Run directory not found: {p}")
    hist = p / "epoch_history.jsonl"
    if not hist.is_file():
        raise FileNotFoundError(f"Not found: {hist}")
    return p


def _short_target_modules(modules: List[str]) -> str:
    """q_proj, v_proj -> qv; 4 modules -> qkvo."""
    keys = [m.replace("_proj", "") for m in modules]
    if keys == ["q", "k", "v", "o"]:
        return "qkvo"
    return "".join(keys)


def _lora_label(run_dir: Path) -> str:
    """used_config.yaml の LoRA 設定から凡例用ラベルを作る。"""
    cfg_path = run_dir / "used_config.yaml"
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        lora = cfg.get("lora", {})
        r = lora.get("r")
        mods = lora.get("target_modules")
        if r is not None and mods:
            return f"r={r}, {_short_target_modules(list(mods))}"

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        ed = json.loads(summary_path.read_text(encoding="utf-8")).get("experiment_design", {})
        r = ed.get("lora_r")
        mods = ed.get("lora_target_modules")
        if r is not None and mods:
            return f"r={r}, {_short_target_modules(list(mods))}"

    return run_dir.name


def _run_label(run_dir: Path) -> str:
    return _lora_label(run_dir)


def plot_compare_overlay(
    runs: List[Tuple[str, List[Dict[str, Any]]]],
    out_path: Path,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(title, fontsize=FS_SUPTITLE)

    panels = [
        ("Train Loss", "loss", False),
        ("||Δθ_client||", "delta", False),
        ("Accuracy", "accuracy", False),
        ("KL(client ‖ surrogate)", "kl", False),
    ]

    for ax, (panel_title, metric, _) in zip(axes.flat, panels):
        for i, (label, history) in enumerate(runs):
            color = _COLORS[i % len(_COLORS)]
            kwargs = {
                "marker": "o",
                "linewidth": 2.5,
                "markersize": 10,
                "label": label,
                "color": color,
            }

            if metric == "loss":
                xs, ys = _train_loss_series(history)
                if xs:
                    ax.plot(xs, ys, **kwargs)
            elif metric == "delta":
                epochs = _epochs(history)
                ax.plot(epochs, [float(r["client_delta"]) for r in history], **kwargs)
            elif metric == "accuracy":
                epochs = _epochs(history)
                ax.plot(epochs, _eval_series(history, "accuracy"), **kwargs)
            elif metric == "kl":
                epochs = _epochs(history)
                ax.plot(
                    epochs,
                    _probe_series(history, "client_vs_surrogate_kl_teacher_a_student_b"),
                    **kwargs,
                )

            ax.set_title(panel_title, fontsize=FS_PANEL_TITLE)
            ax.set_xlabel("Epoch", fontsize=FS_AXIS_LABEL)
            ax.tick_params(axis="both", labelsize=FS_TICK)
            ax.grid(True, alpha=0.3)

        if metric == "loss":
            ax.set_ylabel("Loss", fontsize=FS_AXIS_LABEL)
        elif metric == "delta":
            ax.set_ylabel("||Δθ||", fontsize=FS_AXIS_LABEL)
        elif metric == "accuracy":
            ax.set_ylabel("Accuracy", fontsize=FS_AXIS_LABEL)
        else:
            ax.set_ylabel("KL", fontsize=FS_AXIS_LABEL)
        ax.legend(fontsize=FS_LEGEND, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _disambiguate_labels(run_dirs: List[Path], labels: List[str]) -> List[str]:
    """同一 LoRA ラベルが複数あるとき run 名サフィックスを付ける。"""
    counts: Dict[str, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    out: List[str] = []
    for d, lb in zip(run_dirs, labels):
        if counts[lb] > 1:
            out.append(f"{lb} ({d.name})")
        else:
            out.append(lb)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay-compare Step 4b runs (4 metrics)")
    ap.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory (repeat for each run, at least 2)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: step04b_diff_probe/compare_<run names joined>)",
    )
    args = ap.parse_args()

    if len(args.run_dir) < 2:
        ap.error("At least two --run-dir arguments are required.")

    run_dirs = [_resolve_run_dir(Path(p)) for p in args.run_dir]
    labels = _disambiguate_labels(run_dirs, [_run_label(d) for d in run_dirs])
    histories = [(lb, load_epoch_history(d / "epoch_history.jsonl")) for lb, d in zip(labels, run_dirs)]

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = _ROOT / out_dir
    else:
        short = "_vs_".join(d.name for d in run_dirs)
        out_dir = run_dirs[0].parent / f"compare_{short}"

    _setup_japanese_font()
    out_png = out_dir / "overview_compare_dashboard.png"
    title_labels = " · ".join(labels)
    plot_compare_overlay(
        histories,
        out_png,
        title=f"Step 4b Compare ({title_labels})",
    )
    print(f"Saved: {out_png}")
    for lb, d in zip(labels, run_dirs):
        print(f"  {lb} <- {d.name}")


if __name__ == "__main__":
    main()
