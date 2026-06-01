"""
Step 4b の epoch_history.jsonl からエポック推移グラフを生成する。

使用例:
  python scripts/plot_step04b_epoch_history.py --run-dir artifacts/runs/step04b_diff_probe/run_20260528_035028
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _setup_japanese_font() -> None:
    for family in ("Meiryo", "Yu Gothic", "MS Gothic", "DejaVu Sans"):
        try:
            plt.rcParams["font.family"] = family
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load_epoch_history(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["epoch"]))
    return rows


def _epochs(history: List[Dict[str, Any]]) -> List[int]:
    return [int(r["epoch"]) for r in history]


def _probe_series(history: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(r["probe"][key]) for r in history]


def _eval_series(history: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(r["eval"][key]) for r in history]


def _train_loss_series(history: List[Dict[str, Any]]) -> tuple[List[int], List[float]]:
    xs, ys = [], []
    for r in history:
        loss = r.get("train_loss")
        if loss is not None:
            xs.append(int(r["epoch"]))
            ys.append(float(loss))
    return xs, ys


def _save_line_plot(
    out_path: Path,
    *,
    epochs: List[int],
    series: List[tuple[str, List[float], Optional[str]]],
    title: str,
    ylabel: str,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values, color in series:
        kwargs = {"marker": "o", "linewidth": 1.8, "markersize": 5, "label": label}
        if color:
            kwargs["color"] = color
        ax.plot(epochs, values, **kwargs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(history: List[Dict[str, Any]], out_dir: Path, run_id: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = _epochs(history)
    saved: List[Path] = []

    loss_x, loss_y = _train_loss_series(history)
    if loss_x:
        p = out_dir / "train_loss.png"
        _save_line_plot(
            p,
            epochs=loss_x,
            series=[("Train Loss (epoch avg)", loss_y, None)],
            title=f"Train Loss vs Epoch ({run_id})",
            ylabel="Loss",
        )
        saved.append(p)

    p = out_dir / "client_delta.png"
    _save_line_plot(
        p,
        epochs=epochs,
        series=[("||Δθ_client|| (from init)", [float(r["client_delta"]) for r in history], None)],
        title=f"LoRA Parameter Change ||Δθ|| vs Epoch ({run_id})",
        ylabel="||Δθ||",
    )
    saved.append(p)

    p = out_dir / "eval_metrics.png"
    _save_line_plot(
        p,
        epochs=epochs,
        series=[
            ("Accuracy", _eval_series(history, "accuracy"), "#1f77b4"),
            ("F1 macro", _eval_series(history, "f1_macro"), "#ff7f0e"),
        ],
        title=f"Eval Metrics vs Epoch ({run_id})",
        ylabel="Score",
    )
    saved.append(p)

    p = out_dir / "kl_client_vs_surrogate.png"
    _save_line_plot(
        p,
        epochs=epochs,
        series=[
            (
                "KL(client ‖ surrogate)",
                _probe_series(history, "client_vs_surrogate_kl_teacher_a_student_b"),
                None,
            )
        ],
        title=f"Output KL (Client vs Surrogate) vs Epoch ({run_id})",
        ylabel="KL divergence",
    )
    saved.append(p)

    p = out_dir / "output_diff.png"
    _save_line_plot(
        p,
        epochs=epochs,
        series=[
            ("Logits L2", _probe_series(history, "client_vs_surrogate_logits_l2"), "#2ca02c"),
            ("Prob MSE", _probe_series(history, "client_vs_surrogate_prob_mse"), "#d62728"),
        ],
        title=f"Client vs Surrogate Output Diff vs Epoch ({run_id})",
        ylabel="Distance",
    )
    saved.append(p)

    p = out_dir / "probe_loss_task.png"
    _save_line_plot(
        p,
        epochs=epochs,
        series=[
            ("L_task (client)", _probe_series(history, "loss_task_client"), "#1f77b4"),
            ("L_task (surrogate)", _probe_series(history, "loss_task_surrogate"), "#ff7f0e"),
        ],
        title=f"Probe Sample L_task vs Epoch ({run_id})",
        ylabel="Cross-entropy loss",
    )
    saved.append(p)

    p = out_dir / "grad_metrics.png"
    _save_line_plot(
        p,
        epochs=epochs,
        series=[
            ("||g_client||", _probe_series(history, "grad_norm_client"), "#1f77b4"),
            ("||g_surrogate||", _probe_series(history, "grad_norm_surrogate"), "#ff7f0e"),
            ("||g_c - g_s||", _probe_series(history, "grad_l2_diff"), "#2ca02c"),
        ],
        title=f"Gradient Norms vs Epoch ({run_id})",
        ylabel="Norm",
        logy=True,
    )
    saved.append(p)

    p = out_dir / "grad_cosine.png"
    _save_line_plot(
        p,
        epochs=epochs,
        series=[("cos(g_client, g_surrogate)", _probe_series(history, "grad_cosine"), None)],
        title=f"Gradient Cosine Similarity vs Epoch ({run_id})",
        ylabel="Cosine",
    )
    saved.append(p)

    p = out_dir / "lora_norm.png"
    _save_line_plot(
        p,
        epochs=epochs,
        series=[
            ("||θ_client||", _probe_series(history, "lora_norm_client"), "#1f77b4"),
            ("||θ_surrogate||", _probe_series(history, "lora_norm_surrogate"), "#ff7f0e"),
        ],
        title=f"LoRA Weight Norm vs Epoch ({run_id})",
        ylabel="L2 norm",
    )
    saved.append(p)

    # 概要ダッシュボード（主要4指標）
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Step 4b Epoch Summary ({run_id})", fontsize=14)

    ax = axes[0, 0]
    if loss_x:
        ax.plot(loss_x, loss_y, "o-", color="#1f77b4")
    ax.set_title("Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, [float(r["client_delta"]) for r in history], "o-", color="#ff7f0e")
    ax.set_title("||Δθ_client||")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("||Δθ||")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, _eval_series(history, "accuracy"), "o-", label="Accuracy")
    ax.plot(epochs, _eval_series(history, "f1_macro"), "o-", label="F1 macro")
    ax.set_title("Eval Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(
        epochs,
        _probe_series(history, "client_vs_surrogate_kl_teacher_a_student_b"),
        "o-",
        color="#2ca02c",
    )
    ax.set_title("KL(client ‖ surrogate)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    overview = out_dir / "overview_dashboard.png"
    fig.savefig(overview, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(overview)

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Step 4b epoch history metrics")
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing epoch_history.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots (default: <run-dir>/plots)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _ROOT / run_dir

    history_path = run_dir / "epoch_history.jsonl"
    if not history_path.is_file():
        raise FileNotFoundError(f"Not found: {history_path}")

    run_id = run_dir.name
    meta_path = run_dir / "run_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = str(meta.get("run_id", run_id))

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    _setup_japanese_font()
    history = load_epoch_history(history_path)
    saved = plot_all(history, out_dir, run_id)

    print(f"Saved {len(saved)} plots to {out_dir}:")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
