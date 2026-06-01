"""実験記録: 設定スナップショットと人間向けレポート生成。"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def build_environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def save_config_snapshot(
    run_dir: Path,
    *,
    cfg: Mapping[str, Any],
    cli_args: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    payload: Dict[str, Any] = {
        "saved_at": _now_iso(),
        "environment": build_environment_info(),
        "config": dict(cfg),
        "cli_args": dict(cli_args),
    }
    if extra:
        payload["derived"] = dict(extra)
    path = run_dir / "config_snapshot.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def write_experiment_report(run_dir: Path, summary: Mapping[str, Any]) -> Path:
    """進捗会・ゼミ用の Markdown レポートを自動生成。"""
    d = summary.get("experiment_design", {})
    ld = summary.get("lora_delta", {})
    od = summary.get("output_delta_on_probe", {})
    gd = summary.get("grad_on_probe", {})
    ev = summary.get("eval", {})
    inv = summary.get("lora_inventory", {})
    interp = summary.get("interpretation", {})

    lines = [
        "# Step 4b 実験レポート（LoRA / 出力 / 勾配 差分）",
        "",
        f"- 生成時刻: {_now_iso()}",
        f"- Run ID: `{summary.get('run_id', 'n/a')}`",
        "",
        "## 1. 実験目的",
        "",
        "Ablation 前の基盤確認として、L_task のみで Client LoRA が更新され、",
        "出力・勾配に学習前後の差分が観測できることを記録する。",
        "",
        "## 2. モデル・LoRA 設計（数値一覧）",
        "",
        "| 項目 | 値 |",
        "|------|-----|",
        f"| ベースモデル | {d.get('model_id', 'n/a')} |",
        f"| 注意力実装 | {d.get('attn_implementation', 'n/a')} |",
        f"| 分類クラス数 | {d.get('num_classes', 'n/a')} |",
        f"| 分類ヘッド hidden | {d.get('classifier_hidden', 'n/a')} |",
        f"| LoRA rank (r) | {d.get('lora_r', 'n/a')} |",
        f"| LoRA alpha | {d.get('lora_alpha', 'n/a')} |",
        f"| LoRA スケール (alpha/r) | {d.get('lora_scale', 'n/a')} |",
        f"| LoRA dropout | {d.get('lora_dropout', 'n/a')} |",
        f"| LoRA target_modules | {d.get('lora_target_modules', 'n/a')} |",
        f"| LoRA bias | {d.get('lora_bias', 'n/a')} |",
        f"| Client アダプタ名 | {d.get('client_adapter', 'n/a')} |",
        f"| Surrogate アダプタ名 | {d.get('surrogate_adapter', 'n/a')} |",
        f"| LoRA テンソル数 / アダプタ | {inv.get('client', {}).get('num_tensors', 'n/a')} |",
        f"| LoRA 総パラメータ数 / アダプタ | {inv.get('client', {}).get('total_params', 'n/a'):,} |",
        "",
        "## 3. データ・学習設定",
        "",
        "| 項目 | 値 |",
        "|------|-----|",
        f"| データセット | {d.get('dataset', 'n/a')} |",
        f"| 1 サンプルあたりフレーム数 | {d.get('num_frames', 'n/a')} |",
        f"| フレーム抽出 | {d.get('frame_sampling', 'n/a')} |",
        f"| manifest 総サンプル | {d.get('manifest_total', 'n/a')} |",
        f"| train プール | {d.get('train_pool_size', 'n/a')} |",
        f"| eval サンプル数 | {ev.get('n_eval', 'n/a')} |",
        f"| プローブ sample ID | {summary.get('probe_id', 'n/a')} |",
        f"| 学習ステップ数 | {summary.get('train_steps', 'n/a')} |",
        f"| 学習に使ったユニーク sample 数 | {d.get('train_unique_samples', 'n/a')} |",
        f"| Optimizer | AdamW |",
        f"| 学習率 lr | {d.get('lr', 'n/a')} |",
        f"| weight_decay | {d.get('weight_decay', 'n/a')} |",
        f"| batch_size | {d.get('batch_size', 'n/a')} |",
        f"| max_length (tokens) | {d.get('max_length', 'n/a')} |",
        f"| min_pixels | {d.get('min_pixels', 'n/a')} |",
        f"| max_pixels | {d.get('max_pixels', 'n/a')} |",
        f"| seed | {d.get('seed', 'n/a')} |",
        f"| 混合精度 | {d.get('amp_dtype', 'n/a')} |",
        f"| gradient checkpointing | {d.get('gradient_checkpointing', 'n/a')} |",
        "",
        "## 4. 観測結果サマリ",
        "",
        "### LoRA",
        "",
        f"- `||θ_client||` before: **{ld.get('client_norm_before', 0):.6e}**",
        f"- `||θ_client||` after: **{ld.get('client_norm_after', 0):.6e}**",
        f"- `||Δθ_client||`: **{ld.get('client_delta_norm', 0):.6e}**",
        f"- Surrogate 未更新: **{ld.get('surrogate_unchanged', 'n/a')}**",
        "",
        "### 出力（プローブ 1 サンプル）",
        "",
        f"- Client logits L2 (before→after): **{od.get('logits_l2_before_after_client', 0):.6e}**",
        f"- KL(client‖surrogate) before: **{od.get('kl_client_surrogate_before', 0):.6e}**",
        f"- KL(client‖surrogate) after: **{od.get('kl_client_surrogate_after', 0):.6e}**",
        f"- 予測 class before→after: **{od.get('client_pred_before')} → {od.get('client_pred_after')}**",
        "",
        "### 勾配（L_task, LoRA パラメータ全体のベクトル）",
        "",
        f"- `||g_client||` before: **{gd.get('grad_norm_client_before', 0):.6e}**",
        f"- `||g_client||` after: **{gd.get('grad_norm_client_after', 0):.6e}**",
        f"- `||g_c - g_s||` before: **{gd.get('grad_l2_diff_before', 0):.6e}**",
        f"- `||g_c - g_s||` after: **{gd.get('grad_l2_diff_after', 0):.6e}**",
        f"- cosine(g_c, g_s) before→after: **{gd.get('grad_cosine_before', 0):.4f} → {gd.get('grad_cosine_after', 0):.4f}**",
        "",
        "### 精度（eval セット）",
        "",
        f"- accuracy: **{ev.get('accuracy_before', 'n/a')} → {ev.get('accuracy_after', 'n/a')}**",
        f"- F1 macro: **{ev.get('f1_before', 'n/a')} → {ev.get('f1_after', 'n/a')}**",
        "",
        "## 5. 解釈チェックリスト",
        "",
        f"- LoRA が更新された: **{interp.get('lora_adapted', 'n/a')}**",
        f"- 出力が変化した: **{interp.get('output_changed', 'n/a')}**",
        f"- Client が Surrogate から離れた: **{interp.get('client_surrogate_diverged_after_train', 'n/a')}**",
        f"- 勾配差が増加: **{interp.get('grad_diff_increased', 'n/a')}**",
        "",
        "## 6. 出力ファイル",
        "",
        "- `config_snapshot.json` — 全設定・環境情報",
        "- `probe.jsonl` / `probe.csv` — ステップごとのログ",
        "- `train_steps.jsonl` — 各学習 step の sample ID / loss",
        "- `summary.json` — 数値サマリ（機械可読）",
        "- `experiment_report.md` — 本ファイル",
        "",
    ]
    path = run_dir / "experiment_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
