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
    env = summary.get("environment", {})
    hw = summary.get("hardware_profile", {})
    history = summary.get("history", [])
    ld = summary.get("lora_delta", {})
    od = summary.get("output_delta_on_probe", {})
    gd = summary.get("grad_on_probe", {})
    ev = summary.get("eval", {})
    inv = summary.get("lora_inventory", {})
    interp = summary.get("interpretation", {})

    if history and not ev:
        first = history[0]
        last = history[-1]
        ev = {
            "n_eval": last.get("eval", {}).get("n_eval"),
            "accuracy_before": first.get("eval", {}).get("accuracy"),
            "accuracy_after": last.get("eval", {}).get("accuracy"),
            "f1_before": first.get("eval", {}).get("f1_macro"),
            "f1_after": last.get("eval", {}).get("f1_macro"),
        }
        if not ld and len(history) >= 2:
            ld = {
                "client_delta_norm": last.get("client_delta", 0.0),
            }
        if not od and last.get("probe"):
            pb, pa = first.get("probe", {}), last.get("probe", {})
            od = {
                "kl_client_surrogate_before": pb.get("client_vs_surrogate_kl_teacher_a_student_b"),
                "kl_client_surrogate_after": pa.get("client_vs_surrogate_kl_teacher_a_student_b"),
                "client_pred_before": pb.get("client_pred"),
                "client_pred_after": pa.get("client_pred"),
            }

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
        f"| LoRA rank (r) | {d.get('lora_r', env.get('lora_r', 'n/a'))} |",
        f"| LoRA alpha | {d.get('lora_alpha', 'n/a')} |",
        f"| LoRA スケール (alpha/r) | {d.get('lora_scale', 'n/a')} |",
        f"| LoRA dropout | {d.get('lora_dropout', 'n/a')} |",
        f"| LoRA target_modules | {d.get('lora_target_modules', env.get('lora_target_modules', 'n/a'))} |",
        f"| LoRA bias | {d.get('lora_bias', 'n/a')} |",
        f"| 量子化 | {env.get('quantization_enabled', d.get('quantization_enabled', False))} |",
        f"| Client アダプタ名 | {d.get('client_adapter', 'n/a')} |",
        f"| Surrogate アダプタ名 | {d.get('surrogate_adapter', 'n/a')} |",
        f"| LoRA テンソル数 / アダプタ | {inv.get('client', {}).get('num_tensors', 'n/a')} |",
        f"| LoRA 総パラメータ数 / アダプタ | {inv.get('client', {}).get('total_params', 'n/a'):,} |",
        f"| Adapter サイズ [MB] | {d.get('adapter_size_mb', 'n/a')} |",
        "",
        "## 3. データ・学習設定",
        "",
        "| 項目 | 値 |",
        "|------|-----|",
        f"| データセット | {d.get('dataset', 'n/a')} |",
        f"| モデル入力フレーム数 | {d.get('use_num_frames', env.get('num_frames', 'n/a'))} |",
        f"| 抽出フレーム数（保存） | {d.get('num_frames', 'n/a')} |",
        f"| フレーム抽出 | {d.get('frame_sampling', 'n/a')} |",
        f"| manifest 総サンプル | {d.get('manifest_total', 'n/a')} |",
        f"| train プール | {d.get('train_pool_size', 'n/a')} |",
        f"| eval サンプル数 | {ev.get('n_eval', d.get('eval_max', 'n/a'))} |",
        f"| 学習に使ったユニーク sample 数 | {d.get('train_unique_samples', d.get('train_count', 'n/a'))} |",
        f"| Optimizer | AdamW |",
        f"| 学習率 lr | {d.get('lr', 'n/a')} |",
        f"| weight_decay | {d.get('weight_decay', 'n/a')} |",
        f"| batch_size | {d.get('batch_size', env.get('batch_size', 'n/a'))} |",
        f"| max_length (tokens) | {d.get('max_length', 'n/a')} |",
        f"| seed | {d.get('seed', 'n/a')} |",
        f"| 混合精度 | {env.get('amp_dtype', d.get('amp_dtype', 'bfloat16'))} |",
        f"| gradient checkpointing | {env.get('gradient_checkpointing', d.get('gradient_checkpointing', 'n/a'))} |",
        "",
        "## 4. GPU / CUDA 性能（最終 epoch）",
        "",
        "| 項目 | 値 | 説明 |",
        "|------|-----|------|",
        f"| GPU | {env.get('gpu_name', d.get('gpu_name', 'n/a'))} | 使用デバイス |",
        f"| CUDA available | {env.get('cuda_available', d.get('cuda_available', 'n/a'))} | PyTorch から GPU 利用可 |",
        f"| 学習時間 [s] | {hw.get('train_time_sec', 'n/a')} | 1 epoch の forward+backward |",
        f"| 1 step 平均 [s] | {hw.get('avg_step_time_sec', 'n/a')} | train_time / steps |",
        f"| Peak VRAM [GB] | {hw.get('max_memory_allocated_gb', hw.get('max_vram_gb', 'n/a'))} | torch.max_memory_allocated |",
        f"| 現在 VRAM [GB] | {hw.get('memory_allocated_gb', 'n/a')} | torch.memory_allocated（計測時点） |",
        f"| PyTorch 予約 [GB] | {hw.get('memory_reserved_gb', 'n/a')} | キャッシュプール含む |",
        f"| nvidia-smi used [GB] | {hw.get('nvidia_smi_used_memory_gb', 'n/a')} | ドライバ視点（全プロセス） |",
        f"| GPU 利用率 [%] | {hw.get('avg_gpu_util_percent', 'n/a')} | nvidia-smi 平均 |",
        "",
        "## 5. 観測結果サマリ",
        "",
        "### 精度（eval セット）",
        "",
        f"- accuracy: **{ev.get('accuracy_before', 'n/a')} → {ev.get('accuracy_after', 'n/a')}**",
        f"- F1 macro: **{ev.get('f1_before', 'n/a')} → {ev.get('f1_after', 'n/a')}**",
        "",
    ]
    if ld:
        lines.extend([
            "### LoRA",
            "",
            f"- `||Δθ_client||` (final): **{ld.get('client_delta_norm', 0):.6e}**",
            "",
        ])
    if od:
        lines.extend([
            "### 出力（プローブ 1 サンプル）",
            "",
            f"- KL(client‖surrogate) before→after: **{od.get('kl_client_surrogate_before', 'n/a')} → {od.get('kl_client_surrogate_after', 'n/a')}**",
            f"- 予測 class before→after: **{od.get('client_pred_before')} → {od.get('client_pred_after')}**",
            "",
        ])
    if history:
        lines.extend([
            "### 学習推移（epoch_history.jsonl）",
            "",
            "| Epoch | Loss | Acc | ||Δθ|| | Peak VRAM [GB] | step [s] |",
            "|-------|------|-----|--------|----------------|----------|",
        ])
        for h in history:
            hm = h.get("hardware_metrics") or {}
            loss = h.get("train_loss")
            loss_s = f"{loss:.4f}" if loss is not None else "—"
            acc = h.get("eval", {}).get("accuracy", 0)
            lines.append(
                f"| {h.get('epoch')} | {loss_s} | {acc:.2f} | {h.get('client_delta', 0):.4f} | "
                f"{hm.get('max_memory_allocated_gb', hm.get('max_vram_gb', '—'))} | "
                f"{hm.get('avg_step_time_sec', '—')} |"
            )
        lines.append("")

    if interp:
        lines.extend([
            "## 6. 解釈チェックリスト",
            "",
            f"- LoRA が更新された: **{interp.get('lora_adapted', 'n/a')}**",
            f"- 出力が変化した: **{interp.get('output_changed', 'n/a')}**",
            "",
        ])

    lines.extend([
        "## 7. 出力ファイル",
        "",
        "- `used_config.yaml` — 実行時設定",
        "- `epoch_history.jsonl` — epoch ごとの probe / eval / hardware_metrics",
        "- `summary.json` — 数値サマリ（機械可読）",
        "- `experiment_report.md` — 本ファイル",
        "",
    ])
    path = run_dir / "experiment_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
