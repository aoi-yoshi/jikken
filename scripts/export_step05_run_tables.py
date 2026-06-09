"""
Step 5 FL run の jsonl/json から docs 用 Markdown 表を生成する。

  python scripts/export_step05_run_tables.py --run-dir artifacts/runs/step05_fl/run_20260607_200402
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _fmt(v: Any, *, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.2f}"
        return f"{v:.{digits}f}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, digits: int = 3) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(c, digits=digits) for c in row) + " |")
    return "\n".join(lines)


def _rows_by_event(rows: List[Dict[str, Any]], event: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("event") == event]


def _pl(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("professor_links") or {}


def _legend(lines: Sequence[str]) -> str:
    return "\n".join(f"- **{line.split(' — ', 1)[0]}** — {line.split(' — ', 1)[1]}" if " — " in line else f"- {line}" for line in lines)


# 各表の直下に出す変数説明（再生成しても維持される）
_LEGENDS: Dict[str, List[str]] = {
    "A.1": [
        "manifest — 学習データ一覧ファイル（manifest.jsonl）のパス",
        "partition_mode — データ分割方式（iid_shuffle = ラベル比率を保ったままシャッフルして分割）",
        "partition_seed — 分割の乱数シード（再現性用）",
        "num_clients — 連合学習のクライアント数",
        "pool_size / train_pool_size — FL 学習に使う train プールの件数",
        "eval_size — 全クライアント共通の eval セット件数",
        "total_after_subset — config の上限適用後の manifest 総件数",
        "eval_label_counts — eval 20 件のラベル内訳（0=正常, 1=リスク）",
        "pool_label_counts — train プール 100 件のラベル内訳",
        "client N samples — そのクライアントに割り当てられた train 件数",
        "label_counts — クライアント内のラベル内訳",
        "scene_counts — クライアント内のシーン種別内訳（Urban 等）",
    ],
    "A.2": [
        "trainable_parameters — 勾配更新対象パラメータ数（classifier + client LoRA）",
        "adapter_size_mb — client LoRA アダプタのメモリサイズ [MB]",
        "train_samples — このクライアントの学習サンプル数",
        "eval_samples — 共通 eval セットのサンプル数",
    ],
    "A.3": [
        "client — クライアント ID（0 または 1）",
        "accuracy — 正解率（0〜1）。FL 開始前、凍結 surrogate LoRA で eval した値",
        "f1_macro — F1 スコアのマクロ平均（ラベル 0/1 を平等に扱う）",
    ],
    "A.4": [
        "R — サーバーラウンド番号（1 から開始）",
        "client — クライアント ID",
        "post_redist_acc — サーバーからグローバル重みを受け取った直後の正解率（未学習）",
        "post_redist_f1 — 上記の F1 macro",
        "local_adapt_acc — ローカル学習後の正解率（client LoRA + 更新済み classifier）",
        "local_adapt_f1 — 上記の F1 macro",
        "foundation_acc — 凍結 surrogate（基盤 F）LoRA + そのクライアントの classifier での正解率",
        "foundation_f1 — 上記の F1 macro",
        "w_div_trainable — 学習可能パラメータ全体の L2 距離（基盤スナップショットとの差）",
        "w_div_lora — client LoRA 部分だけの L2 距離",
        "avg_train_loss — ローカル学習の平均交差エントロピー loss（全 train step の平均）",
    ],
    "A.5": [
        "R — サーバーラウンド番号",
        "client — クライアント ID",
        "eval_accuracy — Flower `evaluate()` でサーバーに返した正解率（20 サンプル）",
        "eval_f1_macro — 上記の F1 macro",
    ],
    "A.6": [
        "checkpoint — 評価に使ったグローバル checkpoint ファイルのパス",
        "eval_rows — 評価サンプル数",
        "accuracy / f1_macro — 最終 checkpoint の分類精度・F1",
        "elapsed_sec — post_eval 全体の壁時計時間 [秒]",
        "avg_gpu_util_percent — post_eval 中の GPU 使用率 [%]（nvidia-smi ベース）",
        "memory_allocated_gb — 記録瞬間の PyTorch テンソル VRAM [GB]",
        "memory_reserved_gb — PyTorch キャッシュプール込みの確保 VRAM [GB]",
        "max_memory_allocated_gb — post_eval 中の VRAM ピーク（PyTorch 視点）",
        "max_memory_reserved_gb — reserved のピーク [GB]",
        "nvidia_smi_used_memory_gb — ドライバ報告の VRAM 使用量（全プロセス合計）[GB]",
    ],
    "link1": [
        "R — サーバーラウンド番号（表に R 列がある場合）",
        "client — クライアント ID（表に client 列がある場合）",
        "memory_allocated_gb — 記録した瞬間、テンソルが実際に占有している VRAM [GB]",
        "memory_reserved_gb — PyTorch がキャッシュ用に確保した VRAM（解放後もプールに残る）[GB]",
        "max_memory_allocated_gb — 直前の peak リセット以降の最大テンソル VRAM。教授向け「最大 GPU メモリ」はこれ",
        "max_memory_reserved_gb — reserved のピーク [GB]",
        "total_memory_gb — GPU 物理 VRAM 総容量 [GB]",
        "multi_processor_count — GPU 内 SM（演算ユニット）数",
        "cuda_capability — GPU アーキテクチャ版（12.0 = RTX 5050 / Blackwell 系）",
    ],
    "link3_snapshot": [
        "R — サーバーラウンド番号（表に R 列がある場合）",
        "client — クライアント ID（表に client 列がある場合）",
        "smi_utilization_gpu_percent — GPU コア使用率 [%]（記録した 1 瞬間）",
        "smi_utilization_memory_percent — メモリバス使用率 [%]",
        "smi_memory_used_gb — Windows 上の全 GPU プロセス合計 VRAM [GB]",
        "smi_memory_free_gb — 空き VRAM [GB]",
        "smi_memory_total_gb — 総 VRAM [GB]",
        "smi_power_draw_w — GPU 消費電力 [W]",
        "smi_temperature_gpu_c — GPU 温度 [°C]",
        "smi_clocks_current_sm_mhz — 記録瞬間の SM クロック [MHz]",
        "smi_clocks_current_memory_mhz — 記録瞬間のメモリクロック [MHz]",
        "smi_clocks_max_sm_mhz — 最大 SM クロック [MHz]",
        "smi_clocks_max_memory_mhz — 最大メモリクロック [MHz]",
        "driver_version — NVIDIA ドライバ版",
    ],
    "link4_raw": [
        "allocated_bytes.all.current — 現在のテンソル割当バイト数（Link1 allocated と同系統）",
        "allocated_bytes.all.peak — ピーク割当バイト数（Link1 max_allocated と同系統）",
        "reserved_bytes.all.current — 現在のキャッシュプール確保バイト数",
        "reserved_bytes.all.peak — reserved のピーク",
        "active_bytes.all.peak — 実際に使用中だったバイトのピーク",
        "inactive_split_bytes.all.peak — 断片化で使えない塊のピーク",
        "requested_bytes.all.peak — PyTorch が要求した量のピーク",
        "allocation.all.current — メモリブロック数",
        "segment.all.current — CUDA メモリセグメント数",
        "num_alloc_retries — メモリ確保のリトライ回数",
        "num_ooms — Out Of Memory 発生回数（0 = OOM なし）",
        "num_sync_all_streams — 全ストリーム同期回数",
        "num_device_alloc / num_device_free — GPU メモリ確保・解放回数",
    ],
    "link4_gb": [
        "R — サーバーラウンド番号（表に R 列がある場合）",
        "client — クライアント ID（表に client 列がある場合）",
        "allocated_gb_all_current — 現在のテンソル VRAM [GB]",
        "allocated_gb_all_peak — ピークテンソル VRAM [GB]",
        "reserved_gb_all_current — 現在のキャッシュプール VRAM [GB]",
        "reserved_gb_all_peak — ピークキャッシュプール VRAM [GB]",
        "active_gb_all_current / peak — 実使用中 VRAM [GB]",
        "inactive_split_gb_all_current / peak — 断片化 VRAM [GB]",
        "requested_gb_all_current / peak — PyTorch 要求量 [GB]",
    ],
    "A.7": [
        "R — サーバーラウンド番号",
        "client — クライアント ID（0 または 1）",
        "apply_global_params_sec — グローバル重みをモデルに書き込む時間 [秒]",
        "post_redistribution_eval_sec — 書き込み直後（未学習）の eval 時間 [秒]",
        "fit_duration_sec — ローカル学習本体の時間 [秒]",
        "local_adaptation_eval_sec — 学習後 eval の時間 [秒]",
        "foundation_generalization_eval_sec — 凍結基盤 LoRA eval の時間 [秒]",
        "weight_divergence_sec — 重み divergence 計算時間 [秒]",
        "round_duration_sec — クライアント fit() 1 回の壁時計時間 [秒]",
        "round_overhead_sec — 上記合計との差分（Flower 通信等）[秒]",
        "steps — fit 中の train step 数",
        "avg_step_time_sec — fit_duration_sec ÷ steps（1 step 平均秒）",
    ],
    "A.8": [
        "R — サーバーラウンド番号",
        "client — クライアント ID",
        "elapsed_sec — Flower evaluate 全体の壁時計時間 [秒]",
        "avg_step_time_sec — elapsed_sec ÷ eval サンプル数（1 サンプル平均秒）",
        "steps — evaluate で処理したサンプル数（= eval 件数）",
        "memory_allocated_gb / reserved_gb — evaluate 終了瞬間の PyTorch VRAM [GB]",
        "max_memory_allocated_gb — evaluate 前に peak リセット後の VRAM ピーク [GB]",
        "max_memory_reserved_gb — reserved のピーク [GB]",
        "avg_gpu_util_percent — evaluate 中の GPU 使用率 [%]",
        "nvidia_smi_used_memory_gb — evaluate 終了瞬間の smi used VRAM [GB]",
    ],
    "A.8b": [
        "R — サーバーラウンド番号",
        "client — クライアント ID",
        "phase — 計測フェーズ名（fit）",
        "train_time_sec / elapsed_sec — fit 区間の経過時間 [秒]（≈ fit_duration_sec）",
        "steps — train step 数",
        "avg_step_time_sec — 1 step 平均秒",
        "avg_gpu_util_percent — fit 中 smi サンプルの GPU 使用率平均 [%]",
        "nvidia_smi_used_memory_gb — fit 終了瞬間の smi used VRAM [GB]",
        "memory_allocated_gb — fit 終了瞬間の PyTorch 現在 VRAM [GB]",
        "max_memory_allocated_gb — fit 区間の PyTorch VRAM ピーク [GB]",
    ],
    "A.13": [
        "R — サーバーラウンド番号",
        "client — クライアント ID",
        "metric — 要約対象（gpu_util_percent / memory_util_percent / nvidia_smi_used_memory_gb）",
        "min / max / avg — fit 中に毎 step サンプルした値の最小・最大・平均",
        "n — サンプル数（= train step 数。生時系列は保存せず要約のみ）",
    ],
    "A.16": [
        "R — サーバーラウンド番号",
        "client — クライアント ID",
        "profiler_available — torch.profiler が正常起動したか",
        "cpu_time_total_ms — fit 先頭 1 step の CPU 側合計時間 [ms]",
        "cuda_time_total_ms — 同 step の GPU カーネル合計時間 [ms]（RTX 5050 では 0 になりがち）",
    ],
    "A.17": [
        "name — 処理単位（op）の名前。例: aten::to = テンソル型・デバイス変換",
        "cpu_time_ms — その op に CPU でかかった時間 [ms]",
        "cuda_time_ms — その op の GPU カーネル時間 [ms]",
        "cuda_memory_mb — その op が関与した GPU メモリ量 [MB]",
        "count — その op が 1 step 内で呼ばれた回数",
    ],
    "A.18_agg": [
        "R — サーバーラウンド番号",
        "param_norm — FedAvg 集約後グローバル重みベクトルの L2 ノルム（学習が進むと増える）",
        "aggregate_sec — FedAvg 集約処理そのものの時間 [秒]",
        "global_checkpoint — 最終ラウンドのみ。保存した .pt ファイルのパス",
        "num_clients — 集約に参加したクライアント数",
    ],
    "A.18_eval": [
        "eval_loss — Flower 用 loss = 1.0 − accuracy（交差エントロピーではない）",
        "eval_metrics — サーバー側に集約された evaluate メトリクス（空 dict のこともある）",
    ],
    "A.20": [
        "python_version / platform — 実行環境",
        "cuda_available — CUDA 利用可否",
        "gpu_name / cuda_version / torch_version — GPU・CUDA・PyTorch 版",
        "nvidia_smi_total_memory_gb — smi 報告の GPU 総 VRAM [GB]",
        "batch_size / num_frames — 学習バッチサイズ・入力フレーム数",
        "lora_r / lora_target_modules — LoRA ランク・適用レイヤ",
        "quantization_enabled / load_in_4bit / load_in_8bit — 量子化設定",
        "amp_enabled / amp_dtype — 混合精度（bfloat16 等）",
        "gradient_checkpointing — 勾配チェックポイント有無（VRAM 節約）",
    ],
    "A.21": [
        "fl.strategy — 連合学習アルゴリズム（FedAvg 等）",
        "fl.num_rounds — サーバーラウンド数",
        "fl.local_epochs — クライアント 1 ラウンドあたりのローカル epoch 数",
        "fl.num_clients / min_fit_clients — クライアント数・学習開始に必要な最小数",
        "partition.mode / seed — データ分割方式・シード",
        "train.batch_size / eval_max — バッチサイズ・eval 件数上限",
        "train.max_train_samples — train プール上限",
        "train.lr / max_length — 学習率・トークン最大長",
        "lora.r / target_modules — LoRA 設定",
        "model.id — 使用モデル（HuggingFace ID）",
        "communication.trainable_vector_dim — FedAvg で送受信するベクトル次元",
        "communication.upload_per_client_round_mb — 1 クライアント 1 ラウンドの通信量 [MB]",
    ],
}


def _note(parts: List[str], key: str) -> None:
    if key in _LEGENDS:
        parts.append(_legend(_LEGENDS[key]))
        parts.append("")


def _fit_rows(run_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cid in (0, 1):
        for r in _rows_by_event(_load_jsonl(run_dir / f"client_{cid}/fl_client.jsonl"), "fl_round_metrics"):
            out.append(r)
    return sorted(out, key=lambda r: (int(r["server_round"]), int(r["cid"])))


def _eval_rows(run_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cid in (0, 1):
        for r in _rows_by_event(_load_jsonl(run_dir / f"client_{cid}/fl_client.jsonl"), "evaluate"):
            out.append(r)
    return sorted(out, key=lambda r: (int(r["server_round"]), int(r["cid"])))


def generate_tables(run_dir: Path) -> str:
    run_id = run_dir.name
    l1k = [
        "memory_allocated_gb", "memory_reserved_gb", "max_memory_allocated_gb", "max_memory_reserved_gb",
        "total_memory_gb", "multi_processor_count", "cuda_capability",
    ]
    s3k = [
        "smi_utilization_gpu_percent", "smi_utilization_memory_percent",
        "smi_memory_used_gb", "smi_memory_free_gb", "smi_memory_total_gb",
        "smi_power_draw_w", "smi_temperature_gpu_c",
        "smi_clocks_current_sm_mhz", "smi_clocks_current_memory_mhz",
        "smi_clocks_max_sm_mhz", "smi_clocks_max_memory_mhz", "driver_version",
    ]
    l4_keys = [
        "allocated_bytes.all.current", "allocated_bytes.all.peak",
        "reserved_bytes.all.current", "reserved_bytes.all.peak",
        "active_bytes.all.current", "active_bytes.all.peak",
        "inactive_split_bytes.all.current", "inactive_split_bytes.all.peak",
        "requested_bytes.all.current", "requested_bytes.all.peak",
        "allocation.all.current", "segment.all.current",
        "num_alloc_retries", "num_ooms", "num_sync_all_streams",
        "num_device_alloc", "num_device_free",
    ]
    l4g_keys = [
        "allocated_gb_all_current", "allocated_gb_all_peak",
        "reserved_gb_all_current", "reserved_gb_all_peak",
        "active_gb_all_current", "active_gb_all_peak",
        "inactive_split_gb_all_current", "inactive_split_gb_all_peak",
        "requested_gb_all_current", "requested_gb_all_peak",
    ]

    parts: List[str] = [
        f"<!-- AUTO-GENERATED by scripts/export_step05_run_tables.py from {run_id} -->",
        "",
        "## A. 実測値表（json 全フィールド）",
        "",
        f"対象: `{run_id}`。再生成: `python scripts/export_step05_run_tables.py --run-dir {run_dir.as_posix()}`",
        "",
        "**表に含めるもの:** jsonl/json のスカラー・Link1〜4・精度・時間の全フィールド。",
        "",
        "**意図的に省略:** `t`（Unix 時刻）、`experiment_record.flower_settings` の yaml 全文（→ A.21 抜粋）、`fl_partition.json` のサンプル ID 一覧（件数は A.1）。",
        "",
    ]

    # --- partition ---
    p0 = _rows_by_event(_load_jsonl(run_dir / "client_0/fl_client.jsonl"), "partition")
    if p0:
        p = p0[0]
        parts += ["### A.1 partition（データ分割）", ""]
        parts.append(_md_table(
            ["項目", "値"],
            [
                ("manifest", p.get("manifest")),
                ("partition_mode", p.get("partition_mode")),
                ("partition_seed", p.get("partition_seed")),
                ("num_clients", p.get("num_clients")),
                ("pool_size", p.get("pool_size")),
                ("train_pool_size", p.get("train_pool_size")),
                ("eval_size", p.get("eval_size")),
                ("total_after_subset", p.get("total_after_subset")),
                ("eval_label_counts", p.get("eval_label_counts")),
                ("pool_label_counts", p.get("pool_label_counts")),
            ],
        ))
        parts.append("")
        for c in p.get("clients") or []:
            parts.append(f"**client {c.get('client_id')}** — samples: {c.get('num_samples')}, "
                         f"labels: {c.get('label_counts')}, scenes: {c.get('scene_counts')}")
        parts.append("")
        _note(parts, "A.1")

    # --- model_ready ---
    for cid in (0, 1):
        mr = _rows_by_event(_load_jsonl(run_dir / f"client_{cid}/fl_client.jsonl"), "model_ready")
        if mr:
            r = mr[0]
            parts += [f"### A.2 model_ready（client {cid}）", ""]
            parts.append(_md_table(["項目", "値"], [(k, r[k]) for k in (
                "trainable_parameters", "adapter_size_mb", "train_samples", "eval_samples")]))
            parts.append("")
            _note(parts, "A.2")

    # --- foundation_baseline ---
    parts += ["### A.3 foundation_baseline（FL 開始前の基盤 F）", ""]
    fb_rows: List[List[Any]] = []
    for cid in (0, 1):
        for r in _rows_by_event(_load_jsonl(run_dir / f"client_{cid}/fl_client.jsonl"), "foundation_baseline"):
            fb_rows.append([cid, r.get("foundation_generalization_accuracy"), r.get("foundation_generalization_f1_macro")])
    parts.append(_md_table(["client", "accuracy", "f1_macro"], fb_rows))
    parts.append("")
    _note(parts, "A.3")

    # --- accuracy fl_round_metrics ---
    parts += ["### A.4 精度・重み divergence（fl_round_metrics）", ""]
    acc_headers = [
        "R", "client",
        "post_redist_acc", "post_redist_f1",
        "local_adapt_acc", "local_adapt_f1",
        "foundation_acc", "foundation_f1",
        "w_div_trainable", "w_div_lora",
        "avg_train_loss",
    ]
    acc_rows = []
    for r in _fit_rows(run_dir):
        acc_rows.append([
            r["server_round"], r["cid"],
            r.get("post_redistribution_accuracy"), r.get("post_redistribution_f1_macro"),
            r.get("local_adaptation_accuracy"), r.get("local_adaptation_f1_macro"),
            r.get("foundation_generalization_accuracy"), r.get("foundation_generalization_f1_macro"),
            r.get("weight_divergence_trainable"), r.get("weight_divergence_lora"),
            r.get("avg_train_loss"),
        ])
    parts.append(_md_table(acc_headers, acc_rows))
    parts.append("")
    _note(parts, "A.4")

    # --- evaluate accuracy ---
    parts += ["### A.5 Flower evaluate 精度", ""]
    ev_acc = [[r["server_round"], r["cid"], r.get("eval_accuracy"), r.get("eval_f1_macro")] for r in _eval_rows(run_dir)]
    parts.append(_md_table(["R", "client", "eval_accuracy", "eval_f1_macro"], ev_acc))
    parts.append("")
    _note(parts, "A.5")

    # --- post_eval ---
    pe_rows = _load_jsonl(run_dir / "fl_post_eval.jsonl")
    pe = next((r for r in reversed(pe_rows) if "metrics" in r), None)
    if pe:
        parts += ["### A.6 post_eval（最終 checkpoint 評価）", ""]
        m = pe["metrics"]
        parts.append(_md_table(["項目", "値"], [
            ("checkpoint", pe.get("checkpoint")),
            ("eval_rows", pe.get("eval_rows")),
            ("accuracy", m.get("accuracy")),
            ("f1_macro", m.get("f1_macro")),
            ("elapsed_sec", pe.get("elapsed_sec")),
            ("avg_gpu_util_percent", pe.get("avg_gpu_util_percent")),
            ("memory_allocated_gb", pe.get("memory_allocated_gb")),
            ("memory_reserved_gb", pe.get("memory_reserved_gb")),
            ("max_memory_allocated_gb", pe.get("max_memory_allocated_gb")),
            ("max_memory_reserved_gb", pe.get("max_memory_reserved_gb")),
            ("nvidia_smi_used_memory_gb", pe.get("nvidia_smi_used_memory_gb")),
        ]))
        parts.append("")
        _note(parts, "A.6")
        pl_pe = pe.get("professor_links") or {}
        parts += ["#### post_eval — Link1", ""]
        l1pe = pl_pe.get("link1_pytorch_cuda_semantics") or {}
        parts.append(_md_table(["項目", "値"], [(k, l1pe.get(k)) for k in l1k], digits=4))
        parts.append("")
        _note(parts, "link1")
        parts += ["#### post_eval — Link3 snapshot", ""]
        s3pe = pl_pe.get("link3_nvidia_smi_snapshot") or {}
        parts.append(_md_table(["項目", "値"], [(k, s3pe.get(k)) for k in s3k], digits=2))
        parts.append("")
        _note(parts, "link3_snapshot")
        parts += ["#### post_eval — Link4 memory_stats_raw", ""]
        l4pe = (pl_pe.get("link4_pytorch_memory_stats") or {}).get("memory_stats_raw") or {}
        parts.append(_md_table(["項目", "値"], [(k, l4pe.get(k)) for k in l4_keys], digits=0))
        parts.append("")
        _note(parts, "link4_raw")
        l4gpe = (pl_pe.get("link4_pytorch_memory_stats") or {}).get("memory_stats_gb") or {}
        parts += ["#### post_eval — Link4 memory_stats [GB]", ""]
        parts.append(_md_table(["項目", "値"], [(k, l4gpe.get(k)) for k in l4g_keys], digits=4))
        parts.append("")
        _note(parts, "link4_gb")

    # --- timing ---
    parts += ["### A.7 フェーズ別経過時間 [秒]（round_timing）", ""]
    tk = [
        "apply_global_params_sec", "post_redistribution_eval_sec", "fit_duration_sec",
        "local_adaptation_eval_sec", "foundation_generalization_eval_sec",
        "weight_divergence_sec", "round_duration_sec", "round_overhead_sec",
    ]
    t_headers = ["R", "client"] + tk + ["steps", "avg_step_time_sec"]
    t_rows = []
    for r in _fit_rows(run_dir):
        t_rows.append([r["server_round"], r["cid"]] + [r.get(k) for k in tk] + [r.get("steps"), r.get("avg_step_time_sec")])
    parts.append(_md_table(t_headers, t_rows, digits=2))
    parts.append("")
    _note(parts, "A.7")

    # --- evaluate timing + hw top ---
    parts += ["### A.8 evaluate 時間・ハードウェア（トップレベル）", ""]
    eh = ["R", "client", "elapsed_sec", "avg_step_time_sec", "steps",
          "memory_allocated_gb", "memory_reserved_gb", "max_memory_allocated_gb",
          "max_memory_reserved_gb", "avg_gpu_util_percent", "nvidia_smi_used_memory_gb"]
    eh_rows = []
    for r in _eval_rows(run_dir):
        eh_rows.append([r["server_round"], r["cid"]] + [r.get(k) for k in eh[2:]])
    parts.append(_md_table(eh, eh_rows))
    parts.append("")
    _note(parts, "A.8")

    parts += ["### A.8b fit 終了時ハードウェア（fl_round_metrics トップレベル）", ""]
    fith = ["R", "client", "phase", "train_time_sec", "elapsed_sec", "steps", "avg_step_time_sec",
            "avg_gpu_util_percent", "nvidia_smi_used_memory_gb",
            "memory_allocated_gb", "max_memory_allocated_gb"]
    fith_rows = [[r["server_round"], r["cid"]] + [r.get(k) for k in fith[2:]] for r in _fit_rows(run_dir)]
    parts.append(_md_table(fith, fith_rows))
    parts.append("")
    _note(parts, "A.8b")

    parts += ["### A.8c evaluate — Link4 memory_stats [GB]", ""]
    l4ev_rows = []
    for r in _eval_rows(run_dir):
        gb = (_pl(r).get("link4_pytorch_memory_stats") or {}).get("memory_stats_gb") or {}
        l4ev_rows.append([r["server_round"], r["cid"]] + [gb.get(k) for k in l4g_keys])
    parts.append(_md_table(["R", "client"] + l4g_keys, l4ev_rows, digits=4))
    parts.append("")
    _note(parts, "link4_gb")

    # --- Link1 ---
    parts += ["### A.9 Link1 — PyTorch VRAM [GB]（fit 終了時）", ""]
    l1h = ["R", "client"] + l1k
    l1_rows = []
    for r in _fit_rows(run_dir):
        l1 = _pl(r).get("link1_pytorch_cuda_semantics") or {}
        l1_rows.append([r["server_round"], r["cid"]] + [l1.get(k) for k in l1k])
    parts.append(_md_table(l1h, l1_rows, digits=4))
    parts.append("")
    _note(parts, "link1")

    parts += ["### A.10 Link1 — PyTorch VRAM [GB]（evaluate 終了時）", ""]
    l1e_rows = []
    for r in _eval_rows(run_dir):
        l1 = _pl(r).get("link1_pytorch_cuda_semantics") or {}
        l1e_rows.append([r["server_round"], r["cid"]] + [l1.get(k) for k in l1k[:4]])
    parts.append(_md_table(["R", "client"] + l1k[:4], l1e_rows, digits=4))
    parts.append("")
    parts.append("※ evaluate 時は `max_memory_allocated_gb` 等 4 項目のみ記録（fit より項目が少ない）。")
    parts.append("")
    _note(parts, "link1")

    # --- Link3 snapshot ---
    parts += ["### A.11 Link3 — nvidia-smi スナップショット（fit 終了瞬間）", ""]
    s3_rows = []
    for r in _fit_rows(run_dir):
        s3 = _pl(r).get("link3_nvidia_smi_snapshot") or {}
        s3_rows.append([r["server_round"], r["cid"]] + [s3.get(k) for k in s3k])
    parts.append(_md_table(["R", "client"] + s3k, s3_rows, digits=2))
    parts.append("")
    _note(parts, "link3_snapshot")

    parts += ["### A.12 Link3 — nvidia-smi スナップショット（evaluate 終了瞬間）", ""]
    s3e_rows = []
    for r in _eval_rows(run_dir):
        s3 = _pl(r).get("link3_nvidia_smi_snapshot") or {}
        s3e_rows.append([r["server_round"], r["cid"]] + [s3.get(k) for k in s3k])
    parts.append(_md_table(["R", "client"] + s3k, s3e_rows, digits=2))
    parts.append("")
    _note(parts, "link3_snapshot")

    # --- Link3 timeseries ---
    parts += ["### A.13 Link3 — nvidia-smi 学習中要約（fit 中 min/max/avg）", ""]
    ts_keys = ("gpu_util_percent", "memory_util_percent", "nvidia_smi_used_memory_gb")
    ts_headers = ["R", "client", "metric", "min", "max", "avg", "n"]
    ts_rows: List[List[Any]] = []
    for r in _fit_rows(run_dir):
        ts = _pl(r).get("link3_nvidia_smi_timeseries") or r.get("nvidia_smi_timeseries") or {}
        for metric in ts_keys:
            block = ts.get(metric) or {}
            ts_rows.append([
                r["server_round"], r["cid"], metric,
                block.get("min"), block.get("max"), block.get("avg"), block.get("n"),
            ])
    parts.append(_md_table(ts_headers, ts_rows, digits=2))
    parts.append("")
    _note(parts, "A.13")

    # --- Link4 ---
    parts += ["### A.14 Link4 — PyTorch memory_stats（fit 終了時・bytes）", ""]
    for r in _fit_rows(run_dir):
        rnd, cid = r["server_round"], r["cid"]
        l4 = (_pl(r).get("link4_pytorch_memory_stats") or {}).get("memory_stats_raw") or {}
        parts.append(f"**R{rnd} client {cid}**")
        parts.append("")
        parts.append(_md_table(["項目", "値"], [(k, l4.get(k)) for k in l4_keys], digits=0))
        parts.append("")
    _note(parts, "link4_raw")

    parts += ["### A.15 Link4 — PyTorch memory_stats [GB]（fit 終了時 peak 系）", ""]
    l4g_rows = []
    for r in _fit_rows(run_dir):
        gb = (_pl(r).get("link4_pytorch_memory_stats") or {}).get("memory_stats_gb") or {}
        l4g_rows.append([r["server_round"], r["cid"]] + [gb.get(k) for k in l4g_keys])
    parts.append(_md_table(["R", "client"] + l4g_keys, l4g_rows, digits=4))
    parts.append("")
    _note(parts, "link4_gb")

    # --- Link2 ---
    parts += ["### A.16 Link2 — torch.profiler サマリ（fit 先頭 1 step）", ""]
    l2h = ["R", "client", "profiler_available", "cpu_time_total_ms", "cuda_time_total_ms"]
    l2_rows = []
    for r in _fit_rows(run_dir):
        l2 = _pl(r).get("link2_torch_profiler") or {}
        l2_rows.append([r["server_round"], r["cid"], l2.get("profiler_available"),
                        l2.get("cpu_time_total_ms"), l2.get("cuda_time_total_ms")])
    parts.append(_md_table(l2h, l2_rows, digits=2))
    parts.append("")
    _note(parts, "A.16")

    parts += ["### A.17 Link2 — top_ops（fit 先頭 1 step・全 15 件）", ""]
    for r in _fit_rows(run_dir):
        rnd, cid = r["server_round"], r["cid"]
        ops = (_pl(r).get("link2_torch_profiler") or {}).get("top_ops_by_cuda_time") or []
        parts.append(f"**R{rnd} client {cid}**")
        parts.append("")
        if ops:
            parts.append(_md_table(
                ["#", "name", "cpu_time_ms", "cuda_time_ms", "cuda_memory_mb", "count"],
                [[i + 1, o.get("name"), o.get("cpu_time_ms"), o.get("cuda_time_ms"),
                  o.get("cuda_memory_mb"), o.get("count")] for i, o in enumerate(ops)],
                digits=3,
            ))
        else:
            parts.append("（データなし）")
        parts.append("")
    _note(parts, "A.17")

    # --- server ---
    parts += ["### A.18 サーバー（fl_server.jsonl）", ""]
    srv = _load_jsonl(run_dir / "fl_server.jsonl")
    agg_rows = []
    ev_rows = []
    for r in srv:
        if r.get("event") == "server_round_aggregate":
            agg_rows.append([
                r.get("server_round"), r.get("num_clients"), r.get("param_norm"),
                r.get("server_aggregate_duration_sec"), r.get("global_checkpoint", "—"),
            ])
        elif r.get("event") == "server_evaluate_aggregate":
            ev_rows.append([r.get("server_round"), r.get("num_clients"), r.get("eval_loss"), r.get("eval_metrics")])
    parts.append("**server_round_aggregate**")
    parts.append("")
    parts.append(_md_table(
        ["R", "num_clients", "param_norm", "aggregate_sec", "global_checkpoint"],
        agg_rows, digits=4,
    ))
    parts.append("")
    _note(parts, "A.18_agg")
    parts.append("**server_evaluate_aggregate**")
    parts.append("")
    parts.append(_md_table(["R", "num_clients", "eval_loss", "eval_metrics"], ev_rows))
    parts.append("")
    _note(parts, "A.18_eval")

    # --- baseline from summary ---
    parts += ["### A.19 run 開始 baseline（summary.json → professor_links_baseline）", ""]
    for cid in (0, 1):
        sp = run_dir / f"client_{cid}/summary.json"
        if not sp.is_file():
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        bl = (s.get("environment") or {}).get("professor_links_baseline") or {}
        parts.append(f"**client {cid} — Link1**")
        parts.append("")
        l1 = bl.get("link1_pytorch_cuda_semantics") or {}
        parts.append(_md_table(["項目", "値"], [(k, l1.get(k)) for k in l1k], digits=4))
        parts.append("")
        _note(parts, "link1")
        parts.append(f"**client {cid} — Link3 snapshot**")
        parts.append("")
        s3 = bl.get("link3_nvidia_smi_snapshot") or {}
        parts.append(_md_table(["項目", "値"], [(k, s3.get(k)) for k in s3k], digits=2))
        parts.append("")
        _note(parts, "link3_snapshot")
        parts.append(f"**client {cid} — Link4 memory_stats_raw**")
        parts.append("")
        l4 = (bl.get("link4_pytorch_memory_stats") or {}).get("memory_stats_raw") or {}
        parts.append(_md_table(["項目", "値"], [(k, l4.get(k)) for k in l4_keys], digits=0))
        parts.append("")
        _note(parts, "link4_raw")
    parts.append("※ baseline は run 開始直後・モデル未ロードに近い状態（VRAM ≈ 0）。")
    parts.append("")

    # --- environment scalars ---
    parts += ["### A.20 実行環境（summary.json → environment）", ""]
    sp = run_dir / "client_0/summary.json"
    if sp.is_file():
        env = json.loads(sp.read_text(encoding="utf-8")).get("environment") or {}
        env_keys = [
            "python_version", "platform", "cuda_available", "gpu_name", "cuda_version",
            "torch_version", "nvidia_smi_total_memory_gb", "batch_size", "num_frames",
            "lora_r", "lora_target_modules", "quantization_enabled", "load_in_4bit", "load_in_8bit",
            "amp_enabled", "amp_dtype", "gradient_checkpointing",
        ]
        parts.append(_md_table(["項目", "値"], [(k, env.get(k)) for k in env_keys]))
        parts.append("")
        _note(parts, "A.20")

    # --- fl yaml key scalars from experiment_record ---
    parts += ["### A.21 主要設定（experiment_record → flower_settings 抜粋）", ""]
    er = _rows_by_event(_load_jsonl(run_dir / "client_0/fl_client.jsonl"), "experiment_record")
    if er:
        fs = er[0].get("flower_settings") or {}
        fl = fs.get("fl_yaml") or {}
        train = fs.get("train_yaml") or {}
        lora = fs.get("lora_yaml") or {}
        model = fs.get("model_yaml") or {}
        comm = fs.get("communication") or {}
        cfg_rows = [
            ("fl.strategy", fl.get("strategy")),
            ("fl.num_rounds", fl.get("num_rounds")),
            ("fl.local_epochs", fl.get("local_epochs")),
            ("fl.num_clients", fl.get("num_clients")),
            ("fl.min_fit_clients", fl.get("min_fit_clients")),
            ("partition.mode", (fl.get("partition") or {}).get("mode")),
            ("partition.seed", (fl.get("partition") or {}).get("seed")),
            ("train.batch_size", train.get("batch_size")),
            ("train.eval_max", train.get("eval_max")),
            ("train.max_train_samples", train.get("max_train_samples")),
            ("train.lr", train.get("lr")),
            ("train.max_length", train.get("max_length")),
            ("lora.r", lora.get("r")),
            ("lora.target_modules", lora.get("target_modules")),
            ("model.id", model.get("id")),
            ("communication.trainable_vector_dim", comm.get("trainable_vector_dim")),
            ("communication.upload_per_client_round_mb", comm.get("upload_per_client_round_mb_float32")),
        ]
        parts.append(_md_table(["設定", "値"], cfg_rows))
        parts.append("")
        _note(parts, "A.21")

    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output", default=None, help="default: print to stdout")
    ap.add_argument("--inject-md", default=None, help="replace <!-- RUN_TABLES --> section in this md file")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _ROOT / run_dir

    text = generate_tables(run_dir)

    if args.inject_md:
        md_path = Path(args.inject_md)
        if not md_path.is_absolute():
            md_path = _ROOT / md_path
        content = md_path.read_text(encoding="utf-8")
        start = "<!-- RUN_TABLES -->"
        end = "<!-- /RUN_TABLES -->"
        if start not in content:
            print(f"Marker {start} not found in {md_path}", file=sys.stderr)
            sys.exit(1)
        pre, rest = content.split(start, 1)
        _, post = rest.split(end, 1)
        md_path.write_text(pre + start + "\n" + text + "\n" + end + post, encoding="utf-8")
        print(f"Updated {md_path}")
    elif args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = _ROOT / out
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
