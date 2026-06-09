"""
Step 5 FL run の実測値を冗長性を除いて 1 本の Markdown にまとめる（Word 貼り付け用）。

  python scripts/export_step05_run_compact.py --run-dir artifacts/runs/step05_fl/run_20260607_200402
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def _rows_by_event(rows: List[Dict[str, Any]], event: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("event") == event]


def _pl(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("professor_links") or {}


def _fmt(v: Any, *, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:.1f}"
        if abs(v) >= 100:
            return f"{v:.2f}"
        return f"{v:.{digits}f}"
    if isinstance(v, (dict, list)):
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


def _fit_rows(run_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cid in (0, 1):
        for r in _rows_by_event(_load_jsonl(run_dir / f"client_{cid}/fl_client.jsonl"), "fl_round_metrics"):
            out.append(r)
    return sorted(out, key=lambda r: (int(r["server_round"]), int(r["cid"])))


def _eval_map(run_dir: Path) -> Dict[tuple, Dict[str, Any]]:
    m: Dict[tuple, Dict[str, Any]] = {}
    for cid in (0, 1):
        for r in _rows_by_event(_load_jsonl(run_dir / f"client_{cid}/fl_client.jsonl"), "evaluate"):
            m[(int(r["server_round"]), int(r["cid"]))] = r
    return m


def generate_compact(run_dir: Path) -> str:
    run_id = run_dir.name
    parts: List[str] = [
        f"# Step 5 FL 実験結果（コンパクト版）— `{run_id}`",
        "",
        "Word 用に**重複フィールドを除いた**要約。全フィールド・json パスは "
        "[step05_run_metrics_reference.md](step05_run_metrics_reference.md) を参照。",
        "",
        f"再生成: `python scripts/export_step05_run_compact.py --run-dir {run_dir.as_posix()}`",
        "",
        "---",
        "",
        "## 1. 省略した重複（json にはあるがここでは載せない）",
        "",
        "| カテゴリ | 省略するもの（完全同値のみ） | 省略しないもの（意味が違う） |",
        "|----------|---------------------------|---------------------------|",
        "| 時間 | `train_time_sec`≡`elapsed_sec`（fit行） | `fit_duration_sec` vs `round_duration_sec`、各フェーズ `*_sec` |",
        "| json 構造 | `round_timing` とトップの二重掲載 | フェーズ別 7 項目はすべて別意味 |",
        "| PyTorch VRAM | `professor_links.link1` のコピー | allocated / reserved / max_allocated / max_reserved は別 |",
        "| Link4 | bytes 版と GB 版の同値 | `inactive_split`・`num_ooms` は Link1 にない |",
        "| smi | トップ `nvidia_smi_used` と snapshot の二重 | 瞬間 snapshot vs fit中 min/max/avg、GPU util vs mem util |",
        "| 精度 | — | 再配布後 / 学習後 / 基盤F / Flower eval はすべて別タイミング |",
        "",
        "---",
        "",
    ]

    # --- config ---
    er = _rows_by_event(_load_jsonl(run_dir / "fl_server.jsonl"), "experiment_record")
    if not er:
        er = _rows_by_event(_load_jsonl(run_dir / "client_0/fl_client.jsonl"), "experiment_record")
    if er:
        fs = er[0].get("flower_settings") or {}
        fl, train, lora, model, comm = (
            fs.get("fl_yaml") or {},
            fs.get("train_yaml") or {},
            fs.get("lora_yaml") or {},
            fs.get("model_yaml") or {},
            fs.get("communication") or {},
        )
        part = fl.get("partition") or {}
        parts += [
            "## 2. 実験設定",
            "",
            _md_table(
                ["項目", "値"],
                [
                    ("戦略", fl.get("strategy")),
                    ("ラウンド数", fl.get("num_rounds")),
                    ("local_epochs", fl.get("local_epochs")),
                    ("クライアント数", fl.get("num_clients")),
                    ("分割", f"{part.get('mode')} (seed={part.get('seed')})"),
                    ("モデル", model.get("id")),
                    ("LoRA r / targets", f"r={lora.get('r')}, {lora.get('target_modules')}"),
                    ("batch / eval件数", f"bs={train.get('batch_size')}, eval={train.get('eval_max')}"),
                    ("学習率", f"{float(train.get('lr', 0)):.4g}"),
                    ("通信ベクトル次元", comm.get("trainable_vector_dim")),
                    ("1ラウンド通信量 [MB]", comm.get("upload_per_client_round_mb_float32")),
                ],
            ),
            "",
        ]

    # --- environment ---
    sp = run_dir / "client_0/summary.json"
    if sp.is_file():
        env = json.loads(sp.read_text(encoding="utf-8")).get("environment") or {}
        parts += [
            "## 3. 実行環境",
            "",
            _md_table(
                ["項目", "値"],
                [
                    ("GPU", env.get("gpu_name")),
                    ("VRAM 総量 [GB]", env.get("nvidia_smi_total_memory_gb")),
                    ("PyTorch", env.get("torch_version")),
                    ("CUDA", env.get("cuda_version")),
                    ("Python", env.get("python_version")),
                    ("AMP", f"{env.get('amp_enabled')} ({env.get('amp_dtype')})"),
                    ("勾配checkpoint", env.get("gradient_checkpointing")),
                    ("入力フレーム数", env.get("num_frames")),
                ],
            ),
            "",
        ]

    # --- partition + model ---
    p0 = _rows_by_event(_load_jsonl(run_dir / "client_0/fl_client.jsonl"), "partition")
    mr0 = _rows_by_event(_load_jsonl(run_dir / "client_0/fl_client.jsonl"), "model_ready")
    if p0:
        p = p0[0]
        parts += [
            "## 4. データ分割・モデル",
            "",
            _md_table(
                ["項目", "値"],
                [
                    ("train プール", p.get("train_pool_size")),
                    ("eval 件数", p.get("eval_size")),
                    ("eval ラベル内訳", p.get("eval_label_counts")),
                    ("client0 train", "50 (0:26, 1:24)"),
                    ("client1 train", "50 (0:26, 1:24)"),
                    ("学習可能パラメータ数", mr0[0].get("trainable_parameters") if mr0 else "—"),
                    ("LoRA サイズ [MB]", mr0[0].get("adapter_size_mb") if mr0 else "—"),
                ],
            ),
            "",
        ]

    fb_rows = []
    for cid in (0, 1):
        for r in _rows_by_event(_load_jsonl(run_dir / f"client_{cid}/fl_client.jsonl"), "foundation_baseline"):
            fb_rows.append([cid, r.get("foundation_generalization_accuracy"), r.get("foundation_generalization_f1_macro")])
    parts += [
        "### FL 開始前の基盤 F 精度",
        "",
        _md_table(["client", "accuracy", "F1"], fb_rows),
        "",
        "---",
        "",
    ]

    ev_map = _eval_map(run_dir)

    # --- 5a accuracy ---
    acc_rows: List[List[Any]] = []
    for r in _fit_rows(run_dir):
        rnd, cid = int(r["server_round"]), int(r["cid"])
        ev = ev_map.get((rnd, cid), {})
        acc_rows.append([
            rnd, cid,
            r.get("post_redistribution_accuracy"), r.get("post_redistribution_f1_macro"),
            r.get("local_adaptation_accuracy"), r.get("local_adaptation_f1_macro"),
            r.get("foundation_generalization_accuracy"), r.get("foundation_generalization_f1_macro"),
            ev.get("eval_accuracy"), ev.get("eval_f1_macro"),
            r.get("avg_train_loss"),
            r.get("weight_divergence_trainable"), r.get("weight_divergence_lora"),
        ])
    parts += [
        "## 5. ラウンド別 — 精度・重み",
        "",
        _md_table(
            [
                "R", "client",
                "再配布後 acc", "F1", "学習後 acc", "F1", "基盤F acc", "F1",
                "Flower eval acc", "F1", "avg loss", "w_div 全体", "w_div LoRA",
            ],
            acc_rows,
        ),
        "",
        "- **再配布後** — グローバル重み受信直後（未学習）",
        "- **学習後** — ローカル学習後（client LoRA + classifier）",
        "- **基盤F** — 凍結 surrogate LoRA で eval（fit 内・学習後）",
        "- **Flower eval** — Flower `evaluate()`（fit 返却後の別イベント）",
        "",
        "---",
        "",
    ]

    # --- 5b time ---
    tk = [
        "apply_global_params_sec", "post_redistribution_eval_sec", "fit_duration_sec",
        "local_adaptation_eval_sec", "foundation_generalization_eval_sec",
        "weight_divergence_sec", "round_overhead_sec",
    ]
    t_labels = ["重み適用", "再配布後eval", "fit", "学習後eval", "基盤F eval", "重み差分", "overhead"]
    time_rows: List[List[Any]] = []
    for r in _fit_rows(run_dir):
        rnd, cid = int(r["server_round"]), int(r["cid"])
        ev = ev_map.get((rnd, cid), {})
        time_rows.append(
            [rnd, cid, r.get("steps"), r.get("avg_step_time_sec"),
             float(r.get("round_duration_sec", 0)) / 60.0]
            + [float(r.get(k, 0)) / 60.0 for k in tk]
            + [ev.get("elapsed_sec"), ev.get("avg_step_time_sec")]
        )
    parts += [
        "## 6. ラウンド別 — 時間 [分]（秒は Flower eval のみ）",
        "",
        _md_table(
            ["R", "client", "fit steps", "fit sec/step", "ラウンド計"]
            + t_labels
            + ["Flower eval [s]", "eval sec/件"],
            time_rows,
            digits=2,
        ),
        "",
        "- **fit steps / sec/step** — ローカル学習（`fit_duration_sec ÷ steps`）",
        "- **ラウンド計** — クライアント `fit()` 全体（`round_duration_sec`）",
        "- **Flower eval** — fit とは別イベント。`eval sec/件` は 20 サンプルあたり",
        "",
        "---",
        "",
    ]

    # --- 5c GPU fit ---
    def _ts(block: Dict[str, Any], key: str, stat: str) -> Any:
        return (block.get(key) or {}).get(stat)

    gpu_fit_rows: List[List[Any]] = []
    for r in _fit_rows(run_dir):
        rnd, cid = int(r["server_round"]), int(r["cid"])
        l1 = _pl(r).get("link1_pytorch_cuda_semantics") or {}
        s3 = _pl(r).get("link3_nvidia_smi_snapshot") or {}
        ts = _pl(r).get("link3_nvidia_smi_timeseries") or r.get("nvidia_smi_timeseries") or {}
        l4g = (_pl(r).get("link4_pytorch_memory_stats") or {}).get("memory_stats_gb") or {}
        l4r = (_pl(r).get("link4_pytorch_memory_stats") or {}).get("memory_stats_raw") or {}
        gpu_fit_rows.append([
            rnd, cid,
            l1.get("memory_allocated_gb"), l1.get("max_memory_allocated_gb"),
            l1.get("memory_reserved_gb"), l1.get("max_memory_reserved_gb"),
            s3.get("smi_memory_used_gb"), s3.get("smi_utilization_gpu_percent"),
            s3.get("smi_utilization_memory_percent"),
            s3.get("smi_power_draw_w"), s3.get("smi_temperature_gpu_c"),
            _ts(ts, "gpu_util_percent", "min"), _ts(ts, "gpu_util_percent", "max"), _ts(ts, "gpu_util_percent", "avg"),
            _ts(ts, "nvidia_smi_used_memory_gb", "min"), _ts(ts, "nvidia_smi_used_memory_gb", "max"),
            _ts(ts, "nvidia_smi_used_memory_gb", "avg"),
            _ts(ts, "memory_util_percent", "avg"),
            l4g.get("inactive_split_gb_all_peak"), l4r.get("num_ooms"), l4r.get("segment.all.current"),
        ])
    parts += [
        "## 7. ラウンド別 — GPU（fit 終了時 + fit 中 smi 要約）",
        "",
        "ローカル学習（`fit` ループ）の GPU 指標。計測は **3 系統**（json では `professor_links` に格納）:",
        "",
        "| 系統 | 略称 | 取得元 | 何を見るか |",
        "|------|------|--------|------------|",
        "| Link1 | PyTorch CUDA | `torch.cuda.memory_*` | **この Python プロセス**が確保した VRAM |",
        "| Link3 | nvidia-smi | `nvidia-smi` コマンド | **GPU 全体**（他プロセス含む）の使用率・VRAM・電力・温度 |",
        "| Link4 | PyTorch 内訳 | `torch.cuda.memory_stats` | アロケータの断片化・OOM 回数など Link1 にない詳細 |",
        "",
        "**計測タイミング**: fit 開始直前に `reset_peak_memory_stats()`。学習中は各 step ごとに smi をサンプル（`GpuUtilTracker`）。"
        "表の Link1 / smi スナップショット列は **fit 終了直後の 1 点**、"
        "`fit GPU%` / `fit smi used` 列は **学習 step 中のサンプル**の min / max / avg。",
        "",
        _md_table(
            [
                "R", "client",
                "alloc GB", "peak alloc GB", "reserved GB", "peak res GB",
                "smi used GB", "smi GPU%", "smi mem%",
                "電力 W", "温度 C",
                "fit GPU% min", "max", "avg",
                "fit smi used min", "max", "avg GB",
                "fit mem bus avg%",
                "断片化 peak GB", "OOM回数", "segments",
            ],
            gpu_fit_rows,
            digits=3,
        ),
        "",
        "### §7 列の意味（すべて）",
        "",
        "| 列名 | json キー（主） | 意味 |",
        "|------|-----------------|------|",
        "| **R** | `server_round` | 連合学習ラウンド番号（1 始まり） |",
        "| **client** | `cid` | クライアント ID（0 / 1） |",
        "| **alloc GB** | `memory_allocated_gb` / Link1 `memory_allocated_gb` | fit **終了時点**で PyTorch が実際にテンソルに割り当てている VRAM [GB]。"
        "モデル重み・活性化・勾配など「今使っている」量 |",
        "| **peak alloc GB** | `max_memory_allocated_gb` | fit 開始以降（peak リセット後）の **allocated の最大値** [GB]。"
        "OOM 直前にどこまで膨らんだかの目安 |",
        "| **reserved GB** | `memory_reserved_gb` | PyTorch が CUDA から確保した **キャッシュプール**全体 [GB]。"
        "allocated より大きいことが多い（解放した VRAM を再利用するため保持） |",
        "| **peak res GB** | `max_memory_reserved_gb` | fit 区間中の reserved の最大値 [GB] |",
        "| **smi used GB** | Link3 `smi_memory_used_gb` | fit **終了瞬間**の nvidia-smi `memory.used` [GB]。"
        "**GPU 上の全プロセス合計**（PyTorch 以外も含む） |",
        "| **smi GPU%** | Link3 `smi_utilization_gpu_percent` | fit **終了瞬間**の GPU コア使用率 [%]。"
        "0〜100。瞬間値のため fit 中 avg と大きく違うことがある |",
        "| **smi mem%** | Link3 `smi_utilization_memory_percent` | fit **終了瞬間**の **メモリバス（コントローラ）使用率** [%]。"
        "VRAM 使用量の割合ではない（転送が活発なときに上がる） |",
        "| **電力 W** | Link3 `smi_power_draw_w` | fit 終了瞬間の GPU 消費電力 [W] |",
        "| **温度 C** | Link3 `smi_temperature_gpu_c` | fit 終了瞬間の GPU ダイ温度 [°C] |",
        "| **fit GPU% min** | `nvidia_smi_timeseries.gpu_util_percent.min` | 学習 step ごとの smi サンプルにおける GPU 使用率の **最小** [%] |",
        "| **fit GPU% max** | `…gpu_util_percent.max` | 同上の **最大** [%]。100 に近いほど GPU を使い切っている |",
        "| **fit GPU% avg** | `…gpu_util_percent.avg` / `avg_gpu_util_percent` | 同上の **平均** [%]。fit 全体の GPU 稼働の代表値 |",
        "| **fit smi used min** | `…nvidia_smi_used_memory_gb.min` | 学習中サンプルでの smi VRAM 使用量 [GB] の最小 |",
        "| **fit smi used max** | `…max` | 学習中の smi VRAM 最大 [GB] |",
        "| **fit smi used avg GB** | `…avg` | 学習中の smi VRAM 平均 [GB] |",
        "| **fit mem bus avg%** | `…memory_util_percent.avg` | 学習中のメモリバス使用率の平均 [%]（smi mem% の時系列版） |",
        "| **断片化 peak GB** | Link4 `inactive_split_gb_all_peak` | fit 区間中の **非アクティブ分割ブロック**のピーク [GB]。"
        "大きいほどメモリ断片化が進んでいる（再確保効率低下のサイン） |",
        "| **OOM回数** | Link4 `num_ooms` | fit 区間中に PyTorch アロケータが OOM を記録した回数。0 が正常 |",
        "| **segments** | Link4 `segment.all.current` | 現在の CUDA メモリ **セグメント数**。"
        "増えすぎると断片化・オーバーヘッドのサイン |",
        "",
        "**読み方のコツ**: VRAM 逼迫は `peak alloc GB` と `smi used GB` を §3 の VRAM 総量（~7.96 GB）と比較。"
        "Link1（プロセス内）と smi（GPU 全体）は数 GB 差が出ても正常。",
        "生の smi 時系列（100 点）は jsonl には **min/max/avg のみ**保存。",
        "",
        "---",
        "",
    ]

    # --- 5d GPU evaluate ---
    gpu_ev_rows: List[List[Any]] = []
    for r in _fit_rows(run_dir):
        rnd, cid = int(r["server_round"]), int(r["cid"])
        ev = ev_map.get((rnd, cid), {})
        if not ev:
            continue
        l1 = _pl(ev).get("link1_pytorch_cuda_semantics") or {}
        s3 = _pl(ev).get("link3_nvidia_smi_snapshot") or {}
        gpu_ev_rows.append([
            rnd, cid,
            l1.get("memory_allocated_gb"), l1.get("max_memory_allocated_gb"),
            l1.get("memory_reserved_gb"), l1.get("max_memory_reserved_gb"),
            s3.get("smi_memory_used_gb"), s3.get("smi_utilization_gpu_percent"),
            ev.get("avg_gpu_util_percent"),
        ])
    parts += [
        "## 8. ラウンド別 — GPU（Flower evaluate 終了時）",
        "",
        "FedAvg 集約後、Flower の `evaluate()` RPC で **グローバル重み**を eval セット（20 件）にかけたときの GPU 指標。"
        "fit（§7）とは **別イベント・別計測**（eval 直前に再度 `reset_peak_memory_stats()`）。",
        "学習中の smi 時系列（min/max/avg）は **記録しない**（evaluate では `GpuUtilTracker` 未使用）。",
        "",
        _md_table(
            [
                "R", "client",
                "alloc GB", "peak alloc GB", "reserved GB", "peak res GB",
                "smi used GB", "smi GPU%", "eval中 GPU avg%",
            ],
            gpu_ev_rows,
            digits=3,
        ),
        "",
        "### §8 列の意味（すべて）",
        "",
        "| 列名 | json キー（主） | 意味 |",
        "|------|-----------------|------|",
        "| **R** | `server_round` | その evaluate が属するラウンド番号 |",
        "| **client** | `cid` | 評価を実行したクライアント ID |",
        "| **alloc GB** | Link1 `memory_allocated_gb` | evaluate **終了時点**の PyTorch allocated [GB]（推論のみなので fit よりやや小さいことが多い） |",
        "| **peak alloc GB** | Link1 `max_memory_allocated_gb` | evaluate 開始以降（peak リセット後）の allocated 最大 [GB]。"
        "20 件バッチ推論のピーク VRAM |",
        "| **reserved GB** | Link1 `memory_reserved_gb` | evaluate 終了時点の PyTorch キャッシュプール [GB] |",
        "| **peak res GB** | Link1 `max_memory_reserved_gb` | evaluate 区間中の reserved 最大 [GB] |",
        "| **smi used GB** | Link3 `smi_memory_used_gb` | evaluate **終了瞬間**の nvidia-smi VRAM 使用量 [GB]（GPU 全体） |",
        "| **smi GPU%** | Link3 `smi_utilization_gpu_percent` | evaluate **終了瞬間**の GPU コア使用率 [%] |",
        "| **eval中 GPU avg%** | `avg_gpu_util_percent` | フィールド名は avg だが、evaluate では時系列サンプルがないため **終了時 1 回の smi 読み取り**"
        "（`get_gpu_utilization`）。そのため `smi GPU%` と同値になりやすい |",
        "",
        "**§7 との違い**: §7 は学習（forward+backward+optimizer）の負荷、§8 は推論のみ。"
        "peak alloc は §8 の方が低い（~7.4 GB vs ~7.55 GB）のが典型。"
        "電力・温度・断片化・fit 中時系列は evaluate ログに含めないため表には無し。",
        "",
        "---",
        "",
    ]

    # --- server ---
    srv = _load_jsonl(run_dir / "fl_server.jsonl")
    s_rows = []
    for r in srv:
        if r.get("event") == "server_round_aggregate":
            s_rows.append([r.get("server_round"), r.get("param_norm"), r.get("server_aggregate_duration_sec")])
    ev_srv = [r for r in srv if r.get("event") == "server_evaluate_aggregate"]
    parts += [
        "## 9. サーバー",
        "",
        _md_table(["R", "param_norm", "集約時間 [s]"], s_rows, digits=4),
        "",
        _md_table(
            ["R", "eval_loss (=1-accuracy)"],
            [[r.get("server_round"), r.get("eval_loss")] for r in ev_srv],
        ),
        "",
        "---",
        "",
    ]

    # --- post eval ---
    pe = next((r for r in reversed(_load_jsonl(run_dir / "fl_post_eval.jsonl")) if "metrics" in r), None)
    if pe:
        m = pe["metrics"]
        pl_pe = pe.get("professor_links") or {}
        l1pe = pl_pe.get("link1_pytorch_cuda_semantics") or {}
        s3pe = pl_pe.get("link3_nvidia_smi_snapshot") or {}
        parts += [
            "## 10. 最終評価（post_eval / checkpoint）",
            "",
            _md_table(
                [
                    "accuracy", "F1", "eval件数", "所要 [s]",
                    "alloc GB", "peak alloc GB", "reserved GB", "peak res GB",
                    "smi used GB", "smi GPU%", "GPU util [%]",
                ],
                [[
                    m.get("accuracy"), m.get("f1_macro"), pe.get("eval_rows"), pe.get("elapsed_sec"),
                    l1pe.get("memory_allocated_gb") or pe.get("memory_allocated_gb"),
                    l1pe.get("max_memory_allocated_gb") or pe.get("max_memory_allocated_gb"),
                    l1pe.get("memory_reserved_gb") or pe.get("memory_reserved_gb"),
                    l1pe.get("max_memory_reserved_gb") or pe.get("max_memory_reserved_gb"),
                    s3pe.get("smi_memory_used_gb") or pe.get("nvidia_smi_used_memory_gb"),
                    s3pe.get("smi_utilization_gpu_percent"),
                    pe.get("avg_gpu_util_percent"),
                ]],
            ),
            "",
            f"checkpoint: `{pe.get('checkpoint')}`",
            "",
            "---",
            "",
        ]

    # --- profiler one-liner ---
    parts += [
        "## 11. Profiler（参考・fit 先頭 1 step のみ）",
        "",
        "RTX 5050 では `cuda_time_total_ms` が 0 になりがち。CPU 時間のみ参考。",
        "",
    ]
    prof_rows = []
    for r in _fit_rows(run_dir):
        l2 = _pl(r).get("link2_torch_profiler") or {}
        prof_rows.append([
            r["server_round"], r["cid"],
            l2.get("cpu_time_total_ms"), l2.get("cuda_time_total_ms"),
        ])
    parts.append(_md_table(["R", "client", "cpu_ms", "cuda_ms"], prof_rows, digits=1))
    parts.append("")

    parts += [
        "---",
        "",
        "## 12. 関連ファイル",
        "",
        "| 用途 | パス |",
        "|------|------|",
        f"| 全フィールド版 | [step05_run_metrics_reference.md](step05_run_metrics_reference.md) |",
        f"| グラフ | `artifacts/runs/step05_fl/{run_id}/plots/` |",
        f"| checkpoint | `artifacts/checkpoints/step05_fl/{run_id}/step05_fedavg_global.pt` |",
        "",
    ]

    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--output",
        default="docs/step05_run_metrics_compact.md",
        help="output markdown path (default: docs/step05_run_metrics_compact.md)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _ROOT / run_dir
    out = Path(args.output)
    if not out.is_absolute():
        out = _ROOT / out

    text = generate_compact(run_dir)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
