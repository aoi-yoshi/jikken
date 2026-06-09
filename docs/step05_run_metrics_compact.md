# Step 5 FL 実験結果（コンパクト版）— `run_20260607_200402`

Word 用に**重複フィールドを除いた**要約。全フィールド・json パスは [step05_run_metrics_reference.md](step05_run_metrics_reference.md) を参照。

再生成: `python scripts/export_step05_run_compact.py --run-dir c:/Python/実験_修正/artifacts/runs/step05_fl/run_20260607_200402`

---

## 1. 省略した重複（json にはあるがここでは載せない）

| カテゴリ | 省略するもの（完全同値のみ） | 省略しないもの（意味が違う） |
|----------|---------------------------|---------------------------|
| 時間 | `train_time_sec`≡`elapsed_sec`（fit行） | `fit_duration_sec` vs `round_duration_sec`、各フェーズ `*_sec` |
| json 構造 | `round_timing` とトップの二重掲載 | フェーズ別 7 項目はすべて別意味 |
| PyTorch VRAM | `professor_links.link1` のコピー | allocated / reserved / max_allocated / max_reserved は別 |
| Link4 | bytes 版と GB 版の同値 | `inactive_split`・`num_ooms` は Link1 にない |
| smi | トップ `nvidia_smi_used` と snapshot の二重 | 瞬間 snapshot vs fit中 min/max/avg、GPU util vs mem util |
| 精度 | — | 再配布後 / 学習後 / 基盤F / Flower eval はすべて別タイミング |

---

## 2. 実験設定

| 項目 | 値 |
| --- | --- |
| 戦略 | FedAvg |
| ラウンド数 | 3 |
| local_epochs | 2 |
| クライアント数 | 2 |
| 分割 | iid_shuffle (seed=42) |
| モデル | Qwen/Qwen2.5-VL-3B-Instruct |
| LoRA r / targets | r=8, ['q_proj', 'v_proj'] |
| batch / eval件数 | bs=1, eval=20 |
| 学習率 | 0.0001 |
| 通信ベクトル次元 | 1847298 |
| 1ラウンド通信量 [MB] | 7.047 |

## 3. 実行環境

| 項目 | 値 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| VRAM 総量 [GB] | 7.960 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| Python | 3.10.20 |
| AMP | True (bfloat16) |
| 勾配checkpoint | true |
| 入力フレーム数 | 4 |

## 4. データ分割・モデル

| 項目 | 値 |
| --- | --- |
| train プール | 100 |
| eval 件数 | 20 |
| eval ラベル内訳 | {"0": 8, "1": 12} |
| client0 train | 50 (0:26, 1:24) |
| client1 train | 50 (0:26, 1:24) |
| 学習可能パラメータ数 | 1847298 |
| LoRA サイズ [MB] | 7.031 |

### FL 開始前の基盤 F 精度

| client | accuracy | F1 |
| --- | --- | --- |
| 0 | 0.650 | 0.561 |
| 1 | 0.650 | 0.561 |

---

## 5. ラウンド別 — 精度・重み

| R | client | 再配布後 acc | F1 | 学習後 acc | F1 | 基盤F acc | F1 | Flower eval acc | F1 | avg loss | w_div 全体 | w_div LoRA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0.650 | 0.561 | 0.550 | 0.540 | 0.500 | 0.495 | 0.600 | 0.596 | 0.737 | 1.551 | 19.648 |
| 1 | 1 | 0.650 | 0.561 | 0.600 | 0.596 | 0.600 | 0.596 | 0.600 | 0.596 | 0.726 | 1.480 | 19.643 |
| 2 | 0 | 0.600 | 0.596 | 0.550 | 0.520 | 0.550 | 0.540 | 0.600 | 0.596 | 0.558 | 3.278 | 19.906 |
| 2 | 1 | 0.600 | 0.596 | 0.550 | 0.549 | 0.600 | 0.596 | 0.600 | 0.596 | 0.577 | 3.041 | 19.873 |
| 3 | 0 | 0.600 | 0.596 | 0.550 | 0.520 | 0.650 | 0.627 | 0.600 | 0.583 | 0.360 | 4.992 | 20.295 |
| 3 | 1 | 0.600 | 0.596 | 0.650 | 0.649 | 0.500 | 0.500 | 0.600 | 0.583 | 0.436 | 4.806 | 20.256 |

- **再配布後** — グローバル重み受信直後（未学習）
- **学習後** — ローカル学習後（client LoRA + classifier）
- **基盤F** — 凍結 surrogate LoRA で eval（fit 内・学習後）
- **Flower eval** — Flower `evaluate()`（fit 返却後の別イベント）

---

## 6. ラウンド別 — 時間 [分]（秒は Flower eval のみ）

| R | client | fit steps | fit sec/step | ラウンド計 | 重み適用 | 再配布後eval | fit | 学習後eval | 基盤F eval | 重み差分 | overhead | Flower eval [s] | eval sec/件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 100 | 88.46 | 176.10 | 0.22 | 4.31 | 147.44 | 11.25 | 12.86 | 0.01 | 0.02 | 668.58 | 33.43 |
| 1 | 1 | 100 | 50.80 | 100.54 | 0.04 | 5.15 | 84.66 | 5.34 | 5.32 | 0.00 | 0.02 | 335.24 | 16.76 |
| 2 | 0 | 100 | 36.62 | 71.59 | 0.06 | 3.62 | 61.04 | 3.42 | 3.43 | 0.00 | 0.01 | 206.02 | 10.30 |
| 2 | 1 | 100 | 40.21 | 81.03 | 0.08 | 4.65 | 67.02 | 4.64 | 4.63 | 0.00 | 0.01 | 278.15 | 13.91 |
| 3 | 0 | 100 | 32.05 | 63.73 | 0.00 | 3.43 | 53.42 | 3.43 | 3.43 | 0.00 | 0.01 | 247.71 | 12.39 |
| 3 | 1 | 100 | 39.83 | 80.23 | 0.03 | 4.61 | 66.38 | 4.60 | 4.59 | 0.00 | 0.01 | 275.82 | 13.79 |

- **fit steps / sec/step** — ローカル学習（`fit_duration_sec ÷ steps`）
- **ラウンド計** — クライアント `fit()` 全体（`round_duration_sec`）
- **Flower eval** — fit とは別イベント。`eval sec/件` は 20 サンプルあたり

---

## 7. ラウンド別 — GPU（fit 終了時 + fit 中 smi 要約）

ローカル学習（`fit` ループ）の GPU 指標。計測は **3 系統**（json では `professor_links` に格納）:

| 系統 | 略称 | 取得元 | 何を見るか |
|------|------|--------|------------|
| Link1 | PyTorch CUDA | `torch.cuda.memory_*` | **この Python プロセス**が確保した VRAM |
| Link3 | nvidia-smi | `nvidia-smi` コマンド | **GPU 全体**（他プロセス含む）の使用率・VRAM・電力・温度 |
| Link4 | PyTorch 内訳 | `torch.cuda.memory_stats` | アロケータの断片化・OOM 回数など Link1 にない詳細 |

**計測タイミング**: fit 開始直前に `reset_peak_memory_stats()`。学習中は各 step ごとに smi をサンプル（`GpuUtilTracker`）。表の Link1 / smi スナップショット列は **fit 終了直後の 1 点**、`fit GPU%` / `fit smi used` 列は **学習 step 中のサンプル**の min / max / avg。

| R | client | alloc GB | peak alloc GB | reserved GB | peak res GB | smi used GB | smi GPU% | smi mem% | 電力 W | 温度 C | fit GPU% min | max | avg | fit smi used min | max | avg GB | fit mem bus avg% | 断片化 peak GB | OOM回数 | segments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 7.157 | 7.549 | 7.811 | 7.811 | 7.542 | 28.000 | 6.000 | 10.520 | 54.000 | 33.000 | 100.00 | 97.800 | 7.461 | 7.563 | 7.550 | 1.030 | 0.421 | 0 | 285 |
| 1 | 1 | 7.157 | 7.549 | 7.811 | 7.811 | 7.316 | 34.000 | 7.000 | 10.840 | 54.000 | 24.000 | 100.00 | 96.130 | 7.311 | 7.565 | 7.494 | 1.050 | 0.421 | 0 | 285 |
| 2 | 0 | 7.157 | 7.549 | 7.816 | 7.816 | 7.545 | 100.00 | 1.000 | 8.950 | 53.000 | 9.000 | 100.00 | 96.500 | 7.531 | 7.579 | 7.549 | 1.100 | 0.421 | 0 | 288 |
| 2 | 1 | 7.157 | 7.549 | 7.816 | 7.816 | 7.555 | 87.000 | 1.000 | 7.900 | 53.000 | 2.000 | 100.00 | 98.440 | 7.555 | 7.562 | 7.555 | 1.130 | 0.421 | 0 | 288 |
| 3 | 0 | 7.157 | 7.549 | 7.816 | 7.816 | 7.566 | 2.000 | 5.000 | 7.890 | 54.000 | 3.000 | 100.00 | 98.760 | 7.565 | 7.574 | 7.566 | 1.090 | 0.421 | 0 | 288 |
| 3 | 1 | 7.157 | 7.549 | 7.816 | 7.816 | 7.570 | 2.000 | 5.000 | 8.090 | 53.000 | 2.000 | 100.00 | 98.720 | 7.560 | 7.570 | 7.562 | 1.140 | 0.421 | 0 | 288 |

### §7 列の意味（すべて）

| 列名 | json キー（主） | 意味 |
|------|-----------------|------|
| **R** | `server_round` | 連合学習ラウンド番号（1 始まり） |
| **client** | `cid` | クライアント ID（0 / 1） |
| **alloc GB** | `memory_allocated_gb` / Link1 `memory_allocated_gb` | fit **終了時点**で PyTorch が実際にテンソルに割り当てている VRAM [GB]。モデル重み・活性化・勾配など「今使っている」量 |
| **peak alloc GB** | `max_memory_allocated_gb` | fit 開始以降（peak リセット後）の **allocated の最大値** [GB]。OOM 直前にどこまで膨らんだかの目安 |
| **reserved GB** | `memory_reserved_gb` | PyTorch が CUDA から確保した **キャッシュプール**全体 [GB]。allocated より大きいことが多い（解放した VRAM を再利用するため保持） |
| **peak res GB** | `max_memory_reserved_gb` | fit 区間中の reserved の最大値 [GB] |
| **smi used GB** | Link3 `smi_memory_used_gb` | fit **終了瞬間**の nvidia-smi `memory.used` [GB]。**GPU 上の全プロセス合計**（PyTorch 以外も含む） |
| **smi GPU%** | Link3 `smi_utilization_gpu_percent` | fit **終了瞬間**の GPU コア使用率 [%]。0〜100。瞬間値のため fit 中 avg と大きく違うことがある |
| **smi mem%** | Link3 `smi_utilization_memory_percent` | fit **終了瞬間**の **メモリバス（コントローラ）使用率** [%]。VRAM 使用量の割合ではない（転送が活発なときに上がる） |
| **電力 W** | Link3 `smi_power_draw_w` | fit 終了瞬間の GPU 消費電力 [W] |
| **温度 C** | Link3 `smi_temperature_gpu_c` | fit 終了瞬間の GPU ダイ温度 [°C] |
| **fit GPU% min** | `nvidia_smi_timeseries.gpu_util_percent.min` | 学習 step ごとの smi サンプルにおける GPU 使用率の **最小** [%] |
| **fit GPU% max** | `…gpu_util_percent.max` | 同上の **最大** [%]。100 に近いほど GPU を使い切っている |
| **fit GPU% avg** | `…gpu_util_percent.avg` / `avg_gpu_util_percent` | 同上の **平均** [%]。fit 全体の GPU 稼働の代表値 |
| **fit smi used min** | `…nvidia_smi_used_memory_gb.min` | 学習中サンプルでの smi VRAM 使用量 [GB] の最小 |
| **fit smi used max** | `…max` | 学習中の smi VRAM 最大 [GB] |
| **fit smi used avg GB** | `…avg` | 学習中の smi VRAM 平均 [GB] |
| **fit mem bus avg%** | `…memory_util_percent.avg` | 学習中のメモリバス使用率の平均 [%]（smi mem% の時系列版） |
| **断片化 peak GB** | Link4 `inactive_split_gb_all_peak` | fit 区間中の **非アクティブ分割ブロック**のピーク [GB]。大きいほどメモリ断片化が進んでいる（再確保効率低下のサイン） |
| **OOM回数** | Link4 `num_ooms` | fit 区間中に PyTorch アロケータが OOM を記録した回数。0 が正常 |
| **segments** | Link4 `segment.all.current` | 現在の CUDA メモリ **セグメント数**。増えすぎると断片化・オーバーヘッドのサイン |

**読み方のコツ**: VRAM 逼迫は `peak alloc GB` と `smi used GB` を §3 の VRAM 総量（~7.96 GB）と比較。Link1（プロセス内）と smi（GPU 全体）は数 GB 差が出ても正常。
生の smi 時系列（100 点）は jsonl には **min/max/avg のみ**保存。

---

## 8. ラウンド別 — GPU（Flower evaluate 終了時）

FedAvg 集約後、Flower の `evaluate()` RPC で **グローバル重み**を eval セット（20 件）にかけたときの GPU 指標。fit（§7）とは **別イベント・別計測**（eval 直前に再度 `reset_peak_memory_stats()`）。
学習中の smi 時系列（min/max/avg）は **記録しない**（evaluate では `GpuUtilTracker` 未使用）。

| R | client | alloc GB | peak alloc GB | reserved GB | peak res GB | smi used GB | smi GPU% | eval中 GPU avg% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 7.157 | 7.405 | 7.816 | 7.816 | 7.552 | 100.00 | 100.00 |
| 1 | 1 | 7.157 | 7.405 | 7.816 | 7.816 | 7.561 | 100.00 | 100.00 |
| 2 | 0 | 7.157 | 7.405 | 7.816 | 7.816 | 7.563 | 100.00 | 100.00 |
| 2 | 1 | 7.157 | 7.405 | 7.816 | 7.816 | 7.556 | 100.00 | 100.00 |
| 3 | 0 | 7.157 | 7.405 | 7.816 | 7.816 | 7.613 | 100.00 | 100.00 |
| 3 | 1 | 7.157 | 7.405 | 7.816 | 7.816 | 7.552 | 100.00 | 100.00 |

### §8 列の意味（すべて）

| 列名 | json キー（主） | 意味 |
|------|-----------------|------|
| **R** | `server_round` | その evaluate が属するラウンド番号 |
| **client** | `cid` | 評価を実行したクライアント ID |
| **alloc GB** | Link1 `memory_allocated_gb` | evaluate **終了時点**の PyTorch allocated [GB]（推論のみなので fit よりやや小さいことが多い） |
| **peak alloc GB** | Link1 `max_memory_allocated_gb` | evaluate 開始以降（peak リセット後）の allocated 最大 [GB]。20 件バッチ推論のピーク VRAM |
| **reserved GB** | Link1 `memory_reserved_gb` | evaluate 終了時点の PyTorch キャッシュプール [GB] |
| **peak res GB** | Link1 `max_memory_reserved_gb` | evaluate 区間中の reserved 最大 [GB] |
| **smi used GB** | Link3 `smi_memory_used_gb` | evaluate **終了瞬間**の nvidia-smi VRAM 使用量 [GB]（GPU 全体） |
| **smi GPU%** | Link3 `smi_utilization_gpu_percent` | evaluate **終了瞬間**の GPU コア使用率 [%] |
| **eval中 GPU avg%** | `avg_gpu_util_percent` | フィールド名は avg だが、evaluate では時系列サンプルがないため **終了時 1 回の smi 読み取り**（`get_gpu_utilization`）。そのため `smi GPU%` と同値になりやすい |

**§7 との違い**: §7 は学習（forward+backward+optimizer）の負荷、§8 は推論のみ。peak alloc は §8 の方が低い（~7.4 GB vs ~7.55 GB）のが典型。電力・温度・断片化・fit 中時系列は evaluate ログに含めないため表には無し。

---

## 9. サーバー

| R | param_norm | 集約時間 [s] |
| --- | --- | --- |
| 1 | 13.9429 | 0.2415 |
| 2 | 14.2676 | 0.0186 |
| 3 | 14.7894 | 0.0186 |

| R | eval_loss (=1-accuracy) |
| --- | --- |
| 1 | 0.400 |
| 2 | 0.400 |
| 3 | 0.400 |

---

## 10. 最終評価（post_eval / checkpoint）

| accuracy | F1 | eval件数 | 所要 [s] | alloc GB | peak alloc GB | reserved GB | peak res GB | smi used GB | smi GPU% | GPU util [%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.600 | 0.583 | 20 | 82.949 | 7.127 | 7.376 | 7.652 | 7.652 | 7.566 | 98.000 | 98.000 |

checkpoint: `C:\Python\実験_修正\artifacts\checkpoints\step05_fl\run_20260607_200402\step05_fedavg_global.pt`

---

## 11. Profiler（参考・fit 先頭 1 step のみ）

RTX 5050 では `cuda_time_total_ms` が 0 になりがち。CPU 時間のみ参考。

| R | client | cpu_ms | cuda_ms |
| --- | --- | --- | --- |
| 1 | 0 | 150065.6 | 0.0 |
| 1 | 1 | 167844.5 | 0.0 |
| 2 | 0 | 140251.1 | 0.0 |
| 2 | 1 | 145972.5 | 0.0 |
| 3 | 0 | 111857.8 | 0.0 |
| 3 | 1 | 138702.8 | 0.0 |

---

## 12. 関連ファイル

| 用途 | パス |
|------|------|
| 全フィールド版 | [step05_run_metrics_reference.md](step05_run_metrics_reference.md) |
| グラフ | `artifacts/runs/step05_fl/run_20260607_200402/plots/` |
| checkpoint | `artifacts/checkpoints/step05_fl/run_20260607_200402/step05_fedavg_global.pt` |
