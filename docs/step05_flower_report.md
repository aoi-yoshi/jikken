# Step 5 Flower 設定・ログ一覧（進捗会用）

教授への報告用テンプレート。実際の数値は `artifacts/runs/step05_fl/<run_id>/summary.json` を埋める。

---

## 1. 構成（実験計画 Step 5 = 連合学習一式）

**実験計画（案）260422 より:** Step 5 = Flower 連合学習。Step 6 = Avalanche 継続学習（別フェーズ）。

| 役割 | スクリプト | 実験計画上の Step |
|------|-----------|------------------|
| サーバ（FedAvg 集約） | `scripts/step05_flower_server.py` | **Step 5** |
| クライアント 0 / 1 | `scripts/step05_flower_client.py` | **Step 5** |
| 集約後 eval | `scripts/step05_fl_post_eval.py` | **Step 5** |

**Step 6（継続学習）** は別スクリプト: `scripts/step06_continual_forgetting.py`（計画: Avalanche）。Flower とは最初は別実行（計画書 p.5）。

---

## 2. Flower 設定（`config/default.yaml` → `fl:`）

| 項目 | 現在の default | 意味 |
|------|----------------|------|
| `strategy` | **FedAvg** | `FedAvg` または `FedProx`（`fl.fedprox_mu` で μ） |
| `fedprox_mu` | 0.01 | FedProx 時の proximal 係数 μ |
| `num_clients` | 2 | クライアント数 |
| `min_fit_clients` | 2 | 1 ラウンド開始に必要な学習参加数 |
| `min_available_clients` | 2 | 待機する最小クライアント数 |
| `sample_fraction` | 1.0 | 毎ラウンド全クライアント参加 |
| `local_epochs` | 2 | クライアント側ローカル学習 epoch / ラウンド |
| `gpu_serialize` | **true** | **本プロジェクト独自**。1 GPU 複数 client 時、fit/eval の GPU 区間をファイルロックで直列化（Flower 公式ではない） |
| `server` | `127.0.0.1:8080` | gRPC アドレス |
| `grpc_max_message_length` | 512 MB | LoRA+ヘッド送信の上限 |

### データ分割（`fl.partition`）

| `mode` | 説明 |
|--------|------|
| `iid_shuffle` | シャッフルして均等分割（**現在 default**） |
| `label_skew_dirichlet` | クラス偏り（Non-IID） |
| `label_skew_extreme` | client i → label i%2 中心 |
| `key_skew` | scene 等で分割 |

---

## 3. モデル・学習（FL でも step04b と共通）

| 項目 | default |
|------|---------|
| モデル | Qwen2.5-VL-3B-Instruct |
| LoRA r | 8 |
| target_modules | q_proj, v_proj |
| 損失 | L_task（CE）のみ |
| batch_size | 1 |
| 入力フレーム | 4（8 枚保存の末尾 4 枚） |
| lr | 1e-4 |
| max_train_samples | 300 |
| eval_max | 20 |
| 量子化 | なし（bfloat16 AMP + gradient checkpointing） |

---

## 4. 通信量（1 ラウンドあたり）

- 送受信対象: **分類頭 + client LoRA** を 1 本の float32 ベクトルに連結
- 目安: `trainable_vector_dim × 4 bytes` ≈ **数 MB**（`summary.json` → `flower_settings.communication`）
- Adapter のみ: `adapter_size_mb`（client ログの `model_ready`）

---

## 5. どこに何が記録されるか

```
artifacts/runs/step05_fl/<run_id>/
├── used_config.yaml          … 実行時 merged 設定（CLI 上書き込み）
├── run_meta.json             … environment, fl_partition, cli
├── summary.json              … ★ 進捗会はまずここ（flower_settings 含む）
├── fl_server.jsonl           … サーバ: ラウンド集約・eval 集約
├── fl_partition.json         … クライアント別 sample 数・label 分布
├── client_0/
│   ├── fl_client.jsonl       … ★ 精度 + GPU 計測（fit 毎）
│   └── summary.json
└── client_1/
    └── （同上）

artifacts/checkpoints/step05_fedavg_global.pt  … 最終 FedAvg LoRA
```

### ラウンド指標（`fl_client.jsonl` の `fl_round_metrics`）

| キー | 意味 |
|------|------|
| `post_redistribution_accuracy` | グローバル重み受信直後の eval |
| `local_adaptation_accuracy` | ローカル学習後の eval |
| `foundation_generalization_accuracy` | 凍結 surrogate（基盤 F）での eval |
| `weight_divergence_lora` | 基盤 LoRA との L2 |
| `round_duration_sec` | 1 ラウンド全体 [s] |
| `round_timing` | フェーズ別秒数の dict（下表） |
| `apply_global_params_sec` | グローバル重みの適用 |
| `post_redistribution_eval_sec` | 再配布直後 eval |
| `fit_duration_sec` | ローカル学習 |
| `local_adaptation_eval_sec` | 学習後 eval |
| `foundation_generalization_eval_sec` | 基盤 F eval |
| `weight_divergence_sec` | 重み divergence 計算 |
| `round_overhead_sec` | 上記合計との差分（Flower 等） |
| `avg_train_loss` | ローカル CE loss |

### 教授指定 4 リンク — 記録フィールド（完全版）

`run_meta.json` / `summary.json` / 各 jsonl の `professor_links` ブロックに記録。

| リンク | 記録キー | 内容 |
|--------|---------|------|
| **1 PyTorch CUDA semantics** | `link1_pytorch_cuda_semantics` | `memory_allocated_gb`, `memory_reserved_gb`, `max_memory_allocated_gb`, `max_memory_reserved_gb`, `cuda_capability` |
| **2 torch.profiler** | `link2_torch_profiler` | fit 先頭 1 step: `cuda_time_total_ms`, `cpu_time_total_ms`, `top_ops_by_cuda_time`（上位15 op） |
| **3 nvidia-smi** | `link3_nvidia_smi_snapshot` | GPU/mem 利用率, used/free/total memory, power, temperature, clocks, driver/cuda version |
| | `link3_nvidia_smi_timeseries` / `nvidia_smi_timeseries` | 学習中 min/max/avg（gpu_util, mem_util, used_memory_gb） |
| **4 memory_stats** | `link4_pytorch_memory_stats` | `active_bytes`, `inactive_split_bytes`, `segment`, `num_ooms`, `num_alloc_retries` 等 |

run 開始時: `environment.professor_links_baseline` に Link1/3/4 のスナップショット。

### Flower 設定 — 記録フィールド（完全版）

`run_meta.json` / `summary.json` / `fl_*.jsonl` の `experiment_record` イベント → **`flower_settings`**:

| ブロック | 内容 |
|---------|------|
| `flwr_version` | Flower ライブラリ版 |
| `flower_start_server` | `server_address`, `grpc_max_message_length_bytes` |
| `flower_server_config` | `num_rounds`, `round_timeout` |
| `flower_strategy` | FedAvg 全パラメータ（fraction_fit/evaluate, min_*_clients, accept_failures, inplace, aggregation_fn 等） |
| `flower_client` | server 接続先, `local_epochs`, 環境変数 `THESIS_*`, CLI |
| `fl_yaml` / `partition_yaml` / `train_yaml` / `lora_yaml` / `model_yaml` / `data_yaml` / `consistency_yaml` | default.yaml 該当セクション全文 |
| `communication` | trainable ベクトル次元, 1ラウンド通信量 [MB], checkpoint 名 |
| `sampling_formula` | クライアントサンプリング式 |

checkpoint `.pt` の `meta` にも `flower_settings` と `run_id` を保存。

---

## 6. 実行コマンド（コピー用）

```powershell
$PY = C:\Users\aoi7y\miniconda3\envs\flvl\python.exe
$RUN_ID = "run_YYYYMMDD_HHMMSS"

# Terminal A — サーバ
& $PY scripts\step05_flower_server.py --run-id $RUN_ID

# Terminal B — client 0 (GPU)
$env:THESIS_FL_RUN_ID=$RUN_ID; $env:THESIS_CLIENT_ID="0"
& $PY scripts\step05_flower_client.py

# Terminal C — client 1 (CPU 推奨)
$env:THESIS_FL_RUN_ID=$RUN_ID; $env:THESIS_CLIENT_ID="1"; $env:THESIS_FORCE_CPU="1"
& $PY scripts\step05_flower_client.py

# 集約後 eval
& $PY scripts\step05_fl_post_eval.py --run-id $RUN_ID
```

---

## 7. 進捗会で言えること（例）

- Step 4b で LoRA 単体・GPU ログ確認済み
- **Step 5** で Flower FedAvg、2 client、IID（または Non-IID）で N ラウンド
- 各ラウンドで精度 3 種 + 重み乖離 + fit 時間 + ピーク VRAM を JSONL 記録
- 通信量は adapter サイズ / trainable ベクトル次元で見積もり
- GCP 7B/3B は FL ローカル完走後（実験計画どおり）
