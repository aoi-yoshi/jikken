# Flower 実験ガイド（データ拡大・Non-IID）

**進捗会・教授報告（Step5 設定・ログ一覧）:** [step05_flower_report.md](step05_flower_report.md)

設定は **config/default.yaml のみ**。GPU 計測は **src/resource_metrics.py**（step04b / step05 が自動記録）。
## `fl` / `train` の主なキー
| キー | 用途 |
|------|------|
| `train.max_train_samples` | manifest からの層化サンプル上限（step04b / FL 共通） |
| `train.eval_max` | eval 件数 |
| `fl.num_rounds` | FedAvg ラウンド数 |
| `fl.partition.mode` | 分割方式（下表） |
| `fl.partition.label_skew_alpha` | Dirichlet 偏り（小さいほど Non-IID 強） |
| `fl.partition.key_field` | `key_skew` 時の manifest キー（例: `scene`） |

## `fl.partition.mode`

| mode | 説明 |
|------|------|
| `iid_shuffle` | 全データをシャッフルして均等分割 |
| `label_skew_dirichlet` | クラスごとに Dirichlet で偏らせて割当 |
| `label_skew_extreme` | client i が主に label `i % 2` のみ |
| `key_skew` | `key_field` の値ごとに 1 client にまとめる |

## 条件の切替例

**default.yaml を編集する場合**（IID 本番の例）:

```yaml
train:
  max_train_samples: 300
  eval_max: 40
fl:
  num_rounds: 3
  partition:
    mode: iid_shuffle
```

**CLI で上書きする場合**（別 YAML は不要）:

```powershell
& $PY scripts/step05_flower_server.py --run-id $RUN_ID --num-rounds 3 --eval-max 40
& $PY scripts/step05_flower_client.py --num-rounds 3 --eval-max 40
```

Non-IID（Dirichlet）:

```powershell
& $PY scripts/fl_prepare_partitions.py --run-id $RUN_ID --partition-mode label_skew_dirichlet --label-skew-alpha 0.2
& $PY scripts/step05_flower_server.py --run-id $RUN_ID --partition-mode label_skew_dirichlet --label-skew-alpha 0.2
```

スモーク（少数サンプル・1 round）:

```powershell
& $PY scripts/run_fl_e2e_smoke.py
# 内部: --num-rounds 1 --max-train-samples 8 --eval-max 10
```

## 推奨ワークフロー

### 1. 分割の事前確認（学習なし）

```powershell
$PY = C:\Users\aoi7y\miniconda3\envs\flvl\python.exe
$RUN_ID = "run_fl_non_iid_label_001"

& $PY scripts/fl_prepare_partitions.py --run-id $RUN_ID --partition-mode label_skew_dirichlet --label-skew-alpha 0.2
```

各 client の `label_counts` が大きく異なれば Non-IID 成功。

保存先: `artifacts/runs/step05_fl/<run_id>/fl_partition.json`

### 2. Flower 本番実行

```powershell
# Terminal A
& $PY scripts/step05_flower_server.py --run-id $RUN_ID --partition-mode label_skew_dirichlet --label-skew-alpha 0.2

# Terminal B
$env:THESIS_FL_RUN_ID=$RUN_ID; $env:THESIS_CLIENT_ID="0"
& $PY scripts/step05_flower_client.py --partition-mode label_skew_dirichlet --label-skew-alpha 0.2

# Terminal C
$env:THESIS_FL_RUN_ID=$RUN_ID; $env:THESIS_CLIENT_ID="1"; $env:THESIS_FORCE_CPU="1"
& $PY scripts/step05_flower_client.py --partition-mode label_skew_dirichlet --label-skew-alpha 0.2

# 集約後 eval
& $PY scripts/step05_fl_post_eval.py --run-id $RUN_ID
```

### 3. ローカル FedAvg シミュレーション（Flower なし）

```powershell
& $PY scripts/fl_local_fedavg_sim.py --run-id $RUN_ID_local --partition-mode label_skew_dirichlet --label-skew-alpha 0.2
```

8GB GPU で OOM する場合: `--cpu`

## ラウンド指標（step05 / `fl_local_fedavg_sim`）

| 指標 | キー | 意味 |
|------|------|------|
| 再配布直後の精度 | `post_redistribution_accuracy` | グローバル重みを受信した直後、ローカル eval セットで client 適応前に評価 |
| ローカル適応精度 | `local_adaptation_accuracy` | ローカル学習後の eval 精度 |
| 基盤モデル汎化精度 | `foundation_generalization_accuracy` | 凍結 surrogate（step04b の F）での eval 精度 |
| 重み乖離（全体） | `weight_divergence_trainable` | 初期基盤とローカル trainable ベクトルの L2 |
| 重み乖離（LoRA） | `weight_divergence_lora` | 基盤 surrogate LoRA と client LoRA の L2 |
| 1ラウンド所要時間 | `round_duration_sec` | クライアント側の 1 ラウンド全体 |
| フェーズ別時間 | `round_timing` / `*_sec` | `apply_global_params`, `post_redistribution_eval`, `fit`, `local_adaptation_eval`, `foundation_generalization_eval`, `weight_divergence`, `round_overhead` |

ログ: `artifacts/runs/step05_fl/<run_id>/client_<id>/fl_client.jsonl` または `fl_local_sim.jsonl`

### 最小構成（1 件/クライアント × 2 clients × 1 round）

2 クライアントで各 1 学習サンプルが必要なため `max_train_samples=2`, `max_samples_per_client=1`。

```powershell
& $PY scripts/run_fl_minimal.py
# または Flower（3 ターミナル）で同じ CLI を step05 に渡す
```

## 成果物

| ファイル | 内容 |
|---------|------|
| `artifacts/runs/step05_fl/<run_id>/used_config.yaml` | 実行時にコピーした設定（step04b と同様） |
| `artifacts/runs/step05_fl/<run_id>/fl_partition.json` | 分割統計 |
| `artifacts/checkpoints/step05_fedavg_global.pt` | FedAvg 後の重み（クライアント LoRA + 分類ヘッド） |
