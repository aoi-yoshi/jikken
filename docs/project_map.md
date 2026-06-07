# プロジェクト構成（step00–07）

設定は **config/default.yaml のみ**。実験 run は **04b/05 共通規約**（`src/run_context.py`）。

## フォルダと依存

```
config/default.yaml     … 全 step 共通設定（model / data / train / lora / fl）
scripts/step*.py        … 実行入口（ここから python する）
src/
  config_loader.py      … default.yaml 読込
  run_context.py          … run_id, used_config.yaml, run_meta.json
  logging_utils.py        … RunLogger → jsonl/csv/summary.json
  resource_metrics.py     … GPU/CUDA 計測（environment ブロック）
  vl_model.py             … Qwen2.5-VL + 分類頭
  peft_setup.py           … 双方向 LoRA (client / surrogate)
  dataset_manifest.py     … manifest 読込・層化サンプル
  train_common.py           … device, load_samples
  metrics.py                … Accuracy, F1
  fl_data.py, fl_utils.py   … step05 連合学習
artifacts/
  datasets/               … step02 出力 manifest.jsonl
  checkpoints/            … 学習済み .pt
  runs/<step>/<run_id>/   … ログ（used_config, run_meta, summary, *.jsonl）
docs/                     … FL 手順, GCP 計画, 本ファイル
```

## step 一覧（実験計画 PDF 対応）

| Step | スクリプト | 計画での目的 | run 保存先 |
|------|-----------|-------------|-----------|
| 0 | step00_env_check.py | CUDA / ライブラリ確認 | `runs/step00_env/<run_id>/` |
| 1 | step01_qwen_infer.py | Qwen 推論 | `runs/step01_infer/<run_id>/` |
| 2 | step02_nexar_or_synthetic.py | NEXAR → フレーム → manifest | `runs/step02_data/<run_id>/` + `artifacts/datasets/` |
| 3 | step03_baseline_probe.py | 線形ヘッド単体学習 | `runs/step03_baseline/<run_id>/` |
| 4 | step04_lora_train.py | LoRA 単体学習 | `runs/step04_lora/<run_id>/` |
| 4b | step04b_lora_diff_probe.py | LoRA/勾配/出力差分 + GPU ログ | `runs/step04b_diff_probe/<run_id>/` |
| 5 | step05_flower_server.py + step05_flower_client.py + step05_fl_post_eval.py | Flower 連合学習 | `runs/step05_fl/<run_id>/` |
| 6 | step06_continual_forgetting.py（予定: Avalanche） | 継続学習・forgetting | `runs/step06_continual/` |
| 7 | step07_ablation_modes.py, step07_alt_modes.py | 提案手法 | `runs/step07_ablation_*`, `runs/step07_alt_*` |

## 各 run フォルダの中身（統一）

| ファイル | 内容 |
|---------|------|
| `used_config.yaml` | 実行時の merged 設定（CLI 上書き込み） |
| `run_meta.json` | step, run_id, environment, cli |
| `summary.json` | 最終サマリ |
| `*.jsonl` / `*.csv` | ステップごとの時系列ログ |
