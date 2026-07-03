# プロジェクト構成（step00–09）

設定は **config/default.yaml のみ**。実験 run は **04b/05 共通規約**（`src/run_context.py`）。

**現状進捗（2026-07）:** Step 0–5 完了。Step 7 進行中（loss probe + ablation smoke 済、本番 ablation / FL 統合は継続）。Step 6 後回し。詳細は `docs/code_organization.md` §2。

**runs の読み方:** `docs/runs_layout.md`（step_dir / run_id / suffix の意味）。

**scripts/src 整理方針:** `docs/code_organization.md` + `.cursor/rules/code_organization.mdc`。

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
| 6 | step06_continual_forgetting.py（予定: Avalanche） | 継続学習・forgetting（**一時スキップ**） | `runs/step06_continual/` |
| 7 | step07_ablation_modes.py, step07_loss_probe.py, step07_alt_modes.py | 提案手法 Adaptation Consistency | `runs/step07_ablation_<mode>/<run_id>/` |
| 8 | （未整備）plot/export 等 | 評価・比較 | run 内 plots + `docs/` |
| 9 | （未整備）export / 考察用 | 結果整理 | `docs/` |

**Step 7 進行方針（2026-06-17）:** Phase 1 = standalone `step07_ablation_modes.py` で Lpred/Lgrad/Lpost 計算・安定性確認。Phase 2 = `step05_flower_client.py` の `fl.consistency_mode`（`none` | `lpred` | `lpred_grad` | `lpred_post`）で FL 統合。

## ドキュメント索引

**目次（全 docs の説明）:** [README.md](README.md)

| ファイル | 一言 |
|---------|------|
| `project_map.md` | 本ファイル — 全体地図 |
| `runs_layout.md` | run フォルダ名の読み方 |
| `code_organization.md` | 進捗 + scripts/src 整理方針 |
| `terminology.md` | 用語ルール |
| `deferred_experiments.md` | 延期した実験 |
| `fl_experiments.md` | Step 5 実行手順 |
| `step05_*.md` | Step 5 報告・ログ解説 |
| `gcp_migration_plan.md` | GCP 移行 |
| `progress_meeting_20260604.md` | 進捗会資料（2026-06-04） |

## 各 run フォルダの中身（統一）

| ファイル | 内容 |
|---------|------|
| `used_config.yaml` | 実行時の merged 設定（CLI 上書き込み） |
| `run_meta.json` | step, run_id, environment, cli |
| `summary.json` | 最終サマリ |
| `*.jsonl` / `*.csv` | ステップごとの時系列ログ |
