# scripts / src の構成方針

実験計画 PDF（Step 0–9）に沿った **理想構成** と、**現状のファイル配置**、**今後の整理ルール** をまとめる。

> **原則:** ファイル数は少ないほどよい。似た機能は同一ファイルにまとめる。  
> **禁止:** 情報欠落を起こす大規模リネーム・フォルダ分割・削除。既存 run_id・import パス・ドキュメント参照を壊さない。

---

## 1. 実験計画 Step 0–9 と現状の対応

| 計画 Step | 目的（計画書） | 状態 | 実行入口（scripts/） | run 保存先 |
|-----------|---------------|------|---------------------|-----------|
| **0** | 環境構築 | ✅ ローカル完了 | `step00_env_check.py` | `step00_env/<run_id>/` |
| **1** | Qwen 推論 | ✅ | `step01_qwen_infer.py` | `step01_infer/<run_id>/` |
| **2** | NEXAR 前処理 | ✅ | `step02_nexar_or_synthetic.py` | `step02_data/<run_id>/` |
| **3** | ベースライン学習 | ✅ | `step03_baseline_probe.py` | `step03_baseline/<run_id>/` |
| **4** | LoRA 学習 | ✅ | `step04_lora_train.py` | `step04_lora/<run_id>/` |
| **4 補助** | LoRA 差分・GPU | ✅ | `step04b_lora_diff_probe.py` | `step04b_diff_probe/<run_id>/` |
| **5** | Flower FL | ✅ **完走・教授確認** | `step05_flower_server.py`, `step05_flower_client.py`, `step05_fl_post_eval.py` | `step05_fl/<run_id>/` |
| **6** | Avalanche 継続学習 | ⏸ **後回し** | `step06_continual_forgetting.py`（PyTorch のみ） | `step06_continual/` |
| **7** | Adaptation Consistency | 🔄 **進行中** | `step07_loss_probe.py`, `step07_ablation_modes.py`, FL: `--consistency-mode` | `step07_*` / `step05_fl/` |
| **8** | 評価・比較 | ⬜ 未着手 | （将来）export / eval スクリプト | 未定 |
| **9** | 結果整理 | ⬜ 未着手 | `plot_*`, `export_step05_*` が一部担当 | `docs/` + run 内 plots |

**教授指示（2026-06）:** Step 6 より **Step 7 を優先** → 3B/3B で Lpred/Lgrad/Lpost → FL 統合 → GCP（7B+3B）。

---

## 2. 現段階の進捗（2026-07 時点）

### 完了

- Step 0–5 パイプライン（ローカル 1 GPU、2 クライアント FL、GPU ファイルロック直列化）
- Step 4b ベースライン: `run_20260601_185826`（r=8, q/v, Acc 0.80 付近）
- Step 5 本番 run: `run_20260607_200402`

### Step 7（進行中）

| 項目 | 状態 |
|------|------|
| `step07_loss_probe.py` | 実行済み `run_20260617_012022` |
| `step07_ablation_modes.py` smoke | 3 mode とも `_smoke` run あり（2026-06-17） |
| ablation `_short` / 本番 epoch | 未完了 or 要確認 |
| FL + AC（`run_step07_fl_smoke.py`） | `run_20260617_*_ac_lpred` 等で試行 |
| GCP 7B+3B | Step 7 ローカル確認後 |

### 延期・スキップ

- Step 6 Avalanche、Non-IID、LoRA r=16 比較、7B/3B 本番 → `docs/deferred_experiments.md`

### 未解決

- PyTorch Profiler の GPU カーネル時間 0 問題
- 報告時の **レシピ + 結果** セット整理（教授要求）

---

## 3. scripts/ の役割分担（現状 30 ファイル）

### A. Step 本体（触らない・1 Step ≒ 1 入口を維持）

```
step00_env_check.py … step07_ablation_modes.py
step04b_lora_diff_probe.py
step05_flower_{server,client}.py, step05_fl_post_eval.py
step06_continual_forgetting.py, step07_loss_probe.py, step07_alt_modes.py
```

### B. Step 5 関連の「短時間テスト用」スクリプト（将来の整理候補）

Step 5 の本番入口は **`step05_flower_server.py` + `step05_flower_client.py`** だけ。  
それ以外に、連合学習（FL）を **短時間・小データで試す** ための script が **別ファイルとして** いくつかある。

| ファイル | 何をする script か |
|---------|-------------------|
| `run_fl_minimal.py` | Flower を使わず、データ 1 件×2 クライアント×1 ラウンドだけ FedAvg する最小テスト |
| `run_fl_e2e_smoke.py` | Flower の server と client を **自動で起動** し、1 ラウンドだけ E2E で回す |
| `run_step07_fl_smoke.py` | 上と同様だが **Step 7 の整合損失（AC）付き** で 1 ラウンド試す |
| `fl_local_fedavg_sim.py` | Flower なしで FedAvg パイプライン全体（IID / Non-IID も可）を sim する |
| `fl_prepare_partitions.py` | 学習前にクライアント分割を preview し `fl_partition.json` を保存する |
| `verify_fl_pipeline.py` | モデルをロードせず、重みの flatten / FedAvg / checkpoint 読み書きだけ検証する |

**「統合」と言っていた意味（将来の話・今は未実施）:**

- 上記 6 つは **役割が近い「FL の動作確認・補助」** なので、今後 **新しい同種 script を増やさない** 方針。
- 将来的に整理するなら、例えば `scripts/fl_tools.py` **1 ファイル** にまとめ、  
  `python scripts/fl_tools.py e2e-smoke` のように **サブコマンドで使い分ける** イメージ。
- **今すぐ 6 ファイルを 1 つにまとめる作業はしていない。** 既存の import パス・`RUN.txt`・実行記録を壊さないため。
- 統合する場合も、旧ファイル名から新ファイルへ **中身を委譲する薄いラッパー** を残し、情報を欠落させない。

→ **新規追加時:** 上記と同種の FL スモークは **新ファイルを作らず** 既存のいずれかに `--mode` / サブコマンドで足す。

### C. 可視化・エクスポート（**統合候補** — 将来 `scripts/export_runs.py`）

| ファイル | 対象 Step |
|---------|----------|
| `plot_step04b_epoch_history.py`, `plot_step04b_compare_runs.py` | 4b |
| `plot_step05_fl_run.py` | 5 |
| `export_step05_run_tables.py`, `export_step05_run_compact.py` | 5 |

→ Step 8–9 用 export は **新 plot/export ファイルを増やさず** 既存に `--step` を追加する方向。

### D. 診断 probe（**統合候補**）

| ファイル | 備考 |
|---------|------|
| `probe_batch_size.py` | 汎用 |
| `probe_batch_size_step04b.py` | 4b 条件 |

→ 1 つの `probe_batch_size.py --profile step04b|default` に統合可能。

### E. 環境・ラッパー（そのまま）

`setup_conda_env.ps1`, `setup_venv.ps1`, `run_step07_alt.ps1`, `_smoke.ps1`

---

## 4. src/ の役割分担（現状 21 ファイル）

| グループ | ファイル | 統合方針 |
|---------|---------|---------|
| **基盤** | `paths`, `config_loader`, `run_context`, `logging_utils`, `resource_metrics` | 分割維持（横断インフラ） |
| **モデル・データ** | `vl_model`, `peft_setup`, `dataset_manifest`, `video_frames`, `train_common`, `metrics` | 分割維持 |
| **Step 4b** | `diff_probe` | Step 7 損失と混ぜない |
| **Step 7** | `adaptation_losses` | 単一ファイル維持 |
| **Step 5 FL** | `fl_utils`, `fl_data`, `fl_strategy`, `fl_flower_config`, `fl_metrics`, `gpu_lock` | **将来** `src/fl/` パッケージ化可。今は import パスを壊さないため維持 |
| **報告** | `experiment_record` | export スクリプトと役割分担を明確に |

→ **新規 src ファイルを増やす前に** 既存ファイルに関数追加できないか検討する。

---

## 5. 理想構成（実験計画 Step 0–9 フォルダ）— 将来目標

```
scripts/
  step00/   … 環境
  step01/   … 推論
  ...
  step09/   … 結果整理
```

**現時点では移行しない。** 理由:

- 30+ スクリプトと `RUN.txt` / `docs/` / VS Code tasks の import パスが一括で変わる
- 既存 run の `run_meta.json` に記録された CLI パスと乖離する
- 情報欠落リスクが高い

**移行する場合の条件（すべて満たしてから）:**

1. 旧パスから新パスへの **互換ラッパー**（薄い `scripts/step05_flower_server.py` → `step05/server.py` 委譲）を残す
2. `RUN.txt` と docs を一括更新
3. git 履歴上の run 参照を維持

---

## 6. Cursor / 開発ルール（要約）

`.cursor/rules/code_organization.mdc` に詳細。要点:

1. **新規ファイルより既存ファイルへの追加**を優先
2. **同種の smoke / probe / plot / export** は 1 ファイルに `--step` や `--mode` で集約
3. **step 本体**（`stepNN_*.py`）は 1 Step 1 入口を維持。補助ロジックは `src/` へ
4. **runs 命名**は `docs/runs_layout.md` に従う。legacy run はリネームしない
5. **大規模フォルダ再編**はユーザー明示指示 + 互換ラッパー必須

---

## 関連ドキュメント

- run 命名: `docs/runs_layout.md`
- プロジェクト地図: `docs/project_map.md`
- 延期実験: `docs/deferred_experiments.md`
- 実行手順: `RUN.txt`
- 先生方針: `notes/advisor/context.md`（ローカルのみ）
