# artifacts/runs の命名と読み方

実験ログは **`artifacts/runs/<step_dir>/[<run_id>/]`** に保存される。  
**step_dir** で「どの Step の実行か」、**run_id** で「いつ・何目的の 1 回か」を区別する。

> 迷ったら: run フォルダ内の **`run_meta.json`**（`cli`・`step`）と **`used_config.yaml`** が正本。

---

## 1. フォルダ階層

```
artifacts/runs/
  <step_dir>/              … Step 種別（下表）
    <run_id>/              … 1 回の実行（多くの Step）
      used_config.yaml     … 実行時設定（レシピ）
      run_meta.json        … step, run_id, cli, environment
      summary.json         … 最終サマリ
      *.jsonl / *.csv      … 時系列ログ
    LATEST_RUN_ID          … ポインタ（Step 5 / Step 7 ablation のみ）
    _e2e_logs/             … E2E スモークのプロセスログ（run 本体の外）
  step04b_batch_probe/     … バッチサイズ probe（run_id なし、probe_result.json のみ）
```

**チェックポイント（runs とは別）:**

```
artifacts/checkpoints/step05_fl/<run_id>/step05_fedavg_global.pt   … FedAvg 後の重み（クライアント LoRA + 分類ヘッド）
artifacts/checkpoints/step05_fl/LATEST_CHECKPOINT                    … 最新へのポインタ
```

---

## 2. step_dir 一覧（Step 番号との対応）

| step_dir | 実験計画 | 起動スクリプト | 備考 |
|----------|---------|---------------|------|
| `step00_env` | Step 0 環境 | `step00_env_check.py` | |
| `step01_infer` | Step 1 推論 | `step01_qwen_infer.py` | |
| `step02_data` | Step 2 データ | `step02_nexar_or_synthetic.py` | manifest は `artifacts/datasets/` |
| `step03_baseline` | Step 3 ベースライン | `step03_baseline_probe.py` | |
| `step04_lora` | Step 4 LoRA | `step04_lora_train.py` | |
| `step04b_diff_probe` | Step 4 補助 | `step04b_lora_diff_probe.py` | LoRA/勾配/出力差分 |
| `step04b_batch_probe` | Step 4 補助 | `probe_batch_size_step04b.py` | **run_id なし** |
| `step05_fl` | Step 5 FL | `step05_flower_server.py` + client | 下記「Step 5 特殊構造」 |
| `step06_continual` | Step 6 継続学習 | `step06_continual_forgetting.py` | **run_id なし**（フラット 1 ディレクトリ） |
| `step06_fl` | （レガシー） | 旧 FL 試行 | **現行スクリプトからは未使用**。無視してよい |
| `step07_loss_probe` | Step 7 診断 | `step07_loss_probe.py` | 1 batch で 3 損失確認 |
| `step07_ablation_lpred` | Step 7 | `step07_ablation_modes.py --mode lpred` | モードごとに step_dir が分かれる |
| `step07_ablation_lpred_grad` | Step 7 | `--mode lpred_grad` | |
| `step07_ablation_lpred_post` | Step 7 | `--mode lpred_post` | |
| `step07_alt_self_distill_<mode>` | Step 7 派生 | `step07_alt_modes.py` | **run_id なし**（mode+suffix がディレクトリ名） |

**Step 8–9（評価・結果整理）** は専用 step_dir 未整備。現状は Step 5/7 の run + `docs/`・export スクリプトで対応。

---

## 3. run_id の読み方

### 基本形

```
run_YYYYMMDD_HHMMSS
run_YYYYMMDD_HHMMSS<suffix>
```

- **日時部分** … 実行開始時刻（ローカル PC 時計）
- **suffix** … CLI の `--run-suffix`、またはスクリプトが付与する目的ラベル

### よく使う suffix

| suffix 例 | 意味 | 付与元 |
|-----------|------|--------|
| `_smoke` | 数 step / 小データの動作確認 | `--run-suffix _smoke` |
| `_short` | 短縮本番（例: 100 step） | `--run-suffix _short` |
| `_progress` | 途中経過用（手動命名） | `--run-suffix _progress` |
| `_ac_lpred` 等 | Step 7 + FL 統合スモーク | `run_step07_fl_smoke.py` が自動付与 |

### 特殊な run_id プレフィックス

| run_id 例 | 意味 |
|-----------|------|
| `run_minimal_YYYYMMDD_HHMMSS` | `run_fl_minimal.py`（最小 FedAvg 1 round） |
| `run_YYYYMMDD_HHMMSS_ac_<mode>` | AC 付き FL E2E スモーク（`consistency_mode`） |

### run_id が **ない** ケース

| 場所 | 理由 |
|------|------|
| `step06_continual/` | 旧実装。上書き型の 1 ディレクトリ |
| `step07_alt_self_distill_*` | mode がディレクトリ名そのもの |
| `step04b_batch_probe/` | probe 結果 1 ファイル |
| `step05_fl/_e2e_logs/` | 補助ログのみ（本体 run は別 run_id） |

---

## 4. Step 5 の特殊構造

```
artifacts/runs/step05_fl/
  LATEST_RUN_ID                    … サーバが書く。クライアントはこれを参照可
  run_20260607_200402/             … サーバ側 run ルート
    used_config.yaml
    run_meta.json
    fl_partition.json
    round_*.jsonl                  … サーバ集約ログ
    client_0/                      … クライアント 0 の fit/eval ログ
    client_1/
  _e2e_logs/
    run_*_server.log
    run_*_client_0.log
```

- **サーバ** が `run_id` を決め、`LATEST_RUN_ID` を更新する
- **クライアント** は `THESIS_FL_RUN_ID` または `LATEST_RUN_ID` で同じ `run_id` に接続
- **FedAvg 後の重み** は `artifacts/checkpoints/step05_fl/<run_id>/step05_fedavg_global.pt`

---

## 5. Step 7 ablation の LATEST_RUN_ID

```
artifacts/runs/step07_ablation_lpred/LATEST_RUN_ID
artifacts/runs/step07_ablation_lpred_grad/LATEST_RUN_ID
artifacts/runs/step07_ablation_lpred_post/LATEST_RUN_ID
```

モードごとに **最新 run_id** が 1 つ。比較するときは run_id か `summary.json` を直接指定する。

---

## 6. run_id 以外の「比較用」ディレクトリ

Step 4b では plot スクリプトが **run_id 形式でない** 比較フォルダを作ることがある:

```
compare_3runs_r16qkvo_r8qkvo_r8qv/
compare_run_20260528_035028_vs_run_20260601_185826/
```

これらは **artifacts/runs/step04b_diff_probe/ 直下** の分析出力。1 回の実験 run ではない。

---

## 7. 主要 run の参照（現時点）

| 用途 | run_id | step_dir |
|------|--------|----------|
| Step 4b 本番ベースライン（r=8） | `run_20260601_185826` | `step04b_diff_probe` |
| Step 5 FL 本番 | `run_20260607_200402` | `step05_fl` |
| Step 7 loss probe | `run_20260617_012022` | `step07_loss_probe` |
| Step 7 ablation smoke（各 mode） | `run_20260617_*_smoke` | `step07_ablation_*` |
| Step 7 + FL スモーク | `run_20260617_050349_ac_lpred` 等 | `step05_fl` |

詳細な数値は各 run の `summary.json` と `used_config.yaml` を参照。

---

## 8. 今後の統一方針（破壊的変更はしない）

新規 run から次を推奨する。**既存 run フォルダのリネームはしない**（参照・論文記録が壊れるため）。

1. **run_id** は `run_YYYYMMDD_HHMMSS` + **`--run-suffix`**（`_smoke` / `_short` / `_full`）
2. **step_dir** は上表に合わせる。新 Step は `stepNN_<短い説明>` 
3. **run_id なし** の新規 Step は作らない（`init_run` / `make_run_id` を使う）
4. Step 7 alt など legacy 命名は触らず、新コードは ablation 規約に合わせる

実装の正本: `src/run_context.py` の `make_run_id()` / `init_run()`。
