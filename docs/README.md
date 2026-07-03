# docs/ 目次

このフォルダにある Markdown の **役割と中身** を一覧にしたもの。  
迷ったら **まずこのファイル** を開く。

---

## 全体・地図

| ファイル | 何が書いてあるか | いつ読むか |
|---------|-----------------|-----------|
| **[project_map.md](project_map.md)** | プロジェクト全体の構成（`config/` `scripts/` `src/` `artifacts/`）、Step 0–9 とスクリプト・run 保存先の対応表、各 run フォルダの標準ファイル | 初めて repo を見るとき、どの Step の script を動かすか確認するとき |
| **[code_organization.md](code_organization.md)** | 現時点の **研究進捗**（Step 何が完了/進行中か）、`scripts/` と `src/` のファイル一覧と役割、ファイルを増やさないための整理方針 | 進捗を把握したいとき、新しい script を作る前 |
| **[runs_layout.md](runs_layout.md)** | `artifacts/runs/` の **フォルダ名・run_id の読み方**（`step05_fl` とは何か、`_smoke` suffix とは何か、legacy フォルダの扱い） | run フォルダを開いて意味が分からないとき |
| **[terminology.md](terminology.md)** | 論文・報告・会話で使う用語（**基盤 LoRA** / **クライアント LoRA** / **FedAvg 後の重み** / **配布**）。PDF 記号 F/S との対応 | 説明を書くとき、Step 5 と Step 7 の違いを確認するとき |

---

## 研究の方針・スケジュール

| ファイル | 何が書いてあるか | いつ読むか |
|---------|-----------------|-----------|
| **[deferred_experiments.md](deferred_experiments.md)** | **後回しにした実験** の一覧と理由（Step 6 Avalanche、Non-IID、LoRA r=16 比較、GCP 7B+3B など）と再開条件 | なぜ今やっていない実験があるか確認するとき |
| **[gcp_migration_plan.md](gcp_migration_plan.md)** | ローカル FL 完走 **後** の GCP 移行手順（7B サーバ + 3B クライアント、Phase A/B/C） | GCP に移すタイミングと前提条件 |
| **[progress_meeting_20260604.md](progress_meeting_20260604.md)** | 2026-06-04 進捗会用スライド原稿（教授コメント反映、Step 4b 結果、Flower 次ステップ） | 過去の進捗会で何を話したか振り返るとき |

> **先生との最新方針（Slack 等）** は GitHub に載せない **`notes/advisor/context.md`**（ローカルのみ）。Cursor が実験判断に使う。

---

## Step 5（Flower 連合学習）専用

| ファイル | 何が書いてあるか | いつ読むか |
|---------|-----------------|-----------|
| **[fl_experiments.md](fl_experiments.md)** | Flower FL の **実行手順**、`default.yaml` の `fl` / `train` キー、Non-IID 分割、GPU 直列化、ターミナル起動例 | Step 5 を走らせる・設定を変えるとき |
| **[step05_flower_report.md](step05_flower_report.md)** | 教授報告用 **テンプレート**（構成図、設定項目チェックリスト、ログファイル一覧）。数値は run から埋める | Step 5 の結果を先生に報告するとき |
| **[step05_run_metrics_reference.md](step05_run_metrics_reference.md)** | 1 回の FL run に出る **全ログファイル・全変数の意味**（詳細版。例: `run_20260607_200402`） | jsonl のフィールド名が分からないとき |
| **[step05_run_metrics_compact.md](step05_run_metrics_compact.md)** | 上記の **Word 貼り付け用要約**（重複を除いた表だけ） | 報告書・スライドに数値をコピーするとき |

---

## この目次以外の参照先

| 場所 | 内容 |
|------|------|
| **`RUN.txt`**（プロジェクト直下） | ターミナルでコピペする **実行コマンド** 一覧 |
| **`config/default.yaml`** | 全 Step 共通の設定（唯一の config ファイル） |
| **`notes/advisor/context.md`** | 先生方針・会話ログ（ローカルのみ） |
| **`.cursor/rules/`** | Cursor 向けルール（用語、コード整理方針など） |

---

## ファイル間の関係（Step 5 まわり）

```
fl_experiments.md          … どう実行するか（手順）
        ↓ 実行
artifacts/runs/step05_fl/<run_id>/
        ↓ 中身の意味
step05_run_metrics_reference.md  … 詳細
step05_run_metrics_compact.md    … 要約（報告用）
        ↓ 報告の型
step05_flower_report.md          … テンプレート
```

---

## 更新方針

- **新しい docs を増やす前に** この README に 1 行追加できるか検討する
- 実験の「生ログ」は `artifacts/runs/`、docs は **読み方・手順・報告用** に留める
