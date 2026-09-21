# Step 7 レパートリー — surrogate 適応情報を用いた基盤モデル更新の方式一覧

**更新:** 2026-07-05
**registry:** [`src/step07/repertoire.py`](../src/step07/repertoire.py)
**実行入口:** [`scripts/step07_flower_server.py`](../scripts/step07_flower_server.py) / [`scripts/step07_flower_client.py`](../scripts/step07_flower_client.py)
**用語:** [`terminology.md`](terminology.md) 準拠

---

## 設計思想（2026-07-05 教授方針）

- **提案手法の系列では FedAvg（クライアント重みの平均）を使わない。** クライアントが送る **適応情報**（logits / 勾配ベクトル / 1-step 適応後 logits）だけで **基盤 LoRA** を更新し、**logits 蒸留（distill）** で各クライアントへ還元する。
- FedAvg / FedProx / FedDF / FedOpt は **比較手法（baseline_\*）**。いずれも **FedAvg → 整合 → distill** の統一ループ（`step07.distill_pipeline=true` 既定）
- **`step07.distill_pipeline: true`（既定）:** ローカル 3B/3B も GCP 7B/3B と同じ経路

## distill パイプライン（ローカル 3B/3B = GCP 7B/3B 同一設定）

| 項目 | 値 |
|------|-----|
| config | `step07.distill_pipeline: true`（既定） |
| レパートリー | `hetero_ok=True` かつ logits 送信（提案 `ac_*` + 比較 `baseline_*`） |
| ③ FedAvg | 実行・記録（Flower プロトコル）。サーバモデルへは載せない |
| ④ 整合 | 送信 logits を teacher に **基盤 LoRA** 更新 |
| ⑤ 配布 | `server_train` 上の基盤 logits → クライアント LoRA へ KL 蒸留 |
| 無効化 | `distill_pipeline: false` で従来の同種専用経路（recompute / copy / baseline 重み配布） |

## 5 軸

| 軸 | 値 |
|----|-----|
| 1. 送信内容 | logits / grad_vectors / post_logits（提案系）、weights（baseline のみ） |
| 2. teacher 構成 | logits_aggregation（equal=FedMD / weighted=FedDF）+ use_replay（前ラウンド server logits） |
| 3. サーバ損失 | lpred / lpred_grad / lpred_post / full |
| 4. サーバ optimizer | adamw / fedadam（FedOpt）/ momentum（FedACG） |
| 5. 配布 | distill（全レパートリー既定。baseline も端末へ FedAvg 重み適用後に distill） |

## 比較手法系（baseline 4 種）

| レパートリー | 文献 | 統一ループでの差分 |
|--------------|------|-------------------|
| `baseline_fedavg` | FedAvg | 端末へ **FedAvg 後の重み**適用 + logits **等平均**で基盤 LoRA 整合 |
| `baseline_fedprox` | FedProx | 同上 + proximal（μ=0.01） |
| `baseline_feddf_pure` | FedDF [3] | 端末へ FedAvg 重み適用 + logits **重み付き** ensemble で基盤 LoRA 整合 |
| `baseline_fedopt` | FedOpt [1] + FedDF | `baseline_feddf` と同型、サーバ optimizer を **FedAdam** |

提案 `ac_lpred_df` との差: baseline は **`uses_client_weights=true`**（端末が FedAvg 後の重みを毎ラウンド適用）。提案は適応情報のみで基盤更新（`weights_used_by_server: false`）。

## 提案手法系（10 種）

| レパートリー | 文献 | 送信 | teacher | サーバ損失 | optimizer | 異種可 |
|--------------|------|------|---------|-----------|-----------|--------|
| `ac_lpred_md` | FedMD [2] | logits | 等平均 | Lpred | AdamW | ○ |
| `ac_lpred_df` | FedDF [3] | logits | サンプル数重み付き softmax 平均 | Lpred | AdamW | ○ |
| `ac_lpred_replay` | FedDF [3] + FCL [8][9] | logits | weighted + 前ラウンド server logits（replay） | Lpred + replay | AdamW | ○ |
| `ac_lgrad_tx` | 研究提案 + FedACG [5] | logits + 勾配 | weighted | Lpred + Lgrad(MSE) | AdamW | ×（3B/3B） |
| `ac_lgrad_cos` | FedOMG [6] | logits + 勾配 | weighted | Lpred + Lgrad(1−cos) | AdamW | ×（3B/3B） |
| `ac_lpost_cfed` | 研究提案 + CFeD [7] | logits + post_logits | weighted | Lpred + Lpost | AdamW | ○ |
| `ac_lpost_replay` | CFeD [7] + FCL [8][9] | logits + post_logits | weighted + replay | Lpred + Lpost + replay | AdamW | ○ |
| `ac_full` | 研究提案全成分 | logits + 勾配 + post_logits | weighted | Lpred + Lgrad + Lpost | AdamW | ×（3B/3B） |
| `ac_lpred_fedopt` | FedOpt [1] + FedDF [3] | logits | weighted | Lpred | **FedAdam**（β1=0.9, β2=0.99, eps=1e-3） | ○ |
| `ac_lpred_acg_momentum` | FedACG [5] + FedDF [3] | logits | weighted | Lpred | **SGD+Nesterov**（momentum=0.85） | ○ |

定義のみ（未実行可能）: `ac_lpred_gen` — FedGen [4] 型。合成整合セット（`data/synthetic`）を用意したら有効化。

## 1 ラウンドの流れ（統一ループ）

```
クライアント fit:
  ⑤ 前ラウンド server logits があれば distill（基盤 LoRA の知識を取り込む）
  ② L_task で クライアント LoRA + 分類ヘッド更新
  ②b fit_config の指示に従い送信物を sidecar に書き出し
      logits            → sidecar/client_logits/rN_cK.json
      勾配ベクトル       → sidecar/client_grads/rN_cK.pt
      1-step 適応後 logits → sidecar/client_post_logits/rN_cK.json
サーバ aggregate_fit:
  （FedAvg は記録のみ。サーバモデルに載せない）
  sidecar 読み込み → teacher 構成（equal / weighted / replay）
  基盤 LoRA を transmitted 損失（Lpred / +Lgrad / +Lpost）+ 指定 optimizer で更新
  server logits を sidecar/server_logits/rN.json に書き出し（次ラウンド ⑤ 用）
  foundation_rN.pt 保存
```

baseline 系はサーバ整合をスキップし、`apply_global_weights=1` でクライアントが FedAvg 後の重みを適用する（Step 5 と同じ配布）。

## 先行研究に合わせた設定

| 項目 | 値 | 根拠 |
|------|-----|------|
| 蒸留損失 | KL(teacher‖student) × T²、T=2.0 | FedMD / FedDF |
| logits 集約 | softmax 確率平均（weighted はサンプル数重み） | FedDF |
| FedAdam | β1=0.9, β2=0.99, eps(τ)=1e-3 | FedOpt 論文値 |
| momentum | 0.85（Nesterov） | FedACG λ=0.85 |
| FedProx μ | 0.01 | Step 5 と同値 |
| クライアント学習データ | **合計 80 件**（2 client × 20/class × 2 class） | FedOpt CIFAR（100 件/台）に近い |
| client eval | **合計 30 件**（15/class・全 client 共通） | ラウンド監視用 |
| server_train（整合セット） | **合計 40 件**（20/class × 2 class） | Lpred / distill |
| server eval | **合計 30 件**（15/class × 2 class） | 基盤 LoRA 更新後の監視 |
| 勾配整合の ℓ | L_task（クライアント・サーバで同一定義） | 研究提案 Lgrad の内側損失 |

## 実行方法

```powershell
# サーバ（ターミナル 1）
python scripts/step07_flower_server.py --repertoire ac_lpred_df --run-id run_xxx --num-clients 2

# クライアント（ターミナル 2/3、同じ run_id）
$env:THESIS_FL_RUN_ID="run_xxx"; $env:THESIS_CLIENT_ID="0"
python scripts/step07_flower_client.py --repertoire ac_lpred_df --server 127.0.0.1:8080
```

- 切替は `--repertoire <name>` または `config/default.yaml` の `step07.repertoire` の 1 箇所。
- パラメータは `consistency.*`（w_pred / w_grad / w_post / inner_lr / temperature）と `step07.*`（distill_steps / distill_lr / replay_weight）で調整。
- 一括比較: `python scripts/run_step07_repertoire_compare.py --repertoires ac_lpred_df baseline_fedavg ...`
- スモーク: `python scripts/run_step07_fl_e2e_smoke.py --repertoire <name>`

## 記録（Step 5 と同水準）

- `run_meta.json` / `summary.json` / `config_snapshot.json` / `used_config.yaml` に **repertoire ブロック**（名前・対応文献・5 軸の値）を記録
- run_dir: `artifacts/runs/step07_fl_<repertoire>/<run_id>/`
- 通信コスト: `fit_round` イベントの `communication_bytes`（logits / 勾配 / post_logits の sidecar バイト数と重みベクトルのバイト数）
- 報告は **結果 + レシピ**（LoRA r/α/modules, LR, batch, フレーム数, データ数, AMP/GC, run_id, `used_config.yaml`）をセットで出す

## 異種構成（7B 基盤モデル + 3B クライアント）での可否

- **可:** logits 系（`ac_lpred_*`, `ac_lpost_*`）— 送信物が出力分布のみで形状非依存
- **不可:** 勾配系（`ac_lgrad_*`, `ac_full`）— LoRA 形状一致が前提。共通部分空間への射影等は将来課題
- **不可:** baseline 系 — 重み配布が形状依存
