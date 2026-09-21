# Step 7 設計文書 — 4点整理・Stage 1 設計・関連研究

**更新:** 2026-07-05（レパートリー体系化: FedAvg を比較手法に降格、9 文献の組み合わせで提案 10 方式 + 比較 3 方式。§レパートリー体系化 2026-07-05 参照）  
**用語:** [`terminology.md`](terminology.md) に準拠  
**実装入口:** Flower: [`scripts/step07_flower_server.py`](../scripts/step07_flower_server.py) / [`scripts/step07_flower_client.py`](../scripts/step07_flower_client.py)、monolithic: [`scripts/step07_round_loop.py`](../scripts/step07_round_loop.py)  
**レパートリー一覧:** [`step07_repertoires.md`](step07_repertoires.md)

---

## レパートリー体系化 2026-07-05 — FedAvg を提案手法から除外

教授方針: **FedAvg は提案手法に入らない（入っても比較手法まで）**。

- **提案系（`ac_*`）:** クライアントが送る **適応情報**（logits / 勾配ベクトル / 1-step 適応後 logits）だけで **基盤 LoRA** を更新し、**logits 蒸留** で還元する。クライアント重みの平均は行わない（Flower プロトコル上重みは流れるが、サーバは使わない — `weights_used_by_server: false` を記録）
- **比較系（`baseline_*`）:** FedAvg / FedProx / FedDF そのもの（重み配布）。提案との差分を示す対照
- 9 文献の組み合わせで **提案 10 + 比較 3** を [`src/step07/repertoire.py`](../src/step07/repertoire.py) に登録。切替は `--repertoire` / `step07.repertoire` の 1 箇所。詳細は [`step07_repertoires.md`](step07_repertoires.md)
- これ以前の「FedAvg 後の重み = 参照 adapter」設計（§再設計 2026-07-04）は monolithic `step07_round_loop.py` に残るが、**Flower 経路の提案系では使わない**

---

## 研究の中心概念

**surrogate 適応情報を用いた基盤モデル更新** — エッジの局所適応（L_task + LoRA）から得た情報を、クラウドの基盤 LoRA へ循環的に還元する。Adaptation Consistency（Lpred / Lgrad / Lpost）はその具体的手法。

---

## Ablation 段階（教授指示）

| Stage | 内容 | 状態 |
|-------|------|------|
| **0** | L_task（クライアント）+ サーバ整合ループ。LoRA 重みのみ送信。teacher = サーバ上で FedAvg 後の重みを再 forward | **完走**（スモーク 4R + 本番 lpred 3R） |
| **1** | **Lpred + logits 送信**（FedDistillation 系）。teacher = クライアントが送った出力分布 | **実装済み**（`--stage 1`） |
| **2** | Lpred + **Lgrad**（勾配整合） | 未着手 |
| **3** | Lpred + **Lpost**（Step 6 継続学習と接続） | 未着手 |
| **4** | Lpred + Lgrad + Lpost 全部 | 未着手 |

---

## 教授 4 点整理（Stage 別）

### ① クライアント側の更新

| Stage | 更新対象 | 損失 | コード |
|-------|---------|------|--------|
| 0〜4 | **クライアント LoRA + 分類ヘッド** | **L_task のみ** | `train_client_l_task()` in `step07_round_loop.py` |

エッジ制約のため、クライアントで Lpred/Lgrad/Lpost を合成しない。整合損失はサーバ側のみ。

### ② クライアント → サーバ送信

| Stage | 送信内容 | 備考 |
|-------|---------|------|
| **0** | **クライアント LoRA + 分類ヘッド**（重みベクトル） | Flower: `trainable_state_vector()`。monolithic: メモリ内受け渡し |
| **1** | 上記 + **補助情報: logits** | 下記 Stage 1 設計参照 |
| **2** | 上記 + **勾配ベクトル**（または LoRA 更新方向） | FedACG / FedOMG 系 |
| **3** | 上記 + **1-step 適応後出力** | Lpost 用。Step 6 継続学習設定と接続 |
| **4** | 全部 | — |

### ③ サーバ側計算・基盤 LoRA 更新

| Stage | FedAvg の役割 | サーバ最適化 |
|-------|--------------|-------------|
| **0** | 参照 adapter 作成（1 台なら送信重み = FedAvg 後の重み） | 固定 **FedAvg 後の重み（クライアント LoRA）** を teacher としてサーバデータ上で **再 forward** → **Lpred のみ**（`consistency.w_task=0.0`）→ **基盤 LoRA** 更新 |
| **1** | **参照 adapter 作成に留める**（最終更新ではない） | **送信 logits を teacher として注入** → **Lpred のみ** → **基盤 LoRA** 更新（teacher 側の再 forward 不要） |
| **2** | 同上 | + Lgrad |
| **3** | 同上 | + Lpost |
| **4** | 同上 | 3 成分全部 |

FedAvg は **基盤 LoRA の最終更新手段ではない**。参照 adapter 作成・補助情報集約の候補。最終更新は合成損失に基づく **サーバ側最適化**。

> **2026-07-04 変更:** サーバ整合の損失から **L_task を除外**した（`consistency.w_task: 0.0` がデフォルト。ablation 用に > 0 を指定可能）。根拠は §再設計 2026-07-04 の先行研究比較。2026-07-03 以前の run（`run_20260701_120420` 等）は `total = L_task + w_pred·Lpred` で学習している点に注意。

### ④ 再配布

**2026-07-04 実装済み。** `step07_round_loop.py` の各ラウンド末尾（⑤）で、更新した基盤 LoRA を各クライアント状態へ書き込む。方式は `step07.distribution`（CLI `--distribution`）で切替:

| 方式 | 内容 | 使える構成 |
|------|------|-----------|
| **copy** | 基盤 LoRA の重み + サーバ側分類ヘッドを、各クライアントの **クライアント LoRA + 分類ヘッド** に書き込む | 同種（3B/3B）のみ。形状不一致なら即エラー |
| **distill** | サーバが `server_train`（公開整合セット）上で基盤 LoRA の logits を計算・配布 → クライアントは **FedAvg 後の重みを起点に** その logits へ KL 蒸留（`distill_adapter_from_logits`） | 同種・**異種（7B/3B）両方**。重みを渡さないため LoRA 形状に依存しない（FedMD Digest / FedDF プロトタイプ蒸留と同型） |

---

## Stage 1 設計判断（確定）

### 整合用サンプル集合

- **集合:** `server_train`（`data_splits.json` の `server_train_ids`）
- **理由:** FedMD 型の **public alignment set**。クライアント・サーバ双方が同じ ID のサンプルにアクセスできる（生データはローカル保持、ID のみ共有でも可）
- **タイミング:** クライアントが L_task で **クライアント LoRA** を更新した **直後**、eval mode で forward

### logits 形式

```json
{
  "round": 1,
  "sample_ids": ["id1", "id2"],
  "logits": [[l0_c0, l0_c1], [l1_c0, l1_c1]]
}
```

- ファイル: `artifacts/runs/step07_round_<mode>/<run_id>/client_logits_r<N>.json`
- 参照実装: `src/diff_probe.py` の `probe_logits()`

### 集約（複数クライアント時）

- sample_id ごとに softmax 確率を **重み付き平均**（FedDF 型）
- Stage 1 プロトタイプ（1 クライアント）: 集約不要、そのまま使用

### teacher 注入（サーバ Lpred）

- **Stage 0:** `AdaptationLosses.lpred_only()` — teacher = 固定 **FedAvg 後の重み（クライアント LoRA）** の再 forward
- **Stage 1:** `AdaptationLosses.lpred_from_teacher_logits()` — teacher = 送信 logits テンソル
- student = **基盤 LoRA** の forward（trainable）
- 切替: CLI `--stage 0|1` または `--teacher-source recompute|transmitted`
- config 案: `consistency.teacher_source: recompute | transmitted`

### 1 ラウンドの流れ（2026-07-04 再設計後）

```
② 各クライアント: L_task のみ → クライアント LoRA + 分類ヘッド更新（fresh AdamW / round）
②b Stage 1: server_train 上で forward → logits 送信（複数クライアントは softmax 平均で集約）
③ FedAvg: クライアント LoRA + 分類ヘッドを重み平均（num_clients=1 は素通し）
④ サーバ: teacher（再 forward or 送信 logits）に対し整合損失のみ（w_task=0）→ 基盤 LoRA 更新
⑤ 配布: 更新した基盤 LoRA を各クライアントへ（copy: 重み書き込み / distill: logits 蒸留）
   → 次ラウンドの ② は配布された状態から開始（循環が閉じる）
```

従来実装は ⑤ が存在せず、クライアント LoRA が前ラウンドの続きを学習するだけで、サーバの整合結果がクライアントに還元されていなかった（2026-07-04 修正）。

---

## 再設計 2026-07-04 — L_task 除外・再配布・複数クライアント・異種モデル

### 変更点サマリ

| # | 変更 | 実装 |
|---|------|------|
| 1 | サーバ整合（基盤 LoRA 更新）から **L_task を除外**（`consistency.w_task: 0.0`） | `src/adaptation_losses.py` 全 4 モード |
| 2 | **⑤ 配布** を実装（copy / distill） | `distribute_foundation_to_clients` in `step07_round_loop.py`, `distill_adapter_from_logits` in `src/fl_logits.py`, `copy_adapter_weights` in `src/peft_setup.py` |
| 3 | **複数クライアント**（`--num-clients` / `step07.num_clients`）+ クライアント LoRA + 分類ヘッドの **FedAvg（重み平均）** | `fedavg_client_states` in `step07_round_loop.py` |
| 4 | **異種モデル構成**（`model.server_id` / `model.client_id`）— 7B 基盤モデル + 3B クライアントを config だけで指定 | `step07_round_loop.py`（異種時は 2 モデル + distill + stage 1 + lpred を強制） |

### 根拠 1: サーバ側損失に L_task（教師あり損失）を入れない

紹介論文 9 文献のサーバ側更新を精査した結果:

| 文献 | サーバ側の更新 | ラベル付き損失の使用 |
|------|---------------|---------------------|
| FedMD [2] | なし（logits の単純平均のみ） | なし |
| FedDF [3] | アンサンブル logits への **KL 蒸留のみ**（アンラベルデータ） | **なし**（明示的に unlabeled） |
| FedGen [4] | 生成器の学習に CE を使用（モデル本体はクライアント側更新） | 生成器のみ（例外） |
| FedACG [5] | momentum 集約 | なし |
| FedOMG [6] | 勾配マッチング（data-free） | なし |
| FedOpt [1] | pseudo-gradient への適応 optimizer | なし |
| CFeD [7] | Server Distillation = **蒸留損失のみ** | なし |

研究提案 PDF の最終目的関数も `L = λ₁Lpred + λ₂Lrepr + λ₃Lgrad + λ₄Lpost` で **L_task の単独項はない**（ℓ は Lgrad / Lpost の内部の勾配・1-step 更新の計算にのみ登場し、この部分は実装でも維持）。従来実装の `total = L_task + w_pred·Lpred` は L_task が主成分になっており、基盤 LoRA が「整合」ではなく「サーバデータでの教師あり学習」で動いてしまう。→ `w_task=0.0` をデフォルトとし、L_task の寄与は ablation としてのみ許可。

### 根拠 2: 異種モデル（7B/3B）の配布は蒸留で行う

- FedACG / FedOMG / FedOpt は **同一モデル前提**で異種対応なし。
- 異種対応の先行例は **FedMD**（重みを送らず、公開データ上の logits のみで通信・各クライアントが自モデルで Digest）と **FedDF**（アーキテクチャ別プロトタイプを個別に蒸留して配布）。
- 本実装の `distribution: distill` はこの 2 つと同型: サーバは **`server_train` 上の基盤 LoRA logits** を配布し、クライアントは FedAvg 後の重みを起点に KL で追従する。**重み・LoRA 形状を一切共有しない**ため、基盤モデル 7B（hidden 3584）とクライアント 3B（hidden 2048）で LoRA / 分類ヘッドの形状が違ってもエラーにならない。
- GCP 移行手順: `config/default.yaml` で `model.server_id: "Qwen/Qwen2.5-VL-7B-Instruct"` を設定するだけ。スクリプトが自動で 2 モデル構成 + distill + stage 1 を強制する。

### 根拠 3: データ規模（先行研究比較）

| 文献 | クライアント数（参加率） | ラウンド | クライアントあたり学習データ | サーバ/公開・蒸留データ |
|------|------------------------|---------|---------------------------|------------------------|
| FedMD [2] | 10 | 未記載 | クラスあたり 3〜約 20 件 | 公開 5,000 件/R（ラベル付き） |
| FedDF [3] | 20〜150（0.1〜0.8） | 10〜100 | 約 2,500 件（CIFAR/20台） | アンラベル外部データ（**1% でも有効**と報告） |
| FedGen [4] | 20（50%） | 200 | 未記載（Dirichlet 分割） | なし（生成器） |
| FedACG [5] | 100〜500（2〜5%) | 500〜1,000 | 約 100〜500 件 | なし |
| FedOpt [1] | 500〜342k（10 台/R） | 1,500〜4,000 | **100 件固定**（CIFAR） | なし |
| CFeD [7] | 100（10%） | 20/タスク | 2 shard（surrogate 約 2,300 件） | クライアント/サーバ各 2 shard（アンラベル + 疑似ラベル） |
| **本研究（現設定）** | **2（全参加）** | **3** | **全 client 合計 80 件**（2 台 × 20/class × 2 class） | **server_train 40 件**（20/class × 2） |

**評価（2 class 合計）:** client eval **30**（15/class・全 client 共通）、server eval **30**（15/class）。

**結論:**

- クライアント学習 80 件（合計）は FedOpt CIFAR（100 件/台）に近い。**①②③ 比較 ablation** 用（2026-07-06 更新）。
- サーバ整合セット 40 件は FedMD の 5,000 件/R より小さいが、FedDF の小規模蒸留報告と 2 クラス分類を踏まえ **40〜100 件** が妥当なレンジ。クラスあたり需要 90 件（train+eval+server）で `max_videos_per_class=150` に余裕あり。
- クライアント数（先行研究 10〜500）とラウンド数（10〜4,000）は本研究のローカル環境では 1〜2 台 / 3〜5R に留めるが、**両方とも CLI / config の数値変更だけでスケール可能**（`--num-clients`, `--num-rounds`, `step07.num_clients`）。

### 将来課題（異種構成）

- **Lgrad（Stage 2）:** 7B/3B では LoRA パラメータ数・形状が異なり、勾配ベクトル同士の MSE が定義できない。コサイン類似度・低次元射影・共通部分空間への写像などが候補（FedOMG の on-server matching が参考）。現状は異種時に `--mode lpred` のみ許可。
- **Lpost（Stage 3）:** 1-step 適応のシミュレートは同一モデル上の 2 アダプタ前提。異種では蒸留後のクライアント LoRA を用いた近似が必要。
- **teacher 再 forward（Stage 0）:** FedAvg 後の重みをサーバモデルに書き込めないため異種では不可（logits 送信 = Stage 1 経路のみ）。

---

## 関連研究（比較 5 軸 × 9 文献）

### 比較 5 軸

| 軸 | 本研究 Stage 1 |
|----|---------------|
| (1) 送信内容 | LoRA 差分 + **logits（2 クラス出力分布）** |
| (2) 集約 | sample_id ごとの softmax 平均（複数クライアント時） |
| (3) サーバ最適化 | **FedAvg とは別** — Lpred + L_task 合成損失で **基盤 LoRA** を直接更新 |
| (4) FedAvg の役割 | 参照 adapter 作成（最終更新ではない） |
| (5) 補助情報の使い方 | teacher 分布として KL（温度 T=2.0） |

### 文献マッピング

#### [1] FedOpt — Reddi et al., Adaptive Federated Optimization, ICLR 2021

サーバ側 optimizer（FedAdam / FedYogi）でクライアント更新を集約後に再最適化する系統。本研究では Stage 1 は AdamW を継続し、将来 ablation で FedOpt 型サーバ optimizer と比較候補。**送信内容は重み**であり logits ではない点が Stage 1 と異なる。

#### [2] FedMD — Li & Wang, arXiv:1910.03581, 2019

**Stage 1 の直接参考。** 各クライアントが **public データセット**上で logits を計算しサーバへ送信。サーバが logits を集約し、各クライアントが **蒸留損失**でローカル更新。本研究との差: FedMD はクライアント側蒸留、本研究は **サーバ側で基盤 LoRA を更新**し再配布。

#### [3] FedDF — Lin et al., NeurIPS 2020

複数クライアントの logits を **平均**し、サーバが **アンラベルデータ**上で teacher ensemble として蒸留。集約方式（softmax 平均）が Stage 1 の複数クライアント拡張に参考。**サーバ側 distillation** の位置づけが本研究 Stage 1 に近い。

#### [4] FedGen — Zhu et al., ICML 2021

生成モデルで **疑似データ**を作り logits 通信を補完。送信内容・集約の代替手段として参照。本研究 Stage 1 では生成器は使わず、**整合用サンプル集合（server_train）** で直接 logits を送る。

#### [5] FedACG — Kim et al., CVPR 2024

**Stage 2 参考。** クライアント勾配をサーバでマッチングし、グローバル更新方向を整合。**Lgrad** の理論的背景。Stage 1 では勾配送信なし。

#### [6] On-server Matching Gradient — ICLR 2025

サーバ上で勾配マッチングを行う FedOMG 系。**Lgrad** の別実装候補。Stage 2 で「勾配送信 vs サーバ再計算」の選択肢として参照。

#### [7] CFeD — Gao et al., IJCAI 2022

Federated Continual Learning。クライアント間で **知識を保持しながら**新タスクを学習。**Lpost / Step 6** の接続点。1-step 適応後の出力整合が忘却抑制に対応。

#### [8] Federated Continual Learning: A Comprehensive Survey, arXiv:2606.11272, 2026

FCL 全体像。**Stage 3** で Lpost ablation を継続学習設定（タスク順・忘却計測）で評価する際の枠組み。

#### [9] A Contemporary Survey of Federated Continual Learning, 2025

同上。Non-IID・ドリフト・忘却の baseline 整理。Step 6 Avalanche 再開時の比較対象。

### 本研究の位置づけ（1 段落）

本研究は FedDistillation 系（FedMD / FedDF）の **logits 送信・集約** を Lpred に取り込みつつ、Gradient matching 系（FedACG）の **Lgrad** と FCL 系（CFeD）の **Lpost** を **同一の Adaptation Consistency 枠組み**で ablation する。FedAvg は参照 adapter 作成に留め、**基盤 LoRA の最終更新はサーバ合成損失**で行う点が、従来 FL（FedAvg = グローバルモデル更新）との本質的差分。

---

## アーキテクチャ（2026-07-05 — 段階プロファイル・集約差し替え）

### Step 5（Flower）と Step 7 の関係 — 「未統合」の意味

| | Step 5 Flower 経路 | Step 7 メイン（`step07_round_loop.py`） |
|---|---------------------|----------------------------------------|
| **実行形態** | サーバ + クライアントが **別プロセス**（gRPC）。`step05_flower_server.py` / `step05_flower_client.py` | **1 プロセス・1 GPU** で全クライアントを直列シミュレーション |
| **集約** | `fl_strategy.py`（FedAvg / FedProx）→ **`step05_fedavg_global.pt`** | `src/step07/aggregation.py`（fedavg 実装済み、fedopt/fedacg は stub） |
| **配布** | **FedAvg 後の重み** → **クライアント LoRA + 分類ヘッド** | **更新した基盤 LoRA** → copy / distill（⑤） |
| **サーバ整合** | なし（`fl.consistency_mode=none` が通常） | **AdaptationLosses** で **基盤 LoRA** を更新（④） |
| **連合学習は動くか** | **はい** — Step 5 で配管確認済み | Step 7 は **FL 配管の上に載せる研究ロジック**（別入口） |

**「未統合」= Step 7 の 2 段階ループ（④整合 + ⑤基盤 LoRA 配布）が Flower 経由ではまだ動いていない** という意味。**連合学習そのものが未実装という意味ではない。**

将来: Flower クライアントが logits / 勾配を送り、Flower サーバが AdaptationLosses + aggregation + 配布を実行する統合（context.md の TODO）。

### モジュール構成

```
scripts/step07_round_loop.py     … オーケストレーション（ラウンドループ）
src/step07/stage_profile.py      … 段階レシピ（送信・teacher・整合・集約・配布）
src/step07/aggregation.py        … ③ 集約（fedavg 実装 / fedopt・fedacg stub）
src/step07/client_state.py       … クライアント LoRA + ヘッドの退避・復元
src/step07/distribution.py       … ⑤ 配布（copy / distill）
src/adaptation_losses.py         … ④ 整合損失（AdaptationLosses クラス）
src/fl_logits.py                 … logits 送受信・蒸留
```

**段階の増減:** `src/step07/stage_profile.py` の `STAGE_PROFILES` にエントリを追加・削除。`config/default.yaml` の `step07.stage_profile` または CLI `--stage-profile` で選択。

### 送信内容と整合成分（Lpred / Lgrad / Lpost）

| 成分 | 先行研究の teacher / 信号 | クライアント → サーバ送信 | サーバ側計算 |
|------|---------------------------|---------------------------|--------------|
| **Lpred** | FedMD / FedDF: 公開データ上の **logits** + KL | **`logits`**（server_train 上） | `AdaptationLosses.lpred_*`（FedDF 型 KL × T²） |
| **Lgrad** | FedACG / FedOMG: **更新方向・勾配** | **`grad_vectors`** または model delta（未実装） | サーバ再計算 or 送信勾配の MSE |
| **Lpost** | 研究提案 / CFeD: **1-step 適応後出力** | **`post_logits`**（未実装） | inner step シミュレーション + KL |

**Lgrad / Lpost を有効にする場合、logits 以外の送信が必要** — 認識は正しい。現状 **logits のみ実装済み**（Stage 1 = `stage1_lpred_transmitted`）。

### Lpred と先行研究の対応（なるべく同じに）

| 項目 | FedMD / FedDF | 本実装 Stage 1 |
|------|---------------|----------------|
| 公開整合セット | ラベル付き or アンラベル外部データ | **`server_train`**（ID 共有、生データはローカル） |
| 通信 | クライアント logits | 同左 |
| 集約 | 平均（FedMD: pre-softmax / FedDF: softmax 平均） | **`aggregate_client_logits`**（softmax 平均 = FedDF 型） |
| 蒸留損失 | KL(teacher \|\| student) × T² | **`kl_distillation`** 同形式 |
| 更新対象 | FedMD: **クライアント** / FedDF: **サーバプロトタイプ** | **基盤 LoRA**（本研究の差分） |
| サーバ L_task | なし | **`w_task=0`** |

Stage 0（teacher 再 forward）は **プロトタイプ簡略版** で、FedMD そのものではない。先行研究に揃えるなら **`stage_profile=stage1_lpred_transmitted`** を本番デフォルトにする。

---

## 実装対応表

| 設計要素 | ファイル | 関数 / キー |
|---------|---------|------------|
| Stage 0 ループ | `scripts/step07_round_loop.py` | `train_client_l_task`, `train_server_consistency` |
| Stage 1 logits  export | 同上 | `export_client_logits` |
| Stage 1 teacher 注入 | `src/adaptation_losses.py` | `lpred_from_teacher_logits` |
| logits 集約 | `src/fl_logits.py` | `aggregate_client_logits` |
| teacher 切替 | CLI `--stage` / `--teacher-source` | `consistency.teacher_source` |
| データ分割（クライアント別） | `step07_round_loop.py` | `split_step07_data_pools`, `data_splits.json` |
| サーバ損失の L_task 重み | `src/adaptation_losses.py` | `consistency.w_task`（既定 0.0） |
| FedAvg（重み平均） | `src/step07/aggregation.py` | `aggregate_client_states(method=fedavg)` |
| 集約 ablation | 同上 | `fedopt` / `fedacg` / `fedomg`（stub） |
| 段階レシピ | `src/step07/stage_profile.py` | `step07.stage_profile` / `--stage-profile` |
| ⑤ 配布（copy） | `src/peft_setup.py` | `copy_adapter_weights` |
| ⑤ 配布（distill） | `src/fl_logits.py` | `distill_adapter_from_logits`, `step07.distill_steps/distill_lr` |
| クライアント数 | CLI `--num-clients` | `step07.num_clients` |
| 異種モデル ID | `config/default.yaml` | `model.server_id` / `model.client_id` |

---

## 検証計画

| run | 条件 | 目的 |
|-----|------|------|
| スモーク Stage 1 | 最小データ、`--run-suffix _stage1_smoke --stage 1` | ループ完走 ✅ |
| 本番 Stage 0 vs 1 | 同一 seed・3R・本番データ件数 | [`step07_stage_compare_results.md`](step07_stage_compare_results.md) ✅ |

報告: **結果 + レシピ**（LoRA r/α/modules, LR, batch, フレーム数, データ数, AMP/GC, run_id, `used_config.yaml`）
