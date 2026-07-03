# Step 7 設計文書 — 4点整理・Stage 1 設計・関連研究

**更新:** 2026-07-03（7/3 研究会後）  
**用語:** [`terminology.md`](terminology.md) に準拠  
**実装入口:** [`scripts/step07_round_loop.py`](../scripts/step07_round_loop.py)

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
| **0** | 参照 adapter 作成（1 台なら送信重み = FedAvg 後の重み） | 固定 **FedAvg 後の重み（クライアント LoRA）** を teacher としてサーバデータ上で **再 forward** → Lpred + L_task → **基盤 LoRA** 更新 |
| **1** | **参照 adapter 作成に留める**（最終更新ではない） | **送信 logits を teacher として注入** → Lpred + L_task → **基盤 LoRA** 更新（teacher 側の再 forward 不要） |
| **2** | 同上 | + Lgrad |
| **3** | 同上 | + Lpost |
| **4** | 同上 | 3 成分全部 |

FedAvg は **基盤 LoRA の最終更新手段ではない**。参照 adapter 作成・補助情報集約の候補。最終更新は合成損失に基づく **サーバ側最適化**。

### ④ 再配布

| Stage | 3B/3B（monolithic） | FL 統合 | 7B+3B |
|-------|---------------------|---------|-------|
| **0** | 同一モデル上で基盤 LoRA が次ラウンドに残る | Step 5: **FedAvg 後の重み → クライアント LoRA** のみ | — |
| **1** | 基盤 LoRA state を次ラウンド client phase 前に明示保持 | **基盤 LoRA 配布経路を追加**（Stage 1 FL 統合時） | 蒸留・射影（Stage 1〜2 後） |

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

### 1 ラウンドの流れ（Stage 1）

```
① 基盤 LoRA を保持（前ラウンド配布相当）
② クライアント: L_task → クライアント LoRA 更新
②b クライアント: server_train 上で forward → logits 送信（sidecar JSON）
③ FedAvg（1 台なら省略）→ FedAvg 後の重み
④ サーバ: 送信 logits を teacher に Lpred + L_task → 基盤 LoRA 更新
⑤ 更新した基盤 LoRA を次ラウンドへ
```

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

## 実装対応表

| 設計要素 | ファイル | 関数 / キー |
|---------|---------|------------|
| Stage 0 ループ | `scripts/step07_round_loop.py` | `train_client_l_task`, `train_server_consistency` |
| Stage 1 logits  export | 同上 | `export_client_logits` |
| Stage 1 teacher 注入 | `src/adaptation_losses.py` | `lpred_from_teacher_logits` |
| logits 集約 | `src/fl_logits.py` | `aggregate_client_logits` |
| teacher 切替 | CLI `--stage` / `--teacher-source` | `consistency.teacher_source` |
| データ分割 | `step07_round_loop.py` | `data_splits.json` |

---

## 検証計画

| run | 条件 | 目的 |
|-----|------|------|
| スモーク Stage 1 | 最小データ、`--run-suffix _stage1_smoke --stage 1` | ループ完走 ✅ |
| 本番 Stage 0 vs 1 | 同一 seed・3R・本番データ件数 | [`step07_stage_compare_results.md`](step07_stage_compare_results.md) ✅ |

報告: **結果 + レシピ**（LoRA r/α/modules, LR, batch, フレーム数, データ数, AMP/GC, run_id, `used_config.yaml`）
