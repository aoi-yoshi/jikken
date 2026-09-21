# 提案ループ全体 — 1 ラウンド詳細フローチャート

> **何をしているか（1 文）**  
> 各端末で学ぶ → その「学びの痕跡」でクラウドの大きい AI の追加重み（基盤 LoRA）を整える → 整えた知識を再び端末へ還す。これを毎ラウンド繰り返す。
>
> **いまの実験** … 比較パターン①②（送る情報は logits ＝ Lpred）。①と②の差はクラウド側の更新アルゴリズム（AdamW / FedAdam）だけ。

---

## この文書の読み方

1. 下の **対応表** で「全体の箱」＝「詳細の見出し」を確認する  
2. **全体フロー** を上から一度読む  
3. 知りたい箱だけ、同じ番号の **手順0〜6** を開く  

**番号は全体フローと詳細で同じです。**（§2・§3 のような別番号は使いません）

| 全体フローの箱 | 詳細の見出し | 場所 |
|----------------|--------------|------|
| **手順0** | [手順0 — 還元用知識の保存](#手順0--還元用知識の保存前ラウンド末クラウド) | クラウド |
| **手順1** | [手順1 — 知識還元（蒸留）](#手順1--知識還元蒸留端末) | 端末 |
| **手順2** | [手順2 — L_task](#手順2--ltask端末) | 端末 |
| **手順3** | [手順3 — 適応情報の送信](#手順3--適応情報の送信端末クラウド) | 端末→クラウド |
| **手順4** | [手順4 — 適応情報の集約](#手順4--適応情報の集約クラウド) | クラウド |
| **手順5** | [手順5 — 基盤 LoRA 整合（Lpred）](#手順5--基盤-lora-整合lpredクラウド) | クラウド |
| **手順6** | [手順6 — 次ラウンド用知識の保存](#手順6--次ラウンド用知識の保存クラウド) | クラウド |

※ **手順0 と手順6は同じ操作**（基盤 LoRA の出方を保存する）。手順6の結果が、次ラウンドの手順0／手順1の入力になる。

設計書の「工程⑤／工程④」との対応（紛らわしいので図では使わない）:

| この文書 | 設計書の呼び方 |
|----------|----------------|
| 手順1 | 工程⑤ 蒸留配布 |
| 手順5 | 工程④ クラウド基盤整合（Lpred） |

---

## 0. 登場人物（先にこれだけ覚える）

| 名前 | かみ砕き | 本番想定 | いまの実験 |
|------|----------|----------|------------|
| **端末**（surrogate） | 手元の小さい AI。各クライアントが 1 台ずつ持つ | 3B | 3B |
| **クライアント LoRA + 分類ヘッド** | 端末側の「自分用の小さな追加重み」。端末で学習し、更新する主対象 | — | — |
| **クラウド**（基盤モデル） | みんなで共有する大きい AI | 7B | 3B（手順検証用に同型） |
| **基盤 LoRA** | クラウド側の「共有用の小さな追加重み」。クラウドでだけ更新する | — | — |
| **公開整合セット**（`server_train`） | 全端末とクラウドが **同じサンプル ID** で見られる公開データ。出方合わせの共通の「問題用紙」 | — | — |
| **logits** | モデルが各クラスに付けたスコア（確率の元）。画像そのものは送らない | — | — |

**従来（FedAvg 型）との違い（1 行）**  
従来は「端末の追加重みを平均して、そのまま端末へ配る」。提案は「平均重みは基盤の更新に使わず、端末の出方（適応情報）で **基盤 LoRA** を直し、その知識を端末へ還す」。

**「蒸留」という言葉** … **手順1（クラウド → 端末）だけ**に使う。手順5（クラウドで基盤 LoRA を直す）は「整合」であり蒸留ではない。

---

## 1. 全体フロー（1 ラウンド）

オレンジ = 端末、青 = クラウド、ピンク = 受け渡し。  
各箱の番号は、この下の **手順0〜6** の見出しと 1 対 1。

```mermaid
flowchart TB
  classDef edge fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
  classDef cloud fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
  classDef msg fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
  classDef start fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000

  S([ラウンド N 開始]):::start

  S --> S0
  S0["手順0 クラウド<br/>還元用知識の保存<br/>基盤 LoRA で公開整合セットを解き<br/>出方 logits を保存する<br/>（詳細 → 手順0）"]:::cloud

  S0 --> M1
  M1[["受け渡し クラウド → 全端末<br/>基盤 logits（還元用の知識）"]]:::msg

  M1 --> S1
  S1["手順1 端末<br/>知識還元（ここだけ蒸留）<br/>手本 = 基盤 logits<br/>更新 = クライアント LoRA + 分類ヘッド<br/>（詳細 → 手順1）"]:::edge

  S1 --> S2
  S2["手順2 端末<br/>L_task（普通の学習）<br/>自分のデータ + 正解ラベル<br/>更新 = クライアント LoRA + 分類ヘッド<br/>（詳細 → 手順2）"]:::edge

  S2 --> S3
  S3["手順3 端末 → クラウド<br/>適応情報の送信<br/>公開整合セット上の logits を送る<br/>（詳細 → 手順3）"]:::edge

  S3 --> M2
  M2[["受け渡し 端末 → クラウド<br/>各端末の logits（適応情報）"]]:::msg

  M2 --> S4
  S4["手順4 クラウド<br/>適応情報の集約<br/>端末 logits を重み付き平均 → teacher 1 本<br/>（詳細 → 手順4）"]:::cloud

  S4 --> S5
  S5["手順5 クラウド<br/>基盤 LoRA 整合（Lpred）<br/>teacher = 集約 logits<br/>更新 = 基盤 LoRA だけ<br/>（詳細 → 手順5）"]:::cloud

  S5 --> S6
  S6["手順6 クラウド<br/>次ラウンド用知識の保存<br/>更新後の基盤 LoRA で再び logits を保存<br/>＝ 次ラウンドの手順0<br/>（詳細 → 手順6）"]:::cloud

  S6 --> N([ラウンド N 終了]):::start
  N -.->|"次ラウンドは手順1から<br/>（手順6の成果が手順0相当）"| S1
```

---

## 手順0 — 還元用知識の保存（前ラウンド末・クラウド）

← 全体フローの **手順0** の拡大。手順6 と同じ操作。

**目的:** 次の手順1で端末が真似できる「クラウドの答えの出し方」を残す。

```mermaid
flowchart TB
  classDef cloud fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
  classDef data fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,color:#000
  classDef note fill:#eceff1,stroke:#607d8b,stroke-width:1px,color:#000

  A0([手順0 開始<br/>前ラウンドの手順5が終わった直後]):::cloud
  A1["手元: 更新済みの 基盤 LoRA"]:::cloud
  A2[("公開整合セット server_train<br/>全端末と同じ ID のサンプル一覧")]:::data
  A3["学習は止める（eval / no_grad）<br/>重みは動かさない。出方を測るだけ"]:::cloud
  A4["サンプルを 1 件取り出す"]:::cloud
  A5["基盤モデル + 基盤 LoRA で forward<br/>→ そのサンプルの logits<br/>例: クラス0=1.2, クラス1=-0.3"]:::cloud
  A6{"まだ未処理のサンプルがある?"}
  A7["全サンプル分の logits を保存<br/>形式: sample_id → logits"]:::cloud
  A8([手順0 完了<br/>→ 全体フローの受け渡しへ<br/>→ 各端末の手順1の手本になる]):::note

  A0 --> A1 --> A2 --> A3 --> A4 --> A5 --> A6
  A6 -->|Yes| A4
  A6 -->|No| A7 --> A8
```

**かみ砕き:** クラウドの大きい AI に公開の問題用紙を全部解かせ、「各問題への点数の付け方」を残す。次に端末がそのノートを見る。

---

## 手順1 — 知識還元（蒸留・端末）

← 全体フローの **手順1** の拡大。**ここだけを「蒸留」と呼ぶ。**

**目的:** クラウドの出方に、端末のクライアント LoRA を近づける。基盤 LoRA は触らない。

```mermaid
flowchart TB
  classDef edge fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
  classDef data fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,color:#000
  classDef key fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000
  classDef note fill:#eceff1,stroke:#607d8b,stroke-width:1px,color:#000

  B0([手順1 開始<br/>全体フローから受け渡しで到着]):::edge
  B1["手本 teacher = 手順0（または前R手順6）の基盤 logits"]:::edge
  B2["更新対象 student =<br/>クライアント LoRA + 分類ヘッド<br/>※ FedAvg 後の重みは書き込まない"]:::edge
  B3[("公開整合セット<br/>手本と同じ sample_id を使う")]:::data
  B4["蒸留ステップを繰り返す<br/>いまの設定例: 20 ステップ"]:::edge

  B5a["サンプル x を 1 件取る"]:::edge
  B5b["端末モデル + クライアント LoRA で forward<br/>→ student logits z_S"]:::edge
  B5c["同じ sample_id の teacher logits z_T を取る<br/>（固定・この場では再計算しない）"]:::edge
  B5d["温度 T = 2.0 でやわらかい確率にする<br/>p_T = softmax(z_T / T)<br/>p_S = softmax(z_S / T)"]:::edge
  B5e["差を測る: KL(p_T ‖ p_S) × T²<br/>出方が近いほど小さい"]:::key
  B5f["backward → 学習率で更新<br/>動くのは クライアント LoRA + 分類ヘッド だけ"]:::edge

  B6{"蒸留ステップが残っている?"}
  B7([手順1 完了 → 全体フローの手順2 へ]):::note

  B0 --> B1 --> B2 --> B3 --> B4
  B4 --> B5a --> B5b --> B5c --> B5d --> B5e --> B5f --> B6
  B6 -->|Yes| B5a
  B6 -->|No| B7
```

**かみ砕き:** クラウドの解答ノートを見て、「自分の点数の付け方」を似せる。正解ラベルは使わず、出方の形だけ合わせる。

---

## 手順2 — L_task（端末）

← 全体フローの **手順2** の拡大。

**目的:** 端末固有データで、分類の正解に合わせる。端末では整合損失を混ぜない。

```mermaid
flowchart TB
  classDef edge fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
  classDef data fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,color:#000
  classDef key fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000
  classDef note fill:#eceff1,stroke:#607d8b,stroke-width:1px,color:#000

  C0([手順2 開始 ← 手順1 の直後]):::edge
  C1[("端末固有の学習データ<br/>他端末・クラウドは見ない")]:::data
  C2["更新対象:<br/>クライアント LoRA + 分類ヘッド<br/>（手順1 で少し動いた続き）"]:::edge
  C3["ミニバッチを 1 つ取る"]:::edge
  C4["forward → 予測 logits"]:::edge
  C5["正解ラベルとの差<br/>交差エントロピー等（普通の分類損失）"]:::key
  C6["backward → 学習率で更新"]:::edge
  C7{"まだ学習ステップが残っている?"}
  C8([手順2 完了 → 全体フローの手順3 へ]):::note

  C0 --> C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7
  C7 -->|Yes| C3
  C7 -->|No| C8
```

**かみ砕き:** 自分の写真と正解ラベルで、普通に「当てる」練習をする。

---

## 手順3 — 適応情報の送信（端末→クラウド）

← 全体フローの **手順3** の拡大。

**目的:** 学習後の端末が、公開の問題用紙にどう答えるかをクラウドへ渡す。送るのはスコア表（logits）。画像そのものは送らない。

```mermaid
flowchart TB
  classDef edge fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
  classDef data fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,color:#000
  classDef unused fill:#fafafa,stroke:#9e9e9e,stroke-width:1px,color:#000
  classDef note fill:#eceff1,stroke:#607d8b,stroke-width:1px,color:#000
  classDef ext fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray:4 3,color:#000

  D0([手順3 開始 ← 手順2 の直後]):::edge
  D1["学習を止める（eval / no_grad）<br/>出方の記録だけ"]:::edge
  D2["使うのは今ラウンドで育てた<br/>クライアント LoRA + 分類ヘッド"]:::edge
  D3[("公開整合セット server_train<br/>※自分の L_task 用データではない")]:::data
  D4["サンプルを 1 件ずつ forward<br/>→ logits を記録"]:::edge
  D5{"まだ未処理のサンプルがある?"}
  D6["全サンプル分の logits をクラウドへ送る"]:::edge

  D7["並行して重みベクトルも通信に載ることがある<br/>クライアント LoRA + 分類ヘッド"]:::unused
  D7n["提案の約束: この重みは<br/>クラウドの基盤更新に使わない<br/>従来 FedAvg 型だけがこれを端末へ配る"]:::note

  D8["同じ送信の枠の拡張（いまは未使用）"]:::ext
  D8a["Lgrad: 勾配・LoRA 更新差分"]:::ext
  D8b["Lpost: 1 ステップ適応後の出方"]:::ext

  D9([手順3 完了 → 全体フローの受け渡し → 手順4]):::note

  D0 --> D1 --> D2 --> D3 --> D4 --> D5
  D5 -->|Yes| D4
  D5 -->|No| D6
  D6 --> D7 --> D7n
  D6 -.-> D8 --> D8a & D8b
  D6 --> D9
```

**かみ砕き:** 「公開の問題用紙」への自分の採点表をクラウドに提出する。

---

## 手順4 — 適応情報の集約（クラウド）

← 全体フローの **手順4** の拡大。

**目的:** 端末ごとの採点表を、1 つの「みんなの平均的な出方」（teacher）にする。  
いまの方式 = 確率にしてから、学習件数で重み付き平均（FedDF 型）。

```mermaid
flowchart TB
  classDef cloud fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
  classDef key fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000
  classDef note fill:#eceff1,stroke:#607d8b,stroke-width:1px,color:#000
  classDef unused fill:#fafafa,stroke:#9e9e9e,stroke-width:1px,color:#000

  E0([手順4 開始<br/>全端末の手順3の結果が届いた]):::cloud

  Epar["参考: 同時に重みの平均（FedAvg）も計算されうる"]:::unused
  Epar2["提案ではその平均重みを<br/>端末へ配らず、基盤更新にも使わない"]:::note
  Epar --> Epar2

  E1["各端末から届いた logits を集める"]:::cloud
  E2{"端末は 1 台だけ?"}
  E3["その 1 台の logits をそのまま teacher 候補へ"]:::cloud

  E4["サンプル ID ごとに次を行う"]:::cloud
  E4a["各端末 c の logits → softmax<br/>→ クラス確率 p_c"]:::cloud
  E4b["重み w_c = その端末の L_task 学習件数<br/>※公開整合セットの件数ではない"]:::cloud
  E4c["重み付き平均<br/>p̄ = Σ(w_c · p_c) / Σ w_c"]:::key

  E5["集約結果 = 以降の手順5の teacher（固定）"]:::cloud
  E6([手順4 完了 → 全体フローの手順5 へ]):::note

  E0 --> Epar
  E0 --> E1 --> E2
  E2 -->|Yes| E3 --> E5
  E2 -->|No| E4 --> E4a --> E4b --> E4c --> E5
  E5 --> E6
```

**かみ砕き:** クラスの答案を集めて、「データ量が多い端末の意見を少し厚くした平均答案」を作る。

---

## 手順5 — 基盤 LoRA 整合（Lpred・クラウド）

← 全体フローの **手順5** の拡大。**ここは蒸留ではない。**

**目的:** 集約した端末の出方に、クラウドの基盤 LoRA の出方を近づける。更新は基盤 LoRA だけ。

```mermaid
flowchart TB
  classDef cloud fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
  classDef data fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,color:#000
  classDef key fill:#fff9c4,stroke:#f9a825,stroke-width:2px,color:#000
  classDef opt1 fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
  classDef opt2 fill:#bbdefb,stroke:#1565c0,stroke-width:2px,color:#000
  classDef note fill:#eceff1,stroke:#607d8b,stroke-width:1px,color:#000
  classDef ext fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray:4 3,color:#000

  F0([手順5 開始 ← 手順4 の teacher がある]):::cloud
  F1a["基盤 LoRA を trainable にする<br/>クライアント側は freeze"]:::cloud
  F1b["teacher = 手順4の集約 logits（固定）"]:::cloud
  F1c[("公開整合セット<br/>teacher と同じ sample_id で forward")]:::data

  F2{"比較パターン（optimizer だけ違う）"}
  F2a["① ac_lpred_df → AdamW"]:::opt1
  F2b["② ac_lpred_fedopt → FedAdam 型<br/>β1=0.9, β2=0.99, τ=1e-3"]:::opt2

  F3a["サンプル x を読む"]:::data
  F3b["基盤モデル + 基盤 LoRA で forward<br/>→ student logits z_S"]:::cloud
  F3c["同じ sample_id の teacher logits z_T を取る"]:::cloud
  F3d["温度 T = 2.0<br/>p_T = softmax(z_T / T)<br/>p_S = softmax(z_S / T)"]:::cloud
  F3e["Lpred = w_pred · T² · KL(p_T ‖ p_S)<br/>いま: w_pred = 0.5"]:::key
  F3f["正解ラベルの損失は足さない"]:::note
  F3g["backward → optimizer.step<br/>動くのは 基盤 LoRA のみ"]:::cloud

  F4{"まだ整合ステップが残っている?"}
  F5["loss_pred などを記録"]:::cloud

  F6["同じ場所に後から足す拡張"]:::ext
  F6a["+ Lgrad"]:::ext
  F6b["+ Lpost"]:::ext

  F7([手順5 完了 → 全体フローの手順6 へ]):::note

  F0 --> F1a --> F1b --> F1c --> F2
  F2 -->|①| F2a --> F3a
  F2 -->|②| F2b --> F3a
  F3a --> F3b --> F3c --> F3d --> F3e --> F3f --> F3g --> F4
  F4 -->|Yes| F3a
  F4 -->|No| F5 --> F7
  F5 -.-> F6 --> F6a & F6b
```

**かみ砕き:** 「みんなの平均答案」を手本に、クラウド本体の追加重みだけを直す。①と②は直し方の道具（optimizer）が違うだけ。

---

## 手順6 — 次ラウンド用知識の保存（クラウド）

← 全体フローの **手順6** の拡大。**中身は手順0と同じ。** 更新後の基盤 LoRA でやり直す。

**目的:** 新しい出方を残し、次ラウンドの手順1の手本にする。

```mermaid
flowchart TB
  classDef cloud fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
  classDef data fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,color:#000
  classDef start fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000

  G0([手順6 開始 ← 手順5 で基盤 LoRA 更新済み]):::cloud
  G1["任意: 評価データで精度を測る"]:::cloud
  G2[("公開整合セット server_train")]:::data
  G3["eval / no_grad で 1 件ずつ forward<br/>→ 更新後の基盤 logits"]:::cloud
  G4["全サンプル分の logits を保存"]:::cloud
  G5([手順6 完了 = ラウンド N 終了<br/>保存結果は次ラウンドの手順0相当<br/>→ 次ラウンドは全体フローの手順1 から]):::start

  G0 --> G1 --> G2 --> G3 --> G4 --> G5
```

---

## 付録A — 何が更新され、何が送られるか

| 手順 | 場所 | 入力 | 更新 / 出力 | 蒸留？ |
|------|------|------|-------------|--------|
| 0 | クラウド | 基盤 LoRA × 公開整合セット | 基盤 logits（還元用知識） | ❌ |
| 1 | 端末 | 前Rの基盤 logits | **クライアント LoRA + 分類ヘッド** | ✅ |
| 2 | 端末 | 端末固有データ + 正解 | **クライアント LoRA + 分類ヘッド** | ❌ |
| 3 | 端末→クラウド | 更新後クライアント LoRA × 公開整合セット | 端末 logits（適応情報） | ❌ |
| 4 | クラウド | 各端末 logits | 集約 logits（teacher） | ❌ |
| 5 | クラウド | 集約 logits × 公開整合セット | **基盤 LoRA** | ❌（整合） |
| 6 | クラウド | 更新後基盤 LoRA × 公開整合セット | 次R用の基盤 logits | ❌ |

---

## 付録B — この骨格から出る研究の問い

| 問い | 中身 | 対応 |
|------|------|------|
| **Q1** | どの適応情報が基盤 LoRA 更新に効くか（logits / 勾配・差分 / 適応後挙動） | 柱2・手順3〜5の差し替え |
| **Q2** | クラウド側はどう更新すべきか（損失の組合せ × optimizer） | いまの①②＝手順5の分岐 |
| **Q3** | 公開整合セットなしで手順3〜6を成立させられるか | 柱3 |
