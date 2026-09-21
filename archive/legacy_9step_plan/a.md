# 提案ループ全体 — 1 ラウンド詳細フロー（全 Stage 共通）

> 用語: `terminology.mdc` / `step07_flow_terminology.mdc`  
> 骨格は全 Stage 共通。[ ] 内・破線ノードだけが Stage / 拡張で差し替わる。  
> いまの実験済み: 比較パターン①（`ac_lpred_df`）／②（`ac_lpred_fedopt`）。差はクラウド側 optimizer のみ。  
> 1 文: **端末で学ぶ → 適応情報でクラウドの基盤 LoRA を整える → 基盤の知識を端末へ還す**

---

## 完全フローチャート（端末 ↔ クラウド）

```mermaid
flowchart TB
  classDef cloud fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
  classDef edge fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
  classDef data fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,stroke-dasharray:4 3
  classDef key fill:#fff9c4,stroke:#f9a825,stroke-width:2px
  classDef file fill:#fce4ec,stroke:#c2185b,stroke-width:1px
  classDef note fill:#eceff1,stroke:#607d8b,stroke-width:1px
  classDef ext fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray:5 3
  classDef unused fill:#fafafa,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray:3 3

  %% ========== データ前提 ==========
  subgraph DATA["データ前提（場所ごと）"]
    direction LR
    Dpub[("公開整合セット<br/>server_train<br/>全端末・クラウドが同じ sample ID を参照<br/>※いまの Lpred / 蒸留の担体")]:::data
    Dloc[("端末固有の学習データ<br/>L_task 用<br/>他端末・クラウドは見ない")]:::data
  end

  %% ========== 前ラウンド末 ==========
  subgraph PRE["前ラウンド末 — クラウド（基盤モデル側）"]
    direction TB
    P0["更新済み 基盤 LoRA<br/>（前 R の工程④ の結果）"]:::cloud
    P1["公開整合セット上で forward<br/>adapter = 基盤 LoRA（surrogate）"]:::cloud
    P2["還元用の知識を保存 ★差し替え点"]:::key
    P2now["いま: 基盤 logits<br/>server_logits / r{N-1}.json<br/>{sample_id → logits}"]:::file
    P2ext["拡張候補:<br/>クラス単位統計 / 重み空間表現<br/>（公開整合セットに依存しない還元）"]:::ext
    P0 --> P1 --> P2
    P2 --> P2now
    P2 -.-> P2ext
  end

  %% ========== このラウンド・端末 ==========
  subgraph EDGE["このラウンド — 端末（surrogate / 本番想定 3B・いまの実験は 3B）"]
    direction TB

    subgraph E5["① 工程⑤ 知識還元（クラウド→端末）★「蒸留」と呼ぶのはここだけ"]
      direction TB
      E5in["受信: 前 R の 還元用の知識"]:::file
      E5t["teacher = 還元用の知識（固定）"]:::key
      E5s["student / 更新 =<br/>クライアント LoRA + 分類ヘッド"]:::edge
      E5loss["いま: KL × T²（T=2.0）<br/>distill_adapter_from_logits<br/>※ FedAvg 後の重みは端末に適用しない<br/>apply_global_weights=0"]:::edge
      E5ext["拡張: 還元の担体を<br/>公開整合セット以外へ<br/>（各端末ローカルデータ等）"]:::ext
      E5in --> E5t --> E5s --> E5loss
      E5loss -.-> E5ext
    end

    subgraph Etask["② L_task（端末ローカル学習）"]
      direction TB
      E2["正解ラベルの教師あり学習"]:::edge
      E2u["更新: クライアント LoRA + 分類ヘッド"]:::edge
      E2n["※端末では整合損失（Lpred/Lgrad/Lpost）を合成しない<br/>整合はクラウド側のみ"]:::note
      E2 --> E2u --> E2n
    end

    subgraph Etx["③ 適応情報の送信（端末→クラウド）★差し替え点"]
      direction TB
      E3prep["適応後のクライアント LoRA + 分類ヘッド"]:::edge
      E3slot["送信する surrogate 適応情報"]:::key
      E3lpred["いま（①②・Lpred）:<br/>公開整合セット上の logits<br/>adapter=client / eval / no_grad<br/>→ client_logits / r{N}_c{cid}.json"]:::file
      E3lgrad["拡張 Lgrad:<br/>勾配ベクトル / LoRA 更新差分<br/>※差分はすでに Flower 上を流れている<br/>→ 追加通信ほぼゼロ"]:::ext
      E3lpost["拡張 Lpost:<br/>1-step 適応後 logits・更新後挙動"]:::ext
      E3flower["並行: Flower に載る重みベクトル<br/>クライアント LoRA + 分類ヘッド<br/>※提案系では基盤更新の入力に使わない<br/>weights_used_by_server: false"]:::unused
      E3prep --> E3slot
      E3slot --> E3lpred
      E3slot -.-> E3lgrad
      E3slot -.-> E3lpost
      E3prep -.-> E3flower
    end

    E5 --> Etask --> Etx
  end

  %% ========== このラウンド・クラウド ==========
  subgraph CLOUD["このラウンド — クラウド（基盤モデル / 本番想定 7B・いまの実験は 3B）"]
    direction TB

    subgraph C3["工程③ FedAvg 集約（Flower）— 提案では基盤更新に使わない"]
      direction TB
      C3a["各端末のクライアント LoRA + 分類ヘッドを平均"]:::unused
      C3b["FedAvg 後の重み<br/>step05_fedavg_global.pt 相当"]:::unused
      C3c["提案系: 端末へ配布しない<br/>基盤 LoRA の teacher にもしない"]:::note
      C3a --> C3b --> C3c
    end

    subgraph Cagg["④ 適応情報の集約 ★差し替え点"]
      direction TB
      C4in["受信: 全端末の適応情報<br/>（sidecar / Flower）"]:::file
      C4now["いま（FedDF 型）:<br/>サンプル ID ごと softmax 確率を<br/>サンプル数重み付き平均"]:::cloud
      C4out["集約情報 = 以降の teacher / 参照<br/>aggregated_client_logits / r{N}.json"]:::file
      C4ext["拡張: 等平均（FedMD 型）/<br/>信頼度重み・ロバスト集約"]:::ext
      C4in --> C4now --> C4out
      C4now -.-> C4ext
    end

    subgraph Clpred["⑤ 工程④ クラウド基盤整合 ★研究の中心"]
      direction TB
      C5t["teacher / 参照 = 集約適応情報（固定）"]:::key
      C5s["student / 更新 = 基盤 LoRA のみ"]:::cloud
      C5loss["いま: Lpred のみ<br/>w_pred · T² · KL(p^T ‖ p^S)<br/>担体 = 公開整合セット上の forward"]:::key
      C5opt["optimizer ★①②の差分<br/>① AdamW（ac_lpred_df）<br/>② FedAdam（ac_lpred_fedopt）"]:::cloud
      C5ext["同じ場所に足す拡張:<br/>+ Lgrad（勾配・差分整合）<br/>+ Lpost（適応後挙動整合）"]:::ext
      C5n["※ここは蒸留と呼ばない<br/>（teacher=端末側情報、更新=基盤 LoRA）"]:::note
      C5t --> C5s --> C5loss --> C5opt
      C5loss -.-> C5ext
      C5opt --> C5n
    end

    subgraph Csave["⑥ 還元用知識の保存 → 次ラウンドへ"]
      direction TB
      C6fwd["更新後の基盤 LoRA を<br/>公開整合セット上で forward"]:::cloud
      C6kg["還元用の知識を保存 ★差し替え点"]:::key
      C6now["いま: 基盤 logits<br/>server_logits / r{N}.json"]:::file
      C6ckpt["基盤 checkpoint<br/>foundation_r{N}.pt"]:::file
      C6ext["拡張: クラス単位統計等<br/>（公開整合セット依存の除去）"]:::ext
      C6fwd --> C6kg
      C6kg --> C6now
      C6kg --> C6ckpt
      C6kg -.-> C6ext
    end

    Cagg --> Clpred --> Csave
  end

  %% ========== 横断リンク ==========
  Dpub -.-> P1
  Dpub -.-> E5
  Dpub -.-> E3lpred
  Dpub -.-> C5loss
  Dpub -.-> C6fwd
  Dloc -.-> E2

  P2now -->|"クラウド → 端末<br/>還元用の知識"| E5in
  E3lpred -->|"端末 → クラウド<br/>適応情報（いま: logits）"| C4in
  E3flower -.->|"Flower 重み<br/>（基盤更新には不使用）"| C3a
  C6now -.->|"次ラウンドの①へ"| NEXT["次ラウンド開始"]:::cloud

  %% ========== 従来との差 ==========
  subgraph DIFF["従来 FedAvg 型との差（1 行）"]
    direction LR
    OLD["従来: クライアント LoRA を平均して<br/>端末へ配る（baseline_fedavg）"]:::unused
    NEW["提案: 平均重みは基盤更新に使わず<br/>適応情報 → 基盤 LoRA 更新 → 知識還元の循環"]:::key
  end

  %% ========== 研究の問い ==========
  subgraph Q["この骨格が生む研究の問い（= 目的）"]
    direction TB
    Q1["Q1 どの適応情報が基盤 LoRA 更新に効くか<br/>logits / 勾配・差分 / 適応後挙動 → 柱2"]:::ext
    Q2["Q2 クラウド側はどう更新すべきか<br/>損失の組合せ × optimizer → ①②の軸"]:::key
    Q3["Q3 公開整合セットなしで ③④⑤ を成立させられるか<br/>→ 柱3・スライド F"]:::ext
  end
```

---

## 工程対応表（入力 → 出力 / 更新）

| 工程 | 場所 | 入力 | 出力 / 更新対象 | いま（①②） | 拡張（同じ枠） | 「蒸留」と呼ぶか |
|------|------|------|-----------------|------------|----------------|------------------|
| 工程⑤ 知識還元 | 端末 | 前 R の**還元用の知識** | **クライアント LoRA + 分類ヘッド** | 基盤 logits の KL 蒸留 | 担体を公開整合セット以外へ | ✅ **ここだけ** |
| L_task | 端末 | 端末固有データ | **クライアント LoRA + 分類ヘッド** | 同左 | 継続学習タスク列（Step 6） | ❌ |
| 適応情報送信 | 端末→sidecar/Flower | 更新後クライアント LoRA | 適応情報 JSON / 差分 | logits（公開整合セット上） | 勾配・LoRA 差分 / 1-step 適応後出力 | ❌ |
| 工程③ FedAvg 集約 | クラウド（Flower） | 各端末のクライアント LoRA + 分類ヘッド | FedAvg 後の重み（提案では基盤更新に不使用） | 計算はするが配布・整合に使わない | — | ❌ |
| 適応情報集約 | クラウド | 各端末の適応情報 | **集約情報**（teacher / 参照） | 重み付き softmax 平均（FedDF 型） | 等平均・信頼度重み等 | ❌ |
| 工程④ 基盤整合 | クラウド | 集約情報 + 担体 | **基盤 LoRA** | Lpred × AdamW / FedAdam | +Lgrad / +Lpost | ❌（整合） |
| 還元用知識の保存 | クラウド | 更新後基盤 LoRA | 次 R 用の還元知識 | 公開整合セット上の基盤 logits | クラス単位統計等 | ❌ |

---

## 場所の対応

| | 役割 | 本番想定 | いまの実験 |
|--|------|----------|------------|
| **端末** | surrogate モデル + **クライアント LoRA + 分類ヘッド** | 3B | 3B |
| **クラウド** | 基盤モデル + **基盤 LoRA** | 7B | 3B（同型で手順検証） |

---

## 従来 FedAvg 型との差（1 行）

- 従来（`baseline_fedavg`）: **FedAvg 後の重み**を端末へ配布して終わり
- 提案: 平均重みは基盤更新に使わず、**適応情報 → 基盤 LoRA 更新 → 知識還元**の循環

---

## 図の読み方（差し替え点）

黄色ノードが **全 Stage 共通の骨格上の差し替え点**:

1. **還元用の知識** … いまは基盤 logits。拡張で公開整合セット非依存の表現へ
2. **surrogate 適応情報** … いまは logits（Lpred）。同じ送信スロットに勾配差分（Lgrad）・適応後挙動（Lpost）
3. **集約方式** … いまは FedDF 型重み付き平均。拡張で等平均・ロバスト集約
4. **整合損失** … いまは Lpred のみ。同じクラウド更新場所に +Lgrad / +Lpost
5. **optimizer** … ① AdamW / ② FedAdam（実験済みの軸）

破線・紫 = 未実験の拡張。灰色 = Flower 上は流れるが提案の基盤更新には使わない経路。
