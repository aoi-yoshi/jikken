# Adaptation Consistency — 全体処理フロー

> private データで局所適応した surrogate の出力挙動を、共有整合データを使わずに統合し、代理データと logits のペアを介して異種の基盤モデルへ移す。

```mermaid
flowchart LR
  G0["global surrogate LoRA"]
  C["各client<br/>private dataでsurrogate LoRAを局所適応"]
  L1["第1層<br/>FL／FCLでLoRA更新を統合"]
  G1["更新済みglobal surrogate"]
  S[("cloud側seed")]
  A["任意: 局所適応前後の<br/>集約出力統計"]
  L2["第2層 ★<br/>代理入力を生成・選択・調整"]
  P["蒸留pair<br/>(x̃, z̃ = global surrogate(x̃))"]
  L3["第3層<br/>出力空間で知識蒸留"]
  F["更新済み基盤LoRA"]

  G0 -->|"配布"| C
  C -->|"LoRA更新のみ送信"| L1
  L1 --> G1
  G1 --> L2
  S --> L2
  A -.-> L2
  L2 --> P
  P --> L3
  L3 --> F
  G1 -->|"次round"| C
```

| 層 | 役割 | 出力 |
|---|---|---|
| **第1層** | 同一構造の surrogate 間で更新方向のずれと忘却を扱う | global surrogate LoRA |
| **第2層** | global surrogate の挙動を引き出す代理入力を決め、同じ入力への logits を付ける | `(代理データ, logits)` |
| **第3層** | 共通出力空間を介して異種の大型基盤モデルへ移す | 基盤 LoRA |

## 確認が必要な点

1. 局所学習前後の logits は、private 入力と対応しない**集約統計**として第2層を補助する理解でよいか。
2. surrogate と基盤モデルが共有する出力空間は、現状の分類クラス logits でよいか。
3. 基盤モデルからclientへの知識還元は今回の定義に含めるか、別経路とするか。
