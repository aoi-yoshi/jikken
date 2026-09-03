# 用語ルール（実験・会話・コードコメント）

略称や独自名を避け、**研究提案 PDF・実験計画 PDF と同じ言葉**で書く。

## 文章のルール（目的語を省略しない）

- **主語・目的語・場所を省略しない。** 「読まない」「載せる」「片側」だけでは書かない。
- 悪い例: 「FedAvg 後の重みを読まない」
- 良い例: 「`step07_ablation_modes.py` は、Step 5 が保存したファイル `step05_fedavg_global.pt` を **開いて、モデルの重みに書き込まない**」
- 悪い例: 「どちらのアダプタに載せるか」
- 良い例: 「**FedAvg 後の重み**の数値を、**クライアント LoRA の重みに書き込むか、基盤 LoRA の重みに書き込むか**」
- 操作を書くときは **何のファイル → 何のパラメータ → どのプログラム** を一行で言えるようにする。
- 比較するときは **A と B** を常に明示する（例: **FedAvg 後の重み** と **基盤 LoRA の重み**）。

## 研究 PDF の記号（参照）

| 記号 | PDF の意味 | 本番構成の例 |
|------|-----------|-------------|
| **F** | Foundation model（教師・クラウド） | Qwen2.5-VL-7B |
| **S** | Surrogate model（エッジ） | Qwen2.5-VL-3B |
| **A** | Adapter（LoRA 等） | client / 基盤 LoRA |

ローカル Phase 1（3B/3B）では F と S を同一モデル上の **基盤 LoRA / クライアント LoRA** で近似している。

## 使う言葉（正）

| 言葉 | 意味 | コード・ファイル（参照用） |
|------|------|---------------------------|
| **基盤モデル** | 研究提案の Foundation model（クラウド側） | 7B 本体 |
| **surrogate モデル** | 研究提案の Surrogate model（エッジ側） | 3B 本体 |
| **基盤 LoRA** | 基盤モデル側の PEFT。整合損失の参照・Step 7 配布の主役 | `surrogate` アダプタ |
| **クライアント LoRA** | 各クライアントが学習しサーバへ送る LoRA | `client` アダプタ |
| **分類ヘッド** | 2 クラス出力の線形層 | `classifier` |
| **FedAvg 後の重み** | クライアント LoRA + 分類ヘッドを FedAvg した結果。**LoRA の第3種ではない** | `step05_fedavg_global.pt` |
| **配布** | サーバが次ラウンド用重みをクライアントに送ること | Flower `set_parameters` |
| **整合損失** | Lpred / Lgrad / Lpost の総称 | `AdaptationLosses` |
| **重みを書き込む** | checkpoint の数値を LoRA／ヘッドのパラメータにコピーする | `vector_to_trainable_state` |

## 使わない言葉（避ける）

| 避ける | 理由・代わりに |
|--------|----------------|
| **集約 LoRA** | **PDF にない独自語。** → **FedAvg 後の重み** または `step05_fedavg_global.pt` |
| F / S（単独） | → **基盤モデル / surrogate モデル**、または **基盤 LoRA / クライアント LoRA** |
| surrogate（会話・資料） | → **surrogate モデル** または **基盤 LoRA**（文脈で区別） |
| global / global checkpoint | 曖昧 → **FedAvg 後の重み** または **`step05_fedavg_global.pt`** |
| 代理 LoRA | → **基盤 LoRA** |
| AC | → **整合（Adaptation Consistency）** |

## Step 5（連合学習）で実際に動いていること

1. 各クライアントが **クライアント LoRA + 分類ヘッド** を学習してサーバへ送る。
2. サーバが **FedAvg** → **FedAvg 後の重み**（ファイル: `step05_fedavg_global.pt`）。
3. 次ラウンド、サーバは **FedAvg 後の重みを** クライアントの **クライアント LoRA + ヘッド** に書き込んで配布する。
4. **基盤 LoRA は FedAvg に含めない**。各クライアント内で参照用に保持する。

## Step 7 でやりたいこと（研究の芯）

Step 5 の **FedAvg 後の重みをそのまま配布する代わりに**:

1. **FedAvg 後の重み** と **基盤 LoRA** の間で **整合損失（Lpred / Lgrad / Lpost）** を使い、**基盤 LoRA を更新**する。
2. 更新後の **基盤 LoRA**（＋必要ならヘッド）をクライアントに **配布**する。

## 設定名の読み方

| config | 意味 |
|--------|------|
| `surrogate_ema: 1.0` | 基盤 LoRA をクライアント LoRA に **追従更新しない**（固定） |
| `surrogate_ema: 0.9` | 各学習 step 後、基盤 LoRA を 10% クライアント方向に寄せる |
| `sync_surrogate_at_epoch: true` | epoch 開始時に **基盤 LoRA ← クライアント LoRA** 完全コピー |
