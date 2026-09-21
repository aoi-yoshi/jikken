# NEXAR具体例によるFedPACT Layer 2動作確認（2026-09-13）

## 目的と実験計画上の位置

目的は性能比較ではなく、NEXARの実動画を用いて `client局所適応 → q^{pre}/q^{post}/Δq → class-wise prototype → server seed retrieval → transformation → proxy pair` を追跡し、各段階の入力・出力と失敗箇所を確認することである。

- 実験計画PDF `FedPACT実験計画260901.pdf` の第I段階、特にStep I-A（局所更新とpre/post/deltaの確認）とStep I-B（prototype、seed retrieval、proxy生成）の単体・縦切り確認に相当する。
- 第I段階の完成条件である3〜5 round接続、複数clientの連合集約、Layer 3のfoundation LoRA更新までは未実施である。
- 修正版手法提案PDF `FedPACT手法提案260901.pdf` 5〜6ページのpost/delta prototype、seed retrieval、proxy objectiveの基本経路を実装した。ただし `L_reg` とQinvは未導入である。

## 固定条件

- model: `Qwen/Qwen2.5-VL-3B-Instruct`
- checkpoint: `artifacts/checkpoints/step07_fl/run_20260707_160102_cmp_ac_lpred_fedopt/step07_fedavg_reference.pt`
- split seed: 42
- client private train: risk 3本 + normal 3本
- client評価用動画: risk 3本 + normal 3本（局所学習・prototype作成・server検索に未使用）
- server seed: risk 4本 + normal 4本
- raw video / frame / sample IDはserver payloadへ送信しない。
- `q`、`μ^{post}_{k,r}`、`Δμ_{k,r}`は生logitではなくsoftmax確率（またはその差）である。成分順は `[normal, risk]`。
- 今回の実装上の送信物は、trainable update（`client_lora` + `classification_head`）、`μ^{post}_{k,r}`、`Δμ_{k,r}`、`π_{k,r}`、`class_count`。`μ^{pre}`は独立送信しない。
- PDFの基本protocolで集約重みに使うのは`π_{k,r}`。`class_count`は今回のpayload schemaへ監査用に併記した実装fieldで、現runでは`π_{k,r}`と同値である。
- 現runの目的関数は`λ_post=1`、`λ_Δ=1`で、`L_reg`とQinvは未導入である。

## 局所更新の診断結果

batch size 1で6本を1回ずつ使い、学習順だけを変えた。表はriskクラス確率の平均である。

| split / class | q^{pre} | order 101 q^{post} | Δ | order 202 q^{post} | Δ | order 303 q^{post} | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| train normal | 0.341 | 0.201 | -0.139 | 0.403 | +0.062 | 0.150 | -0.191 |
| train risk | 0.685 | 0.596 | -0.090 | 0.845 | +0.160 | 0.491 | -0.194 |
| 評価用 normal | 0.222 | 0.121 | -0.101 | 0.297 | +0.075 | 0.080 | -0.143 |
| 評価用 risk | 0.596 | 0.466 | -0.130 | 0.707 | +0.111 | 0.350 | -0.246 |

同じsplit・同じ開始checkpointでも、`q^{post}`と`Δq`の符号が学習順で反転した。共通方向の確率移動が支配的で、risk / normalの分離改善は順序間で一貫しない。したがって、現設定では意味的なrisk知識を評価できるほど局所更新が安定していない。

総sample exposureを6本に揃え、6本を同時に1 optimizer stepで更新するfull-batch相当も確認した。

| split / class | q^{pre} | q^{post} | Δ |
|---|---:|---:|---:|
| train normal | 0.341 | 0.300 | -0.041 |
| train risk | 0.685 | 0.698 | +0.013 |
| 評価用 normal | 0.222 | 0.201 | -0.021 |
| 評価用 risk | 0.596 | 0.576 | -0.020 |

batch size 1の三つの順序より共通方向shiftは小さくなった。ただし、batch size 1は6 optimizer steps、full-batch相当は1 optimizer stepであり、batch構成と更新回数が交絡している。したがって「6本同時でshiftが縮小した」とは言えるが、batch化が原因とはまだ言えない。次実験ではoptimizer step数とsample exposureを記録・統制して切り分ける。

## order 101を用いたLayer 2詳細追跡

order 101は結果確認前に最初の詳細追跡runとして固定した。ほかの順序は感度確認であり、都合のよい結果を選ぶためには使わない。

### risk prototype

- `μ^{post}_{k,r} = [0.404, 0.596]`
- `Δμ_{k,r} = [+0.090, -0.090]`
- `π_{k,r} = 3`

risk 3本であってもrisk確率が平均0.090低下している。Layer 2が再現する対象は「有用性が保証されたrisk知識」ではなく、このrunで観測された局所適応挙動である。

server seed 8本 × 変換10種類の全80候補を評価した結果、検索順位6位のseed `4b8432de81`へ`center_crop(0.85)`を適用した候補が最小となった。

seed retrievalは、連合集約後global surrogateの出力を用い、`D_KL(μ^{post}_{k,r} || p_G^{t+1}(s))`で順位付けする。`M_G^t`は連合集約前、`M_G^{t+1}`は連合集約後のglobal surrogateである。今回は1 clientなので、`M_G^{t+1}`は局所学習後surrogateと一致する。proxy側の出力差はclient側の`q^{post}-q^{pre}`と区別し、`p_G^{t+1}(x_tilde)-p_G^t(x_tilde)`と表す。

- post KL: 0.000109
- delta MSE: 0.000169
- `L_match`: 0.000279

検索順位1位だけへ全変換を適用した場合の最小は`brightness(1.10)`、`L_match=0.000697`であった。したがって、server seedが8本しかない現規模ではtop-1固定より全探索が妥当である。

### normal prototype

- `μ^{post}_{k,r} = [0.799, 0.201]`
- `Δμ_{k,r} = [+0.139, -0.139]`
- `π_{k,r} = 3`
- selected seed: `17c2e0d437`（retrieval rank 1）
- transformation: `brightness(0.75)`
- post KL: 0.000080
- delta MSE: 0.000389
- `L_match`: 0.000468

normal側も同じfull-grid条件で代理候補を生成できた。

## 言えること / 言えないこと

言えること：

- class-wise prototypeを作り、private raw dataなしでserverへ渡せた。
- 定義した`μ^{post}_{k,r}`と`Δμ_{k,r}`に近い代理データ候補をserver側で探索する処理経路は動作した。
- top-1 retrieval固定ではfull-grid最小候補を見逃すことを、現在の8 seedで確認した。
- 現行batch size 1の局所更新は学習順に強く依存する。

言えないこと：

- 未見動画へ一般化した。
- risk / normal識別性能が改善した。
- 特定物体・場面特徴を学習した。
- proxyが意味的なclient知識を再現した。
- foundation modelへ知識が転移した。

## 次の優先実験

1. 更新対象parameter、classification headの固定／更新状態、各parameterの更新量、全runでの`q^{pre}`一致を監査する。
2. optimizer step数と総sample exposureを明示的に記録し、batch size 1、risk / normalを必ず含む層化mini-batch、gradient accumulationをscreeningする。
3. 候補条件をorder 101 / 202 / 303で再実行し、`Δq`、class間差、評価用動画のlossの一貫性を確認する。
4. 以上を切り分けた後、必要ならlearning rateを下げる。
5. 局所更新が安定してから2-client化、Layer 3、3〜5 round接続へ進む。

## run一覧

- 詳細・risk full grid: `artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_risk_fullgrid`
- 詳細・normal full grid: `artifacts/runs/fedpact_nexar_example/run_20260913_balanced_order101_normal_fullgrid`
- 順序感度: `run_20260913_balanced_order202_diagnostic`、`run_20260913_balanced_order303_diagnostic`
- full-batch相当: `run_20260913_balanced_fullbatch_diagnostic`
