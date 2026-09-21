# FedPACT Stage I 実行・配線確認（2026-09-17）

第I段階の目的に限定し、性能の優劣ではなく、2 clientの局所更新からFedAvg、更新済みglobal surrogateを用いたserver-side proxy生成、foundation LoRA更新、次roundへの状態継承までを確認しました。sampling条件は修正版、server seedはnormal 30本・risk 30本、検索時のclass制約はなしです。

## 実行順と判定

- 修正版P0：pass（`p0_paired_20260917_02`）
- P1-A：pass（`p1a_paired_20260917_01`）
- P1-B：pass（`p1b_paired_20260917_01`）
- P1-C：初回は更新後損失が増加して停止（`p1c_paired_20260917_01`）。同じ初期状態・同じ勾配でbacktrackingし、`1e-9`を受理してpass（`p1c_paired_20260917_02`）
- P1-D formal round 1：pass（第1層 `p1d_layer1_20260917_01`、全層 `p1d_full_20260917_02`）
- 3 round状態継承：実行・配線確認としてpass。round 2は `p1d_round2_*`、round 3は `p1d_round3_*`

## 教授コメントの11観点に対する記録

1. **同じglobal surrogateからの開始とclient局所更新**
   - 各roundで2 clientの開始時LoRA・classification head digestが一致。
   - client LoRA update norm：round 1 `0.3461 / 0.2916`、round 2 `0.3228 / 0.2803`、round 3 `0.3202 / 0.2804`。
   - classification headも各clientで非ゼロ更新。pre/post/delta、loss、gradient norm、sample exposureをclient別traceへ保存。

2. **client更新の送信、FedAvg、global surrogate更新**
   - LoRA updateとclassification head updateをclient別payloadとして保存。
   - FedAvgの独立再計算との差：round 1 `5.82e-11`、round 2 `5.82e-11`、round 3 `1.16e-10`。
   - `M_G^0→M_G^1→M_G^2→M_G^3` のcheckpointとtensor digestを保存。

3. **集約後global surrogateの第2層での使用**
   - 各roundのseed retrievalとproxy生成が、同roundの集約後 `M_G^1 / M_G^2 / M_G^3` を使用したことをstate ID・checkpoint・digestで確認。
   - 更新前global surrogateやclient-local surrogateの誤使用は検出されず。

4. **delta errorの更新前後global surrogateによる計算**
   - 各候補proxyについて、同一候補に対する `M_G^{t}` と `M_G^{t+1}` のlogit・確率・差分を保存。
   - `delta error` と `L_match` を候補60件すべてに保存。

5. **次roundへのglobal surrogate再配布**
   - round 1→2、round 2→3で、前round出力と次round入力のtensor digestおよびcheckpoint SHA-256が一致。
   - 各clientへの再配布後digestも集約後global surrogateと一致。

6. **prototypeの対応付け**
   - `client ID / round / class / frequency / μ^{post}_{k,r} / Δμ_{k,r}` を4 prototypeすべてに保存。
   - 各round、2 client × 2 class、各frequency `π_{k,r}=3`。

7. **server seed retrieval**
   - 各round、検索対象60本（normal 30・risk 30）、class制約なし。
   - prototypeごとに全60候補の順位、top候補、seed ID、class label、更新済みglobal surrogate出力、検索KLを保存。
   - round 1ではseed labelと更新済みglobal surrogateのargmaxが一致したのは60本中43本。このため、選択seedのlabel不一致には検索挙動だけでなくmodel誤分類も混在する。

8. **proxy transformationと目的関数**
   - 各prototypeについてtop 3 seed × 5変換＝15候補、合計60候補を生成。
   - 元seed、seed ID、変換名・parameter、変換後frame、post KL、delta error、`L_match` を対応付けて保存。
   - round 1の選択4件では、目標 `Δμ` とproxy上の実測deltaの向きは3/4一致。client-1/class 0は目標risk差分 `-0.103` に対しproxy `+0.056` で逆向き。proxy再現成功とはまだ結論しない。

9. **proxy pairによるfoundation LoRA更新**
   - 各round、4組の `P_{k,r}` を第3層へ渡し、foundation LoRAのgradientと非ゼロ更新を確認。
   - foundation LoRA update norm：round 1 `0.001357`、round 2 `0.001357`、round 3 `0.001357`。
   - classification head、inactive surrogate LoRA、その他対象外parameterは不変。
   - round 3では厳しい非増加判定で `1e-6`〜`1e-11` の全候補が不受理となったrunを保存（`p1d_round3_full_20260917_01`）。第I段階の要件に合わせ、有限・非ゼロ更新を必須とするexecution確認で `F^2→F^3` を完了（`p1d_round3_full_20260917_02`）。同じ4 pair上の加重KLは `16.1307→16.1776` で増加しており、性能改善とは扱わない。

10. **client→server通信内容とprivate情報非送信**
    - allowlist監査とnegative controlを実施。
    - server側保存物にprivate動画、raw frame、client private sample IDがないことを3 roundすべてで確認。
    - serverで使用したseed IDは追跡情報として保存。
    - これは実装上の通信監査であり、形式的privacy保証ではない。

11. **複数roundの状態継承と実行時間**
    - global surrogateとfoundation LoRAについてround 1→2、2→3の入力・出力digestがすべて一致。
    - 1 round当たりのおおよその実行時間：client局所更新 `97〜131秒`、seed retrieval `約27秒`、proxy差分・目的関数 `約53秒`、foundation load/update `174〜178秒`。
    - 詳細は `three_round_state_lineage.json` と各runの `summary.json` / traceに保存。

## 研究会で示す観測事実

- 代表sampleから `q^{pre}`、`q^{post}`、`Δq`、`μ^{post}_{k,r}`、`Δμ_{k,r}` を具体的に追跡できる。
- class制約なし・各class 30 seedのround 1では、選択4件すべてでprototype classとseed labelが不一致だった。ただしround 2・3では各3/4が一致しており、1 roundの結果を一般化しない。
- round 1ではdelta方向が3/4一致し、1件は逆向き。選択seedの意味整合性とdelta再現性は第II段階の検討対象。
- 3 roundの処理接続と状態継承は確認できたが、FedPACTの性能向上、未知データへの転移、proxy品質は未評価。

## 保存先

- 3 round横断レポート：`artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.md`
- 機械可読版：`artifacts/reports/fedpact_stage1_20260917/three_round_state_lineage.json`
- 研究会資料：`artifacts/presentations/FedPACT_研究会_StageI_3round_20260918_v3.pptx`
- 各run詳細：`artifacts/runs/fedpact_stage1/<run_id>/`
