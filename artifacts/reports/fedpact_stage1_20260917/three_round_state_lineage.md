# FedPACT Stage I：3 round状態継承確認

判定：**pass**（実行・配線・状態継承のみ。性能結果ではない）

| round | M_G | client LoRA update norm | FedAvg再計算差 | seed / proxy | F | 同じ4 pair上の加重KL |
|---:|---|---|---:|---|---|---|
| 1 | M_G^0→M_G^1 | 0.3461 / 0.2916 | 5.82e-11 | 60 / 4 | F^0→F^1 (nonincrease) | 0.054042→0.053431 |
| 2 | M_G^1→M_G^2 | 0.3228 / 0.2803 | 5.82e-11 | 60 / 4 | F^1→F^2 (nonincrease) | 18.478321→18.339014 |
| 3 | M_G^2→M_G^3 | 0.3202 / 0.2804 | 1.16e-10 | 60 / 4 | F^2→F^3 (execution) | 16.130657→16.177555 |

## round間の照合

- round 1 -> 2: global digest=True, checkpoint SHA-256=True, foundation digest=True
- round 2 -> 3: global digest=True, checkpoint SHA-256=True, foundation digest=True

## 診断上の重要点

- round 3では、厳しい `nonincrease` モードで試した学習率 10^-6〜10^-11 の全てで同じ4 pair上の加重KLが非増加にならず、失敗runとして保存した。
- 第I段階の必須要件は性能改善ではなく更新実行なので、round 3は `execution` モードで有限・非ゼロのfoundation LoRA更新と対象外パラメータ不変を確認した。KLの増減は診断値として残した。
- seed labelと更新済みglobal surrogateのargmaxは、round 1で60本中43本のみ一致した。seed label不一致には検索挙動だけでなくmodel誤分類も混在する。
- round 1の4 pairではdelta方向が3/4一致し、client-1/class 0は逆向きだった。proxy再現成功とはまだ言えない。

## 言えること／言えないこと

3 roundにわたり処理フローと状態継承が接続され、監査可能なログが保存されたことは言える。一方、性能向上、proxy品質、未知データへの転移、形式的なprivacy保証はこの結果からは言えない。
