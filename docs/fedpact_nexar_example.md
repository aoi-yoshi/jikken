# FedPACT Layer 2：NEXAR具体例の縦切り実装

この実装の目的は高いF1を示すことではなく、NEXARの具体例を用いて次の情報の流れを確認・可視化することである。

`private局所適応 → post/delta prototype → server seed retrieval → transformation → proxy objective → proxy pair`

## 実行

```powershell
C:\Users\aoi7y\miniconda3\envs\flvl\python.exe scripts\step07_fedpact_nexar_example.py `
  --private-count 3 `
  --client-eval-count 3 `
  --seed-count-per-class 4 `
  --train-scope balanced `
  --top-k 1 `
  --local-steps 6 `
  --batch-size 1 `
  --train-order-seed 101 `
  --transforms full `
  --screen-all-seeds-identity `
  --run-suffix meeting_example
```

`--checkpoint`を省略すると、既存のStep 7、次いでStep 5のcheckpointから更新日時が最も新しいものを使う。明示する場合は、例えば次のように指定する。

```powershell
--checkpoint artifacts\checkpoints\step07_fl\run_...\step07_fedavg_reference.pt
```

## 出力

`artifacts/runs/fedpact_nexar_example/<run_id>/`以下へ次を保存する。

- `client_local/private_trace.json`: private trainのsample ID、pre/post出力、局所学習ログ。client内限定。
- `client_local/eval_trace.json`: 局所学習に未使用のclient_evalに対するpre/post出力。client内限定。各クラス1本なら診断例であり、汎化性能推定ではない。
- `client_local/data_splits.json`: client private、server seed、evaluation-onlyの重複なしsplit ID。
- `client_upload/server_payload.json`: LoRA更新情報、`mu_post`、`delta_mu_local`、frequency。private sample ID、raw frame、独立した`mu_pre`を含まない。
- `client_upload/trainable_update.npz`: client LoRAと分類headの更新差分。
- `server/layer2_trace.json`: retrieval候補、変換候補、各post KL、delta error、選択結果。
- `server/transformation_candidates/`: retrieved seedから作った全変換候補のフレーム。候補比較の可視化に用いる。
- `server/proxy_pair.json`: Layer 3へ渡す`P=(x_tilde, mu_post)`。
- `presentation_example.png`: client画像、retrieved seed、proxyを一続きに示す説明用画像。client画像を含むためローカル説明用であり、server送信物ではない。

## 現段階の扱い

- 単一clientの縦切りなので、この例ではFedAvg後のglobal surrogateが局所更新後stateと一致する。
- transformation後のproxy optimizationは、小さな変換集合に対する離散探索として実装する。
- 目的関数は`lambda_post * post_kl + lambda_delta * delta_mse`である。`delta_mse`はclientの`delta_mu_local`と、同一proxyに対するglobal surrogate更新前後の出力差を比較する。
- Qinvの具体式とthresholdはscreening待ちのため、このStep I-B実装では値を捏造せず、post KLとdelta errorを保存して一様frequency条件で選択する。
- `--train-scope balanced`ではrisk 3本・normal 3本の両方で局所学習し、各クラスのprototypeを送信する。Layer 2の具体例では`--target-label`で指定したクラスを追跡する。
- `--screen-all-seeds-identity`では、検索上位`top-k`へ全変換を試すだけでなく、それ以外のseedもidentityで評価する。8本×全変換の完全全探索ではないが、検索1位へ固定することの粗い妥当性を診断できる。
- `--diagnostic-only`と異なる`--train-order-seed`を組み合わせると、同じsplitのまま局所学習順だけを変えた感度確認ができる。
- `--local-steps 1 --batch-size 6`なら、balanced 6本を同時に使うfull-batch相当の1更新となる。総sample exposureを6に揃え、`--local-steps 6 --batch-size 1`との順序依存性を比較できる。
- このrunは処理経路の成立確認用であり、F1改善や異種foundationへの転移性能を主張しない。client_evalが各クラス1本の場合も、未知動画で挙動を確認する診断に留める。
