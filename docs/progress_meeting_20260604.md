# 3日後進捗会資料（2026-06-04 想定）

教授コメントを反映した方針変更・Step 4b ベースライン結果・Flower 次ステップをまとめた資料。

---

## 1. 前回からの方針変更

**教授コメント（要旨）**

- LoRA 設定の最適化そのものが現段階の目的ではない
- r=8, q_proj/v_proj で Loss↓・Acc↑・LoRA 差分が確認できれば十分
- その後は **Flower 連合学習パイプラインの完走** を優先
- Non-IID / 7B 本番評価はマイルストーン通過後

**対応**

| 当初計画 | 変更後 |
|---------|--------|
| r=16 vs r=8 比較 | **停止**（記録: [deferred_experiments.md](deferred_experiments.md)） |
| 4 modules vs 2 modules 比較 | **停止** |
| LoRA 最適化に時間を割く | **Flower FedAvg E2E を最優先** |
| Non-IID クライアントドリフト | **FL 完走後** |
| Step 8 Ablation | **FL + GCP 移行後** |

---

## 2. 完了: Step 4b ベースライン確認

**Run ID:** `run_20260601_185826`  
**設定:** Qwen2.5-VL-3B, r=8, q_proj/v_proj, L_task only, 100 samples, 10 epochs, eval 20

### Slide A — 学習・適応の確認

| 指標 | Epoch 0 | 最良付近 | Epoch 10 |
|------|---------|----------|----------|
| Train Loss | — | 0.15 (Ep.5) | **0.01** |
| Eval Accuracy | 0.50 | **0.80** (Ep.4–7) | 0.80 |
| `\|\|Δθ\|\|` | 0 | 6.6 (Ep.5) | **8.25** |
| KL(client‖surrogate) | ≈0 | 0.47 (Ep.5) | 0.62 |

**グラフ（run フォルダ内）**

- `artifacts/runs/step04b_diff_probe/run_20260601_185826/plots/train_loss.png`
- `artifacts/runs/step04b_diff_probe/run_20260601_185826/plots/client_delta.png`
- `artifacts/runs/step04b_diff_probe/run_20260601_185826/plots/eval_metrics.png`
- `artifacts/runs/step04b_diff_probe/run_20260601_185826/plots/overview_dashboard.png`

**結論:** 教授が求めた「LoRA 適応パイプラインの正常動作」は **初期軽量設定で達成**。

### Slide B — 通信・計算コスト指標

| 項目 | 値 |
|------|-----|
| Adapter サイズ（通信量見積） | **7.03 MB** |
| Trainable パラメータ | 1,847,298 |
| Peak VRAM | **7.61 GB** |
| 1 step 平均時間 | **~31 s** |
| GPU 利用率（平均） | **95–99%** |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |

**ログ形式:** `epoch_history.jsonl` の `hardware_metrics` フィールド

再生成コマンド:

```powershell
python scripts/plot_step04b_epoch_history.py --run-dir artifacts/runs/step04b_diff_probe/run_20260601_185826
```

---

## 3. 考察（簡潔版）

- L_task のみでも Client LoRA は更新され、eval accuracy は **0.50 → 0.80** まで改善
- 同時に Surrogate からの KL 乖離・`\|\|Δθ\|\|` 単調増加が観測され、**適応と参照モデルからのドリフトがトレードオフ**関係にある
- 後半 epoch で eval が不安定になる run もあり、**更新量の制御**が今後の課題
- → 提案手法（Adaptation Consistency）は **Flower パイプライン完走後** に Step 8 Ablation で検証予定

---

## 4. 進行中: Flower FedAvg パイプライン

**実装済み（今回）**

- [step05_flower_server.py](../scripts/step05_flower_server.py): 共有初期パラメータ、FedAvg、ラウンドログ、グローバル checkpoint 保存
- [step06_flower_client.py](../scripts/step06_flower_client.py): 同一 seed 初期化、evaluate() 実装、リソースログ
- [step06_fl_post_eval.py](../scripts/step06_fl_post_eval.py): 集約後モデルの eval
- [src/resource_metrics.py](../src/resource_metrics.py): GPU/Adapter 計測の共通化
- [verify_fl_pipeline.py](../scripts/verify_fl_pipeline.py): FedAvg/checkpoint 単体テスト（**PASS**）
- [fl_local_fedavg_sim.py](../scripts/fl_local_fedavg_sim.py): Flower なし 2-client 集約シミュレーション
- [run_fl_e2e_smoke.py](../scripts/run_fl_e2e_smoke.py): 3 プロセス Flower スモーク（手動実行推奨）

**成功条件（Flower gRPC 本番 — 手動実行）**

- [ ] 2 clients × N rounds 完走
- [ ] `artifacts/checkpoints/step06_fedavg_global.pt` 保存
- [ ] 集約後 eval accuracy 記録
- [ ] `artifacts/runs/step06_fl/<run_id>/` に server/client ログ

**実行手順（3 ターミナル）**

```powershell
$PY = C:\Users\aoi7y\miniconda3\envs\flvl\python.exe
$RUN_ID = "run_YYYYMMDD_HHMMSS"

# Terminal A
& $PY scripts/step05_flower_server.py --run-id $RUN_ID

# Terminal B
$env:THESIS_FL_RUN_ID=$RUN_ID; $env:THESIS_CLIENT_ID="0"
& $PY scripts/step06_flower_client.py

# Terminal C（GPU 1枚の場合は CPU 推奨）
$env:THESIS_FL_RUN_ID=$RUN_ID; $env:THESIS_CLIENT_ID="1"; $env:THESIS_FORCE_CPU="1"
& $PY scripts/step06_flower_client.py

# 事後 eval
& $PY scripts/step06_fl_post_eval.py --run-id $RUN_ID
```

---

## 5. 次のマイルストーン

| 順 | 内容 | 時期 |
|----|------|------|
| 1 | Flower E2E 完走（3B, IID） | **今週** |
| 2 | リソースログを step04/06 に統一 | 今週〜来週 |
| 3 | GCP: サーバ 7B / エッジ 3B | FL 完走後（[gcp_migration_plan.md](gcp_migration_plan.md)） |
| 4 | Non-IID・クライアントドリフト | GCP 移行後 |
| 5 | Step 8 Ablation | ベースライン確定後 |
| 6 | LoRA r/modules 詳細比較 | **最後** |

---

## 6. 相談事項

1. **単一 GPU（RTX 5050 8GB）** で client 2 台同時 GPU 学習は OOM リスク → client 1 を CPU にする運用でよいか
2. **GCP 7B** の GPU インスタンス選定（L4 / A100 等）
3. FL スモーク（1 round, 40 samples）を進捗会デモとして提示し、本番 FL は来週フル設定で実行する方針でよいか
