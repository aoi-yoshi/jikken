# GCP 移行計画（FL 完走後）

**前提:** 単一 PC 上の Flower FedAvg（3B × 3B, IID）が E2E 完走していること  
**研究最終構成:** クラウド側 **7B**（集約・推論）、エッジ側 **3B**（ローカル LoRA 学習）

---

## Phase A — ローカル FL 完走（GCP 前）

- [ ] `step05`（server + client）で 2 rounds × 2 clients 完走
- [ ] `step05_fedavg_global.pt` 保存・post eval 確認
- [ ] 統一ログ（accuracy, loss, adapter_size_mb, max_vram_gb, step_time）の run 間比較

---

## Phase B — GCP 環境準備

| タスク | 内容 | 備考 |
|--------|------|------|
| B1 | GCP プロジェクト・課金・IAM | 教授とインスタンス方針を確認 |
| B2 | サーバ VM（7B 用 GPU） | Qwen2.5-VL-7B + Flower Server; L4 24GB 以上を想定 |
| B3 | エッジ想定 VM or ローカル 3B | 既存 RTX 5050 で client 継続も可 |
| B4 | コンテナ or conda 環境の統一 | `flvl` 相当、PyTorch cu128 |
| B5 | モデル・データ配置 | HF cache, manifest, frames を GCS or VM ローカルに |
| B6 | ファイアウォール / gRPC | Flower server 8080 開放、TLS は後回し可 |

---

## Phase C — 7B / 3B 構成への切り替え

| タスク | 内容 |
|--------|------|
| C1 | `config/gcp_server_7b.yaml` — model.id を 7B に |
| C2 | `config/gcp_edge_3b.yaml` — 既存 default ベース |
| C3 | **LoRA 次元・名前の整合** — 7B と 3B で LoRA shape が異なるため **FedAvg 対象を同一アーキテクチャに限定**（まず 3B 同士で検証済みのパイプラインを 7B server のみ差し替え等、段階的に） |
| C4 | 通信量ログ — `adapter_size_mb` をラウンドごとに server 側で記録 |
| C5 | 7B server 上での FedAvg + 3B client からの delta 受信 — **研究上の核心**（実装方式は教授と要相談: 同一 LoRA rank・投影層の扱い） |

> **注意:** 7B と 3B は LoRA パラメータ shape が一致しないため、単純な FedAvg ベクトル平均はそのままでは不可。GCP フェーズでは (1) サーバも 3B のまま GCP 上で scale 確認 → (2) 7B は server-side 集約・蒸留等の設計検討、の 2 段階が現実的。

---

## Phase D — 研究本番実験（GCP 上）

1. IID FedAvg baseline（3B clients → 3B server）
2. Non-IID 分割（天候・時間帯・クラス偏り）
3. Step 7 Ablation vs L_task only
4. 通信効率レポート（Adapter MB × rounds × clients）

---

## 成果物チェックリスト

- [ ] GCP 上 FL run の `summary.json`（server + clients）
- [ ] ラウンドごとの eval accuracy / F1
- [ ] 通信量（MB）と wall-clock 時間の表
- [ ] 7B 移行時の制約と設計判断のメモ（進捗会・論文用）

---

## 参考パス（ローカル）

- FL server: [scripts/step05_flower_server.py](../scripts/step05_flower_server.py)
- FL client: [scripts/step05_flower_client.py](../scripts/step05_flower_client.py)
- グローバル eval: [scripts/step05_fl_post_eval.py](../scripts/step05_fl_post_eval.py)
- チェックポイント: `artifacts/checkpoints/step05_fedavg_global.pt`
