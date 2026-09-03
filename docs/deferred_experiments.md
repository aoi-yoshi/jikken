# 延期した実験一覧（教授指示に基づく）

**記録日:** 2026-06-04（2026-06-17 更新）  
**根拠:** 教授コメント — Step 5 完走後、研究加速のため Step 7 Adaptation Consistency プロトタイプを優先

---

## 一時スキップ

| 項目 | 理由 |
|------|------|
| Step 6 Avalanche 継続学習 | 教授指示により Step 7 プロトタイプを先行 |

---

## 延期する実験

| 実験 | 当初の目的 | 延期理由 | 再開条件 |
|------|-----------|----------|----------|
| LoRA rank r=16 vs r=8 比較 | 表現力とコストのトレードオフ | ベースライン（r=8）で十分確認済み | FL + GCP パイプライン完成後 |
| Target modules 4種 vs q/v のみ | 更新範囲とコストの比較 | 同上 | 同上 |
| Non-IID データ分割 | クライアントドリフトの baseline | IID で FL 完走を先に | Flower FedAvg E2E 成功後 |
| サーバ 7B / エッジ 3B 本番構成 | 研究最終構成 | 3B 同士で FL + AC 動作確認を先行 | 単一 PC FL + Step 7 完走後 |

---

## 再開・進行中

| 実験 | 状態 | 備考 |
|------|------|------|
| **Step 7 Adaptation Consistency プロトタイプ** | **進行中** | standalone 検証 → FL 統合（`fl.consistency_mode`） |
| Step 7 Ablation 本番比較 | Step 7 プロトタイプ完了後 | L_task only vs AC 各モード |

---

## 既に取得済みで再実施不要なデータ

- **r=8, q_proj/v_proj** Step 4b: `run_20260601_185826` — Loss↓, Acc 0.80, `\|\|Δθ\|\|` 8.25, Adapter 7 MB
- **r=16, 4 modules** Step 4b: `run_20260528_035028` 等 — 参考データとして保管のみ
- **Step 5 FL** — Flower FedAvg 3B 完走（教授確認済み）

---

## 再開時の優先順位

1. Step 7 AC プロトタイプ（standalone + FL）
2. Step 7 Ablation 本番比較
3. GCP 移行（7B server / 3B edge）
4. Step 6 Avalanche 継続学習
5. Non-IID baseline
6. LoRA 設定の詳細比較
