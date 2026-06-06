# 延期した実験一覧（教授指示に基づく）

**記録日:** 2026-06-04  
**根拠:** 教授コメント — 「LoRA 設定の最適化そのものが目的ではない」「Flower パイプライン完走を優先」

---

## 延期する実験

| 実験 | 当初の目的 | 延期理由 | 再開条件 |
|------|-----------|----------|----------|
| LoRA rank r=16 vs r=8 比較 | 表現力とコストのトレードオフ | ベースライン（r=8）で十分確認済み | FL + GCP パイプライン完成後 |
| Target modules 4種 vs q/v のみ | 更新範囲とコストの比較 | 同上 | 同上 |
| Non-IID データ分割 | クライアントドリフトの baseline | IID で FL 完走を先に | Flower FedAvg E2E 成功後 |
| Step 8 Ablation（L_pred / L_grad / L_post） | 提案手法の有効性 | 比較 baseline（FL）未確立 | FL baseline + eval 体制確立後 |
| サーバ 7B / エッジ 3B 本番構成 | 研究最終構成 | 3B 同士で FL 動作確認を先行 | 単一 PC FL 完走後 |

---

## 既に取得済みで再実施不要なデータ

- **r=8, q_proj/v_proj** Step 4b: `run_20260601_185826` — Loss↓, Acc 0.80, `\|\|Δθ\|\|` 8.25, Adapter 7 MB
- **r=16, 4 modules** Step 4b: `run_20260528_035028` 等 — 参考データとして保管のみ

---

## 再開時の優先順位

1. Flower FedAvg（IID, 3B）
2. GCP 移行（7B server / 3B edge）
3. Non-IID baseline
4. Step 8 Ablation
5. LoRA 設定の詳細比較
