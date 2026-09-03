# Step 7 Stage 0 vs Stage 1 比較結果

**生成:** 自動（`scripts/run_step07_stage_compare.py`）

## レシピ（共通）

| 項目 | 値 |
|------|-----|
| モデル | Qwen2.5-VL-3B-Instruct（3B/3B） |
| mode | lpred |
| ラウンド数 | 3 |
| client train/eval per class | 50 / 15 |
| server train/eval per class | 25 / 15 |
| max_client_steps | 50 |
| max_server_steps | 25 |
| LoRA | r=8, α=16, q_proj/v_proj（`default.yaml`） |
| LR / batch | 1e-4 / 1（`default.yaml`） |
| w_pred / T | 0.5 / 2.0 |

## run_id

| Stage | run_id | run_dir |
|-------|--------|---------|
| 0（teacher=再forward） | `run_20260703_131436_stage0_cmp3r` | `C:\Python\実験_修正\artifacts\runs\step07_round_lpred\run_20260703_131436_stage0_cmp3r` |
| 1（teacher=送信logits） | `run_20260703_140123_stage1_cmp3r` | `C:\Python\実験_修正\artifacts\runs\step07_round_lpred\run_20260703_140123_stage1_cmp3r` |

## Stage 0 結果

| R | client_acc | server_acc | loss_total | loss_task | loss_pred | teacher | round_min |
|---|------------|------------|------------|-----------|-----------|---------|-----------|
| 1 | 0.6666666666666666 | 0.6666666666666666 | 0.4902 | 0.4902 | 2.38e-07 | recompute | 15.6 |
| 2 | 0.6333333333333333 | 0.7 | 0.3733 | 0.3730 | 5.44e-04 | recompute | 15.2 |
| 3 | 0.6333333333333333 | 0.7 | 0.2957 | 0.2949 | 1.64e-03 | recompute | 15.3 |

## Stage 1 結果

| R | client_acc | server_acc | loss_total | loss_task | loss_pred | teacher | round_min |
|---|------------|------------|------------|-----------|-----------|---------|-----------|
| 1 | 0.6666666666666666 | 0.5666666666666667 | 0.4221 | 0.4062 | 3.17e-02 | transmitted | 13.3 |
| 2 | 0.6333333333333333 | 0.6666666666666666 | 0.3323 | 0.3281 | 8.37e-03 | transmitted | 14.5 |
| 3 | 0.6666666666666666 | 0.7 | 0.2550 | 0.2539 | 2.18e-03 | transmitted | 14.5 |

## 解釈メモ

- Stage 0: サーバが FedAvg 後の重み（クライアント LoRA）を固定し、サーバ整合データ上で **再 forward** して Lpred。
- Stage 1: クライアントが `server_train` 上で計算した **logits を teacher として注入**（再 forward 不要）。
- 両 Stage で **基盤 LoRA** のみサーバ整合で更新。FedAvg は参照 adapter 作成に留める。
