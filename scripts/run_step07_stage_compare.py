"""Stage 0 vs Stage 1 比較 run を実行し、結果 + レシピを Markdown に出力する。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run_stage(stage: int, args: argparse.Namespace) -> Path:
    cmd = [
        PY,
        str(_ROOT / "scripts" / "step07_round_loop.py"),
        "--mode",
        "lpred",
        "--stage",
        str(stage),
        "--num-rounds",
        str(args.num_rounds),
        "--client-train-per-class",
        str(args.client_train_per_class),
        "--client-eval-per-class",
        str(args.client_eval_per_class),
        "--server-train-per-class",
        str(args.server_train_per_class),
        "--server-eval-per-class",
        str(args.server_eval_per_class),
        "--run-suffix",
        f"_stage{stage}_cmp{args.num_rounds}r",
    ]
    if args.max_client_steps is not None:
        cmd.extend(["--max-client-steps", str(args.max_client_steps)])
    if args.max_server_steps is not None:
        cmd.extend(["--max-server-steps", str(args.max_server_steps)])
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_ROOT, check=True)
    latest = _ROOT / "artifacts" / "runs" / "step07_round_lpred" / "LATEST_RUN_ID"
    run_id = latest.read_text(encoding="utf-8").strip()
    return _ROOT / "artifacts" / "runs" / "step07_round_lpred" / run_id


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _round_table(history: List[Dict[str, Any]], stage: int) -> str:
    lines = [
        "| R | client_acc | server_acc | loss_total | loss_task | loss_pred | teacher | round_min |",
        "|---|------------|------------|------------|-----------|-----------|---------|-----------|",
    ]
    for rec in history:
        ce = rec.get("client_eval", {})
        se = rec.get("server_eval", {})
        sl = rec.get("server_loss", {})
        teacher = sl.get("teacher_source", "recompute" if stage == 0 else "transmitted")
        lines.append(
            f"| {rec['round']} "
            f"| {ce.get('accuracy', '—')} "
            f"| {se.get('accuracy', '—')} "
            f"| {sl.get('loss_total', '—'):.4f} "
            f"| {sl.get('loss_task', '—'):.4f} "
            f"| {sl.get('loss_pred', '—'):.2e} "
            f"| {teacher} "
            f"| {rec.get('round_sec', 0) / 60:.1f} |"
        )
    return "\n".join(lines)


def write_report(
    *,
    stage0_dir: Path,
    stage1_dir: Path,
    out_path: Path,
    recipe: Dict[str, Any],
) -> None:
    s0 = _load_summary(stage0_dir)
    s1 = _load_summary(stage1_dir)
    body = f"""# Step 7 Stage 0 vs Stage 1 比較結果

**生成:** 自動（`scripts/run_step07_stage_compare.py`）

## レシピ（共通）

| 項目 | 値 |
|------|-----|
| モデル | Qwen2.5-VL-3B-Instruct（3B/3B） |
| mode | lpred |
| ラウンド数 | {recipe['num_rounds']} |
| client train/eval per class | {recipe['client_train_per_class']} / {recipe['client_eval_per_class']} |
| server train/eval per class | {recipe['server_train_per_class']} / {recipe['server_eval_per_class']} |
| max_client_steps | {recipe.get('max_client_steps', '全件')} |
| max_server_steps | {recipe.get('max_server_steps', '全件')} |
| LoRA | r=8, α=16, q_proj/v_proj（`default.yaml`） |
| LR / batch | 1e-4 / 1（`default.yaml`） |
| w_pred / T | 0.5 / 2.0 |

## run_id

| Stage | run_id | run_dir |
|-------|--------|---------|
| 0（teacher=再forward） | `{s0['run_id']}` | `{stage0_dir}` |
| 1（teacher=送信logits） | `{s1['run_id']}` | `{stage1_dir}` |

## Stage 0 結果

{_round_table(s0['round_history'], 0)}

## Stage 1 結果

{_round_table(s1['round_history'], 1)}

## 解釈メモ

- Stage 0: サーバが FedAvg 後の重み（クライアント LoRA）を固定し、サーバ整合データ上で **再 forward** して Lpred。
- Stage 1: クライアントが `server_train` 上で計算した **logits を teacher として注入**（再 forward 不要）。
- 両 Stage で **基盤 LoRA** のみサーバ整合で更新。FedAvg は参照 adapter 作成に留める。
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print("Wrote:", out_path, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-rounds", type=int, default=3)
    ap.add_argument("--client-train-per-class", type=int, default=50)
    ap.add_argument("--client-eval-per-class", type=int, default=15)
    ap.add_argument("--server-train-per-class", type=int, default=25)
    ap.add_argument("--server-eval-per-class", type=int, default=15)
    ap.add_argument("--max-client-steps", type=int, default=None)
    ap.add_argument("--max-server-steps", type=int, default=None)
    ap.add_argument(
        "--out",
        default="docs/step07_stage_compare_results.md",
        help="比較 Markdown の出力先",
    )
    ap.add_argument("--stage-only", type=int, choices=[0, 1], default=None, help="指定 Stage のみ実行")
    args = ap.parse_args()

    recipe = {
        "num_rounds": args.num_rounds,
        "client_train_per_class": args.client_train_per_class,
        "client_eval_per_class": args.client_eval_per_class,
        "server_train_per_class": args.server_train_per_class,
        "server_eval_per_class": args.server_eval_per_class,
        "max_client_steps": args.max_client_steps,
        "max_server_steps": args.max_server_steps,
    }

    if args.stage_only == 0:
        d0 = _run_stage(0, args)
        print("Stage 0 only:", d0)
        return
    if args.stage_only == 1:
        d1 = _run_stage(1, args)
        print("Stage 1 only:", d1)
        return

    d0 = _run_stage(0, args)
    d1 = _run_stage(1, args)
    write_report(
        stage0_dir=d0,
        stage1_dir=d1,
        out_path=_ROOT / args.out,
        recipe=recipe,
    )


if __name__ == "__main__":
    main()
