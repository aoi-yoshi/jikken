"""
Step 7 レパートリー一括比較 — 同一 seed・同一データ分割条件で複数レパートリーを順次実行し、
summary から比較表 CSV を出力する。

例（スモーク規模で 3 方式比較）:
  python scripts/run_step07_repertoire_compare.py ^
    --repertoires ac_lpred_df ac_lpost_cfed baseline_fedavg ^
    --num-rounds 1 --num-clients 1 --smoke

各レパートリーは Flower サーバ + クライアントを子プロセスで起動する（1 GPU 直列）。
結果: artifacts/metrics/step07_repertoire_compare_<ts>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PY = os.environ.get("THESIS_PYTHON", sys.executable)

SMOKE_ARGS = [
    "--client-train-per-class", "2",
    "--client-eval-per-class", "2",
    "--server-train-per-class", "2",
    "--server-eval-per-class", "2",
    "--max-server-steps", "2",
]
SMOKE_COMMON = ["--max-client-steps", "2", "--distill-steps", "2"]


def _run_bg(cmd: list[str], env: dict | None, log_path: Path) -> subprocess.Popen:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        cmd, cwd=str(_ROOT), env=full_env, stdout=log_f, stderr=subprocess.STDOUT, text=True
    )


def run_one(
    repertoire: str,
    *,
    num_rounds: int,
    num_clients: int,
    smoke: bool,
    server_wait_sec: int,
    port: int,
    client_timeout_sec: int,
    server_timeout_sec: int,
) -> dict:
    run_id = datetime.now().strftime(f"run_%Y%m%d_%H%M%S_cmp_{repertoire}")
    logs_dir = _ROOT / "artifacts" / "runs" / "_e2e_logs"

    common = ["--num-rounds", str(num_rounds), "--num-clients", str(num_clients), "--repertoire", repertoire]
    if smoke:
        common += SMOKE_COMMON
    server_cmd = [PY, "scripts/step07_flower_server.py", *common, "--run-id", run_id, "--port", str(port)]
    if smoke:
        server_cmd += SMOKE_ARGS
    client_cmd = [PY, "scripts/step07_flower_client.py", *common, "--server", f"127.0.0.1:{port}"]

    print(f"[compare] repertoire={repertoire} run_id={run_id}", flush=True)
    server = _run_bg(server_cmd, {"THESIS_FL_RUN_ID": run_id}, logs_dir / f"{run_id}_server.log")
    time.sleep(server_wait_sec)

    clients = []
    for cid in range(num_clients):
        env = {"THESIS_CLIENT_ID": str(cid), "THESIS_FL_RUN_ID": run_id}
        clients.append(_run_bg(client_cmd, env, logs_dir / f"{run_id}_client_{cid}.log"))
        time.sleep(5)

    client_rc = 0
    for p in clients:
        try:
            rc = p.wait(timeout=client_timeout_sec)
        except subprocess.TimeoutExpired:
            p.kill()
            print(
                f"[compare] client timed out after {client_timeout_sec}s "
                f"(repertoire={repertoire}, run_id={run_id})",
                flush=True,
            )
            rc = 124
        client_rc = client_rc or rc
    try:
        server_rc = server.wait(timeout=server_timeout_sec)
    except subprocess.TimeoutExpired:
        server.kill()
        print(
            f"[compare] server timed out after {server_timeout_sec}s "
            f"(repertoire={repertoire}, run_id={run_id})",
            flush=True,
        )
        server_rc = 124

    run_dir = _ROOT / "artifacts" / "runs" / f"step07_fl_{repertoire}" / run_id
    row: dict = {
        "repertoire": repertoire,
        "run_id": run_id,
        "server_rc": server_rc,
        "client_rc": client_rc,
        "status": "ok" if (server_rc == 0 and client_rc == 0) else "failed",
    }

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rep_block = summary.get("repertoire", {})
        row["kind"] = rep_block.get("kind")
        row["papers"] = "; ".join(rep_block.get("papers", []))
        row["consistency_mode"] = rep_block.get("consistency_mode")
        row["server_optimizer"] = rep_block.get("server_optimizer")
        row["distribution"] = rep_block.get("distribution")

    # 最終ラウンドのサーバ集約イベントから指標を拾う
    jsonl = run_dir / "fl_server.jsonl"
    if jsonl.is_file():
        last_agg = None
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "step07_server_aggregate":
                last_agg = rec
        if last_agg:
            for k in (
                "server_eval_accuracy",
                "server_eval_f1_macro",
                "server_loss_pred",
                "server_loss_grad",
                "server_loss_post",
                "server_loss_replay",
                "server_train_sec",
                "server_logits_bytes",
            ):
                if k in last_agg:
                    row[k] = last_agg[k]
            fm = last_agg.get("fit_metrics", {})
            for k in ("post_redistribution_accuracy", "fit_duration_sec", "client_logits_bytes",
                      "client_grads_bytes", "client_post_logits_bytes", "weights_bytes"):
                if k in fm:
                    row[f"client_{k}"] = fm[k]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repertoires", nargs="+", required=True)
    ap.add_argument("--num-rounds", type=int, default=1)
    ap.add_argument("--num-clients", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="最小データ・最小ステップで実行")
    ap.add_argument("--server-wait-sec", type=int, default=90)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--client-timeout-sec",
        type=int,
        default=None,
        help="各 client 子プロセスの待ち上限 [秒]。未指定: smoke=14400, 本番=64800 (18h)",
    )
    ap.add_argument(
        "--server-timeout-sec",
        type=int,
        default=None,
        help="server 子プロセスの待ち上限 [秒]。未指定: smoke=7200, 本番=72000 (20h)",
    )
    args = ap.parse_args()

    if args.smoke:
        client_timeout = int(args.client_timeout_sec or 14400)
        server_timeout = int(args.server_timeout_sec or 7200)
    else:
        client_timeout = int(args.client_timeout_sec or 64800)
        server_timeout = int(args.server_timeout_sec or 72000)

    rows = []
    for rep in args.repertoires:
        rows.append(
            run_one(
                rep,
                num_rounds=args.num_rounds,
                num_clients=args.num_clients,
                smoke=args.smoke,
                server_wait_sec=args.server_wait_sec,
                port=args.port,
                client_timeout_sec=client_timeout,
                server_timeout_sec=server_timeout,
            )
        )

    out_dir = _ROOT / "artifacts" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / datetime.now().strftime("step07_repertoire_compare_%Y%m%d_%H%M%S.csv")
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare] saved {out}", flush=True)
    for r in rows:
        print(f"  {r['repertoire']}: {r['status']} acc={r.get('server_eval_accuracy')}", flush=True)
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
