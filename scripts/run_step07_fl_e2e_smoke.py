"""
Step 7 Flower E2E スモーク（1 round, 小データ）。--repertoire で方式を切替。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("THESIS_PYTHON", r"C:\Users\aoi7y\miniconda3\envs\flvl\python.exe")
CONFIG = "config/default.yaml"


def _run_bg(cmd: list[str], run_id: str, env: dict | None = None, log_name: str = "proc") -> subprocess.Popen:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    log_path = _ROOT / "artifacts" / "runs" / "_e2e_logs" / f"{run_id}_{log_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=str(_ROOT),
        env=full_env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repertoire", default="ac_lpred_df")
    ap.add_argument("--num-rounds", type=int, default=1)
    ap.add_argument("--server-wait-sec", type=int, default=90)
    args = ap.parse_args()

    run_id = datetime.now().strftime(f"run_%Y%m%d_%H%M%S_smoke_{args.repertoire}")

    common = [
        "--config", CONFIG,
        "--num-rounds", str(args.num_rounds),
        "--num-clients", "1",
        "--max-client-steps", "2",
        "--distill-steps", "2",
        "--repertoire", args.repertoire,
    ]
    server_only = [
        "--client-train-per-class", "2",
        "--client-eval-per-class", "2",
        "--server-train-per-class", "2",
        "--server-eval-per-class", "2",
        "--max-server-steps", "2",
        "--run-id", run_id,
    ]
    server_cmd = [PY, "scripts/step07_flower_server.py", *common, *server_only]
    client_cmd = [PY, "scripts/step07_flower_client.py", *common, "--server", "127.0.0.1:8080"]

    print(f"[step07-e2e] repertoire={args.repertoire} run_id={run_id}", flush=True)
    server = _run_bg(server_cmd, run_id, {"THESIS_FL_RUN_ID": run_id}, log_name="server")
    print(f"[step07-e2e] waiting {args.server_wait_sec}s for server init...", flush=True)
    time.sleep(args.server_wait_sec)

    env = {"THESIS_CLIENT_ID": "0", "THESIS_FL_RUN_ID": run_id}
    client = _run_bg(client_cmd, run_id, env, log_name="client_0")
    client_rc = client.wait(timeout=14400)
    server_rc = server.wait(timeout=7200)

    run_dir = _ROOT / "artifacts" / "runs" / f"step07_fl_{args.repertoire}" / run_id
    summary_ok = (run_dir / "summary.json").is_file()
    ok = client_rc == 0 and server_rc == 0 and summary_ok
    print(
        f"[step07-e2e] ok={ok} server_rc={server_rc} client_rc={client_rc} summary={summary_ok} run_dir={run_dir}",
        flush=True,
    )
    if not ok:
        for name in ("server", "client_0"):
            log_file = _ROOT / "artifacts" / "runs" / "_e2e_logs" / f"{run_id}_{name}.log"
            if log_file.is_file():
                tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
                print(f"--- {name} (tail) ---\n{tail}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
