"""
Step 7 + Flower FL スモークテスト（1 round, 小データ, AC 損失）。
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
PY = os.environ.get("THESIS_PYTHON", sys.executable)
CONFIG = "config/default.yaml"


def _run_bg(cmd: list[str], run_id: str, env: dict | None = None, log_name: str = "proc") -> subprocess.Popen:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    log_path = _ROOT / "artifacts" / "runs" / "step05_fl" / "_e2e_logs" / f"{run_id}_{log_name}.log"
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
    ap.add_argument("--consistency-mode", default="lpred", choices=["lpred", "lpred_grad", "lpred_post"])
    ap.add_argument("--num-rounds", type=int, default=1)
    ap.add_argument("--production", action="store_true", help="default.yaml のデータ規模で実行（スモークの小データを使わない）")
    args = ap.parse_args()

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S") + f"_ac_{args.consistency_mode}"
    if args.production:
        smoke_common = [
            "--config", CONFIG,
            "--num-rounds", str(args.num_rounds),
        ]
    else:
        smoke_common = [
            "--config", CONFIG,
            "--num-rounds", str(args.num_rounds),
            "--max-train-samples", "4",
            "--max-samples-per-client", "2",
            "--eval-max", "4",
        ]
    client_extra = ["--consistency-mode", args.consistency_mode]
    server_cmd = [PY, "scripts/step05_flower_server.py", *smoke_common, "--run-id", run_id]
    client_cmd = [PY, "scripts/step05_flower_client.py", *smoke_common, *client_extra, "--server", "127.0.0.1:8080"]

    print(f"[step07-fl-smoke] run_id={run_id} mode={args.consistency_mode}", flush=True)
    server = _run_bg(server_cmd, run_id, env={"THESIS_FL_RUN_ID": run_id}, log_name="server")
    print("[step07-fl-smoke] waiting 180s for server init...", flush=True)
    time.sleep(180)

    clients = []
    for cid in (0, 1):
        print(f"[step07-fl-smoke] starting client {cid}", flush=True)
        env = {"THESIS_CLIENT_ID": str(cid), "THESIS_FL_RUN_ID": run_id}
        if cid == 1:
            env["THESIS_FORCE_CPU"] = "1"
        p = _run_bg(client_cmd, run_id, env=env, log_name=f"client_{cid}")
        clients.append(p)
        time.sleep(5)

    client_rc = 0
    for cid, p in enumerate(clients):
        rc = p.wait(timeout=14400)
        log_file = _ROOT / "artifacts" / "runs" / "step05_fl" / "_e2e_logs" / f"{run_id}_client_{cid}.log"
        if log_file.is_file():
            tail = log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
            try:
                print(f"--- client {cid} (tail) ---\n{tail}", flush=True)
            except UnicodeEncodeError:
                print(f"--- client {cid} (tail, sanitized) ---", flush=True)
                print(tail.encode("ascii", errors="replace").decode("ascii"), flush=True)
        if rc != 0:
            client_rc = rc

    server_rc = server.wait(timeout=7200)
    server_log = _ROOT / "artifacts" / "runs" / "step05_fl" / "_e2e_logs" / f"{run_id}_server.log"
    if server_log.is_file():
        tail = server_log.read_text(encoding="utf-8", errors="replace")[-4000:]
        try:
            print(f"--- server (tail) ---\n{tail}", flush=True)
        except UnicodeEncodeError:
            print("--- server (tail, sanitized) ---", flush=True)
            print(tail.encode("ascii", errors="replace").decode("ascii"), flush=True)

    from src.config_loader import merged_config
    from src.fl_utils import resolve_latest_checkpoint

    ckpt = resolve_latest_checkpoint(merged_config(), _ROOT)
    ok = ckpt is not None and ckpt.is_file() and client_rc == 0 and server_rc == 0
    print(f"[step07-fl-smoke] checkpoint_exists={ckpt is not None and ckpt.is_file()} path={ckpt}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
