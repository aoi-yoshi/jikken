"""
Flower FedAvg E2E スモークテスト（1 round, 小データ）。
RTX 5050 等の単一 GPU 環境向け: クライアントは順次起動する。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("THESIS_PYTHON", r"C:\Users\aoi7y\miniconda3\envs\flvl\python.exe")
CONFIG = "config/default.yaml"
# スモーク時は default.yaml の fl.num_rounds=1, train.max_train_samples=8 等に一時変更するか CLI で指定
RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _run_bg(cmd: list[str], env: dict | None = None, log_name: str = "proc") -> subprocess.Popen:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    log_path = _ROOT / "artifacts" / "runs" / "step05_fl" / "_e2e_logs" / f"{RUN_ID}_{log_name}.log"
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
    smoke_common = [
        "--config", CONFIG,
        "--num-rounds", "1",
        "--max-train-samples", "4",
        "--max-samples-per-client", "2",
        "--eval-max", "4",
    ]
    server_cmd = [PY, "scripts/step05_flower_server.py", *smoke_common, "--run-id", RUN_ID]
    client_cmd = [PY, "scripts/step05_flower_client.py", *smoke_common, "--server", "127.0.0.1:8080"]
    post_cmd = [PY, "scripts/step05_fl_post_eval.py", *smoke_common, "--run-id", RUN_ID]

    print(f"[e2e] run_id={RUN_ID}", flush=True)
    server = _run_bg(server_cmd, {"THESIS_FL_RUN_ID": RUN_ID}, log_name="server")
    print("[e2e] waiting 120s for server init...", flush=True)
    time.sleep(120)

    clients = []
    for cid in (0, 1):
        print(f"[e2e] starting client {cid}", flush=True)
        env = {"THESIS_CLIENT_ID": str(cid), "THESIS_FL_RUN_ID": RUN_ID}
        if cid == 1:
            env["THESIS_FORCE_CPU"] = "1"
        p = _run_bg(client_cmd, env, log_name=f"client_{cid}")
        clients.append(p)
        time.sleep(5)

    print("[e2e] waiting for clients...", flush=True)
    client_rc = 0
    for cid, p in enumerate(clients):
        rc = p.wait(timeout=14400)
        log_file = _ROOT / "artifacts" / "runs" / "step05_fl" / "_e2e_logs" / f"{RUN_ID}_client_{cid}.log"
        if log_file.is_file():
            tail = log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
            print(f"--- client {cid} (tail) ---\n{tail}", flush=True)
        if rc != 0:
            client_rc = rc

    server_rc = server.wait(timeout=7200)
    server_log = _ROOT / "artifacts" / "runs" / "step05_fl" / "_e2e_logs" / f"{RUN_ID}_server.log"
    if server_log.is_file():
        tail = server_log.read_text(encoding="utf-8", errors="replace")[-4000:]
        print(f"--- server (tail) ---\n{tail}", flush=True)

    post = subprocess.run(
        post_cmd,
        cwd=str(_ROOT),
        check=False,
    )

    ckpt = _ROOT / "artifacts" / "checkpoints" / "step05_fedavg_global.pt"
    ok = ckpt.is_file() and client_rc == 0 and server_rc == 0
    print(f"[e2e] checkpoint_exists={ckpt.is_file()} ok={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
