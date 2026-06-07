"""
FL 最小構成: 学習 1 件/クライアント × 2 clients × 1 round。
新指標付きでローカル FedAvg シミュレーションを実行（Flower 手動前の検証用）。

データ数について:
  2 クライアントで各 1 件学習するため max_train_samples=2, max_samples_per_client=1。
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    run_id = datetime.now().strftime("run_minimal_%Y%m%d_%H%M%S")
    cmd = [
        PY,
        str(_ROOT / "scripts" / "fl_local_fedavg_sim.py"),
        "--config",
        "config/default.yaml",
        "--run-id",
        run_id,
        "--num-rounds",
        "1",
        "--num-clients",
        "2",
        "--max-train-samples",
        "2",
        "--max-samples-per-client",
        "1",
        "--train-ratio",
        "1.0",
        "--eval-max",
        "2",
    ]
    if "--cpu" in sys.argv:
        cmd.append("--cpu")
    print("[minimal] ", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(_ROOT)).returncode
    if rc == 0:
        summary = _ROOT / "artifacts" / "runs" / "step05_fl" / run_id / "summary.json"
        print(f"[minimal] OK run_id={run_id}", flush=True)
        if summary.is_file():
            print(summary.read_text(encoding="utf-8"), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
