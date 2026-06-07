"""
FL 用クライアント分割を事前計算し、run フォルダに fl_partition.json を保存する。
学習前に Non-IID の偏りを確認する用途。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_loader import merged_config
from src.fl_data import apply_fl_cli_overrides, build_fl_dataset, save_partition_artifact


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview / save FL client partitions")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--partition-mode", default=None)
    ap.add_argument("--label-skew-alpha", type=float, default=None)
    ap.add_argument("--eval-max", type=int, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    args = ap.parse_args()

    cfg = merged_config()
    cfg = apply_fl_cli_overrides(cfg, args)
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(cfg["artifacts"]["runs"]) / "step05_fl" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "used_config.yaml")

    _pool, eval_rows, client_parts, meta = build_fl_dataset(cfg, root=_ROOT)
    out = save_partition_artifact(run_dir, meta, client_parts)

    print(f"run_id={run_id}")
    print(f"partition_mode={meta['partition_mode']}")
    print(f"pool={meta['train_pool_size']} eval={meta['eval_size']}")
    for c in meta["clients"]:
        print(
            f"  client {c['client_id']}: n={c['num_samples']} "
            f"labels={c['label_counts']} scenes={c.get('scene_counts', {})}"
        )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
