"""
Step 0: ローカル開発 PC 上の Python / GPU 環境チェック。

ネットワーク・GCP・Flower サーバは不要（import と CUDA だけ確認）。
修論の最終構成（7B server / 3B edge on GCP）は docs/gcp_migration_plan.md。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_FLVL_PY = Path(r"C:\Users\aoi7y\miniconda3\envs\flvl\python.exe")


def _warn_if_not_flvl() -> None:
    exe = Path(sys.executable).resolve()
    if _FLVL_PY.is_file() and exe != _FLVL_PY.resolve():
        print("WARNING: flvl 以外の Python です。実験は flvl 推奨。", flush=True)
        print(f"  今: {exe}", flush=True)
        print(f"  推奨: {_FLVL_PY}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-suffix", default="")
    args = ap.parse_args()

    _warn_if_not_flvl()

    from src.config_loader import merged_config
    from src.resource_metrics import build_environment_block, print_gpu_diagnostics
    from src.run_context import init_run

    cfg = merged_config()
    run_id, run_dir, log, env = init_run(
        cfg, step="0", step_dir="step00_env", log_name="check", run_id=args.run_id, run_suffix=args.run_suffix, cli=vars(args)
    )

    import torch
    import transformers

    print("Project root:", _ROOT)
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print("cuda device:", gpu_name)
    print("transformers:", transformers.__version__)
    print_gpu_diagnostics()

    qwen_ok = True
    qwen_err = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
        print("Qwen2.5-VL (transformers): import OK")
    except Exception as e:
        qwen_ok = False
        qwen_err = str(e)
        print("Qwen2.5-VL (transformers): import FAILED:", e)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "environment": env,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "transformers": transformers.__version__,
        "qwen25_vl_import_ok": qwen_ok,
        "qwen25_vl_import_error": qwen_err,
        "status": "ok" if qwen_ok and (torch.cuda.is_available() or torch.version.cuda) else "check_warnings",
    }
    log.save_summary(summary)
    log.log_meta({"status": "completed"})
    print(f"\nOK: Step0 finished. run_dir={run_dir}")
    print("Next: python scripts/step01_qwen_infer.py")


if __name__ == "__main__":
    main()
