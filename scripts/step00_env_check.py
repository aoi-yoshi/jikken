"""
Step 0: GPU / PyTorch / Transformers の確認（このPC内で完結する前提チェック）。
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    root = _ROOT
    print("Project root:", root)

    import torch

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))

    import transformers

    print("transformers:", transformers.__version__)

    # Qwen2.5-VL はトップレベルパッケージではなく transformers 内に含まれる
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401

        print("Qwen2.5-VL (transformers): import OK")
    except Exception as e:
        print("Qwen2.5-VL (transformers): import FAILED:", e)

    if torch.version.cuda is None and not torch.cuda.is_available():
        print(
            "Note: CPU-only torch (+cpu). 3B 学習・FL は GPU 推奨。"
            "CUDA 版は https://pytorch.org/get-started/locally/ のコマンドで"
            "この venv に入れ替えてください。"
        )

    print("\nOK: Step0 checks finished. Next: python scripts/step01_qwen_infer.py")


if __name__ == "__main__":
    main()
