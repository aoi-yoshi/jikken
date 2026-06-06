"""FL パラメータ flatten / FedAvg / checkpoint のユニット検証（モデルロードなし）。"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.fl_utils import fedavg_weights, load_fl_checkpoint, save_fl_checkpoint


def main() -> None:
    names = ["classifier.weight", "lora.client.q_proj"]
    v0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v1 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    _, avg = fedavg_weights([(names, v0), (names, v1)])
    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    assert np.allclose(avg, expected), avg

    out = _ROOT / "artifacts" / "checkpoints" / "_fl_unit_test.pt"
    save_fl_checkpoint(out, names=names, vector=avg, meta={"test": True})
    n2, v2, meta = load_fl_checkpoint(out)
    assert n2 == names
    assert np.allclose(v2, avg)
    assert meta.get("test") is True
    print("verify_fl_pipeline: OK", out)


if __name__ == "__main__":
    main()
