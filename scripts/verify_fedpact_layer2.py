"""モデルをロードせず FedPACT Layer 2 の prototype・検索・監査を検証する。"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.step07.fedpact_layer2 import (
    assert_private_data_absent,
    build_classwise_prototypes,
    build_server_payload,
    proxy_match_metrics,
    rank_server_seeds,
    select_best_proxy,
)


def main() -> None:
    records = [
        {"sample_id": "private-a", "label": 1, "pre_logits": [1.0, 0.0], "post_logits": [0.0, 2.0]},
        {"sample_id": "private-b", "label": 1, "pre_logits": [0.8, 0.2], "post_logits": [0.1, 1.9]},
    ]
    prototype = build_classwise_prototypes(records, client_id="client-0", server_round=1)[0]
    assert prototype["frequency"] == 2
    assert "sample_id" not in prototype and "mu_pre" not in prototype

    payload = build_server_payload(
        [prototype],
        lora_update={"artifact": "trainable_update.npz", "parameter_count": 10},
    )
    assert_private_data_absent(payload)
    try:
        assert_private_data_absent({"sample_ids": ["leak"]})
    except ValueError:
        pass
    else:
        raise AssertionError("privacy audit did not detect sample_ids")

    ranked = rank_server_seeds(
        prototype,
        [
            {"seed_id": "far", "label": 0, "updated_logits": [2.0, 0.0]},
            {"seed_id": "near", "label": 1, "updated_logits": [0.0, 2.0]},
        ],
    )
    assert ranked[0]["seed_id"] == "near"

    near_metrics = proxy_match_metrics(
        prototype,
        global_before_logits=[0.8, 0.2],
        global_after_logits=[0.0, 2.0],
        lambda_post=1.0,
        lambda_delta=1.0,
    )
    far_metrics = proxy_match_metrics(
        prototype,
        global_before_logits=[0.8, 0.2],
        global_after_logits=[2.0, 0.0],
        lambda_post=1.0,
        lambda_delta=1.0,
    )
    selected = select_best_proxy(
        [
            {"seed_id": "near", "transformation": {"name": "identity", "value": None}, "metrics": near_metrics},
            {"seed_id": "far", "transformation": {"name": "identity", "value": None}, "metrics": far_metrics},
        ]
    )
    assert selected["seed_id"] == "near"
    assert selected["qinv"] is None
    print("FedPACT Layer 2 verification: OK")


if __name__ == "__main__":
    main()
