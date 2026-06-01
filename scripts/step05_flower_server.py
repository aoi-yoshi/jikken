"""
Step 5: Flower サーバ（FedAvg）。別ターミナルで step06 を2回起動する想定。
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

from src.config_loader import merged_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default="8080")
    args = ap.parse_args()
    cfg = merged_config()
    fcfg = cfg["fl"]

    import flwr as fl
    from flwr.server import ServerConfig
    from flwr.server.strategy import FedAvg

    addr = f"{args.host}:{args.port}"
    strategy = FedAvg(
        fraction_fit=float(fcfg.get("sample_fraction", 1.0)),
        fraction_evaluate=0.0,
        min_fit_clients=int(fcfg.get("min_fit_clients", 2)),
        min_available_clients=int(fcfg.get("min_available_clients", 2)),
    )
    fl.server.start_server(
        server_address=addr,
        config=ServerConfig(num_rounds=int(fcfg.get("num_rounds", 2))),
        strategy=strategy,
        grpc_max_message_length=512 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
