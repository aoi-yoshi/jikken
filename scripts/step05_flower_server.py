"""
Step 5: Flower サーバ（FedAvg + グローバル LoRA 保存 + 評価集約ログ）。
別ターミナルで step06 を min_fit_clients 回起動する想定。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from src.config_loader import deep_merge, load_yaml, merged_config
from src.fl_utils import build_initial_fl_state, save_fl_checkpoint
from src.logging_utils import RunLogger
from src.paths import ensure_dirs
from src.resource_metrics import build_environment_block
from src.train_common import device_or_auto


def _load_config(config_arg: str) -> dict:
    cfg = merged_config()
    if config_arg and config_arg != "config/default.yaml":
        override = load_yaml(_ROOT / config_arg if not Path(config_arg).is_absolute() else config_arg)
        cfg = deep_merge(cfg, override)
    return cfg


def _make_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default="8080")
    ap.add_argument("--run-id", default=None, help="共有 run_id（クライアントも同じ値を指定）")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    fcfg = cfg["fl"]
    art = cfg["artifacts"]
    num_rounds = int(fcfg.get("num_rounds", 2))

    run_id = args.run_id or _make_run_id()
    run_dir = Path(art["runs"]) / "step06_fl" / run_id
    ensure_dirs(run_dir, Path(art["checkpoints"]))
    latest_ptr = Path(art["runs"]) / "step06_fl" / "LATEST_RUN_ID"
    latest_ptr.write_text(run_id, encoding="utf-8")

    log = RunLogger(run_dir, name="fl_server")
    log.log_meta(
        {
            "step": "5-6",
            "run_id": run_id,
            "status": "started",
            "cli": vars(args),
            "environment": build_environment_block(cfg),
        }
    )

    init_device = torch.device(device_or_auto(prefer_cuda=False))
    print(f"[server] Building shared initial parameters on {init_device}...", flush=True)
    param_names, init_vec = build_initial_fl_state(cfg, init_device)
    ckpt_path = Path(art["checkpoints"]) / "step06_fedavg_global.pt"

    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerConfig
    from flwr.server.strategy import FedAvg

    initial_parameters = ndarrays_to_parameters([init_vec])

    class SavingFedAvg(FedAvg):
        def __init__(
            self,
            *,
            total_rounds: int,
            checkpoint_path: Path,
            names: List[str],
            logger: RunLogger,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.total_rounds = total_rounds
            self.checkpoint_path = checkpoint_path
            self.param_names = names
            self.logger = logger

        def aggregate_fit(self, server_round, results, failures):
            aggregated, metrics = super().aggregate_fit(server_round, results, failures)
            round_metrics: Dict[str, Any] = {
                "server_round": server_round,
                "num_clients": len(results),
                "fit_metrics": dict(metrics) if metrics else {},
            }
            if aggregated is not None:
                ndarrays = parameters_to_ndarrays(aggregated.parameters)
                vec = ndarrays[0] if ndarrays else init_vec
                round_metrics["param_norm"] = float(np.linalg.norm(vec))
                if server_round >= self.total_rounds:
                    save_fl_checkpoint(
                        self.checkpoint_path,
                        names=self.param_names,
                        vector=vec,
                        meta={
                            "run_id": run_id,
                            "num_rounds": self.total_rounds,
                            "strategy": "FedAvg",
                            "param_count": len(self.param_names),
                        },
                    )
                    round_metrics["global_checkpoint"] = str(self.checkpoint_path)
                    print(f"[server] Saved global LoRA checkpoint: {self.checkpoint_path}", flush=True)
            self.logger.log(round_metrics)
            return aggregated, metrics

        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            eval_row: Dict[str, Any] = {
                "server_round": server_round,
                "eval_loss": float(loss) if loss is not None else None,
                "eval_metrics": dict(metrics) if metrics else {},
                "num_clients": len(results),
            }
            self.logger.log(eval_row)
            return loss, metrics

    strategy = SavingFedAvg(
        total_rounds=num_rounds,
        checkpoint_path=ckpt_path,
        names=param_names,
        logger=log,
        fraction_fit=float(fcfg.get("sample_fraction", 1.0)),
        fraction_evaluate=float(fcfg.get("sample_fraction", 1.0)),
        min_fit_clients=int(fcfg.get("min_fit_clients", 2)),
        min_evaluate_clients=int(fcfg.get("min_fit_clients", 2)),
        min_available_clients=int(fcfg.get("min_available_clients", 2)),
        initial_parameters=initial_parameters,
    )

    addr = f"{args.host}:{args.port}"
    print(f"[server] run_id={run_id} rounds={num_rounds} addr={addr}", flush=True)
    print(f"[server] Clients: set THESIS_FL_RUN_ID={run_id}", flush=True)

    fl.server.start_server(
        server_address=addr,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        grpc_max_message_length=512 * 1024 * 1024,
    )

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "num_rounds": num_rounds,
        "global_checkpoint": str(ckpt_path),
        "param_count": len(param_names),
        "trainable_vector_dim": int(init_vec.size),
        "status": "completed",
    }
    log.save_summary(summary)
    log.log_meta({"status": "completed"})
    print("[server] Finished.", flush=True)


if __name__ == "__main__":
    main()
