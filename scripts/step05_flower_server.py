"""
Step 5: Flower サーバ（FedAvg / FedProx + グローバル LoRA 保存 + 評価集約ログ）。
別ターミナルで step05_flower_client を min_fit_clients 回起動する想定。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from src.config_loader import merged_config
from src.fl_data import apply_fl_cli_overrides, build_fl_dataset, save_partition_artifact
from src.fl_flower_config import build_fl_flower_settings_full
from src.fl_strategy import build_flower_strategy_class, fl_strategy_name, is_fedprox
from src.fl_utils import build_initial_fl_state
from src.paths import ensure_dirs
from src.run_context import init_run, make_run_id
from src.resource_metrics import build_environment_block
from src.train_common import device_or_auto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8080")
    ap.add_argument("--run-id", default=None, help="共有 run_id（クライアントも同じ値を指定）")
    ap.add_argument("--run-suffix", default="", help="run_id suffix")
    ap.add_argument("--num-rounds", type=int, default=None)
    ap.add_argument("--partition-mode", default=None)
    ap.add_argument("--label-skew-alpha", type=float, default=None)
    ap.add_argument("--eval-max", type=int, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--train-ratio", type=float, default=None)
    ap.add_argument("--max-samples-per-client", type=int, default=None)
    ap.add_argument("--num-clients", type=int, default=None)
    ap.add_argument("--strategy", default=None, help="FedAvg | FedProx（default.yaml 上書き）")
    ap.add_argument("--fedprox-mu", type=float, default=None)
    args = ap.parse_args()

    cfg = merged_config()
    cfg = apply_fl_cli_overrides(cfg, args)
    fcfg = cfg["fl"]
    art = cfg["artifacts"]
    num_rounds = int(fcfg.get("num_rounds", 2))
    num_clients = int(fcfg.get("num_clients", 2))
    strategy_name = fl_strategy_name(cfg)

    run_id = args.run_id or make_run_id(str(args.run_suffix))
    ensure_dirs(Path(art["checkpoints"]))

    latest_ptr = Path(art["runs"]) / "step05_fl" / "LATEST_RUN_ID"
    latest_ptr.write_text(run_id, encoding="utf-8")

    GRPC_MAX = 512 * 1024 * 1024
    strategy_kwargs = {
        "accept_failures": True,
        "inplace": True,
        "initial_parameters_provided": True,
        "evaluate_fn": None,
        "on_fit_config_fn": None,
        "on_evaluate_config_fn": None,
        "fit_metrics_aggregation_fn": "default_weighted_avg",
        "evaluate_metrics_aggregation_fn": "default_weighted_avg",
    }
    if is_fedprox(cfg):
        strategy_kwargs["proximal_mu"] = float(fcfg.get("fedprox_mu", 0.01))

    run_id, run_dir, log, _env = init_run(
        cfg,
        step="5",
        step_dir="step05_fl",
        log_name="fl_server",
        run_id=run_id,
        cli=vars(args),
    )

    _pool, _eval_rows, client_parts, part_meta = build_fl_dataset(cfg, root=_ROOT)
    part_path = save_partition_artifact(run_dir, part_meta, client_parts)
    print(f"[server] partition_mode={part_meta['partition_mode']} saved {part_path}", flush=True)
    for c in part_meta["clients"]:
        print(f"  client {c['client_id']}: n={c['num_samples']} labels={c['label_counts']}", flush=True)

    init_device = torch.device(device_or_auto(prefer_cuda=False))
    print(f"[server] Building shared initial parameters on {init_device}...", flush=True)
    param_names, init_vec = build_initial_fl_state(cfg, init_device)
    ckpt_name = str(fcfg.get("checkpoint_name", "step05_fedavg_global.pt"))
    ckpt_path = Path(art["checkpoints"]) / ckpt_name

    addr = f"{args.host}:{args.port}"
    flower_settings = build_fl_flower_settings_full(
        cfg,
        role="server",
        server_host=str(args.host),
        server_port=str(args.port),
        trainable_vector_dim=int(init_vec.size),
        param_name_count=len(param_names),
        grpc_max_message_length_bytes=GRPC_MAX,
        round_timeout=None,
        strategy_class=strategy_name,
        strategy_kwargs=strategy_kwargs,
        server_config_kwargs={"num_rounds": num_rounds, "round_timeout": None},
    )
    log.log({"event": "experiment_record", "flower_settings": flower_settings})
    log.log_meta({"fl_partition": part_meta, "flower_settings": flower_settings})

    import flwr as fl
    from flwr.common import ndarrays_to_parameters
    from flwr.server import ServerConfig

    StrategyCls = build_flower_strategy_class(cfg)
    initial_parameters = ndarrays_to_parameters([init_vec])

    strategy = StrategyCls(
        total_rounds=num_rounds,
        checkpoint_path=ckpt_path,
        names=param_names,
        init_vec=init_vec,
        logger=log,
        run_id=run_id,
        partition_mode=part_meta["partition_mode"],
        flower_settings=flower_settings,
        strategy_label=strategy_name,
        fraction_fit=float(fcfg.get("sample_fraction", 1.0)),
        fraction_evaluate=float(fcfg.get("sample_fraction", 1.0)),
        min_fit_clients=int(fcfg.get("min_fit_clients", num_clients)),
        min_evaluate_clients=int(fcfg.get("min_fit_clients", num_clients)),
        min_available_clients=int(fcfg.get("min_available_clients", num_clients)),
        initial_parameters=initial_parameters,
        accept_failures=bool(strategy_kwargs.get("accept_failures", True)),
        inplace=bool(strategy_kwargs.get("inplace", True)),
        **({k: v for k, v in strategy_kwargs.items() if k in ("proximal_mu",)}),
    )

    print(
        f"[server] run_id={run_id} strategy={strategy_name} rounds={num_rounds} clients={num_clients} "
        f"mode={part_meta['partition_mode']} addr={addr}",
        flush=True,
    )
    print(f"[server] Clients: set THESIS_FL_RUN_ID={run_id}", flush=True)

    fl.server.start_server(
        server_address=addr,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        grpc_max_message_length=GRPC_MAX,
    )

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "environment": build_environment_block(cfg),
        "flower_settings": flower_settings,
        "fl_partition_summary": part_meta,
        "num_rounds": num_rounds,
        "strategy": strategy_name,
        "partition_mode": part_meta["partition_mode"],
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
