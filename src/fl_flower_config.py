"""Flower / Step5 実験の全設定を JSON ログ用に展開する。"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import flwr


def build_fl_flower_settings_full(
    cfg: Mapping[str, Any],
    *,
    role: str = "server",
    server_host: str = "127.0.0.1",
    server_port: str = "8080",
    trainable_vector_dim: int | None = None,
    param_name_count: int | None = None,
    grpc_max_message_length_bytes: int = 512 * 1024 * 1024,
    round_timeout: float | None = None,
    strategy_class: str = "FedAvg",
    strategy_kwargs: Mapping[str, Any] | None = None,
    server_config_kwargs: Mapping[str, Any] | None = None,
    client_env: Mapping[str, Any] | None = None,
    client_cli: Mapping[str, Any] | None = None,
    flwr_version: str | None = None,
) -> Dict[str, Any]:
    """教授報告・再現用: Flower API パラメータ + プロジェクト設定をすべて記録。"""
    fcfg = dict(cfg.get("fl", {}))
    pcfg = dict(fcfg.get("partition", {}))
    tcfg = dict(cfg.get("train", {}))
    lora = dict(cfg.get("lora", {}))
    mcfg = dict(cfg.get("model", {}))
    dcfg = dict(cfg.get("data", {}))
    ccfg = dict(cfg.get("consistency", {}))
    art = dict(cfg.get("artifacts", {}))

    sk = dict(strategy_kwargs or {})
    num_clients = int(fcfg.get("num_clients", 2))
    sample_fraction = float(fcfg.get("sample_fraction", 1.0))
    min_fit = int(fcfg.get("min_fit_clients", num_clients))
    min_avail = int(fcfg.get("min_available_clients", num_clients))
    num_rounds = int(fcfg.get("num_rounds", 2))

    comm_mb = None
    if trainable_vector_dim is not None:
        comm_mb = float(trainable_vector_dim * 4 / (1024**2))

    default_strategy = {
        "class": strategy_class,
        "fraction_fit": sample_fraction,
        "fraction_evaluate": sample_fraction,
        "min_fit_clients": min_fit,
        "min_evaluate_clients": min_fit,
        "min_available_clients": min_avail,
        "accept_failures": sk.get("accept_failures", True),
        "inplace": sk.get("inplace", True),
        "initial_parameters_provided": sk.get("initial_parameters_provided", True),
        "evaluate_fn": sk.get("evaluate_fn", None),
        "on_fit_config_fn": sk.get("on_fit_config_fn", None),
        "on_evaluate_config_fn": sk.get("on_evaluate_config_fn", None),
        "fit_metrics_aggregation_fn": sk.get("fit_metrics_aggregation_fn", "default_weighted_avg"),
        "evaluate_metrics_aggregation_fn": sk.get("evaluate_metrics_aggregation_fn", "default_weighted_avg"),
    }
    default_strategy.update({k: v for k, v in sk.items() if k not in default_strategy})

    sc = dict(server_config_kwargs or {})
    server_cfg = {
        "num_rounds": int(sc.get("num_rounds", num_rounds)),
        "round_timeout": sc.get("round_timeout", round_timeout),
    }

    return {
        "flwr_version": flwr_version or getattr(flwr, "__version__", None),
        "role": role,
        "flower_start_server": {
            "server_address": f"{server_host}:{server_port}",
            "grpc_max_message_length_bytes": int(grpc_max_message_length_bytes),
            "grpc_max_message_length_mb": float(grpc_max_message_length_bytes / (1024**2)),
            "certificates": None,
        },
        "flower_server_config": server_cfg,
        "flower_strategy": default_strategy,
        "flower_client": {
            "server_address": f"{server_host}:{server_port}" if role != "server" else None,
            "transport": "grpc",
            "client_type": "NumPyClient",
            "local_epochs_from_yaml": int(fcfg.get("local_epochs", 1)),
            "fit_config_from_server": "empty unless on_fit_config_fn (server_round only implicit)",
            "env": dict(client_env or {}),
            "cli": dict(client_cli or {}),
        },
        "fl_yaml": fcfg,
        "partition_yaml": pcfg,
        "train_yaml": tcfg,
        "lora_yaml": lora,
        "model_yaml": mcfg,
        "data_yaml": dcfg,
        "consistency_yaml": ccfg,
        "artifacts_yaml": art,
        "communication": {
            "trainable_vector_dim": trainable_vector_dim,
            "param_tensor_count": param_name_count,
            "upload_per_client_round_bytes_float32": int(trainable_vector_dim * 4) if trainable_vector_dim else None,
            "upload_per_client_round_mb_float32": comm_mb,
            "download_same_as_upload": True,
            "checkpoint_name": str(fcfg.get("checkpoint_name", "step05_fedavg_global.pt")),
        },
        "sampling_formula": {
            "description": "sample_size = max(floor(num_available * fraction_fit), min_fit_clients)",
            "fraction_fit": sample_fraction,
            "min_fit_clients": min_fit,
            "min_available_clients": min_avail,
        },
    }
