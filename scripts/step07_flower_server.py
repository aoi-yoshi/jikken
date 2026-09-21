"""
Step 7: Flower サーバ — レパートリー切替（提案系: サーバ整合 + logits 配布 / baseline: 重み配布）。

レパートリー一覧: src/step07/repertoire.py / docs/step07_repertoires.md
切替: --repertoire <name> または config step07.repertoire。
別ターミナルで step07_flower_client を min_fit_clients 回起動する。
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
from src.dataset_manifest import read_manifest_filtered
from src.experiment_record import save_config_snapshot
from src.fl_flower_config import build_fl_flower_settings_full
from src.fl_step07.config import apply_step07_cli_overrides, step07_fl_checkpoint_path, step07_run_dir_name
from src.fl_step07.server_phase import Step07ServerContext
from src.fl_step07.strategy import build_step07_flower_strategy_class
from src.fl_utils import build_initial_fl_state
from src.paths import ensure_dirs
from src.resource_metrics import build_environment_block
from src.run_context import init_run, make_run_id
from src.step07.data_splits import save_data_splits_artifact, split_step07_data_pools
from src.step07.repertoire import (
    list_repertoires,
    repertoire_from_stage_profile,
    repertoire_record,
    resolve_repertoire,
    validate_repertoire_for_hetero,
)
from src.step07.pipeline import distill_pipeline_mode, resolve_step07_model_ids
from src.train_common import device_or_auto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8080")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-suffix", default="")
    ap.add_argument("--num-rounds", type=int, default=None)
    ap.add_argument("--num-clients", type=int, default=None)
    ap.add_argument("--eval-max", type=int, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument(
        "--repertoire",
        default=None,
        help=f"レパートリー名（config step07.repertoire より優先）。候補: {list_repertoires(enabled_only=False)}",
    )
    ap.add_argument("--stage-profile", default=None, help="旧 stage_profile 名（repertoire に写像）")
    ap.add_argument("--distill-steps", type=int, default=None)
    ap.add_argument(
        "--client-train-per-class",
        type=int,
        default=20,
        help="1 client・1 class あたり train。2 client 本番: 20×2×2=80 合計",
    )
    ap.add_argument(
        "--client-eval-per-class",
        type=int,
        default=15,
        help="共有 client eval（クラスあたり）。2 class: 15×2=30 合計",
    )
    ap.add_argument(
        "--server-train-per-class",
        type=int,
        default=20,
        help="server_train（クラスあたり）。2 class: 20×2=40 合計",
    )
    ap.add_argument(
        "--server-eval-per-class",
        type=int,
        default=15,
        help="server_eval（クラスあたり）。2 class: 15×2=30 合計",
    )
    ap.add_argument("--max-client-steps", type=int, default=None)
    ap.add_argument("--max-server-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = merged_config()
    cfg = apply_step07_cli_overrides(cfg, args)
    fcfg = cfg["fl"]
    scfg = cfg.setdefault("step07", {})
    art = cfg["artifacts"]
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    seed = int(tcfg["seed"])

    # 旧 stage_profile 名からの互換写像（--repertoire が優先）
    cli_rep = args.repertoire
    if not cli_rep and args.stage_profile:
        mapped = repertoire_from_stage_profile(args.stage_profile)
        if mapped is None:
            raise SystemExit(f"stage_profile={args.stage_profile} に対応する repertoire がありません")
        cli_rep = mapped.name
    rep = resolve_repertoire(cfg, cli_rep)

    server_model_id, client_model_id = resolve_step07_model_ids(cfg)
    validate_repertoire_for_hetero(rep, distill_pipeline_mode(cfg))

    num_rounds = int(args.num_rounds if args.num_rounds is not None else fcfg.get("num_rounds", 2))
    num_clients = int(
        args.num_clients
        if args.num_clients is not None
        else scfg.get("num_clients", fcfg.get("num_clients", 2))
    )
    fcfg["num_clients"] = num_clients
    fcfg["min_fit_clients"] = num_clients
    fcfg["min_available_clients"] = num_clients
    scfg["num_clients"] = num_clients

    run_id = args.run_id or make_run_id(str(args.run_suffix))
    step_dir = step07_run_dir_name(rep.name)
    ensure_dirs(Path(art["checkpoints"]))

    latest_ptr = Path(art["runs"]) / step_dir / "LATEST_RUN_ID"
    latest_ptr.parent.mkdir(parents=True, exist_ok=True)
    latest_ptr.write_text(run_id, encoding="utf-8")

    rep_record = repertoire_record(rep)
    run_id, run_dir, log, _env = init_run(
        cfg,
        step="7-fl",
        step_dir=step_dir,
        log_name="fl_server",
        run_id=run_id,
        cli=vars(args),
        extra_meta={
            "scheme": "step07_flower",
            "repertoire": rep_record,
            "num_clients": num_clients,
            "server_model_id": server_model_id,
            "client_model_id": client_model_id,
        },
    )

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    client_trains, client_eval, server_train, server_eval = split_step07_data_pools(
        rows,
        num_clients=num_clients,
        client_train_per_class=args.client_train_per_class,
        client_eval_per_class=args.client_eval_per_class,
        server_train_per_class=args.server_train_per_class,
        server_eval_per_class=args.server_eval_per_class,
        seed=seed,
    )
    splits_path = save_data_splits_artifact(
        run_dir,
        client_trains=client_trains,
        client_eval=client_eval,
        server_train=server_train,
        server_eval=server_eval,
        seed=seed,
    )
    print(f"[step07-server] repertoire={rep.name} ({rep.kind}) data_splits={splits_path}", flush=True)

    save_config_snapshot(
        run_dir,
        cfg=cfg,
        cli_args=vars(args),
        extra={
            "run_id": run_id,
            "scheme": "step07_flower",
            "repertoire": rep_record,
            "data_splits": str(splits_path),
        },
    )

    init_device = torch.device(device_or_auto(prefer_cuda=False))
    print(f"[step07-server] building initial client parameters on {init_device}...", flush=True)
    param_names, init_vec = build_initial_fl_state(cfg, init_device, model_id=client_model_id)
    ckpt_path = step07_fl_checkpoint_path(cfg, run_id, _ROOT)

    server_device = torch.device(device_or_auto(True))
    server_ctx = Step07ServerContext(
        cfg=cfg,
        root=_ROOT,
        run_dir=run_dir,
        repertoire=rep,
        server_train=server_train,
        server_eval=server_eval,
        device=server_device,
        param_names=param_names,
        max_server_steps=args.max_server_steps,
    )

    addr = f"{args.host}:{args.port}"
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
    flower_settings = build_fl_flower_settings_full(
        cfg,
        role="server",
        server_host=str(args.host),
        server_port=str(args.port),
        trainable_vector_dim=int(init_vec.size),
        param_name_count=len(param_names),
        grpc_max_message_length_bytes=GRPC_MAX,
        round_timeout=None,
        strategy_class=f"Step07Strategy[{rep.name}]",
        strategy_kwargs=strategy_kwargs,
        server_config_kwargs={"num_rounds": num_rounds, "round_timeout": None},
    )
    log.log({"event": "experiment_record", "flower_settings": flower_settings, "repertoire": rep_record})
    log.log_meta({"flower_settings": flower_settings, "repertoire": rep_record, "data_splits": str(splits_path)})

    import flwr as fl
    from flwr.common import ndarrays_to_parameters
    from flwr.server import ServerConfig

    StrategyCls = build_step07_flower_strategy_class(cfg)
    strategy = StrategyCls(
        total_rounds=num_rounds,
        checkpoint_path=ckpt_path,
        names=param_names,
        init_vec=init_vec,
        logger=log,
        run_id=run_id,
        run_dir=run_dir,
        root=_ROOT,
        repertoire=rep,
        server_ctx=server_ctx,
        flower_settings=flower_settings,
        fraction_fit=float(fcfg.get("sample_fraction", 1.0)),
        fraction_evaluate=float(fcfg.get("sample_fraction", 1.0)),
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=ndarrays_to_parameters([init_vec]),
        accept_failures=True,
        inplace=True,
    )

    print(
        f"[step07-server] run_id={run_id} repertoire={rep.name} rounds={num_rounds} "
        f"clients={num_clients} distribution={rep.distribution} addr={addr}",
        flush=True,
    )
    print(f"[step07-server] checkpoint: {ckpt_path}", flush=True)
    print(f"[step07-server] Clients: THESIS_FL_RUN_ID={run_id}", flush=True)

    fl.server.start_server(
        server_address=addr,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        grpc_max_message_length=GRPC_MAX,
    )

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "scheme": "step07_flower",
        "repertoire": rep_record,
        "environment": build_environment_block(cfg),
        "flower_settings": flower_settings,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "checkpoint": str(ckpt_path),
        "data_splits": str(splits_path),
        "status": "completed",
    }
    log.save_summary(summary)
    log.log_meta({"status": "completed"})
    print("[step07-server] Finished.", flush=True)


if __name__ == "__main__":
    main()
