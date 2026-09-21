"""FedPACT P1-A: 2 clientの局所更新、FedAvg、再配布だけを1 round確認する。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_loader import load_yaml, merged_config  # noqa: E402
from src.dataset_manifest import load_sample_row, read_manifest_filtered  # noqa: E402
from src.fedpact_stage1 import (  # noqa: E402
    apply_named_vector,
    audit_communication_bundle,
    audit_server_payload,
    audit_split_video_overlap,
    canonical_video_id,
    flatten_named_tensors,
    l2_norm,
    sha256_file,
    sha256_json,
    state_digests,
    state_named_tensors,
    weighted_average_vectors,
)
from src.fl_utils import load_fl_checkpoint, save_fl_checkpoint, vector_to_trainable_state  # noqa: E402
from src.metrics import set_seed  # noqa: E402
from src.peft_setup import (  # noqa: E402
    attach_dual_lora,
    ensure_client_trainable,
    lora_params_for_adapter,
    sync_surrogate_from_client,
)
from src.step07.data_splits import (  # noqa: E402
    save_data_splits_artifact,
    split_paired_step07_data_pools,
)
from src.step07.fedpact_layer2 import build_classwise_prototypes, softmax_vector  # noqa: E402
from src.step07.train_steps import train_client_l_task  # noqa: E402
from src.train_common import device_or_auto  # noqa: E402
from src.vl_model import build_model, unfreeze_backbone  # noqa: E402


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _ROOT / path
    return path.resolve()


@torch.no_grad()
def _predict_rows(
    model,
    processor,
    rows: Sequence[dict],
    dcfg: dict,
    prompt: str,
    max_length: int,
) -> list[list[float]]:
    model.eval()
    model.backbone.set_adapter("client")
    output: list[list[float]] = []
    for row in rows:
        sample = load_sample_row(row, dcfg)
        logits, _ = model.forward_batch(
            processor, sample.frames, prompt, max_length, output_hidden_states=True
        )
        output.append(logits[0].detach().float().cpu().tolist())
        for frame in sample.frames:
            frame.close()
    return output


def _combined_named_state(model, client_lora) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    grouped = state_named_tensors(model, client_lora)
    lora_names, lora_vector = flatten_named_tensors(grouped["global_surrogate_lora"])
    head_names, head_vector = flatten_named_tensors(grouped["classification_head"])
    names = [*lora_names, *head_names]
    vector = np.concatenate([lora_vector, head_vector]).astype(np.float32, copy=False)
    meta = {
        "lora_names": lora_names,
        "head_names": head_names,
        "lora_count": int(lora_vector.size),
        "head_count": int(head_vector.size),
        "digests": state_digests(model, client_lora),
    }
    return names, vector, meta


def _git_context() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=_ROOT, check=False, capture_output=True, text=True
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": sha256_json(status.splitlines()),
    }


def _model_revision(model, processor) -> dict[str, Any]:
    config = getattr(getattr(model.backbone, "base_model", model.backbone), "config", None)
    if config is None:
        config = getattr(model.backbone, "config", None)
    tokenizer = getattr(processor, "tokenizer", None)
    return {
        "model_commit_hash": getattr(config, "_commit_hash", None),
        "model_name_or_path": getattr(config, "_name_or_path", None),
        "tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        if tokenizer is not None
        else None,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
    }


def _save_update_npz(
    path: Path, names: Sequence[str], delta: np.ndarray
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        names=np.asarray(list(names), dtype=np.str_),
        delta=np.asarray(delta, dtype=np.float32),
    )
    return {
        "artifact": path.name,
        "tensor_count": len(names),
        "parameter_count": int(np.asarray(delta).size),
        "l2_norm": l2_norm(delta),
        "sha256": sha256_file(path),
    }


def _run(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    phase_started = started
    timings: dict[str, float] = {}
    override = load_yaml(args.config)
    cfg = merged_config(override)
    scfg = dict(cfg["fedpact_stage1"])
    dcfg = dict(cfg["data"])
    tcfg = dict(cfg["train"])
    seed = int(tcfg["seed"])
    set_seed(seed)
    profile = str(getattr(args, "profile", "p1a"))
    if profile not in {"p1a", "p1d"}:
        raise ValueError(f"unknown profile: {profile}")
    client_pairs = int(
        scfg["client_train_per_class_p1d"]
        if profile == "p1d"
        else scfg["client_train_per_class_p1a"]
    )
    server_seed_pairs = int(
        scfg["server_seed_per_class_p1d"]
        if profile == "p1d"
        else scfg["server_seed_per_class_p1b"]
    )
    local_max_steps = 2 * client_pairs if profile == "p1d" else int(scfg["local_steps_p1a"])
    server_round = int(getattr(args, "round", None) or scfg["p1a_round"])

    config_path = _resolve_project_path(args.config)
    manifest_path = Path(dcfg["manifest_path"])
    initial_checkpoint = _resolve_project_path(
        str(getattr(args, "initial_global_checkpoint", None) or scfg["initial_global_checkpoint"])
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"paired manifest not found: {manifest_path}")
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(f"explicit M_G^0 checkpoint not found: {initial_checkpoint}")

    snapshot_dir = run_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, snapshot_dir / config_path.name)
    experiment_contract = {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "initial_checkpoint_path": str(initial_checkpoint),
        "initial_checkpoint_file_sha256": sha256_file(initial_checkpoint),
        "git": _git_context(),
        "seed": seed,
        "device_requested": args.device,
        "profile": profile,
        "server_round": server_round,
        "code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "fedpact_stage1": sha256_file(_ROOT / "src" / "fedpact_stage1.py"),
            "train_steps": sha256_file(_ROOT / "src" / "step07" / "train_steps.py"),
            "layer2": sha256_file(_ROOT / "src" / "step07" / "fedpact_layer2.py"),
        },
    }
    _save_json(snapshot_dir / "experiment_contract.json", experiment_contract)

    rows = read_manifest_filtered(manifest_path, dcfg)
    client_pools, validation_rows, server_seed_rows, final_evaluation_rows = (
        split_paired_step07_data_pools(
            rows,
            num_clients=2,
            client_train_pairs=client_pairs,
            client_eval_pairs=int(scfg["validation_per_class"]),
            server_train_pairs=server_seed_pairs,
            server_eval_pairs=int(scfg["final_evaluation_per_class"]),
            seed=seed,
        )
    )
    split_audit = audit_split_video_overlap(
        {
            "client-0-private": client_pools[0],
            "client-1-private": client_pools[1],
            "validation": validation_rows,
            "server-seed": server_seed_rows,
            "final-evaluation": final_evaluation_rows,
        }
    )
    private_control_dir = run_dir / "private_control"
    private_control_dir.mkdir(parents=True, exist_ok=True)
    split_path = save_data_splits_artifact(
        private_control_dir,
        client_trains=client_pools,
        client_eval=validation_rows,
        server_train=server_seed_rows,
        server_eval=final_evaluation_rows,
        seed=seed,
    )
    split_manifest = {
        "schema": "fedpact.stage1.split.v1",
        "source_manifest_sha256": sha256_file(manifest_path),
        "pools": {
            name: [
                {
                    "sample_id": str(row["id"]),
                    "canonical_source_video_id": canonical_video_id(row),
                    "label": int(row["label"]),
                }
                for row in pool
            ]
            for name, pool in {
                "client-0-private": client_pools[0],
                "client-1-private": client_pools[1],
                "validation": validation_rows,
                "server-seed": server_seed_rows,
                "final-evaluation": final_evaluation_rows,
            }.items()
        },
        "audit": split_audit,
    }
    _save_json(private_control_dir / "video_split_manifest.json", split_manifest)
    timings["prepare_and_split"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()

    device_name = device_or_auto(True) if args.device == "auto" else args.device
    device = torch.device(device_name)
    model_id = str(cfg.get("model", {}).get("client_id") or cfg["model"]["id"])
    model, processor = build_model(cfg, device, model_id=model_id)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(
        model.backbone, cfg, client="client", surrogate="surrogate"
    )
    model.backbone.set_adapter("client")
    checkpoint_names, checkpoint_vector, checkpoint_meta = load_fl_checkpoint(initial_checkpoint)
    available = {name for name, _ in model.named_parameters()}
    missing = [name for name in checkpoint_names if name not in available]
    if missing:
        raise KeyError(f"checkpoint parameters missing from current model: {missing[:5]}")
    vector_to_trainable_state(model, checkpoint_names, checkpoint_vector)
    sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")
    client_lora = lora_params_for_adapter(model.backbone, "client")
    surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
    ensure_client_trainable(model, model.backbone, client="client")

    initial_names, initial_vector, initial_meta = _combined_named_state(model, client_lora)
    if checkpoint_names != initial_names:
        raise ValueError("explicit checkpoint parameter names/order do not match Stage I state")
    initial_state_record = {
        "state_id": f"M_G^{server_round - 1}",
        "checkpoint": str(initial_checkpoint),
        "checkpoint_source_run_id": checkpoint_meta.get("run_id"),
        "tensor_count": len(initial_names),
        "parameter_count": int(initial_vector.size),
        **initial_meta,
    }
    _save_json(run_dir / "server" / "M_G_0_state.json", initial_state_record)
    timings["model_load"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()

    prompt = str(dcfg["image_prompt"])
    max_length = int(tcfg.get("max_length", 256))
    learning_rate = float(tcfg["lr"])
    weight_decay = float(tcfg["weight_decay"])
    client_summaries: list[dict[str, Any]] = []
    private_ids_all = [str(row["id"]) for pool in client_pools for row in pool]

    for client_index, train_rows in enumerate(client_pools):
        client_id = f"client-{client_index}"
        set_seed(seed + client_index)
        apply_named_vector(model, initial_names, initial_vector)
        ensure_client_trainable(model, model.backbone, client="client")
        start_digests = state_digests(model, client_lora)
        if start_digests != initial_meta["digests"]:
            raise AssertionError(f"{client_id} did not start from M_G^{server_round - 1}")

        before_grouped = state_named_tensors(model, client_lora)
        lora_names, lora_before = flatten_named_tensors(before_grouped["global_surrogate_lora"])
        head_names, head_before = flatten_named_tensors(before_grouped["classification_head"])
        pre_logits = _predict_rows(model, processor, train_rows, dcfg, prompt, max_length)

        optimizer = torch.optim.AdamW(
            [*model.classifier.parameters(), *client_lora],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        optimizer_state_at_start = len(optimizer.state)
        local_steps, training_metrics = train_client_l_task(
            model=model,
            processor=processor,
            train_rows=list(train_rows),
            dcfg=dcfg,
            client_lora=client_lora,
            frozen_lora=surrogate_lora,
            opt=optimizer,
            prompt=prompt,
            max_length=max_length,
            device=device,
            batch_size=1,
            max_steps=local_max_steps,
            gradient_accumulation_steps=1,
            update_client_lora=True,
            audit_parameter_groups={
                "client_lora": list(client_lora),
                "classification_head": list(model.classifier.parameters()),
            },
        )
        post_logits = _predict_rows(model, processor, train_rows, dcfg, prompt, max_length)
        after_grouped = state_named_tensors(model, client_lora)
        _, lora_after = flatten_named_tensors(after_grouped["global_surrogate_lora"])
        _, head_after = flatten_named_tensors(after_grouped["classification_head"])
        lora_delta = lora_after - lora_before
        head_delta = head_after - head_before
        if not np.all(np.isfinite(lora_delta)) or not np.all(np.isfinite(head_delta)):
            raise FloatingPointError(f"non-finite local update: {client_id}")

        private_trace = []
        for row, pre, post in zip(train_rows, pre_logits, post_logits):
            q_pre = softmax_vector(pre)
            q_post = softmax_vector(post)
            private_trace.append(
                {
                    "sample_id": str(row["id"]),
                    "canonical_source_video_id": canonical_video_id(row),
                    "label": int(row["label"]),
                    "q_pre": q_pre.astype(float).tolist(),
                    "q_post": q_post.astype(float).tolist(),
                    "delta_q": (q_post - q_pre).astype(float).tolist(),
                    "pre_logits": pre,
                    "post_logits": post,
                }
            )
        prototypes = build_classwise_prototypes(
            [
                {
                    "sample_id": item["sample_id"],
                    "label": item["label"],
                    "pre_logits": item["pre_logits"],
                    "post_logits": item["post_logits"],
                }
                for item in private_trace
            ],
            client_id=client_id,
            server_round=server_round,
        )

        client_local_dir = run_dir / "client_local" / client_id
        communication_dir = run_dir / "communication" / client_id
        client_local_dir.mkdir(parents=True, exist_ok=True)
        communication_dir.mkdir(parents=True, exist_ok=True)
        _save_json(
            client_local_dir / "private_trace.json",
            {
                "scope": "client-local-only",
                "client_id": client_id,
                "round": server_round,
                "samples": private_trace,
                "training": {
                    "optimizer": "AdamW",
                    "optimizer_state_entries_at_start": optimizer_state_at_start,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "max_steps": local_max_steps,
                    **training_metrics,
                },
            },
        )
        lora_artifact = _save_update_npz(
            communication_dir / "client_lora_update.npz", lora_names, lora_delta
        )
        head_artifact = _save_update_npz(
            communication_dir / "classification_head_update.npz", head_names, head_delta
        )
        payload = {
            "schema": str(scfg["payload_schema"]),
            "client_id": client_id,
            "round": server_round,
            "local_train_sample_count": len(train_rows),
            "trainable_update": {
                "client_lora": lora_artifact,
                "classification_head": head_artifact,
            },
            "prototypes": prototypes,
            "privacy": {
                "private_raw_data_included": False,
                "private_sample_id_included": False,
                "sample_level_output_included": False,
                "q_pre_sent_independently": False,
            },
        }
        audit_server_payload(payload, forbidden_values=private_ids_all)
        _save_json(communication_dir / "payload.json", payload)
        bundle_audit = audit_communication_bundle(
            communication_dir, forbidden_values=private_ids_all
        )
        _save_json(client_local_dir / "communication_audit.json", bundle_audit)

        after_names, after_vector, after_meta = _combined_named_state(model, client_lora)
        client_summaries.append(
            {
                "client_id": client_id,
                "sample_count": len(train_rows),
                "start_digests": start_digests,
                "after_digests": after_meta["digests"],
                "after_names": after_names,
                "after_vector": after_vector,
                "lora_update_norm": lora_artifact["l2_norm"],
                "head_update_norm": head_artifact["l2_norm"],
                "optimizer_state_entries_at_start": optimizer_state_at_start,
                "local_steps": local_steps,
                "training_metrics": training_metrics,
                "prototype_count": len(prototypes),
                "communication_audit": bundle_audit,
            }
        )

    timings["client_local_updates"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()

    disk_named_vectors: list[tuple[list[str], np.ndarray]] = []
    sample_counts: list[int] = []
    server_received_dir = run_dir / "server" / "received"
    for summary in client_summaries:
        client_id = str(summary["client_id"])
        source_dir = run_dir / "communication" / client_id
        target_dir = server_received_dir / client_id
        shutil.copytree(source_dir, target_dir)
        payload = json.loads((target_dir / "payload.json").read_text(encoding="utf-8"))
        audit_server_payload(payload, forbidden_values=private_ids_all)
        audit_communication_bundle(target_dir, forbidden_values=private_ids_all)
        with np.load(target_dir / "client_lora_update.npz", allow_pickle=False) as data:
            lora_update_names = [str(value) for value in data["names"].tolist()]
            lora_delta = np.asarray(data["delta"], dtype=np.float32)
        with np.load(target_dir / "classification_head_update.npz", allow_pickle=False) as data:
            head_update_names = [str(value) for value in data["names"].tolist()]
            head_delta = np.asarray(data["delta"], dtype=np.float32)
        update_names = [*lora_update_names, *head_update_names]
        update_vector = np.concatenate([lora_delta, head_delta]).astype(np.float32, copy=False)
        if update_names != initial_names:
            raise ValueError(f"server received parameter order mismatch: {client_id}")
        disk_named_vectors.append((update_names, initial_vector + update_vector))
        sample_counts.append(int(payload["local_train_sample_count"]))

    aggregated_names, aggregated_vector, weights = weighted_average_vectors(
        disk_named_vectors, sample_counts
    )
    deltas = [vector - initial_vector for _, vector in disk_named_vectors]
    weighted_delta = sum(
        (np.float32(weight) * delta for weight, delta in zip(weights, deltas)),
        start=np.zeros_like(initial_vector, dtype=np.float32),
    )
    independently_recomputed = initial_vector + weighted_delta
    max_abs_diff = float(np.max(np.abs(aggregated_vector - independently_recomputed)))
    if max_abs_diff > 1e-6:
        raise AssertionError(f"FedAvg offline recomputation mismatch: {max_abs_diff}")

    checkpoint_path = run_dir / "server" / f"M_G_{server_round}.pt"
    save_fl_checkpoint(
        checkpoint_path,
        names=aggregated_names,
        vector=aggregated_vector,
        meta={
            "run_id": args.run_id,
            "round": server_round,
            "method": "sample_count_weighted_fedavg",
            "client_ids": [summary["client_id"] for summary in client_summaries],
            "sample_counts": sample_counts,
            "weights": weights,
            "source_state": f"M_G^{server_round - 1}",
        },
    )
    np.savez_compressed(
        run_dir / "server" / "fedavg_offline_recompute.npz",
        names=np.asarray(aggregated_names, dtype=np.str_),
        vector=independently_recomputed.astype(np.float32),
    )
    apply_named_vector(model, aggregated_names, aggregated_vector)
    aggregated_digests = state_digests(model, client_lora)

    reloaded_names, reloaded_vector, _ = load_fl_checkpoint(checkpoint_path)
    redistribution: list[dict[str, Any]] = []
    for client_index in range(2):
        apply_named_vector(model, reloaded_names, reloaded_vector)
        next_start = state_digests(model, client_lora)
        if next_start != aggregated_digests:
            raise AssertionError(f"redistribution digest mismatch: client-{client_index}")
        redistribution.append(
            {"client_id": f"client-{client_index}", "next_round_start_digests": next_start}
        )

    aggregate_record = {
        "state_id": f"M_G^{server_round}",
        "method": "sample_count_weighted_fedavg",
        "sample_counts": sample_counts,
        "weights": weights,
        "max_abs_diff_offline_recompute": max_abs_diff,
        "digests": aggregated_digests,
        "checkpoint": str(checkpoint_path.relative_to(run_dir)),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "redistribution": redistribution,
    }
    _save_json(run_dir / "server" / "fedavg_and_redistribution.json", aggregate_record)
    timings["fedavg_and_redistribution"] = time.perf_counter() - phase_started

    public_clients = []
    for summary in client_summaries:
        public_clients.append(
            {key: value for key, value in summary.items() if key not in {"after_names", "after_vector"}}
        )
    elapsed = time.perf_counter() - started
    pass_conditions = {
        "same_start_state": all(
            summary["start_digests"] == initial_meta["digests"] for summary in client_summaries
        ),
        "fresh_optimizer_each_client": all(
            int(summary["optimizer_state_entries_at_start"]) == 0 for summary in client_summaries
        ),
        "all_local_samples_exposed_once": all(
            int(summary["training_metrics"]["sample_exposures"]) == int(summary["sample_count"])
            and int(summary["local_steps"]) == int(summary["sample_count"])
            for summary in client_summaries
        ),
        "finite_nonzero_lora_update": all(
            np.isfinite(summary["lora_update_norm"]) and summary["lora_update_norm"] > 0
            for summary in client_summaries
        ),
        "finite_nonzero_head_update": all(
            np.isfinite(summary["head_update_norm"]) and summary["head_update_norm"] > 0
            for summary in client_summaries
        ),
        "fedavg_offline_recompute": max_abs_diff <= 1e-6,
        "redistribution_digest_match": all(
            item["next_round_start_digests"] == aggregated_digests for item in redistribution
        ),
        "raw_data_non_transmission_audit": all(
            summary["communication_audit"]["status"] == "pass" for summary in client_summaries
        ),
    }
    status = "pass" if all(pass_conditions.values()) else "fail"
    return {
        "run_id": args.run_id,
        "gate": "P1-A",
        "status": status,
        "purpose": "2 client Layer 1 one-round smoke: local update, FedAvg, redistribution",
        "claim_limit": "This run checks execution and state lineage only; it is not a performance result.",
        "configuration": experiment_contract,
        "runtime": {
            "device": str(device),
            "model_id": model_id,
            "model_revision": _model_revision(model, processor),
            "dtype_backbone": "torch.bfloat16",
            "dtype_head_and_aggregation": "torch.float32",
        },
        "split_artifacts": {
            "sample_split": str(split_path.relative_to(run_dir)),
            "video_split_manifest": "private_control/video_split_manifest.json",
            "audit": split_audit,
        },
        "M_G_0": initial_state_record,
        "clients": public_clients,
        "M_G_1": aggregate_record,
        "pass_conditions": pass_conditions,
        "timings_sec": timings,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage1.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--profile", choices=["p1a", "p1d"], default="p1a")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--initial-global-checkpoint", default=None)
    args = parser.parse_args()
    args.run_id = args.run_id or datetime.now().strftime("p1a_%Y%m%d_%H%M%S")
    run_dir = _ROOT / "artifacts" / "runs" / "fedpact_stage1" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _save_json(
        run_dir / "run_status.json",
        {
            "run_id": args.run_id,
            "gate": "P1-A",
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        summary = _run(args, run_dir)
        _save_json(run_dir / "summary.json", summary)
        _save_json(
            run_dir / "run_status.json",
            {
                "run_id": args.run_id,
                "gate": "P1-A",
                "status": summary["status"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))
        if summary["status"] != "pass":
            raise SystemExit(2)
    except BaseException as exc:
        _save_json(
            run_dir / "run_status.json",
            {
                "run_id": args.run_id,
                "gate": "P1-A",
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()
