"""FedPACT P1-B: P1-Aの1 prototypeをLayer 2だけへ通す。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_loader import load_yaml, merged_config  # noqa: E402
from src.dataset_manifest import load_sample_row, read_manifest_filtered  # noqa: E402
from src.fedpact_stage1 import sha256_file, state_digests  # noqa: E402
from src.fl_utils import load_fl_checkpoint, vector_to_trainable_state  # noqa: E402
from src.metrics import set_seed  # noqa: E402
from src.peft_setup import attach_dual_lora, lora_params_for_adapter  # noqa: E402
from src.step07.data_splits import load_data_splits_artifact, resolve_rows_by_ids  # noqa: E402
from src.step07.fedpact_layer2 import (  # noqa: E402
    Transformation,
    proxy_match_metrics,
    rank_server_seeds,
    select_best_proxy,
    transform_frames,
)
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


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else _ROOT / path).resolve()


def _load_state(model, checkpoint: Path) -> dict[str, Any]:
    names, vector, meta = load_fl_checkpoint(checkpoint)
    vector_to_trainable_state(model, names, vector)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_file_sha256": sha256_file(checkpoint),
        "source_run_id": meta.get("run_id"),
        "round": meta.get("round"),
        "tensor_count": len(names),
        "parameter_count": int(vector.size),
    }


@torch.no_grad()
def _predict_rows(model, processor, rows, dcfg, prompt, max_length) -> list[list[float]]:
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


@torch.no_grad()
def _predict_frame_sets(model, processor, frame_sets, prompt, max_length) -> list[list[float]]:
    model.eval()
    model.backbone.set_adapter("client")
    output: list[list[float]] = []
    for frames in frame_sets:
        logits, _ = model.forward_batch(
            processor, list(frames), prompt, max_length, output_hidden_states=True
        )
        output.append(logits[0].detach().float().cpu().tolist())
    return output


def _run(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = merged_config(load_yaml(args.config))
    dcfg = dict(cfg["data"])
    scfg = dict(cfg["fedpact_stage1"])
    tcfg = dict(cfg["train"])
    set_seed(int(tcfg["seed"]))

    parent_run = _resolve(args.parent_run)
    parent_summary = json.loads((parent_run / "summary.json").read_text(encoding="utf-8"))
    if parent_summary.get("gate") != "P1-A" or parent_summary.get("status") != "pass":
        raise ValueError("parent run is not a passed P1-A run")
    m_g_0 = _resolve(str(scfg["initial_global_checkpoint"]))
    m_g_1 = parent_run / "server" / "M_G_1.pt"
    payload_path = parent_run / "communication" / "client-0" / "payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    prototype = next(
        item for item in payload["prototypes"] if int(item["class_label"]) == 1
    )
    split = load_data_splits_artifact(parent_run / "private_control" / "data_splits.json")
    manifest_path = Path(dcfg["manifest_path"])
    rows = read_manifest_filtered(manifest_path, dcfg)
    seed_ids = [str(value) for value in split["server_train_ids"]]
    server_seed_rows = resolve_rows_by_ids(rows, seed_ids)
    if len(server_seed_rows) != 8:
        raise ValueError(f"P1-B requires exactly 8 server seeds, got {len(server_seed_rows)}")

    device_name = device_or_auto(True) if args.device == "auto" else args.device
    device = torch.device(device_name)
    model_id = str(cfg.get("model", {}).get("client_id") or cfg["model"]["id"])
    model, processor = build_model(cfg, device, model_id=model_id)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(
        model.backbone, cfg, client="client", surrogate="surrogate"
    )
    model.backbone.set_adapter("client")
    client_lora = lora_params_for_adapter(model.backbone, "client")
    prompt = str(dcfg["image_prompt"])
    max_length = int(tcfg.get("max_length", 256))

    m_g_0_info = _load_state(model, m_g_0)
    m_g_0_info["digests"] = state_digests(model, client_lora)
    expected_m_g_0 = parent_summary["M_G_0"]["digests"]
    if m_g_0_info["digests"] != expected_m_g_0:
        raise AssertionError("loaded M_G^0 digest does not match P1-A")

    m_g_1_info = _load_state(model, m_g_1)
    m_g_1_info["digests"] = state_digests(model, client_lora)
    expected_m_g_1 = parent_summary["M_G_1"]["digests"]
    if m_g_1_info["digests"] != expected_m_g_1:
        raise AssertionError("loaded M_G^1 digest does not match P1-A")
    seed_after_logits = _predict_rows(
        model, processor, server_seed_rows, dcfg, prompt, max_length
    )
    seed_candidates = [
        {
            "seed_id": str(row["id"]),
            "label": int(row["label"]),
            "updated_logits": logits,
        }
        for row, logits in zip(server_seed_rows, seed_after_logits)
    ]
    ranked = rank_server_seeds(prototype, seed_candidates)
    top_k = 2
    row_by_id = {str(row["id"]): row for row in server_seed_rows}
    transformations = [Transformation("identity"), Transformation("brightness", 0.85)]

    variants: list[dict[str, Any]] = []
    frame_sets: list[list[Image.Image]] = []
    candidate_root = run_dir / "server" / "transformation_candidates"
    for retrieved in ranked[:top_k]:
        sample = load_sample_row(row_by_id[retrieved["seed_id"]], dcfg)
        for transformation_index, spec in enumerate(transformations):
            transformed = transform_frames(sample.frames, spec)
            candidate_id = (
                f"rank{int(retrieved['rank']):02d}-{transformation_index:02d}-{spec.name}"
            )
            candidate_dir = candidate_root / candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            frame_files: list[str] = []
            for frame_index, frame in enumerate(transformed):
                path = candidate_dir / f"frame_{frame_index:02d}.png"
                frame.save(path)
                frame_files.append(str(path.relative_to(run_dir)).replace("\\", "/"))
            variants.append(
                {
                    "candidate_id": candidate_id,
                    "seed_id": str(retrieved["seed_id"]),
                    "seed_label": int(retrieved["label"]),
                    "retrieval_rank": int(retrieved["rank"]),
                    "retrieval_post_kl": float(retrieved["retrieval_post_kl"]),
                    "transformation": spec.as_dict(),
                    "frame_files": frame_files,
                }
            )
            frame_sets.append(transformed)
        for frame in sample.frames:
            frame.close()

    _load_state(model, m_g_0)
    variant_before_logits = _predict_frame_sets(
        model, processor, frame_sets, prompt, max_length
    )
    _load_state(model, m_g_1)
    variant_after_logits = _predict_frame_sets(
        model, processor, frame_sets, prompt, max_length
    )
    for variant, before_logits, after_logits in zip(
        variants, variant_before_logits, variant_after_logits
    ):
        variant["global_before_logits"] = before_logits
        variant["global_after_logits"] = after_logits
        variant["metrics"] = proxy_match_metrics(
            prototype,
            global_before_logits=before_logits,
            global_after_logits=after_logits,
            lambda_post=1.0,
            lambda_delta=1.0,
        )
    selected = select_best_proxy(variants)
    selected_variant = next(
        item for item in variants if item["candidate_id"] == selected["candidate_id"]
    )
    pair_id = f"p1b-r1-client-0-class1-{selected['seed_id']}-{selected['candidate_id']}"
    proxy_pair = {
        "schema": "fedpact.proxy_pair.v2",
        "pair_id": pair_id,
        "round": 1,
        "client_id": "client-0",
        "class_label": 1,
        "prototype_id": prototype["prototype_id"],
        "state_lineage": {
            "M_G^t": {"state_id": "M_G^0", "digests": m_g_0_info["digests"]},
            "M_G^{t+1}": {"state_id": "M_G^1", "digests": m_g_1_info["digests"]},
        },
        "s^*_{k,r}": {
            "seed_id": selected["seed_id"],
            "seed_label": int(selected["seed_label"]),
            "retrieval_rank": int(selected["retrieval_rank"]),
        },
        "x_tilde_{k,r}": {
            "frame_files": selected_variant["frame_files"],
            "transformation": selected["transformation"],
        },
        "mu_post_{k,r}": prototype["mu_post"],
        "frequency_pi_{k,r}": int(prototype["frequency"]),
        "layer3_contract": "P_{k,r}=(x_tilde_{k,r}, mu_post_{k,r})",
    }
    _save_json(run_dir / "server" / "proxy_pair.json", proxy_pair)
    trace = {
        "schema": "fedpact.layer2_trace.v2",
        "run_id": args.run_id,
        "parent_p1a_run": str(parent_run),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "configuration": {
            "prototype_id": prototype["prototype_id"],
            "seed_count": len(server_seed_rows),
            "seed_ids": seed_ids,
            "top_k": top_k,
            "transformations": [spec.as_dict() for spec in transformations],
            "lambda_post": 1.0,
            "lambda_delta": 1.0,
        },
        "prototype": prototype,
        "states": {"M_G^0": m_g_0_info, "M_G^1": m_g_1_info},
        "retrieval": {
            "model_state_used": "M_G^1",
            "ranked_seeds": ranked,
        },
        "proxy_candidates": variants,
        "selected_candidate_id": selected["candidate_id"],
        "selected_metrics": selected["metrics"],
        "output_pair_id": pair_id,
    }
    _save_json(run_dir / "server" / "layer2_trace.json", trace)
    for frames in frame_sets:
        for frame in frames:
            frame.close()

    elapsed = time.perf_counter() - started
    pass_conditions = {
        "parent_p1a_pass": True,
        "M_G_0_digest_match": m_g_0_info["digests"] == expected_m_g_0,
        "M_G_1_digest_match": m_g_1_info["digests"] == expected_m_g_1,
        "retrieval_used_M_G_1": trace["retrieval"]["model_state_used"] == "M_G^1",
        "same_proxy_used_for_before_after": all(
            bool(item["global_before_logits"]) and bool(item["global_after_logits"])
            for item in variants
        ),
        "finite_metrics": all(
            all(np.isfinite(float(value)) for value in item["metrics"].values())
            for item in variants
        ),
        "pair_lineage_complete": bool(
            proxy_pair["pair_id"]
            and proxy_pair["prototype_id"]
            and proxy_pair["s^*_{k,r}"]["seed_id"]
            and proxy_pair["x_tilde_{k,r}"]["frame_files"]
        ),
    }
    return {
        "run_id": args.run_id,
        "gate": "P1-B",
        "status": "pending_independent_recompute"
        if all(pass_conditions.values())
        else "fail",
        "purpose": "one-prototype Layer 2 wiring and lineage check",
        "claim_limit": "No retrieval quality, semantic validity, coverage, or performance claim.",
        "parent_p1a_run": str(parent_run),
        "prototype_id": prototype["prototype_id"],
        "seed_count": len(server_seed_rows),
        "candidate_count": len(variants),
        "selected": {
            "candidate_id": selected["candidate_id"],
            "seed_id": selected["seed_id"],
            "seed_label": selected["seed_label"],
            "transformation": selected["transformation"],
            "metrics": selected["metrics"],
        },
        "pair_id": pair_id,
        "pass_conditions": pass_conditions,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage1.yaml")
    parser.add_argument(
        "--parent-run",
        default="artifacts/runs/fedpact_stage1/p1a_20260916_2235",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.run_id = args.run_id or datetime.now().strftime("p1b_%Y%m%d_%H%M%S")
    run_dir = _ROOT / "artifacts" / "runs" / "fedpact_stage1" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _save_json(
        run_dir / "run_status.json",
        {
            "run_id": args.run_id,
            "gate": "P1-B",
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
                "gate": "P1-B",
                "status": summary["status"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))
        if summary["status"] == "fail":
            raise SystemExit(2)
    except BaseException as exc:
        _save_json(
            run_dir / "run_status.json",
            {
                "run_id": args.run_id,
                "gate": "P1-B",
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()

