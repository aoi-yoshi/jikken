"""FedPACT P1-D: 4 prototypes, 60 seeds, 4 proxy pairs, and one foundation update."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_fedpact_stage1_p1b import (  # noqa: E402
    _load_state,
    _predict_frame_sets,
    _predict_rows,
)
from scripts.run_fedpact_stage1_p1c import (  # noqa: E402
    _gradient_vector,
    _kl_teacher_to_student,
    _mapped_foundation_checkpoint,
    _named_lora_state,
)
from src.config_loader import load_yaml, merged_config  # noqa: E402
from src.dataset_manifest import load_sample_row, read_manifest_filtered  # noqa: E402
from src.fedpact_stage1 import (  # noqa: E402
    canonical_tensor_digest,
    named_selected_parameters,
    sha256_file,
)
from src.fl_utils import save_fl_checkpoint, vector_to_trainable_state  # noqa: E402
from src.metrics import set_seed  # noqa: E402
from src.peft_setup import attach_dual_lora, lora_params_for_adapter  # noqa: E402
from src.step07.data_splits import load_data_splits_artifact, resolve_rows_by_ids  # noqa: E402
from src.step07.fedpact_layer2 import (  # noqa: E402
    proxy_match_metrics,
    rank_server_seeds,
    select_best_proxy,
    transformation_candidates,
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


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else _ROOT / path).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _weighted_foundation_loss(model, processor, pair_inputs, prompt: str, max_length: int):
    losses: list[torch.Tensor] = []
    logits_list: list[list[float]] = []
    for item in pair_inputs:
        logits, _ = model.forward_batch(
            processor, item["frames"], prompt, max_length, output_hidden_states=True
        )
        loss = _kl_teacher_to_student(logits, item["teacher"])
        losses.append(loss)
        logits_list.append(logits[0].detach().float().cpu().tolist())
    weighted = sum(
        (float(item["pi"]) * loss for item, loss in zip(pair_inputs, losses)),
        start=torch.zeros((), device=losses[0].device, dtype=losses[0].dtype),
    )
    return weighted, losses, logits_list


def _backward_weighted_foundation_loss(
    model, processor, pair_inputs, prompt: str, max_length: int
):
    """Same weighted sum as _weighted_foundation_loss, with per-pair backward to bound VRAM."""
    weighted_detached: torch.Tensor | None = None
    losses: list[torch.Tensor] = []
    logits_list: list[list[float]] = []
    for item in pair_inputs:
        logits, _ = model.forward_batch(
            processor, item["frames"], prompt, max_length, output_hidden_states=True
        )
        loss = _kl_teacher_to_student(logits, item["teacher"])
        contribution = float(item["pi"]) * loss
        contribution.backward()
        weighted_detached = (
            contribution.detach()
            if weighted_detached is None
            else weighted_detached + contribution.detach()
        )
        losses.append(loss.detach())
        logits_list.append(logits[0].detach().float().cpu().tolist())
        del logits, loss, contribution
    if weighted_detached is None:
        raise ValueError("pair_inputs is empty")
    return weighted_detached, losses, logits_list


def _run(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    cfg = merged_config(load_yaml(args.config))
    dcfg = dict(cfg["data"])
    scfg = dict(cfg["fedpact_stage1"])
    tcfg = dict(cfg["train"])
    set_seed(int(tcfg["seed"]))

    parent_run = _resolve(args.parent_run)
    parent_summary = json.loads((parent_run / "summary.json").read_text(encoding="utf-8"))
    if parent_summary.get("gate") != "P1-A" or parent_summary.get("status") != "pass":
        raise ValueError("parent Layer 1 run is not a passed P1-A-compatible run")
    if parent_summary.get("configuration", {}).get("profile") != "p1d":
        raise ValueError("parent Layer 1 run was not executed with profile=p1d")

    prototypes: list[dict[str, Any]] = []
    for client_index in range(2):
        payload_path = parent_run / "communication" / f"client-{client_index}" / "payload.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        prototypes.extend(payload["prototypes"])
    if len(prototypes) != 4:
        raise AssertionError(f"P1-D requires 4 prototypes, got {len(prototypes)}")
    if any(int(item["frequency"]) != 3 for item in prototypes):
        raise AssertionError("P1-D prototype frequency must be 3")
    server_round = int(prototypes[0]["round"])
    if any(int(item["round"]) != server_round for item in prototypes):
        raise AssertionError("prototype rounds are inconsistent")
    before_global_state_id = f"M_G^{server_round - 1}"
    after_global_state_id = f"M_G^{server_round}"
    before_foundation_state_id = f"F^{server_round - 1}"
    after_foundation_state_id = f"F^{server_round}"

    split = load_data_splits_artifact(parent_run / "private_control" / "data_splits.json")
    manifest_path = Path(dcfg["manifest_path"])
    rows = read_manifest_filtered(manifest_path, dcfg)
    seed_ids = [str(value) for value in split["server_train_ids"]]
    server_seed_rows = resolve_rows_by_ids(rows, seed_ids)
    if len(server_seed_rows) != 60:
        raise AssertionError(f"P1-D requires 60 server seeds, got {len(server_seed_rows)}")
    seed_label_counts = {
        str(label): sum(int(row["label"]) == label for row in server_seed_rows)
        for label in (0, 1)
    }
    if seed_label_counts != {"0": 30, "1": 30}:
        raise AssertionError(f"P1-D seed balance mismatch: {seed_label_counts}")

    phase = time.perf_counter()
    device_name = device_or_auto(True) if args.device == "auto" else args.device
    device = torch.device(device_name)
    model_id = str(cfg.get("model", {}).get("client_id") or cfg["model"]["id"])
    model, processor = build_model(cfg, device, model_id=model_id)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    client_lora = lora_params_for_adapter(model.backbone, "client")
    prompt = str(dcfg["image_prompt"])
    max_length = int(tcfg.get("max_length", 256))
    timings["layer2_model_load"] = time.perf_counter() - phase

    m_g_0 = _resolve(str(parent_summary["M_G_0"]["checkpoint"]))
    m_g_1 = parent_run / str(parent_summary["M_G_1"]["checkpoint"])
    m_g_0_info = _load_state(model, m_g_0)
    from src.fedpact_stage1 import state_digests

    m_g_0_info["digests"] = state_digests(model, client_lora)
    if m_g_0_info["digests"] != parent_summary["M_G_0"]["digests"]:
        raise AssertionError(f"P1-D loaded {before_global_state_id} digest mismatch")
    m_g_1_info = _load_state(model, m_g_1)
    m_g_1_info["digests"] = state_digests(model, client_lora)
    if m_g_1_info["digests"] != parent_summary["M_G_1"]["digests"]:
        raise AssertionError(f"P1-D loaded {after_global_state_id} digest mismatch")

    phase = time.perf_counter()
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
    ranked_by_prototype = {
        str(prototype["prototype_id"]): rank_server_seeds(prototype, seed_candidates)
        for prototype in prototypes
    }
    timings["server_seed_retrieval"] = time.perf_counter() - phase

    phase = time.perf_counter()
    top_k = 3
    transformations = transformation_candidates("tiny")
    row_by_id = {str(row["id"]): row for row in server_seed_rows}
    variants: list[dict[str, Any]] = []
    frame_sets: list[list[Image.Image]] = []
    prototype_by_id = {str(item["prototype_id"]): item for item in prototypes}
    candidate_root = run_dir / "server" / "transformation_candidates"
    for prototype in prototypes:
        prototype_id = str(prototype["prototype_id"])
        for retrieved in ranked_by_prototype[prototype_id][:top_k]:
            sample = load_sample_row(row_by_id[str(retrieved["seed_id"])], dcfg)
            for transformation_index, spec in enumerate(transformations):
                transformed = transform_frames(sample.frames, spec)
                candidate_id = (
                    f"{prototype_id}-rank{int(retrieved['rank']):02d}-"
                    f"{transformation_index:02d}-{spec.name}"
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
                        "prototype_id": prototype_id,
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
    if len(variants) != 4 * 3 * 5:
        raise AssertionError(f"expected 60 proxy candidates, got {len(variants)}")
    timings["proxy_transform_and_save"] = time.perf_counter() - phase

    phase = time.perf_counter()
    _load_state(model, m_g_0)
    before_logits = _predict_frame_sets(model, processor, frame_sets, prompt, max_length)
    _load_state(model, m_g_1)
    after_logits = _predict_frame_sets(model, processor, frame_sets, prompt, max_length)
    for variant, before, after in zip(variants, before_logits, after_logits):
        prototype = prototype_by_id[str(variant["prototype_id"])]
        variant["global_before_logits"] = before
        variant["global_after_logits"] = after
        variant["metrics"] = proxy_match_metrics(
            prototype,
            global_before_logits=before,
            global_after_logits=after,
            lambda_post=1.0,
            lambda_delta=1.0,
        )
    timings["proxy_before_after_and_objective"] = time.perf_counter() - phase

    proxy_pairs: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for prototype in prototypes:
        prototype_id = str(prototype["prototype_id"])
        candidates = [item for item in variants if item["prototype_id"] == prototype_id]
        selected = select_best_proxy(candidates)
        pair_id = f"p1d-{prototype_id}-{selected['seed_id']}-{selected['candidate_id']}"
        pair = {
            "schema": "fedpact.proxy_pair.v2",
            "pair_id": pair_id,
            "round": int(prototype["round"]),
            "client_id": str(prototype["client_id"]),
            "class_label": int(prototype["class_label"]),
            "prototype_id": prototype_id,
            "state_lineage": {
                "M_G^t": {"state_id": before_global_state_id, "digests": m_g_0_info["digests"]},
                "M_G^{t+1}": {"state_id": after_global_state_id, "digests": m_g_1_info["digests"]},
            },
            "s^*_{k,r}": {
                "seed_id": str(selected["seed_id"]),
                "seed_label": int(selected["seed_label"]),
                "retrieval_rank": int(selected["retrieval_rank"]),
                "retrieval_post_kl": float(selected["retrieval_post_kl"]),
            },
            "x_tilde_{k,r}": {
                "frame_files": list(selected["frame_files"]),
                "transformation": dict(selected["transformation"]),
            },
            "mu_post_{k,r}": list(prototype["mu_post"]),
            "delta_mu_{k,r}": list(prototype["delta_mu_local"]),
            "frequency_pi_{k,r}": int(prototype["frequency"]),
            "metrics": dict(selected["metrics"]),
            "layer3_contract": "P_{k,r}=(x_tilde_{k,r}, mu_post_{k,r})",
        }
        proxy_pairs.append(pair)
        selections.append(
            {
                "prototype_id": prototype_id,
                "candidate_id": selected["candidate_id"],
                "seed_id": selected["seed_id"],
                "seed_label": selected["seed_label"],
                "transformation": selected["transformation"],
                "metrics": selected["metrics"],
            }
        )
    _save_json(run_dir / "server" / "proxy_pairs.json", proxy_pairs)
    layer2_trace = {
        "schema": "fedpact.layer2_trace.p1d.v1",
        "parent_layer1_run": str(parent_run),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "configuration": {
            "prototype_count": 4,
            "seed_count": 60,
            "seed_label_counts": seed_label_counts,
            "retrieval_class_constraint": None,
            "top_k": top_k,
            "transformations": [spec.as_dict() for spec in transformations],
            "candidate_count": len(variants),
        },
        "states": {before_global_state_id: m_g_0_info, after_global_state_id: m_g_1_info},
        "prototypes": prototypes,
        "ranked_seeds_by_prototype": ranked_by_prototype,
        "proxy_candidates": variants,
        "selections": selections,
        "proxy_pair_ids": [pair["pair_id"] for pair in proxy_pairs],
    }
    _save_json(run_dir / "server" / "layer2_trace.json", layer2_trace)
    for frames in frame_sets:
        for frame in frames:
            frame.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    phase = time.perf_counter()
    foundation_model_id = str(cfg.get("model", {}).get("server_id") or cfg["model"]["id"])
    foundation_model, foundation_processor = build_model(cfg, device, model_id=foundation_model_id)
    foundation_model.backbone = attach_dual_lora(
        foundation_model.backbone, cfg, client="foundation", surrogate="surrogate"
    )
    foundation_model.backbone.set_adapter("foundation")
    foundation_lora = lora_params_for_adapter(foundation_model.backbone, "foundation")
    inactive_surrogate_lora = lora_params_for_adapter(foundation_model.backbone, "surrogate")
    for parameter in foundation_model.parameters():
        parameter.requires_grad_(False)
    for parameter in foundation_lora:
        parameter.requires_grad_(True)
    source_checkpoint = _resolve(
        str(getattr(args, "foundation_checkpoint", None) or scfg["initial_foundation_checkpoint"])
    )
    mapped_names, mapped_vector, source_meta = _mapped_foundation_checkpoint(source_checkpoint)
    vector_to_trainable_state(foundation_model, mapped_names, mapped_vector)
    foundation_names, foundation_before, foundation_digest_before = _named_lora_state(
        foundation_model, foundation_lora
    )
    head_before = canonical_tensor_digest(
        [(f"classifier.{name}", tensor) for name, tensor in foundation_model.classifier.state_dict().items()]
    )
    inactive_before = canonical_tensor_digest(
        named_selected_parameters(foundation_model, inactive_surrogate_lora)
    )
    target_ids = {id(parameter) for parameter in foundation_lora}
    non_target_versions_before = {
        name: int(parameter._version)
        for name, parameter in foundation_model.named_parameters()
        if id(parameter) not in target_ids
    }
    foundation_dir = run_dir / "foundation"
    save_fl_checkpoint(
        foundation_dir / f"F_{server_round - 1}.pt",
        names=foundation_names,
        vector=foundation_before,
        meta={
            "run_id": args.run_id,
            "state_id": before_foundation_state_id,
            "source_checkpoint": str(source_checkpoint),
            "source_run_id": source_meta.get("run_id"),
        },
    )
    pair_inputs: list[dict[str, Any]] = []
    for pair in proxy_pairs:
        frames = [Image.open(run_dir / relative).convert("RGB") for relative in pair["x_tilde_{k,r}"]["frame_files"]]
        teacher = torch.tensor(
            pair["mu_post_{k,r}"], dtype=torch.float32, device=device
        ).reshape(1, -1)
        pair_inputs.append(
            {
                "pair_id": pair["pair_id"],
                "frames": frames,
                "teacher": teacher,
                "pi": int(pair["frequency_pi_{k,r}"]),
            }
        )
    foundation_model.eval()
    weighted_before, per_pair_before_tensors, logits_before_all = _backward_weighted_foundation_loss(
        foundation_model, foundation_processor, pair_inputs, prompt, max_length
    )
    official_gradient = _gradient_vector(foundation_lora)
    gradient_norm = float(np.linalg.norm(official_gradient))
    gradient_buffers = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in foundation_lora
    ]
    initial_lr = float(scfg.get("foundation_lr_p1c", 1.0e-6))
    max_attempts = int(scfg.get("foundation_backtracking_max_attempts_p1c", 6))
    attempts: list[dict[str, Any]] = []
    accepted = False
    chosen_lr = initial_lr
    weighted_after = weighted_before.detach()
    per_pair_after_tensors = [loss.detach() for loss in per_pair_before_tensors]
    logits_after_all = logits_before_all
    for attempt_index in range(max_attempts):
        candidate_lr = initial_lr * (0.1**attempt_index)
        vector_to_trainable_state(foundation_model, foundation_names, foundation_before)
        optimizer = torch.optim.AdamW(foundation_lora, lr=candidate_lr, weight_decay=0.01)
        optimizer_state_entries_at_start = len(optimizer.state)
        for parameter, gradient in zip(foundation_lora, gradient_buffers):
            parameter.grad = None if gradient is None else gradient.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            trial_weighted, trial_pair_losses, trial_logits = _weighted_foundation_loss(
                foundation_model, foundation_processor, pair_inputs, prompt, max_length
            )
        trial_value = float(trial_weighted.detach().cpu())
        loss_nonincrease = trial_value <= float(weighted_before.detach().cpu()) + 1.0e-8
        trial_finite = math.isfinite(trial_value)
        accepted_trial = (
            loss_nonincrease
            if args.foundation_acceptance == "nonincrease"
            else trial_finite
        )
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "learning_rate": candidate_lr,
                "weighted_loss_after": trial_value,
                "optimizer_state_entries_at_start": optimizer_state_entries_at_start,
                "finite": trial_finite,
                "loss_nonincrease": loss_nonincrease,
                "accepted": accepted_trial,
            }
        )
        if accepted_trial:
            accepted = True
            chosen_lr = candidate_lr
            weighted_after = trial_weighted
            per_pair_after_tensors = trial_pair_losses
            logits_after_all = trial_logits
            break
    if not accepted:
        raise AssertionError(
            f"no P1-D foundation step satisfied {args.foundation_acceptance!r}: {attempts}"
        )

    foundation_after_names, foundation_after, foundation_digest_after = _named_lora_state(
        foundation_model, foundation_lora
    )
    if foundation_after_names != foundation_names:
        raise AssertionError("foundation parameter order changed")
    update_norm = float(np.linalg.norm(foundation_after - foundation_before))
    head_after = canonical_tensor_digest(
        [(f"classifier.{name}", tensor) for name, tensor in foundation_model.classifier.state_dict().items()]
    )
    inactive_after = canonical_tensor_digest(
        named_selected_parameters(foundation_model, inactive_surrogate_lora)
    )
    non_target_versions_after = {
        name: int(parameter._version)
        for name, parameter in foundation_model.named_parameters()
        if id(parameter) not in target_ids
    }
    f1_path = foundation_dir / f"F_{server_round}.pt"
    save_fl_checkpoint(
        f1_path,
        names=foundation_names,
        vector=foundation_after,
        meta={
            "run_id": args.run_id,
            "state_id": after_foundation_state_id,
            "parent_state": before_foundation_state_id,
            "pair_ids": [pair["pair_id"] for pair in proxy_pairs],
            "rho": "Uniform",
            "weights": [int(pair["frequency_pi_{k,r}"]) for pair in proxy_pairs],
            "weight_normalization": "none",
            "learning_rate": chosen_lr,
            "optimizer_steps": 1,
        },
    )
    per_pair = []
    for item, before_loss, after_loss, before_logits_item, after_logits_item in zip(
        pair_inputs,
        per_pair_before_tensors,
        per_pair_after_tensors,
        logits_before_all,
        logits_after_all,
    ):
        per_pair.append(
            {
                "pair_id": item["pair_id"],
                "pi": item["pi"],
                "loss_before": float(before_loss.detach().cpu()),
                "loss_after": float(after_loss.detach().cpu()),
                "student_logits_before": before_logits_item,
                "student_logits_after": after_logits_item,
            }
        )
    timings["foundation_load_and_update"] = time.perf_counter() - phase
    for item in pair_inputs:
        for frame in item["frames"]:
            frame.close()

    weighted_loss_nonincrease = (
        float(weighted_after.detach().cpu())
        <= float(weighted_before.detach().cpu()) + 1e-8
    )
    checks = {
        "parent_layer1_pass": True,
        "two_clients_six_samples_each": [item["sample_count"] for item in parent_summary["clients"]] == [6, 6],
        "four_prototypes_frequency_three": len(prototypes) == 4 and all(int(item["frequency"]) == 3 for item in prototypes),
        "sixty_balanced_seeds": seed_label_counts == {"0": 30, "1": 30},
        "class_unrestricted_retrieval": layer2_trace["configuration"]["retrieval_class_constraint"] is None,
        "sixty_proxy_candidates": len(variants) == 60,
        "four_proxy_pairs": len(proxy_pairs) == 4,
        "retrieval_used_M_G_1": True,
        "proxy_delta_used_same_candidate_M_G_0_M_G_1": all(
            bool(item["global_before_logits"]) and bool(item["global_after_logits"])
            for item in variants
        ),
        "uniform_rho_equals_pi_without_normalization": all(int(item["pi"]) == 3 for item in pair_inputs),
        "foundation_gradient_finite_nonzero": math.isfinite(gradient_norm) and gradient_norm > 0.0,
        "foundation_update_finite_nonzero": math.isfinite(update_norm) and update_norm > 0.0,
        "foundation_acceptance_policy_satisfied": accepted,
        "weighted_loss_nonincrease": weighted_loss_nonincrease,
        "classification_head_unchanged": head_before == head_after,
        "inactive_surrogate_lora_unchanged": inactive_before == inactive_after,
        "non_target_parameter_versions_unchanged": non_target_versions_before == non_target_versions_after,
    }
    required_checks = [
        key
        for key in checks
        if key != "weighted_loss_nonincrease" or args.foundation_acceptance == "nonincrease"
    ]
    status = "pass" if all(checks[key] for key in required_checks) else "fail"
    layer3_trace = {
        "schema": "fedpact.layer3_trace.p1d.v1",
        "state_lineage": {
            "before": before_foundation_state_id,
            "after": after_foundation_state_id,
            "before_digest": foundation_digest_before,
            "after_digest": foundation_digest_after,
            "after_checkpoint": str(f1_path.relative_to(run_dir)).replace("\\", "/"),
        },
        "objective": {
            "rho": "Uniform",
            "weights": [int(item["pi"]) for item in pair_inputs],
            "weight_normalization": "none",
            "definition": "sum_k,r pi_{k,r} * D_KL(mu_post_{k,r} || foundation(x_tilde_{k,r}))",
            "weighted_loss_before": float(weighted_before.detach().cpu()),
            "weighted_loss_after": float(weighted_after.detach().cpu()),
            "per_pair": per_pair,
        },
        "optimization": {
            "optimizer": "AdamW",
            "acceptance_policy": args.foundation_acceptance,
            "accepted_learning_rate": chosen_lr,
            "attempts": attempts,
            "gradient_l2_norm_fp32": gradient_norm,
            "foundation_lora_update_l2_norm_fp32": update_norm,
            "optimizer_steps_in_saved_state": 1,
        },
        "checks": checks,
        "required_checks": required_checks,
    }
    _save_json(foundation_dir / "layer3_trace.json", layer3_trace)

    return {
        "run_id": args.run_id,
        "gate": "P1-D",
        "status": status,
        "purpose": "one-round end-to-end wiring check through Layers 1, 2, and 3",
        "claim_limit": "Execution/state-lineage evidence only; not a performance or proxy-quality result.",
        "parent_layer1_run": str(parent_run),
        "configuration": {
            "round": server_round,
            "client_count": 2,
            "client_samples_per_class": 3,
            "prototype_count": len(prototypes),
            "server_seed_count": len(server_seed_rows),
            "server_seed_label_counts": seed_label_counts,
            "top_k": top_k,
            "transformations_per_seed": len(transformations),
            "proxy_candidate_count": len(variants),
            "proxy_pair_count": len(proxy_pairs),
            "foundation_acceptance": args.foundation_acceptance,
        },
        "selected_proxy_pairs": selections,
        "foundation": {
            "weighted_loss_before": float(weighted_before.detach().cpu()),
            "weighted_loss_after": float(weighted_after.detach().cpu()),
            "accepted_learning_rate": chosen_lr,
            "acceptance_policy": args.foundation_acceptance,
            "weighted_loss_nonincrease": weighted_loss_nonincrease,
            "gradient_l2_norm_fp32": gradient_norm,
            "update_l2_norm_fp32": update_norm,
            "before_digest": foundation_digest_before,
            "after_digest": foundation_digest_after,
            "after_checkpoint": str(f1_path),
        },
        "checks": checks,
        "timings_sec": {
            "layer1_parent": parent_summary.get("timings_sec", {}),
            **timings,
            "p1d_layer2_and_layer3_elapsed": time.perf_counter() - started,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage1.yaml")
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--foundation-checkpoint", default=None)
    parser.add_argument(
        "--foundation-acceptance",
        choices=("nonincrease", "execution"),
        default="nonincrease",
        help=(
            "nonincrease requires the same-pair weighted KL not to increase; "
            "execution accepts the first finite non-zero update for Stage-I wiring checks"
        ),
    )
    args = parser.parse_args()
    run_dir = _ROOT / "artifacts" / "runs" / "fedpact_stage1" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _save_json(
        run_dir / "run_status.json",
        {"run_id": args.run_id, "gate": "P1-D", "status": "running", "started_at": _utc_now()},
    )
    try:
        summary = _run(args, run_dir)
        _save_json(run_dir / "summary.json", summary)
        _save_json(
            run_dir / "run_status.json",
            {"run_id": args.run_id, "gate": "P1-D", "status": summary["status"], "finished_at": _utc_now()},
        )
        print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))
        return 0 if summary["status"] == "pass" else 2
    except BaseException as exc:
        _save_json(
            run_dir / "run_status.json",
            {
                "run_id": args.run_id,
                "gate": "P1-D",
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "finished_at": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
