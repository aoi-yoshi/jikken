"""FedPACT P1-C: P1-Bの1 proxy pairだけで独立foundation LoRAを1 step更新する。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_loader import load_yaml, merged_config  # noqa: E402
from src.fedpact_stage1 import (  # noqa: E402
    canonical_tensor_digest,
    flatten_named_tensors,
    named_selected_parameters,
    sha256_file,
)
from src.fl_utils import (  # noqa: E402
    load_fl_checkpoint,
    save_fl_checkpoint,
    vector_to_trainable_state,
)
from src.metrics import set_seed  # noqa: E402
from src.peft_setup import attach_dual_lora, lora_params_for_adapter  # noqa: E402
from src.train_common import device_or_auto  # noqa: E402
from src.vl_model import build_model  # noqa: E402


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapped_foundation_checkpoint(path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    names, vector, meta = load_fl_checkpoint(path)
    mapped = [
        name.replace(".client.", ".foundation.")
        if "lora_" in name and ".client." in name
        else name
        for name in names
    ]
    return mapped, vector, meta


def _named_lora_state(
    model: torch.nn.Module, parameters: Sequence[torch.nn.Parameter]
) -> tuple[list[str], np.ndarray, str]:
    named = named_selected_parameters(model, parameters)
    names, vector = flatten_named_tensors(named)
    return names, vector, canonical_tensor_digest(named)


def _gradient_vector(parameters: Sequence[torch.nn.Parameter]) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            chunks.append(torch.zeros_like(parameter, device="cpu", dtype=torch.float32).reshape(-1))
        else:
            chunks.append(grad.detach().cpu().float().reshape(-1))
    return torch.cat(chunks).numpy().astype(np.float32, copy=False)


def _kl_teacher_to_student(student_logits: torch.Tensor, teacher_probability: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        teacher_probability,
        reduction="batchmean",
    )


def _run(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = merged_config(load_yaml(args.config))
    scfg = dict(cfg["fedpact_stage1"])
    dcfg = dict(cfg["data"])
    tcfg = dict(cfg["train"])
    seed = int(tcfg["seed"])
    set_seed(seed)

    parent_run = _resolve(args.parent_run)
    parent_summary = json.loads((parent_run / "summary.json").read_text(encoding="utf-8"))
    if parent_summary.get("gate") != "P1-B" or parent_summary.get("status") != "pass":
        raise ValueError("parent run is not a passed P1-B run")
    pair_path = parent_run / "server" / "proxy_pair.json"
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    if pair.get("schema") != "fedpact.proxy_pair.v2":
        raise ValueError(f"unsupported proxy pair schema: {pair.get('schema')}")
    if pair.get("layer3_contract") != "P_{k,r}=(x_tilde_{k,r}, mu_post_{k,r})":
        raise ValueError("proxy pair does not declare the P_{k,r} Layer 3 contract")
    frame_paths = [parent_run / value for value in pair["x_tilde_{k,r}"]["frame_files"]]
    if len(frame_paths) != int(dcfg["use_num_frames"]):
        raise ValueError("proxy pair frame count does not match Stage I input contract")
    frames = [Image.open(path).convert("RGB") for path in frame_paths]

    device_name = device_or_auto(True) if args.device == "auto" else args.device
    device = torch.device(device_name)
    foundation_model_id = str(cfg.get("model", {}).get("server_id") or cfg["model"]["id"])
    model, processor = build_model(cfg, device, model_id=foundation_model_id)
    model.backbone = attach_dual_lora(
        model.backbone, cfg, client="foundation", surrogate="surrogate"
    )
    model.backbone.set_adapter("foundation")
    foundation_lora = lora_params_for_adapter(model.backbone, "foundation")
    inactive_surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in foundation_lora:
        parameter.requires_grad_(True)
    if any(parameter.requires_grad for parameter in model.classifier.parameters()):
        raise AssertionError("foundation classification head must remain fixed")

    source_checkpoint = _resolve(str(scfg["initial_foundation_checkpoint"]))
    mapped_names, mapped_vector, source_meta = _mapped_foundation_checkpoint(source_checkpoint)
    available = {name for name, _ in model.named_parameters()}
    missing = [name for name in mapped_names if name not in available]
    if missing:
        raise KeyError(f"foundation checkpoint parameters missing: {missing[:5]}")
    vector_to_trainable_state(model, mapped_names, mapped_vector)

    foundation_names, foundation_before, foundation_digest_before = _named_lora_state(
        model, foundation_lora
    )
    head_named_before = [
        (f"classifier.{name}", tensor) for name, tensor in model.classifier.state_dict().items()
    ]
    head_digest_before = canonical_tensor_digest(head_named_before)
    surrogate_named_before = named_selected_parameters(model, inactive_surrogate_lora)
    surrogate_digest_before = canonical_tensor_digest(surrogate_named_before)
    target_ids = {id(parameter) for parameter in foundation_lora}
    non_target_versions_before = {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
        if id(parameter) not in target_ids
    }
    non_target_grad_policy = all(
        not parameter.requires_grad
        for parameter in model.parameters()
        if id(parameter) not in target_ids
    )

    foundation_dir = run_dir / "foundation"
    f0_path = foundation_dir / "F_0.pt"
    save_fl_checkpoint(
        f0_path,
        names=foundation_names,
        vector=foundation_before,
        meta={
            "run_id": args.run_id,
            "state_id": "F^0",
            "role": "foundation_lora_only",
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_file_sha256": sha256_file(source_checkpoint),
            "source_run_id": source_meta.get("run_id"),
        },
    )

    teacher = torch.tensor(
        pair["mu_post_{k,r}"], dtype=torch.float32, device=device
    ).reshape(1, -1)
    if not torch.isfinite(teacher).all() or not torch.isclose(
        teacher.sum(), torch.tensor(1.0, device=device), rtol=1e-5, atol=1e-6
    ):
        raise ValueError("mu_post_{k,r} is not a finite probability vector")
    negative_teacher = teacher.flip(dims=(-1,))
    prompt = str(dcfg["image_prompt"])
    max_length = int(tcfg.get("max_length", 256))
    learning_rate = float(scfg.get("foundation_lr_p1c", 1.0e-6))
    weight_decay = 0.01
    temperature = 1.0
    optimizer = torch.optim.AdamW(
        foundation_lora, lr=learning_rate, weight_decay=weight_decay
    )
    optimizer_state_entries_at_start = len(optimizer.state)

    # Stage I配線確認ではdropout等を無効化したままgradientを取得する。
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    negative_logits, _ = model.forward_batch(
        processor, frames, prompt, max_length, output_hidden_states=True
    )
    negative_loss = _kl_teacher_to_student(negative_logits, negative_teacher)
    negative_loss.backward()
    negative_gradient = _gradient_vector(foundation_lora)
    optimizer.zero_grad(set_to_none=True)

    logits_before, _ = model.forward_batch(
        processor, frames, prompt, max_length, output_hidden_states=True
    )
    loss_before = _kl_teacher_to_student(logits_before, teacher)
    if not torch.isfinite(loss_before):
        raise FloatingPointError("non-finite official Layer 3 loss")
    loss_before.backward()
    official_gradient = _gradient_vector(foundation_lora)
    gradient_norm = float(np.linalg.norm(official_gradient))
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise AssertionError("foundation LoRA gradient is zero or non-finite")
    official_gradient_buffers = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in foundation_lora
    ]
    max_attempts = int(scfg.get("foundation_backtracking_max_attempts_p1c", 6))
    learning_rate_attempts: list[dict[str, Any]] = []
    accepted = False
    logits_after = logits_before.detach()
    loss_after = loss_before.detach()
    for attempt_index in range(max_attempts):
        candidate_lr = learning_rate * (0.1**attempt_index)
        vector_to_trainable_state(model, foundation_names, foundation_before)
        trial_optimizer = torch.optim.AdamW(
            foundation_lora, lr=candidate_lr, weight_decay=weight_decay
        )
        trial_state_entries_at_start = len(trial_optimizer.state)
        for parameter, gradient in zip(foundation_lora, official_gradient_buffers):
            parameter.grad = None if gradient is None else gradient.clone()
        trial_optimizer.step()
        trial_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            trial_logits, _ = model.forward_batch(
                processor, frames, prompt, max_length, output_hidden_states=True
            )
            trial_loss = _kl_teacher_to_student(trial_logits, teacher)
        trial_value = float(trial_loss.detach().cpu())
        accepted_trial = trial_value <= float(loss_before.detach().cpu()) + 1.0e-8
        learning_rate_attempts.append(
            {
                "attempt": attempt_index + 1,
                "learning_rate": candidate_lr,
                "loss_after": trial_value,
                "optimizer_state_entries_at_start": trial_state_entries_at_start,
                "accepted": accepted_trial,
            }
        )
        if accepted_trial:
            learning_rate = candidate_lr
            optimizer_state_entries_at_start = trial_state_entries_at_start
            logits_after = trial_logits
            loss_after = trial_loss
            accepted = True
            break
    if not accepted:
        raise AssertionError(f"no non-increasing P1-C step found: {learning_rate_attempts}")

    foundation_after_names, foundation_after, foundation_digest_after = _named_lora_state(
        model, foundation_lora
    )
    if foundation_after_names != foundation_names:
        raise AssertionError("foundation LoRA parameter order changed")
    update = foundation_after - foundation_before
    update_norm = float(np.linalg.norm(update))
    head_named_after = [
        (f"classifier.{name}", tensor) for name, tensor in model.classifier.state_dict().items()
    ]
    head_digest_after = canonical_tensor_digest(head_named_after)
    surrogate_digest_after = canonical_tensor_digest(
        named_selected_parameters(model, inactive_surrogate_lora)
    )
    non_target_versions_after = {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
        if id(parameter) not in target_ids
    }
    non_target_gradients_absent = all(
        parameter.grad is None
        for parameter in model.parameters()
        if id(parameter) not in target_ids
    )
    negative_loss_value = float(negative_loss.detach().cpu())
    loss_before_value = float(loss_before.detach().cpu())
    loss_after_value = float(loss_after.detach().cpu())
    loss_changed = not math.isclose(
        negative_loss_value, loss_before_value, rel_tol=1e-6, abs_tol=1e-8
    )
    gradient_changed = not np.allclose(
        negative_gradient, official_gradient, rtol=1e-5, atol=1e-7
    )

    f1_path = foundation_dir / "F_1.pt"
    save_fl_checkpoint(
        f1_path,
        names=foundation_names,
        vector=foundation_after,
        meta={
            "run_id": args.run_id,
            "state_id": "F^1",
            "role": "foundation_lora_only",
            "parent_state": "F^0",
            "pair_id": pair["pair_id"],
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer_steps": 1,
        },
    )
    reloaded_names, reloaded_vector, _ = load_fl_checkpoint(f1_path)
    saved_state_matches = reloaded_names == foundation_names and np.array_equal(
        reloaded_vector, foundation_after
    )

    checks = {
        "parent_p1b_pass": True,
        "pair_schema_valid": True,
        "teacher_is_mu_post": True,
        "temperature_is_1": temperature == 1.0,
        "optimizer_state_fresh": optimizer_state_entries_at_start == 0,
        "foundation_lora_update_finite_nonzero": math.isfinite(update_norm) and update_norm > 0.0,
        "same_pair_loss_nonincrease": loss_after_value <= loss_before_value + 1.0e-8,
        "classification_head_unchanged": head_digest_before == head_digest_after,
        "inactive_surrogate_lora_unchanged": surrogate_digest_before == surrogate_digest_after,
        "all_non_target_parameters_frozen": non_target_grad_policy,
        "non_target_parameter_versions_unchanged": (
            non_target_versions_before == non_target_versions_after
        ),
        "non_target_gradients_absent": non_target_gradients_absent,
        "negative_teacher_sensitivity": loss_changed or gradient_changed,
        "output_checkpoint_reload_exact": saved_state_matches,
        "all_recorded_values_finite": all(
            math.isfinite(value)
            for value in [
                negative_loss_value,
                loss_before_value,
                loss_after_value,
                gradient_norm,
                update_norm,
            ]
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    trace = {
        "schema": "fedpact.layer3_trace.v1",
        "run_id": args.run_id,
        "gate": "P1-C",
        "status": status,
        "parent_p1b_run": str(parent_run),
        "pair": {
            "path": str(pair_path),
            "file_sha256": sha256_file(pair_path),
            "pair_id": pair["pair_id"],
            "prototype_id": pair["prototype_id"],
            "round": pair["round"],
            "client_id": pair["client_id"],
            "class_label": pair["class_label"],
            "input_frame_files": [str(path) for path in frame_paths],
            "teacher_mu_post": pair["mu_post_{k,r}"],
            "frequency_pi": pair["frequency_pi_{k,r}"],
        },
        "contract": {
            "student": "independent foundation model instance; foundation LoRA adapter only",
            "loss": "D_KL(mu_post_{k,r} || softmax(foundation(x_tilde_{k,r})))",
            "temperature": temperature,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer_steps": 1,
            "foundation_classification_head": "fixed",
            "model_mode": "eval (dropout disabled; gradients enabled)",
        },
        "foundation_state": {
            "initial_source_checkpoint": str(source_checkpoint),
            "initial_source_checkpoint_file_sha256": sha256_file(source_checkpoint),
            "before_checkpoint": str(f0_path.relative_to(run_dir)).replace("\\", "/"),
            "after_checkpoint": str(f1_path.relative_to(run_dir)).replace("\\", "/"),
            "before_checkpoint_file_sha256": sha256_file(f0_path),
            "after_checkpoint_file_sha256": sha256_file(f1_path),
            "parameter_count": int(foundation_before.size),
            "before_tensor_state_digest": foundation_digest_before,
            "after_tensor_state_digest": foundation_digest_after,
            "update_l2_norm_fp32": update_norm,
        },
        "optimization": {
            "student_logits_before": logits_before.detach().float().cpu().tolist()[0],
            "student_logits_after": logits_after.detach().float().cpu().tolist()[0],
            "loss_before": loss_before_value,
            "loss_after": loss_after_value,
            "gradient_l2_norm_fp32": gradient_norm,
            "optimizer_state_entries_at_start": optimizer_state_entries_at_start,
            "learning_rate_attempts": learning_rate_attempts,
        },
        "negative_control": {
            "procedure": "same x_tilde; teacher class probabilities swapped; backward only; no step",
            "teacher_probability": negative_teacher.detach().cpu().tolist()[0],
            "loss": negative_loss_value,
            "gradient_l2_norm_fp32": float(np.linalg.norm(negative_gradient)),
            "loss_changed": loss_changed,
            "gradient_changed": gradient_changed,
            "applied_to_official_checkpoint": False,
        },
        "non_target_state": {
            "classification_head_before_digest": head_digest_before,
            "classification_head_after_digest": head_digest_after,
            "inactive_surrogate_lora_before_digest": surrogate_digest_before,
            "inactive_surrogate_lora_after_digest": surrogate_digest_after,
            "parameter_version_guard_count": len(non_target_versions_before),
        },
        "checks": checks,
        "claim_limit": "Wiring/state-change evidence only; no performance or knowledge-transfer claim.",
        "elapsed_sec": time.perf_counter() - started,
    }
    _save_json(foundation_dir / "layer3_trace.json", trace)
    summary = {
        "run_id": args.run_id,
        "gate": "P1-C",
        "status": status,
        "parent_p1b_run": str(parent_run),
        "pair_id": pair["pair_id"],
        "loss_before": loss_before_value,
        "loss_after": loss_after_value,
        "gradient_l2_norm_fp32": gradient_norm,
        "foundation_lora_update_l2_norm_fp32": update_norm,
        "foundation_before_digest": foundation_digest_before,
        "foundation_after_digest": foundation_digest_after,
        "after_checkpoint": str(f1_path),
        "checks": checks,
        "claim_limit": trace["claim_limit"],
        "elapsed_sec": trace["elapsed_sec"],
    }
    _save_json(run_dir / "summary.json", summary)
    _save_json(
        run_dir / "run_status.json",
        {"run_id": args.run_id, "gate": "P1-C", "status": status, "ended_at": _utc_now()},
    )
    for frame in frames:
        frame.close()
    if status != "pass":
        raise AssertionError(f"P1-C checks failed: {[key for key, value in checks.items() if not value]}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage1.yaml")
    parser.add_argument(
        "--parent-run",
        default="artifacts/runs/fedpact_stage1/p1b_20260917_0110",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_dir = _ROOT / "artifacts" / "runs" / "fedpact_stage1" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _save_json(
        run_dir / "run_status.json",
        {"run_id": args.run_id, "gate": "P1-C", "status": "running", "started_at": _utc_now()},
    )
    try:
        summary = _run(args, run_dir)
    except Exception as exc:
        _save_json(
            run_dir / "run_status.json",
            {
                "run_id": args.run_id,
                "gate": "P1-C",
                "status": "fail",
                "ended_at": _utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        )
        raise
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
