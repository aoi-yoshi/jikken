"""Step 7 ② クライアント L_task / ④ サーバ整合の学習ステップ。"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from src.adaptation_losses import AdaptationLosses
from src.fl_logits import logits_map_to_tensor
from src.train_common import iter_row_chunks, load_sample_row, load_samples
from src.vl_model import logits_loss


def _set_requires_grad(params: List[nn.Parameter], flag: bool) -> None:
    for p in params:
        p.requires_grad = flag


def train_client_l_task(
    *,
    model,
    processor,
    train_rows: List[dict],
    dcfg: dict,
    client_lora: List[nn.Parameter],
    frozen_lora: List[nn.Parameter],
    opt: torch.optim.Optimizer,
    prompt: str,
    max_length: int,
    device: torch.device,
    batch_size: int,
    max_steps: int | None,
    extra_loss_fn: Callable[[], torch.Tensor] | None = None,
    gradient_accumulation_steps: int = 1,
    update_client_lora: bool = True,
    audit_parameter_groups: Mapping[str, List[nn.Parameter]] | None = None,
    class_weights: torch.Tensor | None = None,
) -> Tuple[int, Dict[str, float]]:
    """第1の学習場所: クライアント LoRA + 分類ヘッドを L_task のみで更新。

    extra_loss_fn: baseline_fedprox の proximal 項など、L_task に加算する追加損失。
    戻り値の avg_loss_task / loss_task_last は分類 CE のみ（prox 不含）。
    avg_loss_fit / loss_fit_last は backward した全損失（prox 含む）。
    """
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    model.train()
    _set_requires_grad(client_lora, update_client_lora)
    _set_requires_grad(frozen_lora, False)
    chunks = list(iter_row_chunks(train_rows, batch_size))
    if max_steps is not None:
        chunks = chunks[:max_steps]

    microbatch_steps = 0
    optimizer_steps = 0
    sample_exposures = 0
    loss_task_sum = 0.0
    loss_fit_sum = 0.0
    loss_task_last = 0.0
    loss_fit_last = 0.0
    grad_norm_last: Dict[str, float] = {}
    grad_norm_max: Dict[str, float] = {}

    for group_start in range(0, len(chunks), gradient_accumulation_steps):
        group = chunks[group_start : group_start + gradient_accumulation_steps]
        group_sample_count = sum(len(chunk) for chunk in group)
        opt.zero_grad(set_to_none=True)
        for chunk in group:
            samples = load_samples(chunk, dcfg)
            y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
            frames_batch = [s.frames for s in samples]
            model.backbone.set_adapter("client")
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, _ = model.forward_samples(
                    processor, frames_batch, prompt, max_length, output_hidden_states=True
                )
                if class_weights is None:
                    task_loss = logits_loss(logits.float(), y)
                else:
                    per_sample_loss = F.cross_entropy(logits.float(), y, reduction="none")
                    task_loss = (per_sample_loss * class_weights[y]).mean()
                loss = task_loss
                if extra_loss_fn is not None:
                    loss = loss + extra_loss_fn()

            # 各groupの全sampleに対する平均損失と同じ勾配尺度にする。
            scale = len(samples) / max(1, group_sample_count)
            (loss * scale).backward()
            n_samples = len(samples)
            microbatch_steps += 1
            sample_exposures += n_samples
            loss_task_sum += float(task_loss.detach().cpu()) * n_samples
            loss_fit_sum += float(loss.detach().cpu()) * n_samples
            loss_task_last = float(task_loss.detach().cpu())
            loss_fit_last = float(loss.detach().cpu())
            for sample in samples:
                for frame in sample.frames:
                    frame.close()

        for name, params in (audit_parameter_groups or {}).items():
            squared = 0.0
            for parameter in params:
                if parameter.grad is not None:
                    squared += float(parameter.grad.detach().float().pow(2).sum().cpu())
            norm = squared**0.5
            grad_norm_last[name] = norm
            grad_norm_max[name] = max(grad_norm_max.get(name, 0.0), norm)

        opt.step()
        for p in frozen_lora:
            p.grad = None
        optimizer_steps += 1

    n = max(1, sample_exposures)
    return microbatch_steps, {
        "avg_loss_task": loss_task_sum / n,
        "loss_task_last": loss_task_last,
        "avg_loss_fit": loss_fit_sum / n,
        "loss_fit_last": loss_fit_last,
        "microbatch_steps": microbatch_steps,
        "optimizer_steps": optimizer_steps,
        "sample_exposures": sample_exposures,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "grad_l2_norm_last": grad_norm_last,
        "grad_l2_norm_max": grad_norm_max,
    }


def train_server_consistency_transmitted(
    *,
    model,
    processor,
    losses: AdaptationLosses,
    train_rows: List[dict],
    dcfg: dict,
    frozen_lora: List[nn.Parameter],
    surrogate_lora: List[nn.Parameter],
    opt: torch.optim.Optimizer,
    mode: str,
    device: torch.device,
    max_steps: int | None,
    teacher_logits_map: Dict[str, List[float]],
    grad_teacher: torch.Tensor | None = None,
    post_logits_map: Dict[str, List[float]] | None = None,
    replay_logits_map: Dict[str, List[float]] | None = None,
    grad_match: str = "mse",
    replay_weight: float = 0.5,
) -> Tuple[int, Dict[str, float]]:
    """レパートリー用サーバ整合: 送信された適応情報のみで基盤 LoRA を更新する。

    mode: lpred | lpred_grad | lpred_post | full
    teacher_logits_map: 集約済みクライアント logits（必須）
    grad_teacher:       集約済みクライアント勾配（lpred_grad / full）
    post_logits_map:    集約済み 1-step 適応後 logits（lpred_post / full）
    replay_logits_map:  前ラウンド server logits（use_replay 時）
    """
    from src.dataset_manifest import load_sample_row as _load_row

    model.train()
    _set_requires_grad(frozen_lora, False)
    _set_requires_grad(surrogate_lora, True)
    steps = 0
    last_stats: Dict[str, float] = {}
    for r in train_rows:
        if max_steps is not None and steps >= max_steps:
            break
        sid = str(r["id"])
        if sid not in teacher_logits_map:
            raise KeyError(f"sample_id {sid} missing from transmitted logits")
        sample = _load_row(r, dcfg)
        y = torch.tensor([sample.label], device=device, dtype=torch.long)
        teacher = logits_map_to_tensor(teacher_logits_map, sid, device)
        post_teacher = None
        if post_logits_map is not None:
            if sid not in post_logits_map:
                raise KeyError(f"sample_id {sid} missing from post_logits")
            post_teacher = logits_map_to_tensor(post_logits_map, sid, device)
        replay = None
        if replay_logits_map is not None and sid in replay_logits_map:
            replay = logits_map_to_tensor(replay_logits_map, sid, device)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            total, stats = losses.compute_transmitted(
                mode,
                sample.frames,
                y,
                teacher_logits=teacher,
                grad_teacher=grad_teacher,
                post_teacher_logits=post_teacher,
                replay_logits=replay,
                grad_match=grad_match,
                replay_weight=replay_weight,
            )
        total.backward()
        opt.step()
        for p in frozen_lora:
            p.grad = None
        last_stats = dict(stats)
        steps += 1
    return steps, last_stats


def train_server_consistency(
    *,
    model,
    processor,
    losses: AdaptationLosses,
    train_rows: List[dict],
    dcfg: dict,
    client_lora: List[nn.Parameter],
    surrogate_lora: List[nn.Parameter],
    opt: torch.optim.Optimizer,
    mode: str,
    device: torch.device,
    max_steps: int | None,
    teacher_source: str = "recompute",
    teacher_logits_map: Dict[str, List[float]] | None = None,
) -> Tuple[int, Dict[str, float]]:
    """第2の学習場所: 基盤 LoRA を整合損失で更新（w_task=0 なら L_task を含めない）。"""
    model.train()
    _set_requires_grad(client_lora, False)
    _set_requires_grad(surrogate_lora, True)
    steps = 0
    last_stats: Dict[str, float] = {}
    use_transmitted = teacher_source == "transmitted" and mode == "lpred"
    if use_transmitted and not teacher_logits_map:
        raise RuntimeError("teacher_source=transmitted requires teacher_logits_map")

    for r in train_rows:
        if max_steps is not None and steps >= max_steps:
            break
        sample = load_sample_row(r, dcfg)
        y = torch.tensor([sample.label], device=device, dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        teacher_logits = None
        if use_transmitted:
            sid = str(r["id"])
            if sid not in teacher_logits_map:
                raise KeyError(f"sample_id {sid} missing from transmitted logits")
            teacher_logits = logits_map_to_tensor(teacher_logits_map, sid, device)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            total, stats, _extra = losses.compute(mode, sample.frames, y, teacher_logits=teacher_logits)
        total.backward()
        opt.step()
        for p in client_lora:
            p.grad = None
        last_stats = dict(stats)
        steps += 1
    return steps, last_stats
