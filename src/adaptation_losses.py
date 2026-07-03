from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vl_model import QwenVLClassifier, kl_distillation, logits_loss, softmax_mse


def grad_vec(loss: torch.Tensor, params: List[nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    vec: List[torch.Tensor] = []
    for g in grads:
        if g is None:
            vec.append(torch.zeros(1, device=loss.device, dtype=loss.dtype))
        else:
            vec.append(g.reshape(-1))
    return torch.cat(vec, dim=0)


def grad_vec_create_graph(loss: torch.Tensor, params: List[nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=True,
    )
    vec: List[torch.Tensor] = []
    for g in grads:
        if g is None:
            vec.append(torch.zeros(1, device=loss.device, dtype=loss.dtype))
        else:
            vec.append(g.reshape(-1))
    return torch.cat(vec, dim=0)


@torch.no_grad()
def sgd_step_inplace(params: List[nn.Parameter], grads: List[torch.Tensor | None], lr: float) -> None:
    for p, g in zip(params, grads):
        if g is None:
            continue
        p.data.add_(other=g, alpha=-lr)


@torch.no_grad()
def copy_params(target: List[nn.Parameter], source: List[torch.Tensor]) -> None:
    for p, s in zip(target, source):
        p.data.copy_(s)


@torch.no_grad()
def snapshot_params(params: List[nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().float().clone() for p in params]


class AdaptationLosses:
    """教授定式に基づく Adaptation Consistency 損失群。

    F (foundation) = surrogate adapter（参照、EMA で client から遅れる）
    S (surrogate, student) = client adapter（学習対象）
    Lpred = KL(pF || pS) * T^2 を採用する。
    """

    def __init__(
        self,
        model: QwenVLClassifier,
        processor,
        *,
        w_pred: float,
        w_grad: float,
        w_post: float,
        inner_lr: float,
        max_length: int,
        prompt: str,
        client_adapter: str,
        surrogate_adapter: str,
        client_lora_params: List[nn.Parameter],
        surrogate_lora_params: List[nn.Parameter],
        temperature: float = 2.0,
    ) -> None:
        self.model = model
        self.processor = processor
        self.w_pred = float(w_pred)
        self.w_grad = float(w_grad)
        self.w_post = float(w_post)
        self.inner_lr = float(inner_lr)
        self.max_length = int(max_length)
        self.prompt = str(prompt)
        self.client_adapter = client_adapter
        self.surrogate_adapter = surrogate_adapter
        self.client_lora_params = client_lora_params
        self.surrogate_lora_params = surrogate_lora_params
        self.temperature = float(temperature)

    def _kl(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        return kl_distillation(student, teacher, self.temperature)

    def _set_adapter(self, name: str) -> None:
        self.model.backbone.set_adapter(name)
        # PEFT の set_adapter は非活性アダプタの requires_grad を False に落とすが、
        # Lgrad / Lpost ではどちらのアダプタにも grad が必要なので両方とも True を維持する。
        for p in self.client_lora_params:
            p.requires_grad = True
        for p in self.surrogate_lora_params:
            p.requires_grad = True

    def forward_logits(self, pil_images, adapter: str) -> torch.Tensor:
        self._set_adapter(adapter)
        logits, _ = self.model.forward_batch(
            self.processor,
            pil_images,
            self.prompt,
            self.max_length,
            output_hidden_states=True,
        )
        return logits

    def compute(
        self,
        mode: str,
        pil_images,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        """mode: lpred | lpred_grad | lpred_post. teacher_logits 指定時は Lpred に注入。"""
        if mode == "lpred":
            if teacher_logits is not None:
                total, stats = self.lpred_from_teacher_logits(pil_images, labels, teacher_logits)
            else:
                total, stats = self.lpred_only(pil_images, labels)
            return total, stats, {}
        if mode == "lpred_grad":
            total, stats = self.lpred_lgrad(pil_images, labels)
            return total, stats, {}
        if mode == "lpred_post":
            total, stats, extra = self.lpred_lpost(pil_images, labels)
            return total, stats, extra
        raise ValueError(f"unknown adaptation mode: {mode}")

    def lpred_from_teacher_logits(
        self,
        pil_images,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Stage 1: 送信 logits を teacher として Lpred（基盤 LoRA = client_adapter を更新）。"""
        self._set_adapter(self.client_adapter)
        logits_c, _ = self.model.forward_batch(
            self.processor,
            pil_images,
            self.prompt,
            self.max_length,
            output_hidden_states=True,
        )
        l_task = logits_loss(logits_c, labels)
        l_pred = self._kl(student=logits_c, teacher=teacher_logits.detach())
        total = l_task + self.w_pred * l_pred
        stats = {
            "loss_total": float(total.detach().cpu()),
            "loss_task": float(l_task.detach().cpu()),
            "loss_pred": float(l_pred.detach().cpu()),
            "loss_grad": 0.0,
            "loss_post": 0.0,
            "teacher_source": "transmitted",
        }
        return total, stats

    def lpred_only(
        self,
        pil_images,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._set_adapter(self.client_adapter)
        logits_c, _ = self.model.forward_batch(
            self.processor,
            pil_images,
            self.prompt,
            self.max_length,
            output_hidden_states=True,
        )
        l_task = logits_loss(logits_c, labels)
        with torch.no_grad():
            logits_s = self.forward_logits(pil_images, self.surrogate_adapter)
        l_pred = self._kl(student=logits_c, teacher=logits_s.detach())
        total = l_task + self.w_pred * l_pred
        stats = {
            "loss_total": float(total.detach().cpu()),
            "loss_task": float(l_task.detach().cpu()),
            "loss_pred": float(l_pred.detach().cpu()),
            "loss_grad": 0.0,
            "loss_post": 0.0,
            "teacher_source": "recompute",
        }
        return total, stats

    def lpred_lgrad(
        self,
        pil_images,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._set_adapter(self.client_adapter)
        logits_c, _ = self.model.forward_batch(
            self.processor,
            pil_images,
            self.prompt,
            self.max_length,
            output_hidden_states=True,
        )
        l_task = logits_loss(logits_c, labels)
        with torch.no_grad():
            logits_s0 = self.forward_logits(pil_images, self.surrogate_adapter)
        l_pred = self._kl(student=logits_c, teacher=logits_s0.detach())
        l_combined = l_task + self.w_pred * l_pred

        g_c = grad_vec_create_graph(l_combined, self.client_lora_params)

        self._set_adapter(self.surrogate_adapter)
        logits_s, _ = self.model.forward_batch(
            self.processor,
            pil_images,
            self.prompt,
            self.max_length,
            output_hidden_states=True,
        )
        l_task_s = logits_loss(logits_s, labels)
        with torch.no_grad():
            self._set_adapter(self.client_adapter)
            logits_c_fix, _ = self.model.forward_batch(
                self.processor,
                pil_images,
                self.prompt,
                self.max_length,
                output_hidden_states=True,
            )
        self._set_adapter(self.surrogate_adapter)
        l_pred_s = self._kl(student=logits_s, teacher=logits_c_fix.detach())
        l_combined_s = l_task_s + self.w_pred * l_pred_s
        g_s = grad_vec(l_combined_s, self.surrogate_lora_params)
        self._set_adapter(self.client_adapter)

        l_grad = F.mse_loss(g_c, g_s.detach())
        total = l_combined + self.w_grad * l_grad
        stats = {
            "loss_total": float(total.detach().cpu()),
            "loss_task": float(l_task.detach().cpu()),
            "loss_pred": float(l_pred.detach().cpu()),
            "loss_grad": float(l_grad.detach().cpu()),
            "loss_post": 0.0,
        }
        return total, stats

    def _symmetric_post_update_mse(
        self,
        pil_images,
        labels: torch.Tensor,
        logits_c_ref: torch.Tensor,
    ) -> float:
        snap_c = snapshot_params(self.client_lora_params)
        snap_s = snapshot_params(self.surrogate_lora_params)

        with torch.enable_grad():
            self._set_adapter(self.client_adapter)
            logits_c0, _ = self.model.forward_batch(
                self.processor,
                pil_images,
                self.prompt,
                self.max_length,
                output_hidden_states=True,
            )
            l_task_c = logits_loss(logits_c0, labels)
            l_pred_c = self._kl(student=logits_c0, teacher=logits_c_ref.detach())
            l_combined_c = l_task_c + self.w_pred * l_pred_c
            gc = torch.autograd.grad(
                l_combined_c,
                self.client_lora_params,
                retain_graph=False,
                allow_unused=True,
            )
        with torch.no_grad():
            sgd_step_inplace(self.client_lora_params, list(gc), self.inner_lr)
            logits_c1 = self.forward_logits(pil_images, self.client_adapter)
            copy_params(self.client_lora_params, snap_c)

        with torch.enable_grad():
            self._set_adapter(self.surrogate_adapter)
            logits_s0, _ = self.model.forward_batch(
                self.processor,
                pil_images,
                self.prompt,
                self.max_length,
                output_hidden_states=True,
            )
            l_task_s = logits_loss(logits_s0, labels)
            l_pred_s = self._kl(student=logits_s0, teacher=logits_c_ref.detach())
            l_combined_s = l_task_s + self.w_pred * l_pred_s
            gs = torch.autograd.grad(
                l_combined_s,
                self.surrogate_lora_params,
                retain_graph=False,
                allow_unused=True,
            )
        with torch.no_grad():
            sgd_step_inplace(self.surrogate_lora_params, list(gs), self.inner_lr)
            logits_s1 = self.forward_logits(pil_images, self.surrogate_adapter)
            val = float(F.mse_loss(logits_c1, logits_s1).detach().cpu())
            copy_params(self.surrogate_lora_params, snap_s)

        self._set_adapter(self.client_adapter)
        return val

    def lpred_lpost(
        self,
        pil_images,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        self._set_adapter(self.client_adapter)
        logits_c, _ = self.model.forward_batch(
            self.processor,
            pil_images,
            self.prompt,
            self.max_length,
            output_hidden_states=True,
        )
        l_task = logits_loss(logits_c, labels)
        with torch.no_grad():
            logits_s0 = self.forward_logits(pil_images, self.surrogate_adapter)
        l_pred = self._kl(student=logits_c, teacher=logits_s0.detach())
        l_combined = l_task + self.w_pred * l_pred

        snap_s = snapshot_params(self.surrogate_lora_params)
        self._set_adapter(self.surrogate_adapter)
        logits_s_pre, _ = self.model.forward_batch(
            self.processor,
            pil_images,
            self.prompt,
            self.max_length,
            output_hidden_states=True,
        )
        l_task_s = logits_loss(logits_s_pre, labels)
        l_pred_s = self._kl(student=logits_s_pre, teacher=logits_c.detach())
        l_combined_s = l_task_s + self.w_pred * l_pred_s
        grads_s = torch.autograd.grad(
            l_combined_s,
            self.surrogate_lora_params,
            retain_graph=False,
            allow_unused=True,
        )
        with torch.no_grad():
            sgd_step_inplace(self.surrogate_lora_params, list(grads_s), self.inner_lr)
            logits_s_post = self.forward_logits(pil_images, self.surrogate_adapter)
        copy_params(self.surrogate_lora_params, snap_s)

        l_post = self._kl(student=logits_c, teacher=logits_s_post.detach())
        total = l_combined + self.w_post * l_post

        metrics: Dict[str, float] = {}
        metrics["sym_post_logits_mse"] = self._symmetric_post_update_mse(
            pil_images, labels, logits_c.detach()
        )

        stats = {
            "loss_total": float(total.detach().cpu()),
            "loss_task": float(l_task.detach().cpu()),
            "loss_pred": float(l_pred.detach().cpu()),
            "loss_grad": 0.0,
            "loss_post": float(l_post.detach().cpu()),
        }
        self._set_adapter(self.client_adapter)
        return total, stats, metrics
