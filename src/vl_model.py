from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def _pool_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class QwenVLClassifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        attn_implementation: str = "sdpa",
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dtype = torch_dtype or torch.bfloat16
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=None,
            trust_remote_code=True,
        )
        hidden = int(self.backbone.config.text_config.hidden_size)
        self.classifier = nn.Linear(hidden, num_classes)
        nn.init.zeros_(self.classifier.bias)

    @property
    def processor(self):
        if not hasattr(self, "_processor"):
            raise RuntimeError("processor not set")
        return self._processor

    def bind_processor(self, processor) -> None:
        self._processor = processor

    def forward_samples(
        self,
        processor,
        batch_frames: List[List[Any]],
        prompt: str,
        max_length: int,
        output_hidden_states: bool = True,
        input_mode: str = "images",
        video_fps: List[float | None] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """複数サンプルを1 forward。batch_frames[i] = サンプル i の全フレーム。

        戻り logits は (batch_size, num_classes)。
        """
        if not batch_frames:
            raise ValueError("batch_frames is empty")
        mode = str(input_mode).strip().lower()
        conversations: List[List[Dict[str, Any]]] = []
        if mode == "images":
            for frames in batch_frames:
                content: List[Dict[str, Any]] = []
                for img in frames:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": prompt})
                conversations.append([{"role": "user", "content": content}])
            batch = processor.apply_chat_template(
                conversations,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        elif mode == "video":
            fps_values = list(video_fps or [None] * len(batch_frames))
            if len(fps_values) != len(batch_frames):
                raise ValueError("video_fps must have one value per sample")
            metadata: List[Dict[str, Any]] = []
            for frames, fps_value in zip(batch_frames, fps_values):
                if not frames:
                    raise ValueError("video sample has no frames")
                fps = float(fps_value) if fps_value is not None else 1.0
                if fps <= 0:
                    raise ValueError(f"video fps must be positive, got {fps}")
                conversations.append(
                    [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
                )
                metadata.append(
                    {
                        "fps": fps,
                        "total_num_frames": len(frames),
                        "duration": (len(frames) - 1) / fps if len(frames) > 1 else 0.0,
                        "video_backend": "preextracted_frames",
                        "frames_indices": list(range(len(frames))),
                    }
                )
            texts = [
                processor.apply_chat_template(
                    conversation, add_generation_prompt=False, tokenize=False
                )
                for conversation in conversations
            ]
            batch = processor(
                text=texts,
                videos=batch_frames,
                videos_kwargs={"video_metadata": metadata},
                padding=True,
                return_tensors="pt",
            )
        else:
            raise ValueError(f"Unknown input_mode: {input_mode!r} (expected 'images' or 'video')")
        device = self.classifier.weight.device
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        out = self.backbone(**batch, output_hidden_states=output_hidden_states)
        hs = out.hidden_states[-1]
        pooled = _pool_hidden(hs, batch["attention_mask"])
        pooled = pooled.to(dtype=self.classifier.weight.dtype)
        logits = self.classifier(pooled)
        meta = {
            "attention_mask": batch["attention_mask"],
            "batch_size": len(batch_frames),
            "input_mode": mode,
        }
        if mode == "video":
            meta["video_grid_thw"] = batch["video_grid_thw"]
            meta["second_per_grid_ts"] = batch["second_per_grid_ts"]
        return logits, meta

    def forward_batch(
        self,
        processor,
        pil_images: List[Any],
        prompt: str,
        max_length: int,
        output_hidden_states: bool = True,
        input_mode: str = "images",
        video_fps: float | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """1 サンプルとして pil_images をまとめて処理 (batch=1)。

        pil_images は 1 サンプルの全フレーム (例: 4 枚) を表す。
        戻り logits は (1, num_classes)。
        """
        return self.forward_samples(
            processor,
            [pil_images],
            prompt,
            max_length,
            output_hidden_states=output_hidden_states,
            input_mode=input_mode,
            video_fps=[video_fps],
        )


def build_processor(model_id: str, min_pixels: int, max_pixels: int):
    return AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def build_model(
    cfg: Dict[str, Any], device: torch.device, model_id: str | None = None
) -> Tuple[QwenVLClassifier, Any]:
    """model_id 指定時は cfg["model"]["id"] を上書き（7B サーバ + 3B クライアントの異種構成用）。"""
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    mid = model_id or mcfg["id"]
    model = QwenVLClassifier(
        mid,
        int(cfg["data"]["num_classes"]),
        attn_implementation=str(mcfg.get("attn_implementation", "sdpa")),
    )
    processor = build_processor(
        mid,
        int(tcfg.get("min_pixels", 50176)),
        int(tcfg.get("max_pixels", 602112)),
    )
    model.bind_processor(processor)
    model.to(device)
    model.classifier.to(dtype=torch.float32)
    if bool(tcfg.get("use_gradient_checkpointing", False)):
        try:
            model.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model.backbone, "enable_input_require_grads"):
                model.backbone.enable_input_require_grads()
        except Exception as e:
            print(f"[build_model] gradient checkpointing failed to enable: {e}")
    return model, processor


def freeze_backbone(model: QwenVLClassifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: QwenVLClassifier) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = True


def logits_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def softmax_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    pa = F.softmax(a, dim=-1)
    pb = F.softmax(b, dim=-1)
    return F.mse_loss(pa, pb)


def kl_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(p_teacher || p_student) * T^2, the standard distillation form.

    教授定式に従い、Lpred = KL(pF || pS) を再現する。
    F.kl_div(input=log_pS, target=pT) は ∑ pT (log pT - log pS) = KL(pT||pS) を返す。
    """
    T = float(temperature)
    student_log = F.log_softmax(student_logits / T, dim=-1)
    teacher_prob = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(student_log, teacher_prob, reduction="batchmean") * (T * T)
