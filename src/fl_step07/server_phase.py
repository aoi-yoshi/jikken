"""Step 7 Flower — サーバ aggregate_fit 内の GPU フェーズ（レパートリー分岐）。

proposed 系: 送信された適応情報（logits / 勾配 / post_logits）だけで基盤 LoRA を更新し、
             server logits を書き出す（FedAvg 後の重みはサーバモデルに載せない）。
baseline 系: 同じく基盤 LoRA を logits 整合で更新し distill 配布。
             端末は apply_global_weights=1 で FedAvg 後の重みも適用（7B/3B ではクライアント形状一致時）。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from src.adaptation_losses import AdaptationLosses
from src.fl_logits import (
    aggregate_client_logits,
    distill_adapter_from_logits,
    export_client_logits,
    load_client_logits_artifact,
)
from src.fl_step07.sidecar import (
    aggregated_client_logits_path,
    client_grads_path,
    client_logits_path,
    client_post_logits_path,
    foundation_checkpoint_path,
    server_logits_path,
    write_logits_sidecar,
)
from src.fl_utils import trainable_state_vector, vector_to_trainable_state
from src.metrics import evaluate_classifier
from src.peft_setup import attach_dual_lora, lora_params_for_adapter
from src.step07.pipeline import distill_pipeline_mode, resolve_step07_model_ids
from src.step07.repertoire import Repertoire
from src.step07.server_opt import build_server_optimizer
from src.step07.train_steps import train_server_consistency_transmitted
from src.step07.transmission import (
    aggregate_grad_vectors,
    load_grad_vector_sidecar,
    sidecar_size_bytes,
)
from src.vl_model import build_model, unfreeze_backbone


def _load_sidecar_maps(
    dir_glob_parent: Path,
    pattern: str,
    loader,
) -> Dict[int, Any]:
    """r{N}_c{cid} 形式の sidecar を cid → payload で返す。"""
    out: Dict[int, Any] = {}
    for path in sorted(dir_glob_parent.glob(pattern)):
        cid = int(path.stem.rsplit("_c", 1)[1])
        out[cid] = loader(path)
    return out


def _weights_for(
    cids: List[int],
    sample_counts: Mapping[int, int] | None,
    aggregation: str,
) -> Optional[List[float]]:
    """equal → None（等平均）/ weighted → サンプル数重み（FedDF 型）。"""
    if aggregation != "weighted" or not sample_counts:
        return None
    if any(c not in sample_counts for c in cids):
        return None
    return [float(sample_counts[c]) for c in cids]


class Step07ServerContext:
    """Flower サーバプロセス内で基盤モデルを保持し、各ラウンドのレパートリー処理を実行する。"""

    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        root: Path,
        run_dir: Path,
        repertoire: Repertoire,
        server_train: List[dict],
        server_eval: List[dict],
        device: torch.device,
        param_names: List[str],
        max_server_steps: int | None,
    ) -> None:
        self.cfg = cfg
        self.root = root
        self.run_dir = run_dir
        self.rep = repertoire
        self.server_train = server_train
        self.server_eval = server_eval
        self.device = device
        self.param_names = param_names
        self.max_server_steps = max_server_steps

        mcfg = cfg["model"]
        tcfg = cfg["train"]
        ccfg = cfg["consistency"]
        dcfg = cfg["data"]
        scfg = cfg.get("step07", {})

        self.server_model_id, self.client_model_id = resolve_step07_model_ids(cfg)
        self.distill_pipeline = distill_pipeline_mode(cfg)
        self.model_ids_differ = self.server_model_id != self.client_model_id
        self.prompt = str(dcfg.get("image_prompt", ""))
        self.max_length = int(tcfg.get("max_length", 256))
        self.bs = max(1, int(tcfg.get("batch_size", 1)))
        self.lr = float(tcfg["lr"])
        self.weight_decay = float(tcfg.get("weight_decay", 0.01))
        self.w_task = float(ccfg.get("w_task", repertoire.w_task))
        self.num_clients = int(scfg.get("num_clients", cfg["fl"].get("num_clients", 2)))
        self.distill_lr = float(scfg.get("distill_lr", tcfg["lr"]))
        distill_steps = scfg.get("distill_steps", 20)
        self.distill_steps = int(distill_steps) if distill_steps is not None else None
        self.temperature = float(ccfg.get("temperature", 2.0))
        self.replay_weight = float(scfg.get("replay_weight", repertoire.replay_weight))

        self._model = None
        self._processor = None
        self._losses: AdaptationLosses | None = None
        self._opt: torch.optim.Optimizer | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        model, processor = build_model(self.cfg, self.device, model_id=self.server_model_id)
        unfreeze_backbone(model)
        model.backbone = attach_dual_lora(model.backbone, self.cfg, client="client", surrogate="surrogate")
        surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
        server_side_client_lora = lora_params_for_adapter(model.backbone, "client")

        ccfg = self.cfg["consistency"]
        losses = AdaptationLosses(
            model,
            processor,
            w_task=self.w_task,
            w_pred=float(ccfg.get("w_pred", 0.5)),
            w_grad=float(ccfg.get("w_grad", 0.1)),
            w_post=float(ccfg.get("w_post", 0.5)),
            inner_lr=float(ccfg.get("inner_lr", 1e-4)),
            max_length=self.max_length,
            prompt=self.prompt,
            client_adapter="surrogate",
            surrogate_adapter="client",
            client_lora_params=surrogate_lora,
            surrogate_lora_params=server_side_client_lora,
            temperature=self.temperature,
        )
        opt = build_server_optimizer(
            self.rep.server_optimizer,
            list(model.classifier.parameters()) + surrogate_lora,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self._model = model
        self._processor = processor
        self._losses = losses
        self._opt = opt

    # ------------------------------------------------------------------
    # teacher 構成
    # ------------------------------------------------------------------

    def _collect_client_logits(
        self,
        server_round: int,
        sample_counts: Mapping[int, int] | None,
    ) -> Dict[str, List[float]]:
        logits_dir = client_logits_path(self.run_dir, server_round, 0).parent
        payloads = _load_sidecar_maps(logits_dir, f"r{server_round}_c*.json", load_client_logits_artifact)
        if len(payloads) < self.num_clients:
            raise RuntimeError(
                f"server_round={server_round}: expected {self.num_clients} client logits sidecars, "
                f"found {len(payloads)} under {logits_dir}"
            )
        cids = sorted(payloads.keys())
        maps = [payloads[c] for c in cids]
        weights = _weights_for(cids, sample_counts, self.rep.logits_aggregation)
        if len(maps) == 1:
            teacher = maps[0]
        else:
            teacher = aggregate_client_logits(maps, weights)
        agg_path = aggregated_client_logits_path(self.run_dir, server_round)
        write_logits_sidecar(agg_path, server_round, teacher)
        return teacher

    def _collect_grad_teacher(
        self,
        server_round: int,
        sample_counts: Mapping[int, int] | None,
    ) -> torch.Tensor:
        grads_dir = client_grads_path(self.run_dir, server_round, 0).parent
        payloads = _load_sidecar_maps(grads_dir, f"r{server_round}_c*.pt", load_grad_vector_sidecar)
        if len(payloads) < self.num_clients:
            raise RuntimeError(
                f"server_round={server_round}: expected {self.num_clients} grad sidecars, "
                f"found {len(payloads)} under {grads_dir}"
            )
        cids = sorted(payloads.keys())
        vecs = [payloads[c] for c in cids]
        weights = _weights_for(cids, sample_counts, "weighted")
        return aggregate_grad_vectors(vecs, weights)

    def _collect_post_logits(
        self,
        server_round: int,
        sample_counts: Mapping[int, int] | None,
    ) -> Dict[str, List[float]]:
        post_dir = client_post_logits_path(self.run_dir, server_round, 0).parent
        payloads = _load_sidecar_maps(post_dir, f"r{server_round}_c*.json", load_client_logits_artifact)
        if len(payloads) < self.num_clients:
            raise RuntimeError(
                f"server_round={server_round}: expected {self.num_clients} post_logits sidecars, "
                f"found {len(payloads)} under {post_dir}"
            )
        cids = sorted(payloads.keys())
        maps = [payloads[c] for c in cids]
        weights = _weights_for(cids, sample_counts, self.rep.logits_aggregation)
        return maps[0] if len(maps) == 1 else aggregate_client_logits(maps, weights)

    def _load_replay(self, server_round: int) -> Dict[str, List[float]] | None:
        if not self.rep.use_replay or server_round <= 1:
            return None
        prev = server_logits_path(self.run_dir, server_round - 1)
        return load_client_logits_artifact(prev) if prev.is_file() else None

    # ------------------------------------------------------------------
    # ラウンド処理
    # ------------------------------------------------------------------

    def run_round(
        self,
        *,
        server_round: int,
        fedavg_vec: np.ndarray,
        client_sample_counts: Mapping[int, int] | None = None,
    ) -> Dict[str, Any]:
        rep = self.rep
        if rep.consistency_mode == "none":
            if rep.server_distill_to_weights:
                return self._run_feddf_pure(server_round, fedavg_vec, client_sample_counts)
            return {
                "server_round": server_round,
                "repertoire": rep.name,
                "server_phase": "skipped_baseline",
            }
        return self._run_proposed(server_round, client_sample_counts)

    def _run_proposed(
        self,
        server_round: int,
        sample_counts: Mapping[int, int] | None,
    ) -> Dict[str, Any]:
        """提案系: FedAvg 後の重みを使わず、送信された適応情報のみで基盤 LoRA を更新。"""
        t0 = time.perf_counter()
        self._ensure_model()
        assert self._model is not None and self._losses is not None and self._opt is not None
        rep = self.rep
        model = self._model
        processor = self._processor
        dcfg = self.cfg["data"]
        server_side_client_lora = lora_params_for_adapter(model.backbone, "client")
        surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")

        teacher_map = self._collect_client_logits(server_round, sample_counts)
        grad_teacher = (
            self._collect_grad_teacher(server_round, sample_counts)
            if "grad_vectors" in rep.client_transmits
            else None
        )
        post_map = (
            self._collect_post_logits(server_round, sample_counts)
            if "post_logits" in rep.client_transmits
            else None
        )
        replay_map = self._load_replay(server_round)

        train_t0 = time.perf_counter()
        server_steps, loss_stats = train_server_consistency_transmitted(
            model=model,
            processor=processor,
            losses=self._losses,
            train_rows=self.server_train,
            dcfg=dcfg,
            frozen_lora=server_side_client_lora,
            surrogate_lora=surrogate_lora,
            opt=self._opt,
            mode=rep.consistency_mode,
            device=self.device,
            max_steps=self.max_server_steps,
            teacher_logits_map=teacher_map,
            grad_teacher=grad_teacher.to(self.device) if grad_teacher is not None else None,
            post_logits_map=post_map,
            replay_logits_map=replay_map,
            grad_match=rep.grad_match,
            replay_weight=self.replay_weight,
        )
        server_train_sec = time.perf_counter() - train_t0

        server_eval = evaluate_classifier(
            model,
            processor,
            self.server_eval,
            device=self.device,
            prompt=self.prompt,
            max_length=self.max_length,
            batch_size=self.bs,
            adapter="surrogate",
            dcfg=dcfg,
        )

        server_logits = export_client_logits(
            model,
            processor,
            self.server_train,
            dcfg,
            adapter="surrogate",
            prompt=self.prompt,
            max_length=self.max_length,
            device=self.device,
        )
        logits_out = server_logits_path(self.run_dir, server_round)
        write_logits_sidecar(logits_out, server_round, server_logits)

        foundation_path = self._save_foundation(server_round)

        return {
            "server_round": server_round,
            "repertoire": rep.name,
            "server_steps": server_steps,
            "server_train_sec": server_train_sec,
            "server_consistency_sec": time.perf_counter() - t0,
            "server_eval_accuracy": float(server_eval.get("accuracy", 0.0)),
            "server_eval_f1_macro": float(server_eval.get("f1_macro", 0.0)),
            "server_loss_total": float(loss_stats.get("loss_total", 0.0)),
            "server_loss_pred": float(loss_stats.get("loss_pred", 0.0)),
            "server_loss_grad": float(loss_stats.get("loss_grad", 0.0)),
            "server_loss_post": float(loss_stats.get("loss_post", 0.0)),
            "server_loss_replay": float(loss_stats.get("loss_replay", 0.0)),
            "used_replay": replay_map is not None,
            "server_logits_path": str(logits_out),
            "server_logits_bytes": sidecar_size_bytes(logits_out),
            "foundation_checkpoint": str(foundation_path),
            "hetero": self.model_ids_differ,
            "distill_pipeline": self.distill_pipeline,
        }

    def _run_feddf_pure(
        self,
        server_round: int,
        fedavg_vec: np.ndarray,
        sample_counts: Mapping[int, int] | None,
    ) -> Dict[str, Any]:
        """baseline_feddf_pure: FedAvg 後の重みへアンサンブル logits を蒸留し、配布パラメータを差し替える。"""
        t0 = time.perf_counter()
        self._ensure_model()
        assert self._model is not None
        model = self._model
        processor = self._processor
        dcfg = self.cfg["data"]

        vector_to_trainable_state(model, self.param_names, fedavg_vec)
        teacher_map = self._collect_client_logits(server_round, sample_counts)

        client_lora = lora_params_for_adapter(model.backbone, "client")
        stats = distill_adapter_from_logits(
            model,
            processor,
            self.server_train,
            dcfg,
            teacher_logits_map=teacher_map,
            adapter="client",
            trainable_params=list(model.classifier.parameters()) + client_lora,
            prompt=self.prompt,
            max_length=self.max_length,
            device=self.device,
            lr=self.distill_lr,
            max_steps=self.distill_steps,
            temperature=self.temperature,
        )
        model.backbone.set_adapter("client")
        _, new_vec = trainable_state_vector(model)

        return {
            "server_round": server_round,
            "repertoire": self.rep.name,
            "server_phase": "feddf_pure_distill",
            "server_consistency_sec": time.perf_counter() - t0,
            "distill_steps": stats.get("distill_steps"),
            "distill_kl_last": stats.get("distill_kl_last"),
            "new_global_vec": new_vec,
        }

    def _save_foundation(self, server_round: int) -> Path:
        assert self._model is not None
        art = self.cfg["artifacts"]
        ckpt_base = Path(str(art.get("checkpoints", "artifacts/checkpoints")))
        sub = str(self.cfg.get("step07", {}).get("checkpoint_subdir", "step07_fl"))
        ckpt_dir = ckpt_base / sub if (ckpt_base / sub).is_absolute() else self.root / ckpt_base / sub
        foundation_path = foundation_checkpoint_path(ckpt_dir, self.run_dir.name, server_round)
        foundation_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "classifier": self._model.classifier.state_dict(),
                "backbone_peft": self._model.backbone.state_dict(),
                "meta": {
                    "server_round": server_round,
                    "repertoire": self.rep.name,
                    "consistency_mode": self.rep.consistency_mode,
                    "server_optimizer": self.rep.server_optimizer,
                    "server_model_id": self.server_model_id,
                },
            },
            foundation_path,
        )
        return foundation_path
