"""Step 7 Flower 戦略 — レパートリー分岐。

統一ループ（distill パイプライン）:
  ③ FedAvg → ④ サーバ整合（基盤 LoRA 更新）→ ⑤ server logits → 次ラウンド distill
提案系: apply_global_weights=0（FedAvg 後の重みはサーバ整合に使わない）
baseline 系: apply_global_weights=1（端末が FedAvg 後の重みを適用。7B/3B ではクライアント形状一致時のみ）
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Type

import numpy as np

from src.fl_step07.server_phase import Step07ServerContext
from src.fl_utils import flower_parameters_to_vector, save_fl_checkpoint, write_latest_checkpoint_pointer
from src.logging_utils import RunLogger
from src.step07.repertoire import Repertoire, repertoire_record


def build_step07_flower_strategy_class(cfg: Dict[str, Any]) -> Type:
    from flwr.server.strategy import FedAvg

    scfg = cfg.get("step07", {})
    distill_steps = scfg.get("distill_steps", 20)
    distill_lr = float(scfg.get("distill_lr", cfg["train"]["lr"]))

    class Step07Strategy(FedAvg):
        def __init__(
            self,
            *,
            total_rounds: int,
            checkpoint_path: Path,
            names: List[str],
            init_vec: np.ndarray,
            logger: RunLogger,
            run_id: str,
            run_dir: Path,
            root: Path,
            repertoire: Repertoire,
            server_ctx: Step07ServerContext,
            flower_settings: Dict[str, Any],
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.total_rounds = total_rounds
            self.checkpoint_path = checkpoint_path
            self.param_names = names
            self.init_vec = init_vec
            self.logger = logger
            self.run_id = run_id
            self.run_dir = run_dir
            self.root = root
            self.rep = repertoire
            self.server_ctx = server_ctx
            self.flower_settings = flower_settings
            self._distill_steps = int(distill_steps) if distill_steps is not None else 0
            self._distill_lr = distill_lr

        def configure_fit(self, server_round, parameters, client_manager):
            from flwr.common import FitIns

            pairs = super().configure_fit(server_round, parameters, client_manager)
            rep = self.rep
            fit_config: Dict[str, Any] = {
                "server_round": server_round,
                "repertoire": rep.name,
                "consistency_mode": rep.consistency_mode,
                "distribution": rep.distribution,
                "apply_global_weights": int(rep.uses_client_weights),
                "export_logits": int("logits" in rep.client_transmits),
                "export_grads": int("grad_vectors" in rep.client_transmits),
                "export_post_logits": int("post_logits" in rep.client_transmits),
                "prox_mu": float(rep.client_prox_mu or 0.0),
                "distill_lr": self._distill_lr,
                "distill_steps": self._distill_steps,
            }
            if rep.distribution == "distill" and server_round > 1:
                prev = self.run_dir / "sidecar" / "server_logits" / f"r{server_round - 1}.json"
                if prev.is_file():
                    fit_config["server_logits_path"] = str(prev)
            out = []
            for client, fit_ins in pairs:
                merged = dict(fit_ins.config)
                merged.update(fit_config)
                out.append((client, FitIns(fit_ins.parameters, merged)))
            return out

        def configure_evaluate(self, server_round, parameters, client_manager):
            from flwr.common import EvaluateIns

            pairs = super().configure_evaluate(server_round, parameters, client_manager)
            eval_config = {
                "server_round": server_round,
                "apply_global_weights": int(self.rep.uses_client_weights),
            }
            out = []
            for client, eval_ins in pairs:
                merged = dict(eval_ins.config)
                merged.update(eval_config)
                out.append((client, EvaluateIns(eval_ins.parameters, merged)))
            return out

        def aggregate_fit(self, server_round, results, failures):
            from flwr.common import ndarrays_to_parameters

            from src.fl_step07.sidecar import server_logits_path

            t0 = time.perf_counter()
            aggregated, metrics = super().aggregate_fit(server_round, results, failures)
            round_metrics: Dict[str, Any] = {
                "event": "step07_server_aggregate",
                "server_round": server_round,
                "repertoire": self.rep.name,
                "num_clients": len(results),
                "fit_metrics": dict(metrics) if metrics else {},
            }
            sample_counts: Dict[int, int] = {}
            for _, fit_res in results:
                try:
                    cid = int(fit_res.metrics.get("cid"))
                except (TypeError, ValueError):
                    continue
                sample_counts[cid] = int(fit_res.num_examples)

            if aggregated is not None:
                try:
                    fedavg_vec = flower_parameters_to_vector(
                        aggregated, expected_size=int(self.init_vec.size)
                    )
                except ValueError as exc:
                    round_metrics["checkpoint_error"] = str(exc)
                    fedavg_vec = self.init_vec
                else:
                    server_info = self.server_ctx.run_round(
                        server_round=server_round,
                        fedavg_vec=fedavg_vec,
                        client_sample_counts=sample_counts,
                    )
                    new_vec = server_info.pop("new_global_vec", None)
                    round_metrics.update(server_info)
                    if new_vec is not None:
                        aggregated = ndarrays_to_parameters([np.asarray(new_vec, dtype=np.float32)])
                        fedavg_vec = np.asarray(new_vec, dtype=np.float32)

                round_metrics["param_norm"] = float(np.linalg.norm(fedavg_vec))
                round_metrics["weights_used_by_server"] = False
                round_metrics["clients_apply_fedavg_weights"] = bool(self.rep.uses_client_weights)
                if server_round >= self.total_rounds:
                    save_fl_checkpoint(
                        self.checkpoint_path,
                        names=self.param_names,
                        vector=fedavg_vec,
                        meta={
                            "run_id": self.run_id,
                            "num_rounds": self.total_rounds,
                            "scheme": "step07_flower",
                            "repertoire": repertoire_record(self.rep),
                            "weights_used_by_server": False,
                            "clients_apply_fedavg_weights": bool(self.rep.uses_client_weights),
                            "server_logits_final": str(server_logits_path(self.run_dir, server_round)),
                            "flower_settings": self.flower_settings,
                        },
                    )
                    write_latest_checkpoint_pointer(self.checkpoint_path, self.server_ctx.cfg, self.root)
                    round_metrics["fedavg_checkpoint"] = str(self.checkpoint_path)

            round_metrics["server_aggregate_duration_sec"] = time.perf_counter() - t0
            self.logger.log(round_metrics)
            return aggregated, metrics

    Step07Strategy.__name__ = "Step07Strategy"
    return Step07Strategy
