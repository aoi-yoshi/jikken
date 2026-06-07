"""Flower 戦略ファクトリ（default.yaml の fl.strategy から生成）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Type

import numpy as np

from .logging_utils import RunLogger


def fl_strategy_name(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("fl", {}).get("strategy", "FedAvg")).strip()


def is_fedprox(cfg: Dict[str, Any]) -> bool:
    return fl_strategy_name(cfg).lower().replace("_", "") == "fedprox"


def build_flower_strategy_class(cfg: Dict[str, Any]) -> Type:
    """SavingFedAvg / SavingFedProx のいずれかを返す。"""
    if is_fedprox(cfg):
        from flwr.server.strategy import FedProx

        base = FedProx
        label = "FedProx"
    else:
        from flwr.server.strategy import FedAvg

        base = FedAvg
        label = "FedAvg"

    class SavingStrategy(base):
        def __init__(
            self,
            *,
            total_rounds: int,
            checkpoint_path: Path,
            names: List[str],
            init_vec: np.ndarray,
            logger: RunLogger,
            run_id: str,
            partition_mode: str,
            flower_settings: Dict[str, Any],
            strategy_label: str,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.total_rounds = total_rounds
            self.checkpoint_path = checkpoint_path
            self.param_names = names
            self.init_vec = init_vec
            self.logger = logger
            self.run_id = run_id
            self.partition_mode = partition_mode
            self.flower_settings = flower_settings
            self.strategy_label = strategy_label

        def aggregate_fit(self, server_round, results, failures):
            import time

            from flwr.common import parameters_to_ndarrays

            from .fl_utils import save_fl_checkpoint

            t0 = time.perf_counter()
            aggregated, metrics = super().aggregate_fit(server_round, results, failures)
            round_metrics: Dict[str, Any] = {
                "event": "server_round_aggregate",
                "server_round": server_round,
                "num_clients": len(results),
                "fit_metrics": dict(metrics) if metrics else {},
                "server_aggregate_duration_sec": time.perf_counter() - t0,
            }
            metric_keys = (
                "post_redistribution_accuracy",
                "local_adaptation_accuracy",
                "foundation_generalization_accuracy",
                "weight_divergence_trainable",
                "weight_divergence_lora",
                "round_duration_sec",
                "apply_global_params_sec",
                "post_redistribution_eval_sec",
                "fit_duration_sec",
                "local_adaptation_eval_sec",
                "foundation_generalization_eval_sec",
                "weight_divergence_sec",
                "round_overhead_sec",
            )
            if metrics:
                for key in metric_keys:
                    if key in metrics:
                        round_metrics[f"agg_{key}"] = metrics[key]
            if aggregated is not None:
                params = aggregated.parameters if hasattr(aggregated, "parameters") else aggregated
                ndarrays = parameters_to_ndarrays(params)
                vec = ndarrays[0] if ndarrays else self.init_vec
                round_metrics["param_norm"] = float(np.linalg.norm(vec))
                if server_round >= self.total_rounds:
                    save_fl_checkpoint(
                        self.checkpoint_path,
                        names=self.param_names,
                        vector=vec,
                        meta={
                            "run_id": self.run_id,
                            "num_rounds": self.total_rounds,
                            "strategy": self.strategy_label,
                            "partition_mode": self.partition_mode,
                            "flower_settings": self.flower_settings,
                            "trainable_vector_dim": int(vec.size),
                        },
                    )
                    latest = self.checkpoint_path.parent.parent / "LATEST_CHECKPOINT"
                    latest.parent.mkdir(parents=True, exist_ok=True)
                    latest.write_text(str(self.checkpoint_path.resolve()), encoding="utf-8")
                    round_metrics["global_checkpoint"] = str(self.checkpoint_path)
            self.logger.log(round_metrics)
            return aggregated, metrics

        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            eval_row: Dict[str, Any] = {
                "event": "server_evaluate_aggregate",
                "server_round": server_round,
                "eval_loss": float(loss) if loss is not None else None,
                "eval_metrics": dict(metrics) if metrics else {},
                "num_clients": len(results),
            }
            self.logger.log(eval_row)
            return loss, metrics

    SavingStrategy.__name__ = f"Saving{label}"
    return SavingStrategy
