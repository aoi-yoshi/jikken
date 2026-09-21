"""Step 7 ラウンドループの共通モジュール（段階プロファイル・集約・配布）。

入口スクリプト: scripts/step07_round_loop.py
設計: docs/step07_design.md §アーキテクチャ
"""

from .aggregation import aggregate_client_states, list_aggregation_methods
from .client_state import fedavg_client_states, load_client_state, snapshot_client_state
from .distribution import distribute_foundation_to_clients
from .repertoire import (
    Repertoire,
    list_repertoires,
    repertoire_record,
    resolve_repertoire,
    validate_repertoire_for_hetero,
)
from .pipeline import distill_pipeline_mode, model_ids_differ, resolve_step07_model_ids
from .stage_profile import StageProfile, list_stage_profiles, resolve_stage_profile, validate_profile_for_hetero

__all__ = [
    "distill_pipeline_mode",
    "model_ids_differ",
    "resolve_step07_model_ids",
    "Repertoire",
    "StageProfile",
    "aggregate_client_states",
    "distribute_foundation_to_clients",
    "fedavg_client_states",
    "list_aggregation_methods",
    "list_repertoires",
    "list_stage_profiles",
    "load_client_state",
    "repertoire_record",
    "resolve_repertoire",
    "resolve_stage_profile",
    "snapshot_client_state",
    "validate_profile_for_hetero",
    "validate_repertoire_for_hetero",
]
