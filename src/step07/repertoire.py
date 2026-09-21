"""Step 7 レパートリー registry — 「surrogate 適応情報を用いた基盤モデル更新」の方式群。

設計思想（2026-07-05 教授方針）:
  提案手法の系列では FedAvg（クライアント重みの平均）を使わない。
  クライアントが送る適応情報（logits / 勾配ベクトル / 1-step 適応後 logits）だけで
  基盤 LoRA を更新し、logits 蒸留で還元する。
  FedAvg / FedProx は比較手法（baseline_*）としてのみ登場する。

5 軸:
  1. client_transmits   — 送信内容（logits / grad_vectors / post_logits / weights）
  2. teacher 構成       — logits_aggregation（equal / weighted）+ use_replay
  3. consistency_mode   — サーバ損失（lpred / lpred_grad / lpred_post / full / none）
  4. server_optimizer   — 基盤 LoRA 更新の optimizer（adamw / fedadam / momentum）
  5. distribution       — 配布（distill / weights）

切替: config `step07.repertoire` または CLI `--repertoire`。
一覧・文献対応: docs/step07_repertoires.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Tuple


@dataclass(frozen=True)
class Repertoire:
    """1 レパートリー（基盤モデル更新方式）の宣言。"""

    name: str
    kind: str  # proposed | baseline
    papers: Tuple[str, ...]
    description: str
    # 1. クライアント → サーバの送信内容。
    #    提案系は適応情報のみ（weights は Flower プロトコル上流れるが、サーバは使わない）。
    client_transmits: FrozenSet[str]  # logits | grad_vectors | post_logits | weights
    # 3. サーバ損失（基盤 LoRA 更新）。none = サーバ整合なし（baseline）
    consistency_mode: str  # none | lpred | lpred_grad | lpred_post | full
    # 2. teacher 構成
    logits_aggregation: str = "weighted"  # equal（FedMD 型）| weighted（FedDF 型サンプル数重み）
    use_replay: bool = False  # 前ラウンド server logits を replay teacher に追加（FCL 型忘却抑制）
    grad_match: str = "mse"  # mse | cosine（Lgrad 時のみ）
    # 4. 基盤 LoRA 更新の optimizer
    server_optimizer: str = "adamw"  # adamw | fedadam | momentum
    # 5. 配布
    distribution: str = "distill"  # distill | weights
    # サーバが FedAvg 後の重みを使うか（提案系は False）
    uses_client_weights: bool = False
    # クライアント側 proximal 項（baseline_fedprox 用）。None = なし
    client_prox_mu: float | None = None
    # baseline_feddf_pure: FedAvg 後の重みへサーバ蒸留（基盤 LoRA は更新しない）
    server_distill_to_weights: bool = False
    # 異種モデル（7B/3B）で実行可能か。勾配系・重み配布系は形状依存で不可
    hetero_ok: bool = True
    w_task: float = 0.0
    replay_weight: float = 0.5
    # 先行研究に合わせた設定のメモ（温度・集約・optimizer 係数など）
    prior_settings: str = ""
    # 実行可能か（fedgen は合成整合セット未整備のため定義のみ）
    enabled: bool = True


# 文献番号は docs/step07_design.md §関連研究 に対応:
# [1] FedOpt  [2] FedMD  [3] FedDF  [4] FedGen  [5] FedACG
# [6] FedOMG  [7] CFeD   [8] FCL Survey 2026    [9] FCL Survey 2025

REPERTOIRES: Dict[str, Repertoire] = {}


def _register(rep: Repertoire) -> None:
    REPERTOIRES[rep.name] = rep


# --- 提案手法系（基盤 LoRA をサーバで更新 → distill 配布）---

_register(Repertoire(
    name="ac_lpred_md",
    kind="proposed",
    papers=("FedMD [2]",),
    description="クライアント logits（server_train 上）を等平均 → Lpred で基盤 LoRA 更新",
    client_transmits=frozenset({"logits"}),
    consistency_mode="lpred",
    logits_aggregation="equal",
    prior_settings="FedMD: public set 上の logits を単純平均。蒸留は KL×T²（T=2.0）",
))

_register(Repertoire(
    name="ac_lpred_df",
    kind="proposed",
    papers=("FedDF [3]",),
    description="クライアント logits をサンプル数重み付き softmax 平均（アンサンブル teacher）→ Lpred",
    client_transmits=frozenset({"logits"}),
    consistency_mode="lpred",
    logits_aggregation="weighted",
    prior_settings="FedDF: softmax 確率の重み付き平均をアンサンブル teacher に。KL×T²（T=2.0）",
))

_register(Repertoire(
    name="ac_lpred_replay",
    kind="proposed",
    papers=("FedDF [3]", "FCL Survey [8]", "FCL Survey [9]"),
    description="Lpred + 前ラウンド server logits を replay teacher に追加（忘却抑制）",
    client_transmits=frozenset({"logits"}),
    consistency_mode="lpred",
    logits_aggregation="weighted",
    use_replay=True,
    prior_settings="FCL の knowledge replay を server logits 履歴で近似。replay_weight=0.5",
))

_register(Repertoire(
    name="ac_lgrad_tx",
    kind="proposed",
    papers=("研究提案", "FedACG [5]"),
    description="logits + 勾配ベクトル送信 → Lpred + Lgrad（MSE 整合）",
    client_transmits=frozenset({"logits", "grad_vectors"}),
    consistency_mode="lpred_grad",
    grad_match="mse",
    hetero_ok=False,
    prior_settings="Lgrad: クライアント L_task 勾配と基盤 LoRA 勾配の MSE。LoRA 形状一致が前提（3B/3B）",
))

_register(Repertoire(
    name="ac_lgrad_cos",
    kind="proposed",
    papers=("FedOMG [6]",),
    description="logits + 勾配ベクトル送信 → コサイン類似度による勾配方向マッチング",
    client_transmits=frozenset({"logits", "grad_vectors"}),
    consistency_mode="lpred_grad",
    grad_match="cosine",
    hetero_ok=False,
    prior_settings="FedOMG: 更新方向の整合（大きさでなく方向）。l_grad = 1 - cos(g_c, g_s)",
))

_register(Repertoire(
    name="ac_lpost_cfed",
    kind="proposed",
    papers=("研究提案", "CFeD [7]"),
    description="logits + 1-step 適応後 logits 送信 → Lpred + Lpost",
    client_transmits=frozenset({"logits", "post_logits"}),
    consistency_mode="lpred_post",
    hetero_ok=True,
    prior_settings="CFeD: 適応後の出力分布を teacher に KL。inner_lr は consistency.inner_lr",
))

_register(Repertoire(
    name="ac_lpost_replay",
    kind="proposed",
    papers=("CFeD [7]", "FCL Survey [8]", "FCL Survey [9]"),
    description="Lpred + Lpost + replay teacher の併用（継続学習向け）",
    client_transmits=frozenset({"logits", "post_logits"}),
    consistency_mode="lpred_post",
    use_replay=True,
    hetero_ok=True,
    prior_settings="CFeD の Server Distillation + FCL replay。Step 6 継続学習設定と接続",
))

_register(Repertoire(
    name="ac_full",
    kind="proposed",
    papers=("研究提案", "FedMD [2]", "FedDF [3]", "FedACG [5]", "FedOMG [6]", "CFeD [7]"),
    description="研究提案全成分。Lpred + Lgrad + Lpost（すべて transmitted）",
    client_transmits=frozenset({"logits", "grad_vectors", "post_logits"}),
    consistency_mode="full",
    hetero_ok=False,
    prior_settings="L = λ₁Lpred + λ₃Lgrad + λ₄Lpost（研究提案の最終目的関数、Lrepr 除く）",
))

_register(Repertoire(
    name="ac_lpred_fedopt",
    kind="proposed",
    papers=("FedOpt [1]", "FedDF [3]"),
    description="Lpred は ac_lpred_df と同じ。基盤 LoRA 更新の optimizer を FedAdam 型 adaptive に差し替え",
    client_transmits=frozenset({"logits"}),
    consistency_mode="lpred",
    server_optimizer="fedadam",
    prior_settings="FedOpt (FedAdam): β1=0.9, β2=0.99, τ(eps)=1e-3。AdamW との optimizer ablation",
))

_register(Repertoire(
    name="ac_lpred_acg_momentum",
    kind="proposed",
    papers=("FedACG [5]", "FedDF [3]"),
    description="Lpred は ac_lpred_df と同じ。基盤 LoRA の更新に SGD + Nesterov momentum（lookahead 型）",
    client_transmits=frozenset({"logits"}),
    consistency_mode="lpred",
    server_optimizer="momentum",
    prior_settings="FedACG: momentum λ=0.85 の加速集約をサーバ optimizer として再解釈（SGD+Nesterov, momentum=0.85）",
))

# --- FedGen [4] は合成整合セットが必要（data/synthetic 未整備）→ 定義のみ ---

_register(Repertoire(
    name="ac_lpred_gen",
    kind="proposed",
    papers=("FedGen [4]",),
    description="整合セットを実データでなく合成データにする変種（data/synthetic に整合セットを用意したら有効化）",
    client_transmits=frozenset({"logits"}),
    consistency_mode="lpred",
    prior_settings="FedGen: 生成器による疑似データで logits 通信を補完。合成整合セットの生成手順が前提",
    enabled=False,
))

# --- 比較手法系（baseline。提案には含めない）---
# distill パイプライン統一: ③ FedAvg → ④ logits 整合で基盤 LoRA 更新 → ⑤ distill 配布。
# 提案系との差: uses_client_weights=True（端末が FedAvg 後の重みを適用。7B/3B では形状一致時のみ）。

_register(Repertoire(
    name="baseline_fedavg",
    kind="baseline",
    papers=("FedAvg（McMahan 等）",),
    description="FedAvg 後の重みを端末へ適用 + logits 等平均で基盤 LoRA 整合 + distill 配布",
    client_transmits=frozenset({"logits", "weights"}),
    consistency_mode="lpred",
    logits_aggregation="equal",
    distribution="distill",
    uses_client_weights=True,
    hetero_ok=True,
    prior_settings="FedAvg: 重み平均を端末へ配布（形状一致時）。整合は FedMD 型等平均 logits → 基盤 LoRA",
))

_register(Repertoire(
    name="baseline_fedprox",
    kind="baseline",
    papers=("FedProx",),
    description="baseline_fedavg + クライアント側 proximal 項（μ=0.01）",
    client_transmits=frozenset({"logits", "weights"}),
    consistency_mode="lpred",
    logits_aggregation="equal",
    distribution="distill",
    uses_client_weights=True,
    client_prox_mu=0.01,
    hetero_ok=True,
    prior_settings="FedProx: μ=0.01（fl.fedprox_mu と同値）+ 統一 distill ループ",
))

_register(Repertoire(
    name="baseline_feddf_pure",
    kind="baseline",
    papers=("FedDF [3]",),
    description="FedAvg + FedDF 型重み付き ensemble logits で基盤 LoRA 整合 + distill（端末は FedAvg 重みも適用）",
    client_transmits=frozenset({"logits", "weights"}),
    consistency_mode="lpred",
    logits_aggregation="weighted",
    distribution="distill",
    uses_client_weights=True,
    hetero_ok=True,
    prior_settings="FedDF: softmax 重み付き平均 teacher。提案 ac_lpred_df との差は端末への FedAvg 重み適用の有無",
))

_register(Repertoire(
    name="baseline_fedopt",
    kind="baseline",
    papers=("FedOpt [1]", "FedDF [3]"),
    description="baseline_feddf と同型だが基盤 LoRA 更新 optimizer を FedAdam（β1=0.9, β2=0.99, ε=1e-3）",
    client_transmits=frozenset({"logits", "weights"}),
    consistency_mode="lpred",
    logits_aggregation="weighted",
    distribution="distill",
    uses_client_weights=True,
    server_optimizer="fedadam",
    hetero_ok=True,
    prior_settings="FedOpt (FedAdam) 係数をサーバ整合 optimizer に適用。重み集約は FedAvg、配布は distill",
))


def list_repertoires(*, kind: str | None = None, enabled_only: bool = True) -> List[str]:
    names = []
    for name, rep in sorted(REPERTOIRES.items()):
        if kind is not None and rep.kind != kind:
            continue
        if enabled_only and not rep.enabled:
            continue
        names.append(name)
    return names


def resolve_repertoire(cfg: Dict[str, Any], cli_repertoire: str | None = None) -> Repertoire:
    """CLI --repertoire 優先、次に config step07.repertoire。"""
    scfg = cfg.get("step07", {})
    key = str(cli_repertoire or scfg.get("repertoire") or "ac_lpred_df").strip()
    if key not in REPERTOIRES:
        raise ValueError(
            f"Unknown repertoire={key!r}. Available: {list_repertoires(enabled_only=False)}"
        )
    rep = REPERTOIRES[key]
    if not rep.enabled:
        raise SystemExit(
            f"repertoire={key} は定義のみ（enabled=False）。{rep.prior_settings}"
        )
    return rep


def validate_repertoire_for_hetero(rep: Repertoire, distill_pipeline: bool) -> None:
    """distill パイプライン（7B/3B または step07.distill_pipeline=true）で実行可能なレパートリーのみ許可。

    統一ループ: FedAvg → 基盤 LoRA 整合更新（適応情報）→ distill 再配布。
    勾配送信・重み配布のみ（整合なし）は不可。
    """
    if not distill_pipeline:
        return
    if not rep.hetero_ok:
        raise SystemExit(
            f"repertoire={rep.name} は distill パイプラインでは実行不可: "
            f"{'勾配ベクトルの形状一致' if 'grad_vectors' in rep.client_transmits else '重み配布の形状一致'}が前提。"
            f"使えるのは hetero_ok=True の logits 系（例: ac_lpred_df, ac_lpred_md）。"
        )
    if rep.distribution != "distill":
        raise SystemExit(
            f"repertoire={rep.name} は distill パイプラインでは distribution=distill 必須。"
            f"現在: {rep.distribution}"
        )
    if rep.consistency_mode == "none":
        raise SystemExit(
            f"repertoire={rep.name} は distill パイプラインではサーバ整合（基盤 LoRA 更新）必須。"
        )
    if "logits" not in rep.client_transmits:
        raise SystemExit(
            f"repertoire={rep.name} は distill パイプラインでは logits 送信必須。"
        )


def repertoire_record(rep: Repertoire) -> Dict[str, Any]:
    """run_meta / summary / config_snapshot に埋める記録ブロック。"""
    return {
        "name": rep.name,
        "kind": rep.kind,
        "papers": list(rep.papers),
        "description": rep.description,
        "client_transmits": sorted(rep.client_transmits),
        "consistency_mode": rep.consistency_mode,
        "logits_aggregation": rep.logits_aggregation,
        "use_replay": rep.use_replay,
        "replay_weight": rep.replay_weight,
        "grad_match": rep.grad_match,
        "server_optimizer": rep.server_optimizer,
        "distribution": rep.distribution,
        "uses_client_weights": rep.uses_client_weights,
        "client_prox_mu": rep.client_prox_mu,
        "server_distill_to_weights": rep.server_distill_to_weights,
        "hetero_ok": rep.hetero_ok,
        "w_task": rep.w_task,
        "prior_settings": rep.prior_settings,
    }


# --- stage_profile 互換（旧 CLI / config を新 registry に写像）---

_STAGE_PROFILE_ALIASES: Dict[str, str] = {
    "stage1_lpred_transmitted": "ac_lpred_df",
    "stage2_lpred_lgrad": "ac_lgrad_tx",
    "stage3_lpred_lpost": "ac_lpost_cfed",
    "stage4_full_ac": "ac_full",
    "stage1_lpred_fedopt": "ac_lpred_fedopt",
    "stage1_lpred_fedacg": "ac_lpred_acg_momentum",
}


def repertoire_from_stage_profile(profile_name: str) -> Repertoire | None:
    key = _STAGE_PROFILE_ALIASES.get(str(profile_name).strip())
    return REPERTOIRES[key] if key else None
