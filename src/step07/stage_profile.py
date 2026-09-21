"""Step 7 段階プロファイル — 送信内容・teacher・整合モード・集約・配布を 1 か所で定義。

教授 ablation 段階（Stage 0〜4）に加え、FedOpt / FedACG 等の集約方式試行用に
stage_profile を増やせる。段階の増減はこの registry の追加・削除で行う。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional


@dataclass(frozen=True)
class StageProfile:
    """1 つの実験段階（レシピ）の宣言。"""

    name: str
    description: str
    # AdaptationLosses の --mode
    consistency_mode: str  # lpred | lpred_grad | lpred_post
    # Lpred の teacher: recompute=FedAvg 後の重みをサーバ上で再 forward / transmitted=FedMD 型 logits
    teacher_source: str  # recompute | transmitted
    # クライアント → サーバで送る補助情報（重みは常に FedAvg 集約の入力）
    client_transmits: FrozenSet[str]  # weights は暗黙、ここは logits | grad_vectors | post_logits
    # ③ クライアント LoRA + 分類ヘッドの集約（FedAvg 後の重みを作る方式）
    aggregation_method: str  # fedavg | fedopt | fedacg | fedomg
    # ⑤ 基盤 LoRA の再配布
    distribution_method: str  # copy | distill
    # サーバ整合の L_task 重み（0 = FedDF/CFeD 型サーバ蒸留）
    w_task: float = 0.0
    # 先行研究との対応（docs 用メモ）
    prior_work: str = ""


# --- 教授 ablation 段階（研究提案 Adaptation Consistency）---
STAGE_PROFILES: Dict[str, StageProfile] = {
    "stage0_lpred_recompute": StageProfile(
        name="stage0_lpred_recompute",
        description="Lpred のみ。teacher = FedAvg 後の重みをサーバ上で再 forward（3B/3B プロトタイプ）",
        consistency_mode="lpred",
        teacher_source="recompute",
        client_transmits=frozenset(),
        aggregation_method="fedavg",
        distribution_method="distill",
        prior_work="Lpred 形式は FedDF 型 KL；teacher 生成は簡略版（サーバ再 forward）",
    ),
    "stage1_lpred_transmitted": StageProfile(
        name="stage1_lpred_transmitted",
        description="Lpred + server_train 上の logits 送信（FedMD 型 public alignment + FedDF 型集約）",
        consistency_mode="lpred",
        teacher_source="transmitted",
        client_transmits=frozenset({"logits"}),
        aggregation_method="fedavg",
        distribution_method="distill",
        prior_work="FedMD [2] logits 通信 + FedDF [3] softmax 平均；サーバ側基盤 LoRA 更新は本研究",
    ),
    "stage2_lpred_lgrad": StageProfile(
        name="stage2_lpred_lgrad",
        description="Lpred + Lgrad。logits + LoRA 勾配ベクトル（または更新方向）を送信",
        consistency_mode="lpred_grad",
        teacher_source="transmitted",
        client_transmits=frozenset({"logits", "grad_vectors"}),
        aggregation_method="fedavg",
        distribution_method="distill",
        prior_work="Lgrad: 研究提案 + FedACG [5] / FedOMG [6] 背景",
    ),
    "stage3_lpred_lpost": StageProfile(
        name="stage3_lpred_lpost",
        description="Lpred + Lpost。logits + 1-step 適応後出力を送信（Step 6 継続学習と接続）",
        consistency_mode="lpred_post",
        teacher_source="transmitted",
        client_transmits=frozenset({"logits", "post_logits"}),
        aggregation_method="fedavg",
        distribution_method="distill",
        prior_work="Lpost: 研究提案 + CFeD [7] 背景",
    ),
    "stage4_full_ac": StageProfile(
        name="stage4_full_ac",
        description="Lpred + Lgrad + Lpost 全部",
        consistency_mode="lpred_post",  # AdaptationLosses: lpred_post に Lgrad 含む実装は要確認→現状 lpred_grad+post 未統合
        teacher_source="transmitted",
        client_transmits=frozenset({"logits", "grad_vectors", "post_logits"}),
        aggregation_method="fedavg",
        distribution_method="distill",
        prior_work="研究提案の全成分",
    ),
}

# --- 集約 ablation（③ のみ差し替え。整合・配布は stage1 ベース）---
for _agg, _label in (
    ("fedopt", "FedOpt / FedAdam [1] — サーバ側 adaptive pseudo-gradient 集約"),
    ("fedacg", "FedACG [5] — クライアント更新方向の加速集約"),
):
    STAGE_PROFILES[f"stage1_lpred_{_agg}"] = StageProfile(
        name=f"stage1_lpred_{_agg}",
        description=f"Stage 1 Lpred + 集約={_agg.upper()}（実装: {_agg} は未実装 stub）",
        consistency_mode="lpred",
        teacher_source="transmitted",
        client_transmits=frozenset({"logits"}),
        aggregation_method=_agg,
        distribution_method="distill",
        prior_work=_label,
    )


def list_stage_profiles() -> List[str]:
    return sorted(STAGE_PROFILES.keys())


def resolve_stage_profile(
    cfg: Dict[str, Any],
    *,
    cli_stage: int | None = None,
    cli_mode: str | None = None,
    cli_teacher_source: str | None = None,
    cli_aggregation: str | None = None,
    cli_distribution: str | None = None,
) -> StageProfile:
    """config step07.stage_profile 優先。未指定時は従来 CLI (--stage / --mode) から推定。"""
    scfg = cfg.get("step07", {})
    profile_name = scfg.get("stage_profile")
    if profile_name:
        key = str(profile_name).strip()
        if key not in STAGE_PROFILES:
            raise ValueError(
                f"Unknown step07.stage_profile={key!r}. "
                f"Available: {list_stage_profiles()}"
            )
        base = STAGE_PROFILES[key]
    else:
        stage = int(cli_stage if cli_stage is not None else 0)
        mode = str(cli_mode or "lpred")
        if stage == 0 and mode == "lpred":
            base = STAGE_PROFILES["stage0_lpred_recompute"]
        elif stage == 1 and mode == "lpred":
            base = STAGE_PROFILES["stage1_lpred_transmitted"]
        elif mode == "lpred_grad":
            base = STAGE_PROFILES["stage2_lpred_lgrad"]
        elif mode == "lpred_post":
            base = STAGE_PROFILES["stage3_lpred_lpost"]
        else:
            raise ValueError(f"Cannot infer stage profile from stage={stage} mode={mode}")

    # CLI / config で個別上書き（プロファイルをベースに patch）
    agg = cli_aggregation or scfg.get("aggregation", {}).get("method") or base.aggregation_method
    dist = cli_distribution or scfg.get("distribution") or base.distribution_method
    teacher = cli_teacher_source or base.teacher_source

    if agg != base.aggregation_method or dist != base.distribution_method or teacher != base.teacher_source:
        return StageProfile(
            name=f"{base.name}_override",
            description=base.description,
            consistency_mode=base.consistency_mode,
            teacher_source=teacher,
            client_transmits=base.client_transmits,
            aggregation_method=str(agg),
            distribution_method=str(dist),
            w_task=base.w_task,
            prior_work=base.prior_work,
        )
    return base


def validate_profile_for_hetero(profile: StageProfile, distill_pipeline: bool) -> None:
    """distill パイプライン（7B/3B または step07.distill_pipeline=true）で禁止する経路を検査。

    統一ループ: ③ FedAvg → ④ 整合（送信 logits 等で基盤 LoRA 更新）→ ⑤ distill 配布。
    FedAvg 後の重みをサーバモデルへ載せる recompute teacher / copy 配布は形状非依存の経路のみ。
    """
    if not distill_pipeline:
        return
    if profile.distribution_method != "distill":
        raise SystemExit(
            "distill パイプラインでは distribution=distill 必須（LoRA 重み copy 不可）。"
            f"現在: {profile.distribution_method}"
        )
    if profile.teacher_source != "transmitted":
        raise SystemExit(
            "distill パイプラインでは teacher=transmitted（logits 送信）必須。"
            "stage_profile=stage1_lpred_transmitted または --stage 1 を使う。"
        )
    if "logits" not in profile.client_transmits:
        raise SystemExit(
            "distill パイプラインでは client_transmits に logits が必要。"
            f"現在: {sorted(profile.client_transmits)}"
        )
    if profile.consistency_mode != "lpred":
        raise SystemExit(
            "distill パイプラインでは Lgrad/Lpost（勾配・適応後 logits の形状依存）は未対応。"
            "consistency_mode=lpred のみ。GCP 7B/3B と同条件に揃える。"
        )
