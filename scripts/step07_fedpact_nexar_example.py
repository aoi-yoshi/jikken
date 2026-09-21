"""NEXAR の具体例で private adaptation → Layer 2 → proxy pair を追跡する。

これは Step I-B の縦切り実装であり、性能比較を目的としない。balanced モードでは
risk / normal の両方で局所学習し、未使用 client_eval でも出力前後差を診断する。
単一 client のため、この run 内の連合集約後 global surrogate は局所更新後 state と一致する。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.config_loader import load_yaml, merged_config
from src.dataset_manifest import load_sample_row, read_manifest_filtered
from src.fl_utils import load_fl_checkpoint, vector_to_trainable_state
from src.metrics import set_seed
from src.peft_setup import attach_dual_lora, lora_params_for_adapter, sync_surrogate_from_client
from src.step07.client_state import load_client_state, snapshot_client_state
from src.step07.data_splits import save_data_splits_artifact, split_step07_data_pools
from src.step07.fedpact_layer2 import (
    assert_private_data_absent,
    build_classwise_prototypes,
    build_server_payload,
    proxy_match_metrics,
    rank_server_seeds,
    select_best_proxy,
    softmax_vector,
    transform_frames,
    transformation_candidates,
)
from src.step07.train_steps import train_client_l_task
from src.train_common import device_or_auto
from src.vl_model import build_model, unfreeze_backbone


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _flat_parameter_vector(parameters: Sequence[torch.nn.Parameter]) -> np.ndarray:
    tensors = [parameter.detach().cpu().float().reshape(-1) for parameter in parameters]
    if not tensors:
        return np.zeros((0,), dtype=np.float32)
    return torch.cat(tensors).numpy()


def _named_parameter_vector(
    model: torch.nn.Module,
    parameters: Sequence[torch.nn.Parameter],
) -> Tuple[List[str], np.ndarray]:
    """requires_gradの一時変化に依存せず、指定parameterだけを固定順で直列化する。"""
    selected_ids = {id(parameter) for parameter in parameters}
    named = [(name, parameter) for name, parameter in model.named_parameters() if id(parameter) in selected_ids]
    names = [name for name, _ in named]
    if len(names) != len(parameters):
        raise RuntimeError(
            f"could not resolve every selected parameter name: expected={len(parameters)}, resolved={len(names)}"
        )
    return names, _flat_parameter_vector([parameter for _, parameter in named])


def _direction_diagnostics(records: Sequence[Mapping[str, Any]], epsilon: float) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        label = int(record["label"])
        pre = float(record["pre_probability"][1])
        post = float(record["post_probability"][1])
        delta = post - pre
        desired = delta < -epsilon if label == 0 else delta > epsilon
        rows.append(
            {
                "sample_id": str(record["sample_id"]),
                "label": label,
                "delta_q_risk": delta,
                "desired_direction_with_margin": desired,
            }
        )

    by_class: Dict[str, Any] = {}
    for label in (0, 1):
        selected = [row for row in rows if int(row["label"]) == label]
        source = [record for record in records if int(record["label"]) == label]
        mean_pre = float(np.mean([float(record["pre_probability"][1]) for record in source]))
        mean_post = float(np.mean([float(record["post_probability"][1]) for record in source]))
        by_class[str(label)] = {
            "count": len(selected),
            "mean_q_risk_pre": mean_pre,
            "mean_q_risk_post": mean_post,
            "mean_delta_q_risk": mean_post - mean_pre,
            "desired_direction_count": sum(bool(row["desired_direction_with_margin"]) for row in selected),
        }

    normal = by_class["0"]
    risk = by_class["1"]
    return {
        "epsilon": float(epsilon),
        "classwise": by_class,
        "common_direction_shift": float(np.mean([row["delta_q_risk"] for row in rows])),
        "class_gap_pre": float(risk["mean_q_risk_pre"] - normal["mean_q_risk_pre"]),
        "class_gap_post": float(risk["mean_q_risk_post"] - normal["mean_q_risk_post"]),
        "class_mean_direction_success": bool(
            normal["mean_delta_q_risk"] < -epsilon and risk["mean_delta_q_risk"] > epsilon
        ),
        "at_least_two_of_three_per_class_success": bool(
            normal["desired_direction_count"] >= min(2, normal["count"])
            and risk["desired_direction_count"] >= min(2, risk["count"])
        ),
        "all_samples_success": all(bool(row["desired_direction_with_margin"]) for row in rows),
        "pre_correct_count": sum(
            (
                (int(record["label"]) == 1 and float(record["pre_probability"][1]) >= 0.5)
                or (int(record["label"]) == 0 and float(record["pre_probability"][1]) < 0.5)
            )
            for record in records
        ),
        "post_correct_count": sum(
            (
                (int(record["label"]) == 1 and float(record["post_probability"][1]) >= 0.5)
                or (int(record["label"]) == 0 and float(record["post_probability"][1]) < 0.5)
            )
            for record in records
        ),
        "prediction_flip_count": sum(
            (float(record["pre_probability"][1]) >= 0.5)
            != (float(record["post_probability"][1]) >= 0.5)
            for record in records
        ),
        "samples": rows,
    }


def _resolve_checkpoint(explicit: str | None) -> Path | None:
    if explicit:
        if explicit.strip().lower() == "none":
            return None
        path = Path(explicit)
        if not path.is_absolute():
            path = _ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    candidates = list((_ROOT / "artifacts" / "checkpoints" / "step07_fl").glob("run_*/step07_fedavg_reference.pt"))
    candidates.extend((_ROOT / "artifacts" / "checkpoints" / "step05_fl").glob("run_*/step05_fedavg_global.pt"))
    if not candidates:
        legacy = _ROOT / "artifacts" / "checkpoints" / "step05_fedavg_global.pt"
        return legacy if legacy.is_file() else None
    return max(candidates, key=lambda path: path.stat().st_mtime)


@torch.no_grad()
def _predict_rows(model, processor, rows: Sequence[dict], dcfg: dict, prompt: str, max_length: int) -> List[List[float]]:
    model.eval()
    model.backbone.set_adapter("client")
    outputs: List[List[float]] = []
    for row in rows:
        sample = load_sample_row(row, dcfg)
        logits, _ = model.forward_batch(processor, sample.frames, prompt, max_length, output_hidden_states=True)
        outputs.append(logits[0].detach().float().cpu().tolist())
        for frame in sample.frames:
            frame.close()
    return outputs


@torch.no_grad()
def _predict_frame_sets(
    model,
    processor,
    frame_sets: Sequence[Sequence[Image.Image]],
    prompt: str,
    max_length: int,
) -> List[List[float]]:
    model.eval()
    model.backbone.set_adapter("client")
    outputs: List[List[float]] = []
    for frames in frame_sets:
        logits, _ = model.forward_batch(processor, list(frames), prompt, max_length, output_hidden_states=True)
        outputs.append(logits[0].detach().float().cpu().tolist())
    return outputs


def _load_initial_state(model, checkpoint: Path | None) -> Dict[str, Any]:
    if checkpoint is None:
        return {"checkpoint": None, "loaded": False, "warning": "random classifier / zero-initialized LoRA"}
    names, vector, meta = load_fl_checkpoint(checkpoint)
    available = {name for name, _ in model.named_parameters()}
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"checkpoint has names not present in model: {missing[:3]}")
    vector_to_trainable_state(model, names, vector)
    return {
        "checkpoint": str(checkpoint),
        "loaded": True,
        "parameter_count": int(vector.size),
        "tensor_count": len(names),
        "source_run_id": meta.get("run_id"),
    }


def _font(size: int, *, bold: bool = False):
    names = ["meiryob.ttc" if bold else "meiryo.ttc", "YuGothB.ttc" if bold else "YuGothR.ttc"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _thumbnail_strip(frames: Sequence[Image.Image], *, thumb_size: Tuple[int, int], gap: int = 10) -> Image.Image:
    width, height = thumb_size
    strip = Image.new("RGB", (len(frames) * width + max(0, len(frames) - 1) * gap, height), "white")
    for index, frame in enumerate(frames):
        thumb = frame.copy()
        thumb.thumbnail((width, height), Image.Resampling.LANCZOS)
        x = index * (width + gap) + (width - thumb.width) // 2
        y = (height - thumb.height) // 2
        strip.paste(thumb, (x, y))
    return strip


def _draw_board(
    path: Path,
    *,
    private_frames: Sequence[Image.Image],
    seed_frames: Sequence[Image.Image],
    proxy_frames: Sequence[Image.Image],
    prototype: Mapping[str, Any],
    private_trace: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
) -> None:
    canvas = Image.new("RGB", (1800, 1180), "#F7F9FC")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    head_font = _font(28, bold=True)
    body_font = _font(22)
    small_font = _font(18)
    draw.text((60, 38), "NEXAR 1例で追う FedPACT Layer 2", fill="#13233A", font=title_font)
    draw.text((60, 94), "目的：精度競争ではなく、private適応がproxy pairになるまでを可視化", fill="#40566F", font=body_font)

    rows = [
        ("CLIENT（端末内のみ）", private_frames, "#DFF3EA"),
        ("SERVER SEED（検索結果）", seed_frames, "#E4EEFF"),
        ("PROXY（変換・目的関数で選択）", proxy_frames, "#F2E8FF"),
    ]
    y_positions = [180, 500, 820]
    for (label, frames, color), y in zip(rows, y_positions):
        draw.rounded_rectangle((45, y - 28, 1755, y + 250), radius=22, fill=color, outline="#90A4AE", width=2)
        draw.text((70, y - 12), label, fill="#172B4D", font=head_font)
        strip = _thumbnail_strip(frames, thumb_size=(250, 150))
        canvas.paste(strip, (70, y + 48))

    draw.text(
        (70, 442),
        "↓ client → server：LoRA更新情報 + μ_post + Δμ_local + frequency（private画像は送らない）",
        fill="#355D4B",
        font=small_font,
    )
    draw.text(
        (70, 762),
        "↓ retrieval → transformation → proxy objective → quality evaluation / selection",
        fill="#405D8A",
        font=small_font,
    )

    mean_pre = np.mean([softmax_vector(item["pre_logits"]) for item in private_trace], axis=0)
    mean_post = np.asarray(prototype["mu_post"], dtype=float)
    delta = np.asarray(prototype["delta_mu_local"], dtype=float)
    info_x = 1150
    draw.multiline_text(
        (info_x, 245),
        f"適応前 [normal, risk] = [{mean_pre[0]:.3f}, {mean_pre[1]:.3f}]\n"
        f"μ_post = [{mean_post[0]:.3f}, {mean_post[1]:.3f}]\n"
        f"Δμ_local = [{delta[0]:+.3f}, {delta[1]:+.3f}]\n"
        f"frequency = {prototype['frequency']}",
        fill="#193B2B",
        font=body_font,
        spacing=12,
    )
    metrics = selected["metrics"]
    transform = selected["transformation"]
    transform_text = transform["name"] if transform.get("value") is None else f"{transform['name']}({transform['value']})"
    draw.multiline_text(
        (info_x, 565),
        f"retrieved seed = {selected['seed_id']}\n"
        f"retrieval rank = {selected['retrieval_rank']}\n"
        f"transformation = {transform_text}",
        fill="#173C72",
        font=body_font,
        spacing=14,
    )
    draw.multiline_text(
        (info_x, 885),
        f"post KL = {metrics['post_kl']:.6f}\n"
        f"delta MSE = {metrics['delta_mse']:.6f}\n"
        f"L_match = {metrics['proxy_objective']:.6f}\n"
        f"Layer 3へ：P = (x̃, μ_post)",
        fill="#4B236B",
        font=body_font,
        spacing=12,
    )
    draw.text(
        (60, 1135),
        "※ 上段のprivate画像・sample IDは説明用のローカル表示であり、server payloadには含めない。",
        fill="#8A3B12",
        font=small_font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="省略時は最新 Step 7/5 checkpoint、none で未学習初期値")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--client-id", default="client-0")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--target-label", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--train-scope",
        choices=["target-class", "balanced"],
        default="target-class",
        help="target-class は対象クラスだけ、balanced は client train の両クラスで局所学習する",
    )
    parser.add_argument("--private-count", type=int, default=3)
    parser.add_argument("--client-eval-count", type=int, default=1)
    parser.add_argument("--confirmation-count-per-class", type=int, default=3)
    parser.add_argument(
        "--evaluate-confirmation",
        action="store_true",
        help="設定確定後だけ、screeningに未使用の確認用動画を評価する",
    )
    parser.add_argument("--seed-count-per-class", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--update-scope",
        choices=["lora-head", "lora-only", "head-only", "logit-scale-only"],
        default="lora-head",
        help="局所学習で更新するparameter群。logit-scale-onlyは判断境界を固定した対照条件",
    )
    parser.add_argument(
        "--train-order-mode",
        choices=["shuffled", "stratified"],
        default="shuffled",
        help="stratifiedはnormal/riskを交互に並べ、batch size 2なら各batchを1本ずつにする",
    )
    parser.add_argument(
        "--train-order-seed",
        type=int,
        default=None,
        help="splitを変えず、局所学習行だけをこのseedでshuffleする",
    )
    parser.add_argument("--transforms", choices=["identity", "tiny", "full"], default="tiny")
    parser.add_argument(
        "--screen-all-seeds-identity",
        action="store_true",
        help="top-k外もidentityだけ評価し、検索1位への固定が不利でないか診断する",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="client train/evalの前後差とpayloadまで保存し、server proxy探索は省略する",
    )
    parser.add_argument("--lambda-post", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "initial-error-balanced"],
        default="none",
        help="initial-error-balancedは学習前の各classの誤差寄与が同程度になる重みを固定して使う",
    )
    parser.add_argument("--direction-epsilon", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        args.private_count < 1
        or args.client_eval_count < 1
        or args.confirmation_count_per_class < 1
        or args.seed_count_per_class < 1
        or args.top_k < 1
        or args.local_steps < 1
        or args.batch_size < 1
        or args.gradient_accumulation_steps < 1
        or args.direction_epsilon < 0
    ):
        raise ValueError(
            "counts, top-k, local-steps, batch-size, and gradient-accumulation-steps must be >= 1; direction-epsilon must be >= 0"
        )
    if args.train_order_mode == "stratified" and args.train_scope != "balanced":
        raise ValueError("train-order-mode=stratified requires train-scope=balanced")
    if args.train_order_mode == "stratified" and args.batch_size != 2:
        raise ValueError("train-order-mode=stratified requires batch-size=2 (normal/risk one sample each)")
    if args.class_weight_mode != "none" and args.train_scope != "balanced":
        raise ValueError("class weighting requires train-scope=balanced")
    if args.update_scope == "logit-scale-only" and args.class_weight_mode != "none":
        raise ValueError("logit-scale-only does not use class weighting")

    cfg = merged_config(load_yaml(args.config))
    seed = int(args.seed if args.seed is not None else cfg["train"]["seed"])
    set_seed(seed)
    device_name = device_or_auto(True) if args.device == "auto" else args.device
    device = torch.device(device_name)
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.run_suffix:
        run_id = f"{run_id}_{args.run_suffix.strip('_')}"
    run_dir = Path(cfg["artifacts"]["runs"]) / "fedpact_nexar_example" / run_id
    client_dir = run_dir / "client_local"
    upload_dir = run_dir / "client_upload"
    server_dir = run_dir / "server"
    for path in (client_dir, upload_dir, server_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = Path(dcfg["manifest_path"])
    rows = read_manifest_filtered(manifest, dcfg)
    client_pools, client_eval, server_seed_rows, server_eval = split_step07_data_pools(
        rows,
        num_clients=1,
        client_train_per_class=int(args.private_count),
        client_eval_per_class=int(args.client_eval_count),
        server_train_per_class=int(args.seed_count_per_class),
        server_eval_per_class=int(args.confirmation_count_per_class),
        seed=seed,
    )
    client_train_rows = list(client_pools[0])
    target_private_rows = [
        row for row in client_train_rows if int(row["label"]) == int(args.target_label)
    ]
    if not target_private_rows:
        raise RuntimeError(f"no private rows for target label {args.target_label}")
    train_rows = target_private_rows if args.train_scope == "target-class" else client_train_rows
    trace_rows = target_private_rows if args.train_scope == "target-class" else client_train_rows
    order_seed = int(args.train_order_seed if args.train_order_seed is not None else seed)
    if args.train_order_mode == "stratified":
        rng = random.Random(order_seed)
        normal_rows = [row for row in train_rows if int(row["label"]) == 0]
        risk_rows = [row for row in train_rows if int(row["label"]) == 1]
        rng.shuffle(normal_rows)
        rng.shuffle(risk_rows)
        train_rows = [row for pair in zip(normal_rows, risk_rows) for row in pair]
    elif args.train_order_seed is not None:
        train_rows = list(train_rows)
        random.Random(order_seed).shuffle(train_rows)
    split_path = save_data_splits_artifact(
        client_dir,
        client_trains=client_pools,
        client_eval=client_eval,
        server_train=server_seed_rows,
        server_eval=server_eval,
        seed=seed,
    )

    checkpoint = _resolve_checkpoint(args.checkpoint)
    print(f"[1/6] model load: device={device}, checkpoint={checkpoint}", flush=True)
    started = time.perf_counter()
    model_id = str(cfg.get("model", {}).get("client_id") or cfg["model"]["id"])
    model, processor = build_model(cfg, device, model_id=model_id)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")
    checkpoint_info = _load_initial_state(model, checkpoint)
    sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")

    client_lora = lora_params_for_adapter(model.backbone, "client")
    surrogate_lora = lora_params_for_adapter(model.backbone, "surrogate")
    classifier_parameters = list(model.classifier.parameters())
    update_lora = args.update_scope in {"lora-head", "lora-only"}
    update_head = args.update_scope in {"lora-head", "head-only"}
    logit_scale_only = args.update_scope == "logit-scale-only"
    for parameter in client_lora:
        parameter.requires_grad_(update_lora)
    for parameter in classifier_parameters:
        parameter.requires_grad_(update_head)
    optimizer_parameters = [
        parameter
        for parameter in [*classifier_parameters, *client_lora]
        if parameter.requires_grad
    ]
    state_parameters = [
        *([*client_lora] if update_lora else []),
        *([*classifier_parameters] if update_head or logit_scale_only else []),
    ]
    if not state_parameters:
        raise RuntimeError("update-scope selected no trainable parameters")

    before_names, before_vector = _named_parameter_vector(model, state_parameters)
    before_state = snapshot_client_state(model, client_lora)
    parameter_group_before = {
        "client_lora": _flat_parameter_vector(client_lora),
        "classification_head": _flat_parameter_vector(classifier_parameters),
    }

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    print(
        f"[2/6] client pre logits: train={len(trace_rows)}, evaluation={len(client_eval)} samples",
        flush=True,
    )
    pre_logits = _predict_rows(model, processor, trace_rows, dcfg, prompt, max_length)
    eval_pre_logits = _predict_rows(model, processor, client_eval, dcfg, prompt, max_length)
    confirmation_pre_logits = (
        _predict_rows(model, processor, server_eval, dcfg, prompt, max_length)
        if args.evaluate_confirmation
        else []
    )

    class_weight_values = [1.0, 1.0]
    if args.class_weight_mode == "initial-error-balanced":
        pre_probabilities = [softmax_vector(logits) for logits in pre_logits]
        normal_errors = [
            float(probability[1])
            for row, probability in zip(trace_rows, pre_probabilities)
            if int(row["label"]) == 0
        ]
        risk_errors = [
            float(1.0 - probability[1])
            for row, probability in zip(trace_rows, pre_probabilities)
            if int(row["label"]) == 1
        ]
        if not normal_errors or not risk_errors:
            raise RuntimeError("initial-error-balanced requires both normal and risk samples")
        inverse_errors = np.asarray(
            [1.0 / max(np.mean(normal_errors), 1e-6), 1.0 / max(np.mean(risk_errors), 1e-6)],
            dtype=np.float64,
        )
        inverse_errors /= float(np.mean(inverse_errors))
        class_weight_values = inverse_errors.tolist()
    class_weights = torch.tensor(class_weight_values, device=device, dtype=torch.float32)

    exposure_count = int(args.local_steps) * int(args.batch_size)
    repeated_train_rows = [train_rows[index % len(train_rows)] for index in range(exposure_count)]
    learning_rate = float(args.learning_rate if args.learning_rate is not None else tcfg["lr"])
    if logit_scale_only:
        print("[3/6] client local adaptation: logit scale fitting", flush=True)
        logits_tensor = torch.tensor(pre_logits, device=device, dtype=torch.float32)
        labels_tensor = torch.tensor(
            [int(row["label"]) for row in trace_rows], device=device, dtype=torch.long
        )
        log_scale = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
        scale_optimizer = torch.optim.LBFGS(
            [log_scale], lr=0.25, max_iter=100, tolerance_grad=1e-9, tolerance_change=1e-12
        )
        initial_scale_loss = float(torch.nn.functional.cross_entropy(logits_tensor, labels_tensor).detach().cpu())

        def scale_closure() -> torch.Tensor:
            scale_optimizer.zero_grad(set_to_none=True)
            scale_loss = torch.nn.functional.cross_entropy(
                logits_tensor * torch.exp(log_scale), labels_tensor
            )
            scale_loss.backward()
            return scale_loss

        scale_optimizer.step(scale_closure)
        fitted_logit_scale = float(torch.exp(log_scale.detach()).cpu())
        final_scale_loss = float(
            torch.nn.functional.cross_entropy(
                logits_tensor * fitted_logit_scale, labels_tensor
            ).detach().cpu()
        )
        with torch.no_grad():
            model.classifier.weight.mul_(fitted_logit_scale)
            if model.classifier.bias is not None:
                model.classifier.bias.mul_(fitted_logit_scale)
        local_steps = 1
        train_metrics = {
            "avg_loss_task": final_scale_loss,
            "loss_task_last": final_scale_loss,
            "avg_loss_fit": final_scale_loss,
            "loss_fit_last": final_scale_loss,
            "microbatch_steps": 0,
            "optimizer_steps": 1,
            "sample_exposures": len(trace_rows),
            "gradient_accumulation_steps": 0,
            "logit_scale": fitted_logit_scale,
            "loss_before_logit_scale": initial_scale_loss,
            "loss_after_logit_scale": final_scale_loss,
            "note": "diagnostic control: decision boundary and predicted labels are unchanged",
        }
    else:
        opt = torch.optim.AdamW(
            optimizer_parameters,
            lr=learning_rate,
            weight_decay=float(tcfg["weight_decay"]),
        )
        print(f"[3/6] client local adaptation: {args.local_steps} step(s)", flush=True)
        local_steps, train_metrics = train_client_l_task(
            model=model,
            processor=processor,
            train_rows=repeated_train_rows,
            dcfg=dcfg,
            client_lora=client_lora,
            frozen_lora=surrogate_lora,
            opt=opt,
            prompt=prompt,
            max_length=max_length,
            device=device,
            batch_size=int(args.batch_size),
            max_steps=int(args.local_steps),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            update_client_lora=update_lora,
            audit_parameter_groups={
                "client_lora": client_lora,
                "classification_head": classifier_parameters,
            },
            class_weights=class_weights,
        )
    post_logits = _predict_rows(model, processor, trace_rows, dcfg, prompt, max_length)
    eval_post_logits = _predict_rows(model, processor, client_eval, dcfg, prompt, max_length)
    confirmation_post_logits = (
        _predict_rows(model, processor, server_eval, dcfg, prompt, max_length)
        if args.evaluate_confirmation
        else []
    )
    after_names, after_vector = _named_parameter_vector(model, state_parameters)
    if after_names != before_names:
        raise RuntimeError("trainable parameter order changed during local adaptation")
    after_state = snapshot_client_state(model, client_lora)
    update = after_vector - before_vector
    parameter_group_after = {
        "client_lora": _flat_parameter_vector(client_lora),
        "classification_head": _flat_parameter_vector(classifier_parameters),
    }
    parameter_audit = {
        name: {
            "parameter_count": int(parameter_group_before[name].size),
            "before_l2_norm": float(np.linalg.norm(parameter_group_before[name])),
            "after_l2_norm": float(np.linalg.norm(parameter_group_after[name])),
            "update_l2_norm": float(
                np.linalg.norm(parameter_group_after[name] - parameter_group_before[name])
            ),
            "updated": bool(
                np.linalg.norm(parameter_group_after[name] - parameter_group_before[name]) > 0
            ),
        }
        for name in parameter_group_before
    }

    private_trace = [
        {
            "sample_id": str(row["id"]),
            "label": int(row["label"]),
            "pre_logits": pre,
            "post_logits": post,
            "pre_probability": softmax_vector(pre).tolist(),
            "post_probability": softmax_vector(post).tolist(),
        }
        for row, pre, post in zip(trace_rows, pre_logits, post_logits)
    ]
    eval_trace = [
        {
            "sample_id": str(row["id"]),
            "label": int(row["label"]),
            "pre_logits": pre,
            "post_logits": post,
            "pre_probability": softmax_vector(pre).tolist(),
            "post_probability": softmax_vector(post).tolist(),
        }
        for row, pre, post in zip(client_eval, eval_pre_logits, eval_post_logits)
    ]
    confirmation_trace = [
        {
            "sample_id": str(row["id"]),
            "label": int(row["label"]),
            "pre_logits": pre,
            "post_logits": post,
            "pre_probability": softmax_vector(pre).tolist(),
            "post_probability": softmax_vector(post).tolist(),
        }
        for row, pre, post in zip(server_eval, confirmation_pre_logits, confirmation_post_logits)
    ]
    direction_diagnostics = {
        "train": _direction_diagnostics(private_trace, float(args.direction_epsilon)),
        "evaluation": _direction_diagnostics(eval_trace, float(args.direction_epsilon)),
    }
    if args.evaluate_confirmation:
        direction_diagnostics["confirmation"] = _direction_diagnostics(
            confirmation_trace, float(args.direction_epsilon)
        )
    prototypes = build_classwise_prototypes(private_trace, client_id=args.client_id, server_round=args.round)
    prototype = next(item for item in prototypes if int(item["class_label"]) == int(args.target_label))

    update_path = upload_dir / "trainable_update.npz"
    np.savez_compressed(update_path, names=np.asarray(before_names), delta=update.astype(np.float32))
    lora_update_info = {
        "artifact": update_path.name,
        "format": "npz",
        "parameter_count": int(update.size),
        "tensor_count": len(before_names),
        "l2_norm": float(np.linalg.norm(update)),
        "includes": [
            name
            for name, enabled in (
                ("client_lora", update_lora),
                ("classification_head", update_head or logit_scale_only),
            )
            if enabled
        ],
    }
    server_payload = build_server_payload(prototypes, lora_update=lora_update_info)
    assert_private_data_absent(server_payload)
    serialized_payload = json.dumps(server_payload, ensure_ascii=False)
    client_local_rows = [*client_train_rows, *client_eval]
    leaked_private_ids = [str(row["id"]) for row in client_local_rows if str(row["id"]) in serialized_payload]
    if leaked_private_ids:
        raise RuntimeError(f"private sample ID leaked into server payload: {leaked_private_ids}")
    payload_path = upload_dir / "server_payload.json"
    _save_json(payload_path, server_payload)
    _save_json(
        client_dir / "private_trace.json",
        {
            "scope": "client-local-only",
            "private_samples": private_trace,
            "direction_diagnostics": direction_diagnostics["train"],
            "training": {
                "scope": args.train_scope,
                "steps": local_steps,
                "batch_size": len(trace_rows) if logit_scale_only else int(args.batch_size),
                "sample_exposures": (
                    len(trace_rows) if logit_scale_only else len(repeated_train_rows)
                ),
                "sample_count": len(train_rows),
                "order_seed": args.train_order_seed,
                "order_mode": (
                    "full-set-order-invariant" if logit_scale_only else args.train_order_mode
                ),
                "sample_order": [
                    str(row["id"])
                    for row in (trace_rows if logit_scale_only else repeated_train_rows)
                ],
                "update_scope": args.update_scope,
                "learning_rate": learning_rate,
                "class_weight_mode": args.class_weight_mode,
                "class_weights": class_weight_values,
                "parameter_audit": parameter_audit,
                "class_counts": {
                    str(label): sum(int(row["label"]) == label for row in train_rows)
                    for label in (0, 1)
                },
                **train_metrics,
            },
            "note": "sample IDs and raw frames are not copied into server payload",
        },
    )
    _save_json(
        client_dir / "eval_trace.json",
        {
            "scope": "client-local-only",
            "split": "evaluation-only: excluded from local training, prototype creation, and server seed retrieval",
            "samples": eval_trace,
            "direction_diagnostics": direction_diagnostics["evaluation"],
            "note": "This evaluation set is used for setting screening and is not a final generalization estimate.",
        },
    )
    if args.evaluate_confirmation:
        _save_json(
            client_dir / "confirmation_trace.json",
            {
                "scope": "client-local-only",
                "split": "confirmation-only: first evaluated after the setting was selected",
                "samples": confirmation_trace,
                "direction_diagnostics": direction_diagnostics["confirmation"],
                "note": "This split must not be used to tune the selected setting.",
            },
        )

    if args.diagnostic_only:
        elapsed = time.perf_counter() - started
        summary = {
            "run_id": run_id,
            "purpose": "balanced NEXAR client diagnostic; server proxy search intentionally skipped",
            "model_id": model_id,
            "checkpoint": checkpoint_info,
            "device": str(device),
            "seed": seed,
            "train_order_seed": args.train_order_seed,
            "train_order_mode": args.train_order_mode,
            "update_scope": args.update_scope,
            "learning_rate": learning_rate,
            "class_weight_mode": args.class_weight_mode,
            "class_weights": class_weight_values,
            "batch_size": len(trace_rows) if logit_scale_only else int(args.batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "split_artifact": str(split_path.relative_to(run_dir)),
            "client_train_scope": args.train_scope,
            "client_train_count": len(train_rows),
            "client_train_class_counts": {
                str(label): sum(int(row["label"]) == label for row in train_rows)
                for label in (0, 1)
            },
            "client_eval_count": len(client_eval),
            "client_eval_class_counts": {
                str(label): sum(int(row["label"]) == label for row in client_eval)
                for label in (0, 1)
            },
            "confirmation_evaluated": bool(args.evaluate_confirmation),
            "confirmation_count": len(confirmation_trace),
            "prototypes": prototypes,
            "direction_diagnostics": direction_diagnostics,
            "parameter_audit": parameter_audit,
            "training_metrics": train_metrics,
            "artifacts": {
                "client_local_trace": "client_local/private_trace.json",
                "client_local_eval_trace": "client_local/eval_trace.json",
                **(
                    {"client_local_confirmation_trace": "client_local/confirmation_trace.json"}
                    if args.evaluate_confirmation
                    else {}
                ),
                "client_to_server_payload": "client_upload/server_payload.json",
                "lora_update": "client_upload/trainable_update.npz",
            },
            "privacy_audit": {
                "server_payload_has_private_raw_data": False,
                "mu_pre_sent_independently": False,
            },
            "limitations": [
                "diagnostic-only run: server seed retrieval and proxy generation were skipped",
                "the small evaluation set is used only for setting screening and is not a generalization estimate",
                "the confirmation set is inspected only when --evaluate-confirmation is explicitly enabled",
                *(
                    ["logit-scale-only changes confidence without changing the decision boundary; it is a diagnostic control, not evidence that Layer 1 learned better features"]
                    if logit_scale_only
                    else []
                ),
            ],
            "elapsed_sec": elapsed,
        }
        _save_json(run_dir / "summary.json", summary)
        print("[4/4] client diagnostic completed", flush=True)
        print(f"run_dir={run_dir}", flush=True)
        return

    print(f"[4/6] server seed retrieval: {len(server_seed_rows)} candidates", flush=True)
    load_client_state(model, client_lora, before_state)
    load_client_state(model, client_lora, after_state)
    seed_after_logits = _predict_rows(model, processor, server_seed_rows, dcfg, prompt, max_length)
    seed_candidates = [
        {
            "seed_id": str(row["id"]),
            "label": int(row["label"]),
            "updated_logits": logits,
        }
        for row, logits in zip(server_seed_rows, seed_after_logits)
    ]
    ranked = rank_server_seeds(prototype, seed_candidates)
    top_k = min(int(args.top_k), len(ranked))
    row_by_id = {str(row["id"]): row for row in server_seed_rows}

    variants: List[Dict[str, Any]] = []
    frame_sets: List[List[Image.Image]] = []
    specs = transformation_candidates(args.transforms)
    transformation_dir = server_dir / "transformation_candidates"
    transformation_dir.mkdir(parents=True, exist_ok=True)
    identity_spec = next(spec for spec in specs if spec.name == "identity")
    retrieved_for_screening = ranked if args.screen_all_seeds_identity else ranked[:top_k]
    for retrieved in retrieved_for_screening:
        sample = load_sample_row(row_by_id[retrieved["seed_id"]], dcfg)
        seed_specs = specs if int(retrieved["rank"]) <= top_k else [identity_spec]
        for spec_index, spec in enumerate(seed_specs):
            transformed = transform_frames(sample.frames, spec)
            value_text = "none" if spec.value is None else str(spec.value).replace(".", "p")
            candidate_dir = transformation_dir / (
                f"rank{int(retrieved['rank']):02d}_{spec_index:02d}_{spec.name}_{value_text}"
            )
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_frame_files: List[str] = []
            for frame_index, frame in enumerate(transformed):
                candidate_path = candidate_dir / f"frame_{frame_index:02d}.png"
                frame.save(candidate_path)
                candidate_frame_files.append(str(candidate_path.relative_to(server_dir)))
            variants.append(
                {
                    "seed_id": retrieved["seed_id"],
                    "seed_label": int(retrieved["label"]),
                    "retrieval_rank": int(retrieved["rank"]),
                    "retrieval_post_kl": float(retrieved["retrieval_post_kl"]),
                    "transformation": spec.as_dict(),
                    "frame_files": candidate_frame_files,
                }
            )
            frame_sets.append(transformed)
        for frame in sample.frames:
            frame.close()

    print(f"[5/6] proxy optimization: {len(frame_sets)} transformed candidates", flush=True)
    load_client_state(model, client_lora, before_state)
    variant_before_logits = _predict_frame_sets(model, processor, frame_sets, prompt, max_length)
    load_client_state(model, client_lora, after_state)
    variant_after_logits = _predict_frame_sets(model, processor, frame_sets, prompt, max_length)
    for variant, before_logits, after_logits in zip(variants, variant_before_logits, variant_after_logits):
        variant["global_before_probability"] = softmax_vector(before_logits).tolist()
        variant["global_after_probability"] = softmax_vector(after_logits).tolist()
        variant["metrics"] = proxy_match_metrics(
            prototype,
            global_before_logits=before_logits,
            global_after_logits=after_logits,
            lambda_post=float(args.lambda_post),
            lambda_delta=float(args.lambda_delta),
        )
    selected = select_best_proxy(variants)
    selected_index = next(
        index
        for index, item in enumerate(variants)
        if item["seed_id"] == selected["seed_id"]
        and item["transformation"] == selected["transformation"]
    )
    selected_frames = frame_sets[selected_index]
    proxy_frame_dir = server_dir / "proxy_frames"
    proxy_frame_dir.mkdir(parents=True, exist_ok=True)
    proxy_frame_files: List[str] = []
    for index, frame in enumerate(selected_frames):
        frame_path = proxy_frame_dir / f"proxy_{index:02d}.png"
        frame.save(frame_path)
        proxy_frame_files.append(str(frame_path.relative_to(server_dir)))

    proxy_pair = {
        "schema": "fedpact.proxy_pair.v1",
        "prototype_id": prototype["prototype_id"],
        "proxy": {
            "source_seed_id": selected["seed_id"],
            "frame_files": proxy_frame_files,
            "transformation": selected["transformation"],
        },
        "teacher_mu_post": prototype["mu_post"],
        "frequency": prototype["frequency"],
        "layer3_contract": "P=(x_tilde, mu_post)",
    }
    _save_json(server_dir / "proxy_pair.json", proxy_pair)
    _save_json(
        server_dir / "layer2_trace.json",
        {
            "schema": "fedpact.layer2_trace.v1",
            "inputs": {
                "payload": str(payload_path.relative_to(run_dir)),
                "global_surrogate_before": "checkpoint state before client local adaptation",
                "global_surrogate_after": "single-client FedAvg state; equals this client's locally adapted state",
                "server_seed_split": "cloud-only; not distributed to client",
            },
            "retrieval": {
                "rule": "ascending D_KL(mu_post || updated_global_surrogate(seed))",
                "top_k": top_k,
                "candidates": ranked,
            },
            "transformation_and_optimization": {
                "preset": args.transforms,
                "screening": (
                    "all server seeds evaluated with identity; top-k also evaluated with full transform preset"
                    if args.screen_all_seeds_identity
                    else "only retrieved top-k evaluated with transform preset"
                ),
                "objective": "lambda_post * post_kl + lambda_delta * delta_mse",
                "delta_comparison": "client delta_mu_local vs global surrogate after-before probability difference on proxy",
                "candidates": variants,
            },
            "quality_and_selection": {
                "selected": selected,
                "qinv_note": "Qinv formula/threshold is intentionally not fixed in Step I-B; screening is required before Qinv-based weighting.",
            },
            "output": proxy_pair,
        },
    )

    private_sample = load_sample_row(target_private_rows[0], dcfg)
    seed_sample = load_sample_row(row_by_id[selected["seed_id"]], dcfg)
    board_path = run_dir / "presentation_example.png"
    _draw_board(
        board_path,
        private_frames=private_sample.frames,
        seed_frames=seed_sample.frames,
        proxy_frames=selected_frames,
        prototype=prototype,
        private_trace=private_trace,
        selected=selected,
    )
    for frame in private_sample.frames + seed_sample.frames:
        frame.close()
    for frames in frame_sets:
        for frame in frames:
            frame.close()

    elapsed = time.perf_counter() - started
    summary = {
        "run_id": run_id,
        "purpose": "NEXAR concrete-task vertical slice for explaining FedPACT Layer 2; not a performance result",
        "model_id": model_id,
        "checkpoint": checkpoint_info,
        "device": str(device),
        "seed": seed,
        "train_order_seed": args.train_order_seed,
        "train_order_mode": args.train_order_mode,
        "update_scope": args.update_scope,
        "learning_rate": learning_rate,
        "class_weight_mode": args.class_weight_mode,
        "class_weights": class_weight_values,
        "batch_size": len(trace_rows) if logit_scale_only else int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "split_artifact": str(split_path.relative_to(run_dir)),
        "client_train_scope": args.train_scope,
        "client_train_count": len(train_rows),
        "client_train_class_counts": {
            str(label): sum(int(row["label"]) == label for row in train_rows)
            for label in (0, 1)
        },
        "client_eval_count": len(client_eval),
        "client_eval_class_counts": {
            str(label): sum(int(row["label"]) == label for row in client_eval)
            for label in (0, 1)
        },
        "confirmation_evaluated": bool(args.evaluate_confirmation),
        "confirmation_count": len(confirmation_trace),
        "server_seed_count": len(server_seed_rows),
        "server_seed_identity_screened": len(ranked) if args.screen_all_seeds_identity else top_k,
        "prototypes": prototypes,
        "prototype": prototype,
        "selected_proxy": selected,
        "direction_diagnostics": direction_diagnostics,
        "parameter_audit": parameter_audit,
        "training_metrics": train_metrics,
        "artifacts": {
            "client_local_trace": "client_local/private_trace.json",
            "client_local_eval_trace": "client_local/eval_trace.json",
            **(
                {"client_local_confirmation_trace": "client_local/confirmation_trace.json"}
                if args.evaluate_confirmation
                else {}
            ),
            "client_to_server_payload": "client_upload/server_payload.json",
            "lora_update": "client_upload/trainable_update.npz",
            "server_layer2_trace": "server/layer2_trace.json",
            "layer3_proxy_pair": "server/proxy_pair.json",
            "presentation_board": "presentation_example.png",
        },
        "privacy_audit": {
            "server_payload_has_private_raw_data": False,
            "mu_pre_sent_independently": False,
        },
        "limitations": [
            "single-client vertical slice: FedAvg after-state equals the local after-state",
            "discrete transformation search is the minimal proxy optimization",
            "Qinv weighting is deferred until its screening rule is fixed",
            "this run does not claim F1 or transfer improvement",
            "the small evaluation set is used only for setting screening and is not a generalization estimate",
            "the confirmation set is inspected only when --evaluate-confirmation is explicitly enabled",
            *(
                ["logit-scale-only changes confidence without changing the decision boundary; it is a diagnostic control, not evidence that Layer 1 learned better features"]
                if logit_scale_only
                else []
            ),
        ],
        "elapsed_sec": elapsed,
    }
    _save_json(run_dir / "summary.json", summary)
    print("[6/6] completed", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"board={board_path}", flush=True)
    print(f"selected_seed={selected['seed_id']} transform={selected['transformation']}", flush=True)
    print(f"post_kl={selected['metrics']['post_kl']:.6f} delta_mse={selected['metrics']['delta_mse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
