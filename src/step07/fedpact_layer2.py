"""FedPACT Layer 2 の最小構成に使う prototype・検索・proxy 評価。

このモジュールはモデルに依存しない。モデル forward と局所学習は入口スクリプト側で行い、
ここでは client から送れる表現と server 側の比較規則を明示的に分離する。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


_PRIVATE_PAYLOAD_KEYS = {
    "private_row",
    "private_rows",
    "sample_id",
    "sample_ids",
    "frame_path",
    "frame_paths",
    "video_path",
}


def _as_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains a non-finite value")
    return arr


def softmax_vector(logits: Sequence[float]) -> np.ndarray:
    """1 サンプルの logits を安定な確率ベクトルへ変換する。"""
    arr = _as_vector(logits, name="logits")
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def kl_divergence(target: Sequence[float], actual: Sequence[float], eps: float = 1e-8) -> float:
    """D_KL(target || actual)。入力は確率分布。"""
    p = _as_vector(target, name="target")
    q = _as_vector(actual, name="actual")
    if p.shape != q.shape:
        raise ValueError(f"distribution shape mismatch: {p.shape} vs {q.shape}")
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def build_classwise_prototypes(
    records: Sequence[Mapping[str, Any]],
    *,
    client_id: str,
    server_round: int,
) -> List[Dict[str, Any]]:
    """同一 private sample の pre/post logits から class-wise prototype を作る。

    出力には private sample ID や path を含めない。mu_post と delta_mu は確率空間で
    平均するため、server 側の KL と global-surrogate 出力差にそのまま利用できる。
    """
    by_label: Dict[int, List[Mapping[str, Any]]] = {}
    for record in records:
        by_label.setdefault(int(record["label"]), []).append(record)

    prototypes: List[Dict[str, Any]] = []
    for label, items in sorted(by_label.items()):
        pre = np.stack([softmax_vector(item["pre_logits"]) for item in items], axis=0)
        post = np.stack([softmax_vector(item["post_logits"]) for item in items], axis=0)
        mu_post = np.mean(post, axis=0)
        delta_mu = np.mean(post - pre, axis=0)
        prototypes.append(
            {
                "prototype_id": f"r{int(server_round)}-{client_id}-class{label}",
                "client_id": str(client_id),
                "round": int(server_round),
                "class_label": int(label),
                "mu_post": mu_post.astype(float).tolist(),
                "delta_mu_local": delta_mu.astype(float).tolist(),
                "frequency": len(items),
                "class_count": len(items),
                "representation": "classwise_mean_probability",
            }
        )
    return prototypes


def assert_private_data_absent(payload: Mapping[str, Any]) -> None:
    """client→server payload に raw/private sample を示すキーがないことを監査する。"""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text.lower() in _PRIVATE_PAYLOAD_KEYS:
                    raise ValueError(f"private-data key found in server payload: {child_path}")
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(payload, "")


def build_server_payload(
    prototypes: Sequence[Mapping[str, Any]],
    *,
    lora_update: Mapping[str, Any],
) -> Dict[str, Any]:
    """FedPACT basic payload: LoRA 更新情報 + post/delta prototype + frequency。"""
    payload = {
        "schema": "fedpact.client_payload.v1",
        "lora_update": dict(lora_update),
        "prototypes": [dict(prototype) for prototype in prototypes],
        "privacy": {
            "private_raw_data_included": False,
            "mu_pre_sent_independently": False,
        },
    }
    assert_private_data_absent(payload)
    return payload


def retrieval_score(prototype: Mapping[str, Any], updated_logits: Sequence[float]) -> float:
    """updated global surrogate の seed 出力と mu_post の post KL。"""
    return kl_divergence(prototype["mu_post"], softmax_vector(updated_logits))


def rank_server_seeds(
    prototype: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """updated global surrogate の出力だけを使い、post prototype に近い順へ並べる。"""
    ranked: List[Dict[str, Any]] = []
    for candidate in candidates:
        post_kl = retrieval_score(prototype, candidate["updated_logits"])
        ranked.append(
            {
                "seed_id": str(candidate["seed_id"]),
                "label": int(candidate["label"]),
                "updated_probability": softmax_vector(candidate["updated_logits"]).astype(float).tolist(),
                "retrieval_post_kl": post_kl,
            }
        )
    ranked.sort(key=lambda item: (float(item["retrieval_post_kl"]), item["seed_id"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def proxy_match_metrics(
    prototype: Mapping[str, Any],
    *,
    global_before_logits: Sequence[float],
    global_after_logits: Sequence[float],
    lambda_post: float,
    lambda_delta: float,
) -> Dict[str, float]:
    """最小 proxy objective の post KL、delta error、加重和を返す。

    delta_mu_local は client の局所適応前後差であり、比較対象は proxy 上で測った
    global surrogate の集約前後差である。この二つを同じ値として流用しない。
    """
    before = softmax_vector(global_before_logits)
    after = softmax_vector(global_after_logits)
    target_delta = _as_vector(prototype["delta_mu_local"], name="delta_mu_local")
    if before.shape != target_delta.shape:
        raise ValueError(f"delta shape mismatch: {before.shape} vs {target_delta.shape}")
    post_kl = kl_divergence(prototype["mu_post"], after)
    observed_global_delta = after - before
    delta_mse = float(np.mean(np.square(target_delta - observed_global_delta)))
    objective = float(lambda_post) * post_kl + float(lambda_delta) * delta_mse
    return {
        "post_kl": post_kl,
        "delta_mse": delta_mse,
        "proxy_objective": objective,
        "lambda_post": float(lambda_post),
        "lambda_delta": float(lambda_delta),
    }


@dataclass(frozen=True)
class Transformation:
    name: str
    value: float | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value}


def transformation_candidates(preset: str) -> List[Transformation]:
    """Stage I-B 用の小さな離散探索集合。最終採用値は screening で固定する。"""
    mode = str(preset).strip().lower()
    if mode == "identity":
        return [Transformation("identity")]
    if mode == "tiny":
        return [
            Transformation("identity"),
            Transformation("brightness", 0.85),
            Transformation("brightness", 1.15),
            Transformation("contrast", 1.15),
            Transformation("center_crop", 0.90),
        ]
    if mode == "full":
        return [
            Transformation("identity"),
            Transformation("brightness", 0.75),
            Transformation("brightness", 0.90),
            Transformation("brightness", 1.10),
            Transformation("brightness", 1.25),
            Transformation("contrast", 0.80),
            Transformation("contrast", 1.20),
            Transformation("center_crop", 0.85),
            Transformation("center_crop", 0.92),
            Transformation("gaussian_blur", 0.75),
        ]
    raise ValueError(f"unknown transformation preset: {preset!r}")


def transform_frames(frames: Sequence[Image.Image], spec: Transformation) -> List[Image.Image]:
    """同じ変換を動画サンプルの全フレームへ適用する。"""
    if not frames:
        raise ValueError("frames is empty")
    out: List[Image.Image] = []
    for frame in frames:
        image = frame.convert("RGB")
        if spec.name == "identity":
            transformed = image.copy()
        elif spec.name == "brightness":
            transformed = ImageEnhance.Brightness(image).enhance(float(spec.value))
        elif spec.name == "contrast":
            transformed = ImageEnhance.Contrast(image).enhance(float(spec.value))
        elif spec.name == "gaussian_blur":
            transformed = image.filter(ImageFilter.GaussianBlur(radius=float(spec.value)))
        elif spec.name == "center_crop":
            scale = float(spec.value)
            if not 0.0 < scale <= 1.0:
                raise ValueError(f"center_crop scale must be in (0, 1], got {scale}")
            width, height = image.size
            crop_w = max(1, int(math.floor(width * scale)))
            crop_h = max(1, int(math.floor(height * scale)))
            left = (width - crop_w) // 2
            top = (height - crop_h) // 2
            transformed = image.crop((left, top, left + crop_w, top + crop_h)).resize(
                (width, height), Image.Resampling.BICUBIC
            )
        else:
            raise ValueError(f"unknown transformation: {spec.name!r}")
        out.append(transformed)
    return out


def select_best_proxy(candidates: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Stage I-B の一様利用条件: Qinv weighting はせず L_match 最小を採用する。"""
    items = [dict(item) for item in candidates]
    if not items:
        raise ValueError("proxy candidates is empty")
    items.sort(
        key=lambda item: (
            float(item["metrics"]["proxy_objective"]),
            str(item["seed_id"]),
            str(item["transformation"]["name"]),
            str(item["transformation"].get("value")),
        )
    )
    selected = items[0]
    selected["selected"] = True
    selected["selection_rule"] = "minimum_proxy_objective_uniform_frequency"
    selected["qinv"] = None
    selected["qinv_status"] = "not_computed_until_screening_rule_is_fixed"
    return selected
