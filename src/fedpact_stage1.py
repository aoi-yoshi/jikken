"""FedPACT Stage I の state・FedAvg・通信契約を検証する小さな共通部品。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


FORBIDDEN_SERVER_KEYS = {
    "private_row",
    "private_rows",
    "private_sample",
    "private_samples",
    "sample_id",
    "sample_ids",
    "frame_path",
    "frame_paths",
    "video_path",
    "video_paths",
    "raw_frame",
    "raw_frames",
    "q_pre",
    "q^{pre}",
    "pre_logits",
    "post_logits",
}

PAYLOAD_V2_TOP_LEVEL_KEYS = {
    "schema",
    "client_id",
    "round",
    "local_train_sample_count",
    "trainable_update",
    "prototypes",
    "privacy",
}
PAYLOAD_V2_UPDATE_KEYS = {"client_lora", "classification_head"}
PAYLOAD_V2_ARTIFACT_KEYS = {
    "artifact",
    "tensor_count",
    "parameter_count",
    "l2_norm",
    "sha256",
}
PAYLOAD_V2_PROTOTYPE_KEYS = {
    "prototype_id",
    "client_id",
    "round",
    "class_label",
    "mu_post",
    "delta_mu_local",
    "frequency",
    "class_count",
    "representation",
}
PAYLOAD_V2_PRIVACY_KEYS = {
    "private_raw_data_included",
    "private_sample_id_included",
    "sample_level_output_included",
    "q_pre_sent_independently",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def canonical_tensor_digest(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Tensor名・shape・dtype・raw bytesを固定順でhashする。"""
    digest = hashlib.sha256()
    items = sorted((str(name), tensor.detach().cpu().contiguous()) for name, tensor in named_tensors)
    for name, tensor in items:
        header = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def named_selected_parameters(
    model: torch.nn.Module, parameters: Sequence[torch.nn.Parameter]
) -> list[tuple[str, torch.Tensor]]:
    selected = {id(parameter) for parameter in parameters}
    items = [(name, parameter) for name, parameter in model.named_parameters() if id(parameter) in selected]
    if len(items) != len(parameters):
        raise RuntimeError(
            f"selected parameter name mismatch: expected={len(parameters)}, resolved={len(items)}"
        )
    return items


def state_named_tensors(
    model: torch.nn.Module, client_lora: Sequence[torch.nn.Parameter]
) -> dict[str, list[tuple[str, torch.Tensor]]]:
    lora = named_selected_parameters(model, client_lora)
    head = [(f"classifier.{name}", tensor) for name, tensor in model.classifier.state_dict().items()]
    return {"global_surrogate_lora": lora, "classification_head": head}


def state_digests(
    model: torch.nn.Module, client_lora: Sequence[torch.nn.Parameter]
) -> dict[str, str]:
    groups = state_named_tensors(model, client_lora)
    return {name: canonical_tensor_digest(tensors) for name, tensors in groups.items()}


def flatten_named_tensors(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> tuple[list[str], np.ndarray]:
    names = [name for name, _ in named_tensors]
    if not named_tensors:
        return names, np.zeros((0,), dtype=np.float32)
    vector = torch.cat([tensor.detach().cpu().float().reshape(-1) for _, tensor in named_tensors])
    return names, vector.numpy().astype(np.float32, copy=False)


def weighted_average_vectors(
    named_vectors: Sequence[tuple[Sequence[str], np.ndarray]], sample_counts: Sequence[int]
) -> tuple[list[str], np.ndarray, list[float]]:
    if not named_vectors:
        raise ValueError("named_vectors is empty")
    if len(named_vectors) != len(sample_counts):
        raise ValueError("named_vectors/sample_counts length mismatch")
    if any(int(count) <= 0 for count in sample_counts):
        raise ValueError("sample_counts must be positive")
    names = list(named_vectors[0][0])
    for other_names, vector in named_vectors:
        if list(other_names) != names:
            raise ValueError("parameter names/order mismatch")
        if np.asarray(vector).shape != np.asarray(named_vectors[0][1]).shape:
            raise ValueError("parameter vector shape mismatch")
    counts = np.asarray(sample_counts, dtype=np.float64)
    weights = counts / counts.sum()
    averaged = np.zeros_like(np.asarray(named_vectors[0][1]), dtype=np.float32)
    for weight, (_, vector) in zip(weights, named_vectors):
        averaged += np.asarray(vector, dtype=np.float32) * np.float32(weight)
    return names, averaged, weights.astype(float).tolist()


def apply_named_vector(
    model: torch.nn.Module, names: Sequence[str], vector: np.ndarray
) -> None:
    name_to_parameter = dict(model.named_parameters())
    values = torch.as_tensor(vector, dtype=torch.float32)
    offset = 0
    with torch.no_grad():
        for name in names:
            if name not in name_to_parameter:
                raise KeyError(f"parameter not found: {name}")
            parameter = name_to_parameter[name]
            count = parameter.numel()
            chunk = values[offset : offset + count]
            if chunk.numel() != count:
                raise ValueError(f"vector ended before {name}")
            parameter.copy_(chunk.view_as(parameter).to(parameter.device, parameter.dtype))
            offset += count
    if offset != values.numel():
        raise ValueError(f"unused vector values: used={offset}, total={values.numel()}")


def l2_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float32)))


def canonical_video_id(row: Mapping[str, Any]) -> str:
    raw = str(row.get("video_path", "")).strip()
    if not raw:
        raise ValueError("manifest row has no video_path")
    normalized = raw.replace("\\", "/").casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_split_video_overlap(pools: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    seen: dict[str, str] = {}
    overlaps: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for pool_name, rows in pools.items():
        ids = [canonical_video_id(row) for row in rows]
        counts[str(pool_name)] = len(ids)
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate source video inside pool: {pool_name}")
        for video_id in ids:
            previous = seen.get(video_id)
            if previous is not None:
                overlaps.append({"video_id": video_id, "first": previous, "second": str(pool_name)})
            else:
                seen[video_id] = str(pool_name)
    if overlaps:
        raise ValueError(f"source video overlap across pools: {overlaps[:3]}")
    return {"status": "pass", "pool_counts": counts, "unique_video_count": len(seen)}


def audit_server_payload(
    payload: Mapping[str, Any], *, forbidden_values: Sequence[str] = ()
) -> dict[str, Any]:
    violations: list[str] = []
    forbidden_text = [str(value) for value in forbidden_values if str(value)]

    def require_exact_keys(value: Any, expected: set[str], path: str) -> None:
        if not isinstance(value, Mapping):
            violations.append(f"expected object at {path}")
            return
        actual = {str(key) for key in value.keys()}
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing:
            violations.append(f"missing allowlisted keys at {path}: {missing}")
        if unexpected:
            violations.append(f"unexpected keys at {path}: {unexpected}")

    if str(payload.get("schema", "")) != "fedpact.client_payload.v2":
        violations.append("unsupported payload schema")
    require_exact_keys(payload, PAYLOAD_V2_TOP_LEVEL_KEYS, "payload")
    trainable_update = payload.get("trainable_update", {})
    require_exact_keys(trainable_update, PAYLOAD_V2_UPDATE_KEYS, "payload.trainable_update")
    if isinstance(trainable_update, Mapping):
        for name in sorted(PAYLOAD_V2_UPDATE_KEYS):
            require_exact_keys(
                trainable_update.get(name, {}),
                PAYLOAD_V2_ARTIFACT_KEYS,
                f"payload.trainable_update.{name}",
            )
    prototypes = payload.get("prototypes", [])
    if not isinstance(prototypes, list):
        violations.append("payload.prototypes must be a list")
    else:
        for index, prototype in enumerate(prototypes):
            require_exact_keys(
                prototype, PAYLOAD_V2_PROTOTYPE_KEYS, f"payload.prototypes[{index}]"
            )
    privacy = payload.get("privacy", {})
    require_exact_keys(privacy, PAYLOAD_V2_PRIVACY_KEYS, "payload.privacy")
    if isinstance(privacy, Mapping):
        for key in PAYLOAD_V2_PRIVACY_KEYS:
            if privacy.get(key) is not False:
                violations.append(f"privacy flag must be false: payload.privacy.{key}")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text.casefold() in FORBIDDEN_SERVER_KEYS:
                    violations.append(f"forbidden key: {child_path}")
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str):
            if any(token in value for token in forbidden_text):
                violations.append(f"forbidden private identifier value at {path}")
            lowered = value.casefold()
            if lowered.endswith((".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png")):
                violations.append(f"raw media/path-like value at {path}")
            if ":\\" in value or value.startswith(("/", "\\\\")):
                violations.append(f"absolute path at {path}")

    visit(payload, "")
    if violations:
        raise ValueError("; ".join(violations[:10]))
    return {
        "status": "pass",
        "schema": "fedpact.client_payload.v2",
        "policy": "exact_field_allowlist_plus_private_value_scan",
        "checked_forbidden_keys": sorted(FORBIDDEN_SERVER_KEYS),
    }


def audit_communication_bundle(
    bundle_dir: Path, *, forbidden_values: Sequence[str] = ()
) -> dict[str, Any]:
    allowed_suffixes = {".json", ".npz"}
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in bundle_dir.rglob("*") if item.is_file()):
        if path.suffix.casefold() not in allowed_suffixes:
            raise ValueError(f"unexpected communication artifact: {path.name}")
        relative = str(path.relative_to(bundle_dir)).replace("\\", "/")
        if path.suffix.casefold() == ".json":
            audit_server_payload(
                json.loads(path.read_text(encoding="utf-8")), forbidden_values=forbidden_values
            )
        else:
            with np.load(path, allow_pickle=False) as data:
                if sorted(data.files) != ["delta", "names"]:
                    raise ValueError(f"unexpected npz fields in {relative}: {data.files}")
                names = [str(value) for value in data["names"].tolist()]
                if any(token in name for token in forbidden_values for name in names):
                    raise ValueError(f"private identifier in npz names: {relative}")
                delta = np.asarray(data["delta"], dtype=np.float32)
                if not np.all(np.isfinite(delta)):
                    raise ValueError(f"non-finite update in {relative}")
        files.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    if not files:
        raise ValueError("empty communication bundle")
    return {"status": "pass", "files": files}
