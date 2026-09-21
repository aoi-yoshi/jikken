"""Model-independent visual manipulation check for the Stage-II coverage split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_descriptor(path: str) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb01 = rgb.astype(np.float32) / 255.0
    hsv01 = hsv.astype(np.float32) / np.array([179.0, 255.0, 255.0], dtype=np.float32)
    gray01 = gray.astype(np.float32) / 255.0
    gray_hist = np.histogram(gray01, bins=16, range=(0.0, 1.0), density=True)[0]
    hue_hist = np.histogram(hsv01[..., 0], bins=12, range=(0.0, 1.0), density=True)[0]
    sat_hist = np.histogram(hsv01[..., 1], bins=8, range=(0.0, 1.0), density=True)[0]
    edges = cv2.Canny(gray, 80, 160)
    dct = cv2.dct(cv2.resize(gray01, (16, 16)))[:8, :8].reshape(-1)[1:]
    return np.concatenate(
        [
            rgb01.mean(axis=(0, 1)),
            rgb01.std(axis=(0, 1)),
            hsv01.mean(axis=(0, 1)),
            hsv01.std(axis=(0, 1)),
            np.percentile(gray01, [10, 25, 50, 75, 90]),
            gray_hist,
            hue_hist,
            sat_hist,
            np.asarray(
                [
                    float((edges > 0).mean()),
                    float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())),
                ]
            ),
            dct,
        ]
    ).astype(np.float32)


def video_descriptor(row: dict) -> np.ndarray:
    frames = np.stack([frame_descriptor(path) for path in row["frame_paths"]])
    temporal = np.abs(np.diff(frames, axis=0)).mean(axis=0)
    return np.concatenate([frames.mean(axis=0), frames.std(axis=0), temporal]).astype(np.float32)


def audit_pair(target_rows: list[dict], seed_rows: list[dict], random_seed: int) -> dict:
    rows = [*target_rows, *seed_rows]
    x = np.stack([video_descriptor(row) for row in rows])
    y = np.asarray([1] * len(target_rows) + [0] * len(seed_rows), dtype=np.int64)
    groups = np.asarray([row["pair_id"] for row in rows])
    role = np.asarray([row["pair_role"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
    probability = np.zeros(len(rows), dtype=np.float64)
    folds: list[float] = []
    for train_index, test_index in splitter.split(x, y, groups):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=random_seed),
        )
        classifier.fit(x[train_index], y[train_index])
        probability[test_index] = classifier.predict_proba(x[test_index])[:, 1]
        folds.append(float(roc_auc_score(y[test_index], probability[test_index])))

    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    target_count = len(target_rows)
    target_z = z[:target_count]
    seed_z = z[target_count:]
    target_groups = groups[:target_count]
    target_role = role[:target_count]
    seed_role = role[target_count:]
    cross_distances: list[float] = []
    within_distances: list[float] = []
    for index, vector in enumerate(target_z):
        seed_mask = seed_role == target_role[index]
        cross_distances.append(float(np.linalg.norm(seed_z[seed_mask] - vector, axis=1).min()))
        within_mask = (target_role == target_role[index]) & (target_groups != target_groups[index])
        within_distances.append(float(np.linalg.norm(target_z[within_mask] - vector, axis=1).min()))
    cross_median = float(np.median(cross_distances))
    within_median = float(np.median(within_distances))
    return {
        "target_videos": len(target_rows),
        "seed_videos": len(seed_rows),
        "descriptor_dimensions": int(x.shape[1]),
        "domain_classifier_oof_auc": float(roc_auc_score(y, probability)),
        "fold_auc": folds,
        "median_target_to_seed_nn_distance": cross_median,
        "median_target_leave_pair_out_nn_distance": within_median,
        "nn_distance_ratio": cross_median / within_median if within_median > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("artifacts/datasets/fedpact_stage2_coverage_v2_20260920"),
    )
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    root = args.dataset_dir.resolve()
    target = read_jsonl(root / "manifest_target_test.jsonl")
    conditions = {
        "c_high": read_jsonl(root / "manifest_s_seed_c_high.jsonl"),
        "c_factor_separated": read_jsonl(root / "manifest_s_seed_c_factor_separated.jsonl"),
        "c_nosupport": read_jsonl(root / "manifest_s_seed_c_nosupport.jsonl"),
    }
    result = {
        "schema": "fedpact-stage2-visual-coverage-audit-v1",
        "descriptor": "fixed handcrafted RGB/HSV/histogram/edge/DCT descriptor",
        "interpretation_limit": (
            "A distributional manipulation check only; it does not prove semantic support "
            "or foundation-model coverage."
        ),
        "conditions": {
            name: audit_pair(target, seed_rows, args.seed)
            for name, seed_rows in conditions.items()
        },
    }
    output = root / "visual_coverage_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
