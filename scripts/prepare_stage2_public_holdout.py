"""Prepare the untouched public-test holdout for frozen local adaptation.

The holdout contains only Normal-light, Cloudy, Highway videos.  Each positive
video is paired with one unused negative video, and both receive the positive
video's four alert-event quartile timestamps.  No model output is inspected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_manifest import read_manifest, write_manifest
from src.video_frames import extract_frames_at_times, video_frame_info


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_class(root: Path, class_name: str, label: int) -> list[dict[str, Any]]:
    folder = root / class_name
    with (folder / "metadata.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    output: list[dict[str, Any]] = []
    for row in metadata:
        if not (
            str(row.get("light_conditions", "")).strip() == "Normal"
            and str(row.get("weather", "")).strip() == "Cloudy"
            and str(row.get("scene", "")).strip() == "Highway"
        ):
            continue
        path = folder / str(row["file_name"])
        total_frames, fps, duration = video_frame_info(path)
        item = dict(row)
        item.update(
            {
                "label": label,
                "video_path": str(path.resolve()),
                "source_video_frame_count": total_frames,
                "source_video_fps": fps,
                "source_video_duration_sec": duration,
            }
        )
        output.append(item)
    return output


def quartiles(start: float, end: float) -> list[float]:
    if end <= start:
        raise ValueError(f"invalid alert-event interval: {start}, {end}")
    return [start + (end - start) * index / 3 for index in range(4)]


def sample_id(path: Path, times: list[float]) -> str:
    raw = str(path.resolve()) + "|" + ",".join(f"{value:.6f}" for value in times)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=Path(r"C:\nexar_collision_prediction\test-public")
    )
    parser.add_argument(
        "--training-master",
        type=Path,
        default=ROOT / "artifacts" / "datasets" / "fedpact_stage2_coverage_v2_20260920" / "manifest_master.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "datasets" / "fedpact_stage2_public_holdout_20260921",
    )
    parser.add_argument("--max-side", type=int, default=720)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    negatives = read_class(args.dataset_root.resolve(), "negative", 0)
    positives = read_class(args.dataset_root.resolve(), "positive", 1)
    positives.sort(key=lambda row: str(row["file_name"]))
    available_negatives = sorted(negatives, key=lambda row: str(row["file_name"]))
    if len(positives) != 19 or len(available_negatives) < len(positives):
        raise RuntimeError(f"unexpected target counts: negative={len(negatives)}, positive={len(positives)}")

    training_hashes = {
        str(row["source_video_sha256"])
        for row in read_manifest(args.training_master.resolve())
    }
    rows: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    pair_records: list[dict[str, Any]] = []
    for pair_index, risk in enumerate(positives):
        alert = float(risk["time_of_alert"])
        event = float(risk["time_of_event"])
        times = quartiles(alert, event)
        eligible = [
            normal for normal in available_negatives
            if float(normal["source_video_duration_sec"]) >= event
        ]
        if not eligible:
            raise RuntimeError(f"no negative video long enough for {risk['file_name']} at {event}s")
        normal = min(
            eligible,
            key=lambda item: (
                abs(float(item["source_video_duration_sec"]) - float(risk["source_video_duration_sec"])),
                str(item["file_name"]),
            ),
        )
        available_negatives.remove(normal)
        pair_id = f"public-cloudy-highway-{pair_index:02d}"
        pair_records.append(
            {
                "pair_id": pair_id,
                "normal_file": normal["file_name"],
                "risk_file": risk["file_name"],
                "times_sec": times,
            }
        )
        for role, item in (("normal", normal), ("risk", risk)):
            path = Path(item["video_path"])
            source_hash = file_hash(path)
            if source_hash in training_hashes:
                raise RuntimeError(f"public holdout overlaps training content: {path}")
            if source_hash in selected_hashes:
                raise RuntimeError(f"duplicate public holdout content: {path}")
            selected_hashes.add(source_hash)
            frames = extract_frames_at_times(path, times, max_side=int(args.max_side))
            sid = sample_id(path, times)
            frame_dir = output_dir / "frames" / sid
            frame_dir.mkdir(parents=True, exist_ok=False)
            frame_paths: list[str] = []
            frame_hashes: list[str] = []
            for frame_index, frame in enumerate(frames):
                frame_path = frame_dir / f"{frame_index:03d}.jpg"
                frame.save(frame_path, quality=88)
                frame_paths.append(str(frame_path.resolve()))
                frame_hashes.append(file_hash(frame_path))
                frame.close()
            rows.append(
                {
                    "id": sid,
                    "pair_id": pair_id,
                    "pair_role": role,
                    "video_path": str(path.resolve()),
                    "frame_paths": frame_paths,
                    "label": int(item["label"]),
                    "source": "nexar_test_public",
                    "sampling_rule": "risk_alert_event_quartiles_shared_absolute_seconds",
                    "frame_times_sec": times,
                    "frame_indices": [
                        max(0, min(int(item["source_video_frame_count"]) - 1, round(t * float(item["source_video_fps"]))))
                        for t in times
                    ],
                    "time_of_alert": risk["time_of_alert"],
                    "time_of_event": risk["time_of_event"],
                    "light_conditions": "Normal",
                    "weather": "Cloudy",
                    "scene": "Highway",
                    "source_video_sha256": source_hash,
                    "source_video_frame_count": int(item["source_video_frame_count"]),
                    "source_video_fps": float(item["source_video_fps"]),
                    "source_video_duration_sec": float(item["source_video_duration_sec"]),
                    "frame_sha256": frame_hashes,
                }
            )

    write_manifest(output_dir / "manifest_public_holdout.jsonl", rows)
    audit = {
        "schema": "fedpact-stage2-public-holdout-v1",
        "status": "pass",
        "task_contract": "alert_interval_clip_classification_not_early_anticipation",
        "environment": ["Normal", "Cloudy", "Highway"],
        "pair_count": len(pair_records),
        "video_count": len(rows),
        "class_counts": {
            "normal": sum(int(row["label"]) == 0 for row in rows),
            "risk": sum(int(row["label"]) == 1 for row in rows),
        },
        "training_content_overlap": 0,
        "within_holdout_content_duplicates": 0,
        "pairs": pair_records,
    }
    (output_dir / "holdout_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
