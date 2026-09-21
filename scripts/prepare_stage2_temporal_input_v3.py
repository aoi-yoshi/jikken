"""Create an 8-frame temporal-input derivative of the fixed Stage-II split.

Pair membership and source videos are inherited from coverage split v2.  Only
the sampling density changes: both risk and its matched normal video use eight
uniform timestamps spanning the risk video's annotated alert-event interval.
The original v2 artifacts remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_manifest import read_manifest, write_manifest
from src.video_frames import extract_frames_at_times


DEFAULT_MANIFESTS = (
    "client_train",
    "gate_validation",
    "d_server_lab",
    "retention_test",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(path: Path, times: list[float]) -> str:
    raw = str(path.resolve()) + "|" + ",".join(f"{value:.6f}" for value in times)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def uniform_times(start: float, end: float, count: int) -> list[float]:
    if count < 2 or end <= start:
        raise ValueError(f"invalid temporal interval: start={start}, end={end}, count={count}")
    return [start + (end - start) * index / (count - 1) for index in range(count)]


def json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "artifacts" / "datasets" / "fedpact_stage2_coverage_v2_20260920",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "datasets" / "fedpact_stage2_temporal_v3_20260921",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=720)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    manifests = {
        name: read_manifest(source_dir / f"manifest_{name}.jsonl")
        for name in DEFAULT_MANIFESTS
    }
    rows_by_old_id: dict[str, dict[str, Any]] = {}
    risk_interval_by_pair: dict[str, tuple[float, float]] = {}
    for rows in manifests.values():
        for row in rows:
            rows_by_old_id.setdefault(str(row["id"]), row)
            if int(row["label"]) == 1:
                risk_interval_by_pair[str(row["pair_id"])] = (
                    float(row["time_of_alert"]),
                    float(row["time_of_event"]),
                )

    converted: dict[str, dict[str, Any]] = {}
    for number, (old_id, row) in enumerate(sorted(rows_by_old_id.items()), start=1):
        pair_id = str(row["pair_id"])
        if pair_id not in risk_interval_by_pair:
            raise RuntimeError(f"missing risk interval for pair {pair_id}")
        start, end = risk_interval_by_pair[pair_id]
        times = uniform_times(start, end, int(args.num_frames))
        video_path = Path(row["video_path"])
        frames = extract_frames_at_times(video_path, times, max_side=int(args.max_side))
        if len(frames) != int(args.num_frames):
            raise RuntimeError(f"expected {args.num_frames} frames: {video_path}")
        sid = sample_id(video_path, times)
        frame_dir = output_dir / "frames" / sid
        frame_dir.mkdir(parents=True, exist_ok=False)
        frame_paths: list[str] = []
        frame_hashes: list[str] = []
        for index, frame in enumerate(frames):
            frame_path = frame_dir / f"{index:03d}.jpg"
            frame.save(frame_path, quality=88)
            frame_paths.append(str(frame_path.resolve()))
            frame_hashes.append(file_hash(frame_path))
            frame.close()
        new_row = dict(row)
        new_row.update(
            {
                "id": sid,
                "parent_v2_id": old_id,
                "frame_paths": frame_paths,
                "frame_times_sec": times,
                "frame_indices": [
                    max(0, min(int(row["source_video_frame_count"]) - 1, round(t * float(row["source_video_fps"]))))
                    for t in times
                ],
                "frame_sha256": frame_hashes,
                "sampling_rule": f"risk_alert_event_uniform_{args.num_frames}_shared_absolute_seconds",
            }
        )
        converted[old_id] = new_row
        print(f"[{number}/{len(rows_by_old_id)}] {sid}", flush=True)

    output_counts: dict[str, int] = {}
    for name, rows in manifests.items():
        converted_rows = [converted[str(row["id"])] for row in rows]
        write_manifest(output_dir / f"manifest_{name}.jsonl", converted_rows)
        output_counts[name] = len(converted_rows)

    durations = [end - start for start, end in risk_interval_by_pair.values()]
    audit = {
        "schema": "fedpact-stage2-temporal-input-v3",
        "source_dir": str(source_dir),
        "num_frames": int(args.num_frames),
        "sampling_rule": f"risk_alert_event_uniform_{args.num_frames}_shared_absolute_seconds",
        "input_task": "alert-interval clip normal/risk classification; not early anticipation",
        "unique_videos": len(converted),
        "manifest_counts": output_counts,
        "pair_count": len(risk_interval_by_pair),
        "alert_interval_sec": {
            "min": min(durations),
            "mean": sum(durations) / len(durations),
            "max": max(durations),
        },
        "content_hash": json_hash(
            {
                key: {"id": value["id"], "frame_sha256": value["frame_sha256"]}
                for key, value in sorted(converted.items())
            }
        ),
    }
    (output_dir / "temporal_input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
