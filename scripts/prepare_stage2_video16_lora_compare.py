"""Prepare the fixed video16 split for the two-scope LoRA comparison.

The selected 16 client pairs come from the completed data-count preflight.
Server, confirmation, and retention rows remain disjoint from client training.
Every pair uses 16 uniform timestamps over the risk video's annotated
alert-to-event interval; the matched normal video uses the same timestamps.
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


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(path: Path, times: list[float]) -> str:
    value = str(path.resolve()) + "|" + ",".join(f"{item:.6f}" for item in times)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def uniform_times(start: float, end: float, count: int) -> list[float]:
    if end <= start or count < 2:
        raise ValueError(f"invalid interval: {start=}, {end=}, {count=}")
    return [start + (end - start) * index / (count - 1) for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "artifacts" / "datasets" / "fedpact_stage2_coverage_v2_20260920",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT
        / "artifacts"
        / "runs"
        / "fedpact_stage2"
        / "stage2_data_count_preflight_20260921_01"
        / "split.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "datasets" / "fedpact_stage2_video16_lora_20260921",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-side", type=int, default=720)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    train_pair_ids = set(selection["pair_ids_by_count"]["16"])

    master = read_manifest(source_dir / "manifest_master.jsonl")
    manifests = {
        "client_train": [row for row in master if str(row["pair_id"]) in train_pair_ids],
        "d_server_lab": read_manifest(source_dir / "manifest_d_server_lab.jsonl"),
        "confirmation": read_manifest(source_dir / "manifest_seed_a_target.jsonl"),
        "retention": read_manifest(source_dir / "manifest_retention_test.jsonl"),
    }
    pair_sets = {
        name: {str(row["pair_id"]) for row in rows} for name, rows in manifests.items()
    }
    overlap = {
        f"{left}__{right}": sorted(pair_sets[left] & pair_sets[right])
        for index, left in enumerate(manifests)
        for right in list(manifests)[index + 1 :]
    }
    if any(overlap.values()):
        raise RuntimeError(f"pair overlap detected: {overlap}")
    if len(manifests["client_train"]) != 32:
        raise RuntimeError("client_train must contain 16 complete pairs")

    all_rows = [row for rows in manifests.values() for row in rows]
    risk_interval_by_pair = {
        str(row["pair_id"]): (float(row["time_of_alert"]), float(row["time_of_event"]))
        for row in all_rows
        if int(row["label"]) == 1
    }
    converted: dict[str, dict[str, Any]] = {}
    unique_rows = {str(row["id"]): row for row in all_rows}
    for number, (old_id, row) in enumerate(sorted(unique_rows.items()), start=1):
        start, end = risk_interval_by_pair[str(row["pair_id"])]
        times = uniform_times(start, end, int(args.num_frames))
        video_path = Path(row["video_path"])
        frames = extract_frames_at_times(video_path, times, max_side=int(args.max_side))
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
        converted_row = dict(row)
        converted_row.update(
            {
                "id": sid,
                "parent_v2_id": old_id,
                "frame_paths": frame_paths,
                "frame_times_sec": times,
                "frame_indices": [
                    max(
                        0,
                        min(
                            int(row["source_video_frame_count"]) - 1,
                            round(value * float(row["source_video_fps"])),
                        ),
                    )
                    for value in times
                ],
                "frame_sha256": frame_hashes,
                "sampling_rule": "risk_alert_event_uniform_16_shared_absolute_seconds",
            }
        )
        converted[old_id] = converted_row
        print(f"[{number}/{len(unique_rows)}] {sid}", flush=True)

    counts = {}
    for name, rows in manifests.items():
        output_rows = [converted[str(row["id"])] for row in rows]
        write_manifest(output_dir / f"manifest_{name}.jsonl", output_rows)
        counts[name] = len(output_rows)
    audit = {
        "schema": "fedpact-stage2-video16-lora-compare",
        "num_frames": int(args.num_frames),
        "sampling_rule": "risk_alert_event_uniform_16_shared_absolute_seconds",
        "task": "alert-interval clip classification, not early prediction",
        "counts": counts,
        "pair_overlap": overlap,
        "selection_source": str(args.selection.resolve()),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
