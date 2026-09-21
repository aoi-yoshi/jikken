"""Build the pre-registered FedPACT Stage-II coverage split.

The split unit is a risk/normal video pair with identical environment metadata.
Frame times are derived from the risk alert-event interval and are reused for
the matched normal video.  Coverage conditions intentionally share nested
server-seed subsets; no server video is shared with client/evaluation pools.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_manifest import write_manifest
from src.video_frames import extract_frames_at_times, video_frame_info


TARGET = ("Normal", "Cloudy", "Highway")


def json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def environment(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(key, "") or "").strip() for key in (
        "light_conditions", "weather", "scene"
    ))  # type: ignore[return-value]


def read_candidates(dataset_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for class_dir, label in (("negative", 0), ("positive", 1)):
        folder = dataset_root / class_dir
        with (folder / "metadata.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            name = str(row.get("file_name", "") or "").strip()
            path = folder / name
            if not name or not path.exists():
                continue
            try:
                total_frames, fps, duration = video_frame_info(path)
            except Exception:
                continue
            item = dict(row)
            item.update(
                {
                    "label": label,
                    "video_path": str(path.resolve()),
                    "total_frames": int(total_frames),
                    "fps": float(fps),
                    "duration": float(duration),
                    "env": environment(row),
                }
            )
            candidates.append(item)
    return candidates


def read_blacklist_paths(manifest_path: Path | None) -> set[str]:
    if manifest_path is None or not manifest_path.exists():
        return set()
    paths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = str(row.get("video_path", "") or "").strip()
            if raw:
                paths.add(str(Path(raw).resolve()).casefold())
    return paths


def quartile_times(risk: dict[str, Any]) -> list[float] | None:
    alert = as_float(risk.get("time_of_alert"))
    event = as_float(risk.get("time_of_event"))
    if alert is None or event is None or alert < 0 or event <= alert:
        return None
    if float(risk["duration"]) <= event:
        return None
    return [alert + (event - alert) * fraction for fraction in (0.0, 1 / 3, 2 / 3, 1.0)]


def pair_candidates(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for item in candidates:
        grouped[item["env"]][int(item["label"])].append(item)

    all_pairs: list[dict[str, Any]] = []
    for env in sorted(grouped):
        env_seed = int(hashlib.sha256((str(seed) + "|" + "|".join(env)).encode()).hexdigest()[:16], 16)
        rng = random.Random(env_seed)
        normals = list(grouped[env][0])
        risks = list(grouped[env][1])
        rng.shuffle(normals)
        rng.shuffle(risks)
        for risk in risks:
            times = quartile_times(risk)
            if times is None:
                continue
            eligible = [index for index, normal in enumerate(normals) if float(normal["duration"]) > max(times)]
            if not eligible:
                continue
            normal = normals.pop(eligible[0])
            pair_id = hashlib.sha256(
                (str(risk["video_path"]) + "|" + str(normal["video_path"])).encode("utf-8")
            ).hexdigest()[:16]
            all_pairs.append(
                {
                    "pair_id": pair_id,
                    "environment": list(env),
                    "times_sec": [float(value) for value in times],
                    "risk": risk,
                    "normal": normal,
                }
            )
    return all_pairs


def classify_pair(pair: dict[str, Any]) -> str:
    light, weather, scene = pair["environment"]
    if (light, weather, scene) == TARGET:
        return "target"
    if light == "Normal" and weather == "Cloudy" and scene != "Highway":
        return "cloudy_non_highway"
    if light == "Normal" and weather != "Cloudy" and scene == "Highway":
        return "noncloudy_highway"
    if light == "Normal" and weather != "Cloudy" and scene != "Highway":
        return "neither"
    return "other"


def shuffled(values: Iterable[dict[str, Any]], seed: int, tag: str) -> list[dict[str, Any]]:
    out = list(values)
    rng = random.Random(int(hashlib.sha256(f"{seed}|{tag}".encode()).hexdigest()[:16], 16))
    rng.shuffle(out)
    return out


def choose_retention(
    available: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    by_env: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in available:
        env = tuple(pair["environment"])
        if env != TARGET:
            by_env[env].append(pair)
    for env in by_env:
        by_env[env] = shuffled(by_env[env], seed, "retention|" + "|".join(env))
    envs = sorted(by_env, key=lambda env: (-len(by_env[env]), env))
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        changed = False
        for env in envs:
            if by_env[env] and len(selected) < count:
                selected.append(by_env[env].pop())
                changed = True
        if not changed:
            break
    if len(selected) != count:
        raise RuntimeError(f"retention pool shortfall: {len(selected)}/{count}")
    return selected


def allocate_pairs(all_pairs: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in all_pairs:
        groups[classify_pair(pair)].append(pair)
    target = shuffled(groups["target"], seed, "target")
    cloudy = shuffled(groups["cloudy_non_highway"], seed, "cloudy_non_highway")
    highway = shuffled(groups["noncloudy_highway"], seed, "noncloudy_highway")
    neither = shuffled(groups["neither"], seed, "neither")
    required_target = 8 + 8 + 20 + 15
    if len(target) < required_target:
        raise RuntimeError(f"target pool shortfall: {len(target)}/{required_target}")
    if len(cloudy) < 30:
        raise RuntimeError(f"cloudy/non-highway pool shortfall: {len(cloudy)}/30")
    if len(highway) < 30:
        raise RuntimeError(f"non-cloudy/highway pool shortfall: {len(highway)}/30")
    if len(neither) < 15:
        raise RuntimeError(f"neither pool shortfall: {len(neither)}/15")

    pools = {
        "client_train": target[:8],
        "gate_validation": target[8:16],
        "target_test": target[16:36],
        "seed_a_target": target[36:51],
        "dserver_lab_b_cloudy": cloudy[:15],
        "seed_b_cloudy": cloudy[15:30],
        "dserver_lab_c_highway": highway[:15],
        "seed_c_highway": highway[15:30],
        "seed_d_neither": neither[:15],
    }
    reserved = {
        role["video_path"]
        for pairs in pools.values()
        for pair in pairs
        for role in (pair["risk"], pair["normal"])
    }
    retention_candidates = [
        pair
        for pair in all_pairs
        if all(role["video_path"] not in reserved for role in (pair["risk"], pair["normal"]))
    ]
    pools["retention_test"] = choose_retention(retention_candidates, 20, seed)
    pools["d_server_lab"] = pools["dserver_lab_b_cloudy"] + pools["dserver_lab_c_highway"]
    pools["s_seed_c_high"] = pools["seed_a_target"] + pools["seed_d_neither"]
    pools["s_seed_c_factor_separated"] = pools["seed_b_cloudy"] + pools["seed_c_highway"]
    pools["s_seed_c_nosupport"] = pools["seed_b_cloudy"] + pools["seed_d_neither"]
    return pools


def pair_public(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "environment": pair["environment"],
        "times_sec": pair["times_sec"],
        "risk_video": pair["risk"]["video_path"],
        "normal_video": pair["normal"]["video_path"],
    }


def audit_allocations(pools: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    fixed_names = [
        "client_train", "gate_validation", "target_test", "retention_test",
        "seed_a_target", "dserver_lab_b_cloudy", "seed_b_cloudy",
        "dserver_lab_c_highway", "seed_c_highway", "seed_d_neither",
    ]
    owner: dict[str, str] = {}
    collisions: list[dict[str, str]] = []
    for name in fixed_names:
        for pair in pools[name]:
            if tuple(pair["risk"]["env"]) != tuple(pair["normal"]["env"]):
                raise RuntimeError(f"environment mismatch in pair {pair['pair_id']}")
            for role in (pair["risk"], pair["normal"]):
                path = str(role["video_path"])
                if path in owner:
                    collisions.append({"video": path, "first": owner[path], "second": name})
                else:
                    owner[path] = name
    if collisions:
        raise RuntimeError(f"unexpected source overlap: {collisions[:3]}")
    expected = {
        "s_seed_c_high": (30, 15, 15, 15),
        "s_seed_c_factor_separated": (30, 0, 15, 15),
        "s_seed_c_nosupport": (30, 0, 15, 0),
    }
    coverage: dict[str, Any] = {}
    for name, (pair_count, target_count, cloudy_count, highway_count) in expected.items():
        pairs = pools[name]
        actual_target = sum(tuple(pair["environment"]) == TARGET for pair in pairs)
        actual_cloudy = sum(pair["environment"][1] == "Cloudy" for pair in pairs)
        actual_highway = sum(pair["environment"][2] == "Highway" for pair in pairs)
        if (
            len(pairs) != pair_count
            or actual_target != target_count
            or actual_cloudy != cloudy_count
            or actual_highway != highway_count
        ):
            raise RuntimeError(f"coverage contract failed for {name}")
        coverage[name] = {
            "pairs": len(pairs),
            "target_pairs": actual_target,
            "cloudy_pairs": actual_cloudy,
            "highway_pairs": actual_highway,
        }
    return {
        "status": "pass",
        "fixed_pool_unique_videos": len(owner),
        "fixed_pool_overlap_count": len(collisions),
        "coverage": coverage,
        "intentional_server_condition_nesting": True,
    }


def sample_id(path: Path, times: list[float]) -> str:
    raw = str(path.resolve()) + "|" + ",".join(f"{value:.6f}" for value in times)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_rows(
    selected_pairs: list[dict[str, Any]],
    output_dir: Path,
    max_side: int,
    prior_content_hashes: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    rows_by_pair: dict[str, list[dict[str, Any]]] = {}
    content_owner: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for number, pair in enumerate(selected_pairs, start=1):
        pair_rows: list[dict[str, Any]] = []
        times = list(pair["times_sec"])
        for role_name, label in (("normal", 0), ("risk", 1)):
            item = pair[role_name]
            path = Path(item["video_path"])
            frames = extract_frames_at_times(path, times, max_side=max_side)
            if len(frames) != 4:
                raise RuntimeError(f"expected four frames: {path}")
            sid = sample_id(path, times)
            frame_dir = output_dir / "frames" / sid
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_paths: list[str] = []
            frame_hashes: list[str] = []
            for index, frame in enumerate(frames):
                frame_path = frame_dir / f"{index:03d}.jpg"
                frame.save(frame_path, quality=88)
                frame_paths.append(str(frame_path.resolve()))
                frame_hashes.append(file_hash(frame_path))
            source_hash = file_hash(path)
            if source_hash in prior_content_hashes:
                raise RuntimeError(f"selected video content was used by the initial checkpoint: {path}")
            prior = content_owner.get(source_hash)
            if prior and prior != str(path.resolve()):
                duplicates[str(path.resolve())] = prior
            content_owner[source_hash] = str(path.resolve())
            indices = [max(0, min(int(item["total_frames"]) - 1, round(t * float(item["fps"])))) for t in times]
            row = {
                "id": sid,
                "pair_id": pair["pair_id"],
                "pair_role": role_name,
                "video_path": str(path.resolve()),
                "frame_paths": frame_paths,
                "label": label,
                "source": "nexar",
                "sampling_rule": "risk_alert_event_quartiles_shared_absolute_seconds",
                "frame_times_sec": times,
                "frame_indices": indices,
                "time_of_alert": item.get("time_of_alert", "") or "",
                "time_of_event": item.get("time_of_event", "") or "",
                "light_conditions": pair["environment"][0],
                "weather": pair["environment"][1],
                "scene": pair["environment"][2],
                "source_video_sha256": source_hash,
                "source_video_frame_count": int(item["total_frames"]),
                "source_video_fps": float(item["fps"]),
                "source_video_duration_sec": float(item["duration"]),
                "frame_sha256": frame_hashes,
            }
            pair_rows.append(row)
        rows_by_pair[pair["pair_id"]] = pair_rows
        print(f"[{number}/{len(selected_pairs)}] {pair['pair_id']}")
    if duplicates:
        raise RuntimeError(f"duplicate video content across selected paths: {duplicates}")
    return rows_by_pair, content_owner


def ordered_unique_pairs(pools: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in (
        "client_train", "gate_validation", "target_test", "retention_test",
        "seed_a_target", "dserver_lab_b_cloudy", "seed_b_cloudy",
        "dserver_lab_c_highway", "seed_c_highway", "seed_d_neither",
    ):
        for pair in pools[name]:
            if pair["pair_id"] not in seen:
                result.append(pair)
                seen.add(pair["pair_id"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"C:\nexar_collision_prediction\train"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "datasets" / "fedpact_stage2_coverage_v2_20260920",
    )
    parser.add_argument(
        "--prior-manifest",
        type=Path,
        default=ROOT / "artifacts" / "datasets" / "manifest.jsonl",
        help="All source videos in this historical manifest are excluded conservatively.",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    blacklist_paths = read_blacklist_paths(args.prior_manifest.resolve())
    candidates_all = read_candidates(args.dataset_root.resolve())
    candidates = [
        item for item in candidates_all if str(Path(item["video_path"]).resolve()).casefold() not in blacklist_paths
    ]
    all_pairs = pair_candidates(candidates, args.seed)
    pools = allocate_pairs(all_pairs, args.seed)
    audit = audit_allocations(pools)
    plan = {
        "schema": "fedpact-stage2-coverage-split-v2",
        "seed": args.seed,
        "target_environment": list(TARGET),
        "dataset_root": str(args.dataset_root.resolve()),
        "candidate_videos_readable_before_blacklist": len(candidates_all),
        "prior_manifest_video_blacklist_count": len(blacklist_paths),
        "candidate_videos_after_blacklist": len(candidates),
        "feasible_matched_pairs": len(all_pairs),
        "pool_pairs": {name: [pair_public(pair) for pair in values] for name, values in pools.items()},
        "audit": audit,
    }
    plan["allocation_sha256"] = json_hash(plan["pool_pairs"])
    plan_path = output_dir / "split_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "plan": str(plan_path),
        "allocation_sha256": plan["allocation_sha256"],
        "pair_counts": {name: len(values) for name, values in pools.items()},
        "audit": audit,
    }, ensure_ascii=False, indent=2))
    if args.plan_only:
        return

    prior_content_hashes = {
        file_hash(Path(path))
        for path in sorted(blacklist_paths)
        if Path(path).exists()
    }
    selected = ordered_unique_pairs(pools)
    rows_by_pair, content_owner = build_rows(
        selected, output_dir, args.max_side, prior_content_hashes
    )
    manifests: dict[str, str] = {}
    for name, pairs in pools.items():
        rows = [row for pair in pairs for row in rows_by_pair[pair["pair_id"]]]
        manifest_path = output_dir / f"manifest_{name}.jsonl"
        write_manifest(manifest_path, rows)
        manifests[name] = str(manifest_path)
    all_rows = [row for pair in selected for row in rows_by_pair[pair["pair_id"]]]
    master_path = output_dir / "manifest_master.jsonl"
    write_manifest(master_path, all_rows)
    manifests["master"] = str(master_path)

    g0 = {
        **audit,
        "status": "pass",
        "selected_unique_pairs": len(selected),
        "selected_unique_videos": len(content_owner),
        "source_content_hash_count": len(content_owner),
        "prior_checkpoint_content_hash_blacklist_count": len(prior_content_hashes),
        "risk_frames_inside_alert_event_rate": 1.0,
        "missing_frame_count": 0,
        "manifests": manifests,
        "allocation_sha256": plan["allocation_sha256"],
        "master_manifest_sha256": file_hash(master_path),
    }
    (output_dir / "g0_audit.json").write_text(
        json.dumps(g0, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(g0, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
