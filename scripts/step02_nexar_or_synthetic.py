"""
Step 2: NEXAR Collision Prediction を読み込み、各動画から num_frames 枚を抽出する。

抽出方式:
- frame_sampling: "event_window"（推奨）
    positive: time_of_event を最終フレームとし、frame_interval_sec 間隔で num_frames 枚（後方に遡る）
    negative: 動画内のランダム始点から frame_interval_sec 間隔で num_frames 枚
- frame_sampling: "alert_event"
    positive: time_of_alert と time_of_event の 2 フレーム (metadata.csv から取得)
    negative: positive の平均 alert/event 時刻の 2 フレーム
- frame_sampling: "paired_alert_event_quartiles"
    positive: time_of_alert から time_of_event までを 0, 1/3, 2/3, 1 で 4 枚
    negative: 対応する positive と同じ絶対秒を用いる
- frame_sampling: "uniform"
    動画全体から等間隔に num_frames 枚

NEXAR が見つからない場合は合成動画で fallback。
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import hashlib
import json
import random
from statistics import mean
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.config_loader import load_yaml, merged_config
from src.dataset_manifest import write_manifest
from src.paths import ensure_dirs
from src.run_context import init_run
from src.video_frames import (
    clamp_times_to_duration,
    extract_frames,
    extract_frames_at_times,
    video_frame_info,
    video_duration_sec,
)


CLASS_DIRS = [("negative", 0), ("positive", 1)]


def ensure_dummy_video(path: Path, frames: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 360, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed.")
    for t in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (t * 7) % 255, (t * 13) % 255, (t * 3) % 255
        cv2.putText(frame, f"t={t}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def read_metadata(meta_csv: Path) -> Dict[str, Dict[str, str]]:
    if not meta_csv.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with meta_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("file_name") or row.get("filename")
            if fn:
                out[fn] = row
    return out


def to_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def collect_videos_with_meta(
    root: Path,
    subdir: str,
    max_per_class: int,
    seed: int,
    *,
    weather_filter: str = "",
) -> Tuple[List[Tuple[Path, int, Dict]], Tuple[float, float]]:
    rng = random.Random(seed)
    items: List[Tuple[Path, int, Dict]] = []
    base = root / subdir
    if not base.exists():
        return items, (18.5, 19.7)

    pos_alerts: List[float] = []
    pos_events: List[float] = []
    weather_filter = str(weather_filter or "").strip()

    for name, label in CLASS_DIRS:
        cls_dir = base / name
        if not cls_dir.exists():
            continue
        meta = read_metadata(cls_dir / "metadata.csv")
        vids = sorted(cls_dir.glob("*.mp4"))
        rng.shuffle(vids)
        picked = 0
        for v in vids:
            if picked >= max_per_class:
                break
            m = meta.get(v.name, {})
            if weather_filter and str(m.get("weather", "") or "").strip() != weather_filter:
                continue
            items.append((v, label, m))
            picked += 1
            if label == 1:
                a = to_float(m.get("time_of_alert"))
                e = to_float(m.get("time_of_event"))
                if a is not None:
                    pos_alerts.append(a)
                if e is not None:
                    pos_events.append(e)
        if picked < max_per_class:
            print(
                f"warning: {subdir}/{name} only {picked}/{max_per_class} videos "
                f"(weather_filter={weather_filter!r})"
            )

    if pos_alerts:
        mean_alert = float(mean(pos_alerts))
    else:
        mean_alert = 18.5
    if pos_events:
        mean_event = float(mean(pos_events))
    else:
        mean_event = 19.7
    return items, (mean_alert, mean_event)


def times_for_video(
    label: int,
    meta: Dict,
    mean_alert: float,
    mean_event: float,
    fallback_alert: float,
    fallback_event: float,
) -> List[float]:
    if label == 1:
        a = to_float(meta.get("time_of_alert"))
        e = to_float(meta.get("time_of_event"))
        if a is None:
            a = fallback_alert
        if e is None:
            e = fallback_event
        return [a, e]
    return [mean_alert, mean_event]


def window_times_ending_at(
    end_sec: float, num_frames: int, interval_sec: float
) -> List[float]:
    """最終フレームを end_sec に置き、interval_sec 間隔で過去へ num_frames 枚。"""
    span = (num_frames - 1) * interval_sec
    start = end_sec - span
    return [start + i * interval_sec for i in range(num_frames)]


def window_times_random_start(
    duration_sec: float,
    num_frames: int,
    interval_sec: float,
    rng: random.Random,
) -> Tuple[List[float], float]:
    """ランダム始点から interval_sec 間隔で num_frames 枚。始点 [秒] も返す。"""
    span = (num_frames - 1) * interval_sec
    max_start = max(0.0, duration_sec - span - 1e-3)
    start = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
    times = [start + i * interval_sec for i in range(num_frames)]
    return times, start


def rng_for_video(global_seed: int, video_path: Path) -> random.Random:
    digest = hashlib.md5(f"{global_seed}:{video_path.name}".encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def alert_event_quartile_times(meta: Dict[str, Any]) -> List[float]:
    alert = to_float(meta.get("time_of_alert"))
    event = to_float(meta.get("time_of_event"))
    if alert is None or event is None:
        raise ValueError("paired sampling requires time_of_alert and time_of_event")
    if alert < 0.0 or event <= alert:
        raise ValueError(f"invalid alert/event interval: alert={alert}, event={event}")
    return [alert + (event - alert) * fraction for fraction in (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)]


def deterministic_sample_id(split_tag: str, video_path: Path, times: List[float]) -> str:
    raw = f"{split_tag}|{video_path.resolve()}|" + ",".join(f"{value:.6f}" for value in times)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_paired_sampling_plan(
    videos: List[Tuple[Path, int, Dict]],
    *,
    target_pairs: int,
) -> Dict[str, Dict[str, Any]]:
    positives = [(path, meta) for path, label, meta in videos if int(label) == 1]
    negatives = [
        (path, meta, video_duration_sec(path))
        for path, label, meta in videos
        if int(label) == 0
    ]
    plan: Dict[str, Dict[str, Any]] = {}
    unused_negatives = list(negatives)
    for positive_path, positive_meta in positives:
        if len(plan) // 2 >= target_pairs:
            break
        times = alert_event_quartile_times(positive_meta)
        eligible = [
            (duration, index)
            for index, (_, _, duration) in enumerate(unused_negatives)
            if duration > max(times)
        ]
        chosen_index = min(eligible)[1] if eligible else None
        if chosen_index is None:
            continue
        negative_path, _, _ = unused_negatives.pop(chosen_index)
        pair_id = hashlib.sha256(
            f"{positive_path.resolve()}|{negative_path.resolve()}".encode("utf-8")
        ).hexdigest()[:16]
        common = {
            "pair_id": pair_id,
            "paired_risk_times_sec": [float(value) for value in times],
            "sampling_rule": "risk_alert_event_quartiles_shared_absolute_seconds",
        }
        plan[str(positive_path.resolve())] = {**common, "times": times, "pair_role": "risk"}
        plan[str(negative_path.resolve())] = {**common, "times": times, "pair_role": "normal"}
    if len(plan) != 2 * target_pairs:
        raise RuntimeError(
            f"paired sampling produced {len(plan) // 2}/{target_pairs} feasible pairs"
        )
    return plan


def resolve_frame_times(
    *,
    vpath: Path,
    label: int,
    meta: Dict,
    cfg_data: Dict,
    mean_alert: float,
    mean_event: float,
    fallback_alert: float,
    fallback_event: float,
    seed: int,
) -> Tuple[List[float], Dict[str, float | str]]:
    sampling = str(cfg_data.get("frame_sampling", "alert_event"))
    num_frames = int(cfg_data.get("num_frames", 2))
    extra: Dict[str, float | str] = {"frame_sampling": sampling}

    if sampling == "event_window":
        interval = float(cfg_data.get("frame_interval_sec", 0.5))
        duration = video_duration_sec(vpath)
        extra["video_duration_sec"] = duration
        extra["frame_interval_sec"] = interval
        if label == 1:
            event = to_float(meta.get("time_of_event"))
            if event is None:
                event = fallback_event
            ts = window_times_ending_at(event, num_frames, interval)
            extra["anchor"] = "time_of_event"
            extra["anchor_sec"] = float(event)
        else:
            vid_rng = rng_for_video(seed, vpath)
            ts, start = window_times_random_start(duration, num_frames, interval, vid_rng)
            extra["anchor"] = "random_start"
            extra["anchor_sec"] = float(start)
        ts = clamp_times_to_duration(ts, duration)
        extra["frame_times_sec"] = ",".join(f"{t:.3f}" for t in ts)
        return ts, extra

    if sampling == "alert_event":
        ts = times_for_video(
            label, meta, mean_alert, mean_event, fallback_alert, fallback_event
        )
        ts = ts[:num_frames] if len(ts) > num_frames else ts
        extra["frame_times_sec"] = ",".join(f"{t:.3f}" for t in ts)
        return ts, extra

    # uniform: 等間隔の時刻列を返す（extract_frames が内部で index 計算するため空で渡す）
    extra["frame_times_sec"] = "uniform"
    return [], extra


def write_split(
    videos: List[Tuple[Path, int, Dict]],
    frame_root: Path,
    cfg_data: Dict,
    manifest_path: Path,
    split_tag: str,
    mean_alert: float,
    mean_event: float,
    seed: int,
) -> None:
    rows = []
    fallback_alert = float(cfg_data.get("fallback_alert_sec", 18.5))
    fallback_event = float(cfg_data.get("fallback_event_sec", 19.7))
    sampling = str(cfg_data.get("frame_sampling", "alert_event"))
    num_frames = int(cfg_data.get("num_frames", 2))
    max_side = int(cfg_data.get("frame_max_side", 720))
    target_pairs = int(
        cfg_data.get(
            "max_test_videos_per_class" if split_tag == "test" else "max_videos_per_class",
            0,
        )
    )
    paired_plan = (
        build_paired_sampling_plan(videos, target_pairs=target_pairs)
        if sampling == "paired_alert_event_quartiles"
        else {}
    )

    for vpath, label, meta in tqdm(videos, desc=f"frames[{split_tag}]"):
        if sampling == "paired_alert_event_quartiles" and str(vpath.resolve()) not in paired_plan:
            continue
        try:
            if sampling == "paired_alert_event_quartiles":
                pair = paired_plan[str(vpath.resolve())]
                ts = list(pair["times"])
                time_meta = {
                    "frame_sampling": sampling,
                    "frame_times_sec": ",".join(f"{t:.6f}" for t in ts),
                    "pair_id": str(pair["pair_id"]),
                    "pair_role": str(pair["pair_role"]),
                    "sampling_rule": str(pair["sampling_rule"]),
                    "paired_risk_times_sec": ",".join(
                        f"{t:.6f}" for t in pair["paired_risk_times_sec"]
                    ),
                }
            else:
                ts, time_meta = resolve_frame_times(
                    vpath=vpath,
                    label=label,
                    meta=meta,
                    cfg_data=cfg_data,
                    mean_alert=mean_alert,
                    mean_event=mean_event,
                    fallback_alert=fallback_alert,
                    fallback_event=fallback_event,
                    seed=seed,
                )
            if sampling == "uniform":
                frames = extract_frames(vpath, num_frames=num_frames, max_side=max_side)
            else:
                frames = extract_frames_at_times(vpath, ts, max_side=max_side)
        except Exception as e:
            print(f"skip {vpath}: {e}")
            continue

        total_frames, fps, duration = video_frame_info(vpath)
        frame_indices = [max(0, min(total_frames - 1, int(round(float(t) * fps)))) for t in ts]
        sid = deterministic_sample_id(split_tag, vpath, ts)
        out_dir = frame_root / split_tag / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, im in enumerate(frames):
            fp = out_dir / f"{i:03d}.jpg"
            im.save(fp, quality=88)
            paths.append(str(fp.resolve()))

        if len(paths) != num_frames:
            raise RuntimeError(f"expected {num_frames} frames, got {len(paths)} for {vpath}")

        rows.append(
            {
                "id": sid,
                "video_path": str(vpath),
                "frame_paths": paths,
                "label": int(label),
                "source": "nexar",
                "split": split_tag,
                "time_of_alert": meta.get("time_of_alert", "") or "",
                "time_of_event": meta.get("time_of_event", "") or "",
                "light_conditions": meta.get("light_conditions", "") or "",
                "weather": meta.get("weather", "") or "",
                "scene": meta.get("scene", "") or "",
                "source_video_sha256": sha256_path(vpath),
                "source_video_frame_count": total_frames,
                "source_video_fps": fps,
                "source_video_duration_sec": duration,
                "frame_indices": frame_indices,
                "frame_sha256": [sha256_path(Path(path)) for path in paths],
                **{k: v for k, v in time_meta.items() if isinstance(v, (str, int, float))},
            }
        )
    write_manifest(manifest_path, rows)
    print(f"[{split_tag}] wrote {manifest_path} samples={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="default.yaml に重ねる設定ファイル")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--run-suffix", default="")
    args = ap.parse_args()
    cfg = merged_config(load_yaml(args.config) if args.config else None)
    run_id, run_dir, log, env = init_run(
        cfg, step="2", step_dir="step02_data", log_name="prepare", run_id=args.run_id, run_suffix=args.run_suffix, cli=vars(args)
    )
    dcfg = cfg["data"]
    art = cfg["artifacts"]
    seed = int(cfg["train"]["seed"])

    ensure_dirs(Path(art["root"]))
    frame_root = Path(art["root"]) / "datasets" / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)

    train_manifest = Path(dcfg["manifest_path"])
    test_manifest = Path(dcfg["test_manifest_path"])
    train_manifest.parent.mkdir(parents=True, exist_ok=True)

    nexar_root = Path(dcfg["nexar_root"])
    weather_filter = str(dcfg.get("weather_filter", "") or "").strip()
    if nexar_root.exists():
        print(f"Using NEXAR at: {nexar_root}")
        if weather_filter:
            print(f"weather_filter={weather_filter!r}")
        candidate_multiplier = (
            int(dcfg.get("paired_candidate_multiplier", 3))
            if str(dcfg.get("frame_sampling", "")) == "paired_alert_event_quartiles"
            else 1
        )
        train_videos, (mean_alert, mean_event) = collect_videos_with_meta(
            nexar_root,
            str(dcfg["nexar_train_subdir"]),
            int(dcfg.get("max_videos_per_class", 40)) * candidate_multiplier,
            seed,
            weather_filter=weather_filter,
        )
        print(f"mean_alert={mean_alert:.3f}s, mean_event={mean_event:.3f}s")
        test_videos, _ = collect_videos_with_meta(
            nexar_root,
            str(dcfg["nexar_test_subdir"]),
            int(dcfg.get("max_test_videos_per_class", 20)) * candidate_multiplier,
            seed + 1,
            weather_filter=weather_filter,
        )
        if not train_videos:
            raise RuntimeError(f"No training videos under {nexar_root}/{dcfg['nexar_train_subdir']}")

        write_split(
            train_videos, frame_root, dcfg, train_manifest, "train",
            mean_alert, mean_event, seed,
        )
        test_pairing_status: Dict[str, Any] = {"status": "not_requested"}
        if test_videos:
            try:
                write_split(
                    test_videos, frame_root, dcfg, test_manifest, "test",
                    mean_alert, mean_event, seed + 1,
                )
                test_pairing_status = {"status": "completed", "manifest": str(test_manifest)}
            except RuntimeError as exc:
                if str(dcfg.get("frame_sampling", "")) != "paired_alert_event_quartiles":
                    raise
                print(f"warning: paired public-test manifest unavailable: {exc}")
                test_pairing_status = {"status": "unavailable", "reason": str(exc)}
        meta_summary = {
            "mean_alert_sec": mean_alert,
            "mean_event_sec": mean_event,
            "frames_per_sample": int(dcfg.get("num_frames", 2)),
            "use_num_frames": int(dcfg.get("use_num_frames") or 0),
            "use_frame_slice": str(dcfg.get("use_frame_slice", "last")),
            "frame_interval_sec": float(dcfg.get("frame_interval_sec", 0.5)),
            "frame_max_side": int(dcfg.get("frame_max_side", 720)),
            "sampling": str(dcfg.get("frame_sampling", "alert_event")),
            "weather_filter": weather_filter,
            "max_videos_per_class": int(dcfg.get("max_videos_per_class", 0)),
            "num_classes": int(dcfg["num_classes"]),
            "test_pairing": test_pairing_status,
        }
    else:
        print(f"NEXAR not found at {nexar_root}; falling back to synthetic videos.")
        synth_dir = Path(dcfg["synthetic_dir"])
        if not synth_dir.is_absolute():
            synth_dir = _ROOT / synth_dir
        synth_videos_dir = synth_dir / "videos"
        synth_videos_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        videos: list[tuple[Path, int, dict]] = []
        for i in range(4):
            v = synth_videos_dir / f"dummy_{i}.mp4"
            if not v.exists():
                ensure_dummy_video(v)
            videos.append((v, rng.randint(0, int(dcfg["num_classes"]) - 1), {}))
        write_split(videos, frame_root, dcfg, train_manifest, "train", 18.5, 19.7, seed)
        meta_summary = {"fallback": True}

    (train_manifest.parent / "manifest_meta.json").write_text(
        json.dumps(meta_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.save_summary(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "environment": env,
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
            "meta": meta_summary,
        }
    )
    log.log_meta({"status": "completed"})


if __name__ == "__main__":
    main()
