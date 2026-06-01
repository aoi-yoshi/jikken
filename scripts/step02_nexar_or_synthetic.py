"""
Step 2: NEXAR Collision Prediction を読み込み、各動画から num_frames 枚を抽出する。

抽出方式:
- frame_sampling: "event_window"（推奨）
    positive: time_of_event を最終フレームとし、frame_interval_sec 間隔で num_frames 枚（後方に遡る）
    negative: 動画内のランダム始点から frame_interval_sec 間隔で num_frames 枚
- frame_sampling: "alert_event"
    positive: time_of_alert と time_of_event の 2 フレーム (metadata.csv から取得)
    negative: positive の平均 alert/event 時刻の 2 フレーム
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
import uuid
from statistics import mean
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.config_loader import merged_config
from src.dataset_manifest import write_manifest
from src.paths import ensure_dirs
from src.video_frames import (
    clamp_times_to_duration,
    extract_frames,
    extract_frames_at_times,
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

    for vpath, label, meta in tqdm(videos, desc=f"frames[{split_tag}]"):
        try:
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

        sid = uuid.uuid4().hex[:10]
        out_dir = frame_root / split_tag / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, im in enumerate(frames):
            fp = out_dir / f"{i:03d}.jpg"
            im.save(fp, quality=88)
            paths.append(str(fp.resolve()))

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
                **{k: v for k, v in time_meta.items() if isinstance(v, (str, int, float))},
            }
        )
    write_manifest(manifest_path, rows)
    print(f"[{split_tag}] wrote {manifest_path} samples={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = merged_config()
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
        train_videos, (mean_alert, mean_event) = collect_videos_with_meta(
            nexar_root,
            str(dcfg["nexar_train_subdir"]),
            int(dcfg.get("max_videos_per_class", 40)),
            seed,
            weather_filter=weather_filter,
        )
        print(f"mean_alert={mean_alert:.3f}s, mean_event={mean_event:.3f}s")
        test_videos, _ = collect_videos_with_meta(
            nexar_root,
            str(dcfg["nexar_test_subdir"]),
            int(dcfg.get("max_test_videos_per_class", 20)),
            seed + 1,
            weather_filter=weather_filter,
        )
        if not train_videos:
            raise RuntimeError(f"No training videos under {nexar_root}/{dcfg['nexar_train_subdir']}")

        write_split(
            train_videos, frame_root, dcfg, train_manifest, "train",
            mean_alert, mean_event, seed,
        )
        if test_videos:
            write_split(
                test_videos, frame_root, dcfg, test_manifest, "test",
                mean_alert, mean_event, seed + 1,
            )
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


if __name__ == "__main__":
    main()
