from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def _open(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    if n <= 0:
        cap.release()
        raise RuntimeError(f"No frames in video: {video_path}")
    return cap, n, fps


def _read_at(cap, frame_idx: int, max_side: int) -> Image.Image:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx) - 1))
        ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed reading frame {frame_idx}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return Image.fromarray(frame)


def extract_frames(
    video_path: Path, num_frames: int, max_side: int = 720
) -> List[Image.Image]:
    cap, n, _ = _open(video_path)
    idxs = np.linspace(0, n - 1, num=num_frames, dtype=int).tolist()
    frames: List[Image.Image] = []
    for i in idxs:
        try:
            frames.append(_read_at(cap, int(i), max_side))
        except Exception:
            continue
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to read frames: {video_path}")
    return frames


def video_duration_sec(video_path: Path) -> float:
    cap, n, fps = _open(video_path)
    cap.release()
    return float(n) / fps


def clamp_times_to_duration(times_sec: List[float], duration_sec: float) -> List[float]:
    end = max(0.0, duration_sec - 1e-3)
    return [max(0.0, min(end, float(t))) for t in times_sec]


def extract_frames_at_times(
    video_path: Path,
    times_sec: List[float],
    max_side: int = 720,
) -> List[Image.Image]:
    cap, n, fps = _open(video_path)
    frames: List[Image.Image] = []
    for t in times_sec:
        idx = int(round(float(t) * fps))
        idx = max(0, min(n - 1, idx))
        try:
            frames.append(_read_at(cap, idx, max_side))
        except Exception:
            continue
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to read frames at {times_sec}: {video_path}")
    return frames


def resize_frames(frames: List[Image.Image], size: Tuple[int, int]) -> List[Image.Image]:
    return [im.resize(size, Image.BICUBIC) for im in frames]
