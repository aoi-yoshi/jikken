"""1 GPU + 複数 Flower client プロセス: fit/eval をファイルロックで直列化。"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        for line in lock_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("pid="):
                return int(line.split("=", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def _clear_stale_lock(lock_path: Path) -> bool:
    """ロック保持プロセスが死んでいればファイルを削除。削除できたら True。"""
    pid = _read_lock_pid(lock_path)
    if pid is not None and _pid_alive(pid):
        return False
    try:
        lock_path.unlink()
    except OSError:
        return False
    return True


@contextmanager
def gpu_file_lock(
    lock_path: Path,
    *,
    client_id: int = 0,
    timeout_sec: float = 86400,
) -> Iterator[None]:
    """同一 PC 上の client 0/1 が同時に fit しても、GPU 区間だけ 1 本ずつ実行。"""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(timeout_sec)
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, f"cid={client_id} pid={os.getpid()}\n".encode())
            break
        except FileExistsError:
            if _clear_stale_lock(lock_path):
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"GPU lock wait timeout ({timeout_sec}s): {lock_path}")
            time.sleep(2)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except OSError:
            pass
