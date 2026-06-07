"""1 GPU + 複数 Flower client プロセス: fit/eval をファイルロックで直列化。"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


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
