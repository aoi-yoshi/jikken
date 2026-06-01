from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


class RunLogger:
    def __init__(self, run_dir: Path, name: str = "metrics") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []
        self.csv_path = self.run_dir / f"{name}.csv"
        self.jsonl_path = self.run_dir / f"{name}.jsonl"
        self.meta_path = self.run_dir / "run_meta.json"
        self._fieldnames: Optional[List[str]] = None

    def log_meta(self, meta: Mapping[str, Any]) -> None:
        payload = dict(meta)
        payload["saved_at"] = time.time()
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def log(self, row: Mapping[str, Any]) -> None:
        r = dict(row)
        r["t"] = time.time()
        self.rows.append(r)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if self._fieldnames is None:
            self._fieldnames = sorted(r.keys())
        else:
            for k in r.keys():
                if k not in self._fieldnames:
                    self._fieldnames.append(k)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({k: r.get(k, "") for k in self._fieldnames})

    def save_summary(self, summary: Mapping[str, Any]) -> None:
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(dict(summary), f, ensure_ascii=False, indent=2)
