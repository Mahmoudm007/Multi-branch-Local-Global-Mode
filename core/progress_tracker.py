from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(_json_sanitize(payload), indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


class CSVProgressTracker:
    """Append-only CSV progress file with a resumable completed-key index."""

    def __init__(self, path: Path, fieldnames: Iterable[str], key_field: str = "key") -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.key_field = key_field
        self._lock = Lock()
        ensure_dir(self.path.parent)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def append(self, row: dict[str, Any]) -> None:
        with self._lock:
            normalized = {field: row.get(field, "") for field in self.fieldnames}
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writerow(normalized)

    def rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def completed_keys(self, output_root: Path | None = None, output_field: str = "output_path") -> set[str]:
        completed: set[str] = set()
        for row in self.rows():
            if row.get("status") != "ok":
                continue
            key = row.get(self.key_field)
            if not key:
                continue
            if output_root is not None:
                rel = row.get(output_field, "")
                if rel:
                    candidate = output_root / rel
                    if not candidate.exists() or candidate.stat().st_size <= 0:
                        continue
            completed.add(key)
        return completed
