"""Summary CSV generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from utils.path_utils import ensure_dir


SUMMARY_FIELDS = [
    'image_path',
    'split',
    'class',
    'sky_ratio',
    'road_ratio',
    'lane_ratio',
    'glare_ratio',
    'snow_ratio_within_road',
    'warnings',
]



def _normalize_row(row: dict) -> dict:
    return {field: row.get(field, '') for field in SUMMARY_FIELDS}



def read_summary_rows(path: Path) -> list[dict]:
    """Read summary rows from CSV if it exists."""

    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get('image_path'):
                rows.append(_normalize_row(row))
    return rows



def append_summary_row(path: Path, row: dict) -> None:
    """Append a single summary row, creating the file if needed."""

    ensure_dir(path.parent)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(_normalize_row(row))



def write_summary_csv(path: Path, rows: Iterable[dict]) -> None:
    """Write summary rows to CSV."""

    ensure_dir(path.parent)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))
