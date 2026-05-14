"""Filesystem and path helpers."""

from __future__ import annotations

import re
from pathlib import Path



def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path



def normalize_name(name: str) -> str:
    """Normalize names so underscores, spaces, and dashes match."""

    return re.sub(r'[^a-z0-9]+', '', name.lower())



def resolve_child_by_name(parent: Path, target_name: str) -> Path | None:
    """Resolve a direct child using tolerant name normalization."""

    if not parent.exists():
        return None
    direct = parent / target_name
    if direct.exists():
        return direct
    target_norm = normalize_name(target_name)
    for child in parent.iterdir():
        if normalize_name(child.name) == target_norm:
            return child
    return None



def resolve_defined_dir(input_root: Path, defined_folder: str) -> Path:
    """Resolve the configured defined-folder path."""

    resolved = resolve_child_by_name(input_root, defined_folder)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not resolve defined folder '{defined_folder}' under '{input_root}'."
        )
    return resolved



def relative_parent_under_root(file_path: Path, root: Path) -> Path:
    """Return the file parent relative to a root folder."""

    return file_path.relative_to(root).parent



def stem_with_prefix(prefix: str, file_path: Path) -> str:
    """Create a prefixed output filename stem."""

    return f'{prefix}_{file_path.stem}'
