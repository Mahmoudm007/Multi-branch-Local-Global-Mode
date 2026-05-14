"""Dataset scanning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import ProjectConfig
from utils.io_utils import is_image_file
from utils.path_utils import normalize_name, relative_parent_under_root, resolve_defined_dir


@dataclass
class DatasetItem:
    """A single dataset image and its metadata."""

    input_path: Path
    split: str
    class_name: str
    relative_parent: Path



def _resolve_class_dirs(split_dir: Path, cfg: ProjectConfig) -> List[Path]:
    """Resolve configured class directories using tolerant aliases."""

    discovered = {normalize_name(path.name): path for path in split_dir.iterdir() if path.is_dir()}
    resolved: List[Path] = []
    for class_name in cfg.dataset.class_folder_names:
        aliases = cfg.dataset.class_aliases.get(class_name, [class_name])
        match = None
        for alias in aliases:
            match = discovered.get(normalize_name(alias))
            if match is not None:
                break
        if match is not None:
            resolved.append(match)
    return resolved



def scan_dataset(cfg: ProjectConfig, logger) -> tuple[Path, List[DatasetItem]]:
    """Scan the dataset and return mirrored metadata items."""

    defined_dir = resolve_defined_dir(cfg.dataset.input_root, cfg.dataset.defined_folder_name)
    dataset_root = defined_dir.parent
    items: List[DatasetItem] = []
    for split in cfg.dataset.splits:
        split_dir = defined_dir / split
        if not split_dir.exists():
            logger.warning('Missing split folder: %s', split_dir)
            continue
        class_dirs = _resolve_class_dirs(split_dir, cfg)
        if not class_dirs:
            logger.warning('No configured class folders found under: %s', split_dir)
        for class_dir in class_dirs:
            files = [
                path
                for path in sorted(class_dir.iterdir())
                if is_image_file(path, cfg.dataset.image_extensions)
            ]
            if not files:
                logger.warning('Empty class folder: %s', class_dir)
                continue
            for path in files:
                items.append(
                    DatasetItem(
                        input_path=path,
                        split=split,
                        class_name=class_dir.name,
                        relative_parent=relative_parent_under_root(path, dataset_root),
                    )
                )
                if cfg.max_images is not None and len(items) >= cfg.max_images:
                    return dataset_root, items
    return dataset_root, items
