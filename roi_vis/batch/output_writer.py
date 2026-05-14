"""Mirrored output-writing utilities."""

from __future__ import annotations

from pathlib import Path

from batch.prefix_naming import get_artifact_spec
from config import ProjectConfig
from utils.io_utils import write_image
from utils.path_utils import ensure_dir, stem_with_prefix



def choose_output_extension(input_ext: str, default_ext: str, file_type: str) -> str:
    """Choose a save extension for the artifact."""

    if file_type == 'mask':
        return '.png'
    if input_ext.lower() in {'.jpg', '.jpeg', '.png'}:
        return input_ext.lower()
    return default_ext



def write_artifact(
    cfg: ProjectConfig,
    relative_parent: Path,
    input_file: Path,
    artifact_key: str,
    data,
) -> Path:
    """Write a single artifact under a mirrored folder hierarchy."""

    spec = get_artifact_spec(artifact_key)
    ext = choose_output_extension(input_file.suffix, spec.default_ext, spec.file_type)
    out_dir = ensure_dir(cfg.dataset.output_root / spec.folder / relative_parent)
    out_path = out_dir / f"{stem_with_prefix(spec.prefix, input_file)}{ext}"
    write_image(
        out_path,
        data,
        jpeg_quality=cfg.jpeg_quality,
        png_compression=cfg.png_compression,
    )
    return out_path
