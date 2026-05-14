"""Image I/O helpers that work well with Windows paths."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.path_utils import ensure_dir



def is_image_file(path: Path, extensions: tuple[str, ...]) -> bool:
    """Return True when the file extension is a supported image type."""

    return path.is_file() and path.suffix.lower() in extensions



def read_image(path: Path) -> np.ndarray | None:
    """Read an image using imdecode so unicode paths stay safe."""

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)



def write_image(
    path: Path,
    image: np.ndarray,
    jpeg_quality: int = 95,
    png_compression: int = 3,
) -> None:
    """Write an image using imencode."""

    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix in {'.jpg', '.jpeg'}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif suffix == '.png':
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
    else:
        params = []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise IOError(f"Failed to encode image for '{path}'.")
    encoded.tofile(str(path))
