"""Color-space helpers used across the classical pipelines."""

from __future__ import annotations

import cv2
import numpy as np



def get_color_spaces(image: np.ndarray) -> dict[str, np.ndarray]:
    """Return the main color-space representations."""

    return {
        'hsv': cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        'hls': cv2.cvtColor(image, cv2.COLOR_BGR2HLS),
        'lab': cv2.cvtColor(image, cv2.COLOR_BGR2LAB),
        'gray': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    }



def compute_chroma(lab: np.ndarray) -> np.ndarray:
    """Compute Lab chroma."""

    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    return np.sqrt(a * a + b * b)



def compute_whiteness(lab: np.ndarray) -> np.ndarray:
    """Compute a simple whiteness score."""

    lightness = lab[:, :, 0].astype(np.float32)
    chroma = compute_chroma(lab)
    return lightness - 1.6 * chroma



def blue_dominance(image: np.ndarray) -> np.ndarray:
    """Compute blue dominance over the red and green channels."""

    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    return b - np.maximum(g, r)
