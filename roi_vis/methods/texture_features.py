"""Texture and homogeneity feature maps."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from utils.image_utils import colorize_map, normalize_to_uint8, to_gray



def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude from Sobel derivatives."""

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)



def compute_local_variance(gray: np.ndarray, window: int = 9) -> np.ndarray:
    """Local variance using box filters."""

    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (window, window))
    mean_sq = cv2.blur(gray_f * gray_f, (window, window))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return variance



def compute_laplacian_energy(gray: np.ndarray) -> np.ndarray:
    """Absolute Laplacian response."""

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return np.abs(lap)



def compute_entropy_map(
    gray: np.ndarray,
    window: int = 9,
    bins: int = 16,
    downscale: float = 0.5,
) -> np.ndarray:
    """Approximate local entropy with a quantized generic filter."""

    if min(gray.shape[:2]) > 320:
        small = cv2.resize(gray, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    else:
        small = gray
    quantized = np.clip((small.astype(np.int32) * bins) // 256, 0, bins - 1)

    def entropy_fn(values):
        hist = np.bincount(values.astype(np.int32), minlength=bins).astype(np.float32)
        probs = hist / (hist.sum() + 1e-6)
        probs = probs[probs > 0]
        return float(-(probs * np.log2(probs)).sum())

    entropy_small = ndi.generic_filter(quantized, entropy_fn, size=window, mode='nearest')
    if small.shape != gray.shape:
        entropy = cv2.resize(
            entropy_small.astype(np.float32),
            (gray.shape[1], gray.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        entropy = entropy_small.astype(np.float32)
    return entropy



def compute_texture_bundle(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return colorized texture, entropy, and gradient maps."""

    gray = to_gray(image)
    variance = compute_local_variance(gray)
    lap_energy = compute_laplacian_energy(gray)
    gradient = compute_gradient_magnitude(gray)
    entropy = compute_entropy_map(gray)
    texture = 0.45 * normalize_to_uint8(variance) + 0.30 * normalize_to_uint8(lap_energy) + 0.25 * normalize_to_uint8(gradient)
    texture_uint8 = np.clip(texture, 0, 255).astype(np.uint8)
    entropy_uint8 = normalize_to_uint8(entropy)
    gradient_uint8 = normalize_to_uint8(gradient)
    return colorize_map(texture_uint8), colorize_map(entropy_uint8), colorize_map(gradient_uint8)
