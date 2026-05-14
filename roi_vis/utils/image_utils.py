"""Image manipulation helpers."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np



def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an image to width, height."""

    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)



def gray_world_normalize(image: np.ndarray) -> np.ndarray:
    """Apply a mild gray-world white balance."""

    img = image.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    scale = means.mean() / means
    balanced = np.clip(img * scale, 0, 255).astype(np.uint8)
    return balanced



def resize_and_normalize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize then normalize color slightly."""

    resized = resize_image(image, size)
    return gray_world_normalize(resized)



def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a numeric map to uint8."""

    arr = arr.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)



def apply_binary_mask(
    image: np.ndarray,
    mask: np.ndarray,
    fill_value: int | tuple[int, int, int] = 0,
) -> np.ndarray:
    """Apply a binary mask to a color image."""

    out = image.copy()
    if isinstance(fill_value, int):
        fill = np.array([fill_value, fill_value, fill_value], dtype=np.uint8)
    else:
        fill = np.array(fill_value, dtype=np.uint8)
    out[mask == 0] = fill
    return out



def colorize_map(gray_map: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Convert a single-channel map to a color heatmap."""

    return cv2.applyColorMap(gray_map, colormap)



def masked_median(values: np.ndarray, mask: np.ndarray) -> float:
    """Median under a binary mask."""

    pixels = values[mask > 0]
    if pixels.size == 0:
        return float(np.median(values))
    return float(np.median(pixels))



def masked_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    """Percentile under a binary mask."""

    pixels = values[mask > 0]
    if pixels.size == 0:
        return float(np.percentile(values, percentile))
    return float(np.percentile(pixels, percentile))



def blend_images(images: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    """Blend multiple images using normalized weights."""

    images = list(images)
    weights = np.asarray(list(weights), dtype=np.float32)
    weights /= weights.sum() + 1e-6
    acc = np.zeros_like(images[0], dtype=np.float32)
    for image, weight in zip(images, weights):
        acc += image.astype(np.float32) * weight
    return np.clip(acc, 0, 255).astype(np.uint8)
