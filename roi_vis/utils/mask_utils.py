"""Binary mask helpers."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi



def to_mask_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean or integer mask to 0/255 uint8."""

    return np.where(mask > 0, 255, 0).astype(np.uint8)



def kernel(size: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """Build a morphology kernel."""

    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(shape, (size, size))



def morph_open(mask: np.ndarray, size: int) -> np.ndarray:
    """Open a binary mask."""

    return cv2.morphologyEx(to_mask_uint8(mask), cv2.MORPH_OPEN, kernel(size))



def morph_close(mask: np.ndarray, size: int) -> np.ndarray:
    """Close a binary mask."""

    return cv2.morphologyEx(to_mask_uint8(mask), cv2.MORPH_CLOSE, kernel(size))



def morph_dilate(mask: np.ndarray, size: int) -> np.ndarray:
    """Dilate a binary mask."""

    return cv2.dilate(to_mask_uint8(mask), kernel(size))



def morph_erode(mask: np.ndarray, size: int) -> np.ndarray:
    """Erode a binary mask."""

    return cv2.erode(to_mask_uint8(mask), kernel(size))



def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill binary holes."""

    return to_mask_uint8(ndi.binary_fill_holes(mask > 0))



def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(to_mask_uint8(mask), 8)
    if num_labels <= 1:
        return to_mask_uint8(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return to_mask_uint8(labels == largest_label)



def filter_components_by_area(
    mask: np.ndarray,
    min_area: int,
    max_area: int | None = None,
) -> np.ndarray:
    """Filter mask components by area."""

    src = to_mask_uint8(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, 8)
    out = np.zeros_like(src)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        out[labels == label] = 255
    return out



def border_connected(mask: np.ndarray, border: str = 'top') -> np.ndarray:
    """Keep components connected to a given image border."""

    src = to_mask_uint8(mask)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(src, 8)
    if num_labels <= 1:
        return src
    if border == 'top':
        border_labels = np.unique(labels[0, :])
    elif border == 'bottom':
        border_labels = np.unique(labels[-1, :])
    elif border == 'left':
        border_labels = np.unique(labels[:, 0])
    else:
        border_labels = np.unique(labels[:, -1])
    keep = [int(x) for x in border_labels if x != 0]
    return to_mask_uint8(np.isin(labels, keep))



def connected_to_seed(mask: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    """Keep components connected to a provided seed mask."""

    src = to_mask_uint8(mask)
    seeds = to_mask_uint8(seed_mask)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(src, 8)
    if num_labels <= 1:
        return src
    keep = np.unique(labels[seeds > 0])
    keep = [int(x) for x in keep if x != 0]
    if not keep:
        return np.zeros_like(src)
    return to_mask_uint8(np.isin(labels, keep))



def mask_ratio(mask: np.ndarray, region_mask: np.ndarray | None = None) -> float:
    """Compute the positive fraction of a binary mask."""

    binary = mask > 0
    if region_mask is None:
        return float(binary.mean())
    support = region_mask > 0
    denom = int(support.sum())
    if denom == 0:
        return 0.0
    return float((binary & support).sum() / denom)
