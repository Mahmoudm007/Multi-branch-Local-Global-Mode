"""Convenience wrappers for morphology."""

from __future__ import annotations

import numpy as np

from utils.mask_utils import fill_holes, morph_close, morph_dilate, morph_erode, morph_open



def smooth_mask(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    """Apply a light open-close smoothing pass."""

    return morph_close(morph_open(mask, open_size), close_size)



def refine_binary_mask(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    """Refine a binary mask and fill holes."""

    return fill_holes(smooth_mask(mask, open_size, close_size))


__all__ = [
    'fill_holes',
    'morph_close',
    'morph_dilate',
    'morph_erode',
    'morph_open',
    'refine_binary_mask',
    'smooth_mask',
]
