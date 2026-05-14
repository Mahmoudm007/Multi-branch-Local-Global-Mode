"""Non-road suppression helpers."""

from __future__ import annotations

import cv2
import numpy as np

from config import ProjectConfig
from methods.ipm import estimate_road_trapezoid, polygon_mask
from utils.image_utils import apply_binary_mask
from utils.mask_utils import morph_erode, to_mask_uint8



def suppress_nonroad_objects(
    image: np.ndarray,
    road_mask: np.ndarray,
    sky_mask: np.ndarray,
    horizon_row: int,
    vanishing_point: tuple[int, int],
    cfg: ProjectConfig,
    lane_mask: np.ndarray | None = None,
    glare_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the drivable corridor and suppress side clutter."""

    trapezoid = estimate_road_trapezoid(image.shape, horizon_row, vanishing_point, cfg)
    corridor = polygon_mask(image.shape, trapezoid)
    conservative_mask = cv2.bitwise_and(corridor, road_mask)
    conservative_mask[sky_mask > 0] = 0
    conservative_mask = morph_erode(conservative_mask, 3)
    conservative_mask = to_mask_uint8(conservative_mask)
    cleaned = apply_binary_mask(image, conservative_mask, fill_value=0)
    if lane_mask is not None and glare_mask is not None:
        suppress_mask = cv2.bitwise_or(lane_mask, glare_mask)
        suppress_mask = cv2.bitwise_and(suppress_mask, conservative_mask)
        if np.any(suppress_mask):
            cleaned = cv2.inpaint(cleaned, suppress_mask, 3, cv2.INPAINT_TELEA)
    return conservative_mask, cleaned
