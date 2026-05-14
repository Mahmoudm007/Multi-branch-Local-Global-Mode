"""Simple perspective helpers for an approximate road patch."""

from __future__ import annotations

import cv2
import numpy as np

from config import ProjectConfig



def estimate_road_trapezoid(
    image_shape: tuple[int, int],
    horizon_row: int,
    vanishing_point: tuple[int, int] | None,
    cfg: ProjectConfig,
) -> np.ndarray:
    """Estimate a conservative road trapezoid."""

    h, w = image_shape[:2]
    bottom_y = int(h * (1.0 - cfg.thresholds.road_bottom_exclusion_ratio))
    top_y = int(np.clip(horizon_row + 0.05 * h, 0.20 * h, bottom_y - 40))
    center_x = w // 2 if vanishing_point is None else int(np.clip(vanishing_point[0], 0.35 * w, 0.65 * w))
    bottom_half = int(0.5 * cfg.thresholds.corridor_bottom_width_ratio * w)
    top_half = int(0.5 * cfg.thresholds.corridor_top_width_ratio * w)
    polygon = np.array(
        [
            [center_x - top_half, top_y],
            [center_x + top_half, top_y],
            [center_x + bottom_half, bottom_y],
            [center_x - bottom_half, bottom_y],
        ],
        dtype=np.int32,
    )
    polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)
    return polygon



def polygon_mask(image_shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """Rasterize a polygon to a binary mask."""

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
    return mask



def warp_trapezoid_patch(
    image: np.ndarray,
    trapezoid: np.ndarray,
    output_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Warp a road trapezoid into a square patch."""

    width, height = output_size
    src = trapezoid.astype(np.float32)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, output_size)
