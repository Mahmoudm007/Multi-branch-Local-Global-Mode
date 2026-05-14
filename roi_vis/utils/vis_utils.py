"""Visualization helpers for overlays and contours."""

from __future__ import annotations

import cv2
import numpy as np

from utils.mask_utils import to_mask_uint8



def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay a colorized mask on an image."""

    out = image.copy()
    binary = mask > 0
    tint = np.zeros_like(out)
    tint[:, :] = np.array(color, dtype=np.uint8)
    blended = cv2.addWeighted(out, 1.0 - alpha, tint, alpha, 0)
    out[binary] = blended[binary]
    return out



def draw_mask_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw contours for a binary mask."""

    out = image.copy()
    contours, _ = cv2.findContours(to_mask_uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out



def draw_horizon(image: np.ndarray, horizon_row: int) -> np.ndarray:
    """Draw a horizon estimate."""

    out = image.copy()
    cv2.line(out, (0, int(horizon_row)), (out.shape[1] - 1, int(horizon_row)), (0, 255, 255), 2)
    return out



def draw_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a closed polygon."""

    out = image.copy()
    pts = polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], True, color, thickness)
    return out



def draw_point(
    image: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 5,
) -> np.ndarray:
    """Draw a point marker."""

    out = image.copy()
    cv2.circle(out, (int(point[0]), int(point[1])), radius, color, -1)
    return out
