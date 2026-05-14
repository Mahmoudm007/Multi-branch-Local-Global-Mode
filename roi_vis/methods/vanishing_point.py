"""Vanishing-point estimation from line intersections."""

from __future__ import annotations

import cv2
import numpy as np

from utils.image_utils import to_gray



def _line_intersection(line_a, line_b):
    x1, y1, x2, y2 = [float(v) for v in line_a]
    x3, y3, x4, y4 = [float(v) for v in line_b]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return px, py



def estimate_vanishing_point(
    image: np.ndarray,
    horizon_row: int | None = None,
) -> tuple[int, int]:
    """Estimate a simple vanishing point from dominant left/right road lines."""

    gray = cv2.GaussianBlur(to_gray(image), (5, 5), 0)
    h, w = gray.shape
    if horizon_row is None:
        horizon_row = int(0.38 * h)
    edges = cv2.Canny(gray, 70, 170)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=45,
        minLineLength=max(30, w // 12),
        maxLineGap=20,
    )
    left = []
    right = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            slope = dy / float(dx)
            length = float(np.hypot(dx, dy))
            if length < w * 0.08 or abs(slope) < 0.25 or abs(slope) > 6.0:
                continue
            y_mid = 0.5 * (y1 + y2)
            if y_mid < 0.18 * h or y_mid > 0.92 * h:
                continue
            if slope < 0:
                left.append((length, line))
            else:
                right.append((length, line))
    left = [line for _, line in sorted(left, key=lambda item: item[0], reverse=True)[:8]]
    right = [line for _, line in sorted(right, key=lambda item: item[0], reverse=True)[:8]]
    intersections = []
    for line_a in left:
        for line_b in right:
            point = _line_intersection(line_a, line_b)
            if point is None:
                continue
            x, y = point
            if -0.25 * w <= x <= 1.25 * w and 0 <= y <= 0.85 * h:
                intersections.append((x, y))
    if intersections:
        xs = np.asarray([point[0] for point in intersections], dtype=np.float32)
        ys = np.asarray([point[1] for point in intersections], dtype=np.float32)
        return int(np.median(xs)), int(np.median(ys))
    return w // 2, int(horizon_row)
