"""Simple horizon estimation from lines and vertical gradients."""

from __future__ import annotations

import cv2
import numpy as np

from utils.image_utils import to_gray



def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = cumulative[-1] * 0.5
    return float(values[np.searchsorted(cumulative, cutoff)])



def estimate_horizon_row(image: np.ndarray) -> int:
    """Estimate a horizon row using Hough lines with a gradient fallback."""

    gray = cv2.GaussianBlur(to_gray(image), (5, 5), 0)
    h, w = gray.shape
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=70,
        minLineLength=max(40, w // 8),
        maxLineGap=25,
    )
    ys = []
    weights = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            slope = abs(dy / float(dx))
            length = float(np.hypot(dx, dy))
            y_mid = 0.5 * (y1 + y2)
            if slope < 0.18 and y_mid < 0.72 * h and length > w * 0.10:
                ys.append(y_mid)
                weights.append(length)
    if ys:
        row = int(_weighted_median(np.asarray(ys, dtype=np.float32), np.asarray(weights, dtype=np.float32)))
        return int(np.clip(row, int(0.18 * h), int(0.68 * h)))
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    row_grad = np.abs(sobel_y[:, w // 5 : 4 * w // 5]).mean(axis=1)
    kernel = np.ones(31, dtype=np.float32) / 31.0
    smooth = np.convolve(row_grad, kernel, mode='same')
    lo = int(0.18 * h)
    hi = int(0.68 * h)
    row = lo + int(np.argmax(smooth[lo:hi]))
    return int(np.clip(row, lo, hi))
