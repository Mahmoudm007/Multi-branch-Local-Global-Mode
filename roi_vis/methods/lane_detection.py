"""Classical lane-mark enhancement and masking."""

from __future__ import annotations

import cv2
import numpy as np

from config import ProjectConfig
from methods.color_spaces import get_color_spaces
from methods.texture_features import compute_gradient_magnitude
from utils.mask_utils import filter_components_by_area, morph_dilate, morph_open, to_mask_uint8



def detect_lane_mask(image: np.ndarray, road_mask: np.ndarray, cfg: ProjectConfig) -> np.ndarray:
    """Detect white and yellow lane paint inside the road ROI."""

    spaces = get_color_spaces(image)
    hsv = spaces['hsv']
    hls = spaces['hls']
    gray = spaces['gray']
    gradient = compute_gradient_magnitude(gray)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hls_l = hls[:, :, 1]
    white = (hls_l > cfg.thresholds.lane_white_l) & (sat < cfg.thresholds.lane_white_s) & (val > 150)
    yellow = (
        (hue >= cfg.thresholds.lane_yellow_h_low)
        & (hue <= cfg.thresholds.lane_yellow_h_high)
        & (sat > 70)
        & (val > 110)
    )
    grad_u8 = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(gray, 70, 160)
    grad_thr = np.percentile(grad_u8[road_mask > 0], 68) if np.any(road_mask) else 80
    candidate = (white | yellow) & ((grad_u8 > grad_thr) | (edges > 0)) & (road_mask > 0)
    candidate = morph_open(candidate, cfg.morphology.open_size)
    line_mask = np.zeros_like(road_mask)
    lines = cv2.HoughLinesP(
        to_mask_uint8(candidate),
        rho=1,
        theta=np.pi / 180.0,
        threshold=25,
        minLineLength=max(15, image.shape[1] // 40),
        maxLineGap=12,
    )
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                slope = 999.0
            else:
                slope = dy / float(dx)
            if abs(slope) < 0.15 or abs(slope) > 8.0:
                continue
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
    if np.any(line_mask):
        candidate = candidate & (morph_dilate(line_mask, 5) > 0)
    lane_mask = morph_dilate(candidate, 3)
    max_area = int(max(1, (road_mask > 0).sum()) * 0.18)
    lane_mask = filter_components_by_area(lane_mask, cfg.morphology.min_lane_area, max_area=max_area)
    return to_mask_uint8(lane_mask)
