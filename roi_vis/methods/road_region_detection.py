"""Conservative road-region extraction using geometry and color consistency."""

from __future__ import annotations

import cv2
import numpy as np

from config import ProjectConfig
from methods.color_spaces import compute_chroma, get_color_spaces
from methods.ipm import estimate_road_trapezoid, polygon_mask
from methods.morphology_ops import refine_binary_mask
from methods.texture_features import compute_gradient_magnitude
from methods.vanishing_point import estimate_vanishing_point
from utils.image_utils import masked_median
from utils.mask_utils import connected_to_seed, filter_components_by_area, keep_largest_component, morph_dilate, to_mask_uint8



def estimate_hood_cut_row(gray: np.ndarray, cfg: ProjectConfig) -> int:
    """Estimate the top of the dashboard/hood from strong lower horizontal edges."""

    h, w = gray.shape
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    row_signal = grad_y[:, w // 8 : 7 * w // 8].mean(axis=1)
    smooth = np.convolve(row_signal, np.ones(21, dtype=np.float32) / 21.0, mode='same')
    lo = int(0.60 * h)
    hi = int(0.94 * h)
    fallback = int(h * (1.0 - cfg.thresholds.road_bottom_exclusion_ratio))
    band = smooth[lo:hi]
    if band.size == 0:
        return fallback
    threshold = float(np.percentile(band, 88))
    strong = np.where(band >= threshold)[0]
    if strong.size == 0:
        return fallback
    hood_row = lo + int(strong[0])
    return int(np.clip(hood_row, int(0.68 * h), fallback))



def detect_road_roi(
    image: np.ndarray,
    sky_mask: np.ndarray,
    horizon_row: int,
    cfg: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Detect a conservative road ROI connected to the lower-center corridor."""

    h, w = image.shape[:2]
    spaces = get_color_spaces(image)
    lab = spaces['lab']
    gray = spaces['gray']
    gradient = compute_gradient_magnitude(gray)
    chroma = compute_chroma(lab)
    hood_cut = max(int(0.68 * h), estimate_hood_cut_row(gray, cfg) - 10)
    vanishing_point = estimate_vanishing_point(image, horizon_row=horizon_row)
    trapezoid = estimate_road_trapezoid(image.shape, horizon_row, vanishing_point, cfg)
    trapezoid[2:, 1] = hood_cut
    prior = polygon_mask(image.shape, trapezoid)
    prior[sky_mask > 0] = 0

    seed_mask = np.zeros((h, w), dtype=np.uint8)
    x_half = max(8, int(0.5 * cfg.thresholds.road_seed_width_ratio * w))
    y0 = int(cfg.thresholds.road_seed_y0_ratio * h)
    y1 = min(hood_cut - 6, int(cfg.thresholds.road_seed_y1_ratio * h))
    if y1 <= y0:
        y0 = max(int(0.55 * h), hood_cut - 80)
        y1 = max(y0 + 12, hood_cut - 8)
    cx = int(np.clip(vanishing_point[0], 0.40 * w, 0.60 * w))
    seed_mask[y0:y1, max(0, cx - x_half) : min(w, cx + x_half)] = 255
    seed_mask = cv2.bitwise_and(seed_mask, prior)
    if not np.any(seed_mask):
        seed_mask[max(y0 - 20, 0) : y1, w // 2 - 10 : w // 2 + 10] = 255
        seed_mask = cv2.bitwise_and(seed_mask, prior)

    ref_l = masked_median(lab[:, :, 0], seed_mask)
    ref_a = masked_median(lab[:, :, 1], seed_mask)
    ref_b = masked_median(lab[:, :, 2], seed_mask)
    ref_chroma = masked_median(chroma, seed_mask)
    ref_gray = masked_median(gray, seed_mask)
    ab_distance = np.sqrt((lab[:, :, 1].astype(np.float32) - ref_a) ** 2 + (lab[:, :, 2].astype(np.float32) - ref_b) ** 2)
    l_distance = np.abs(lab[:, :, 0].astype(np.float32) - ref_l)
    chroma_distance = np.abs(chroma - ref_chroma)
    grad_limit = np.percentile(gradient[prior > 0], 76) if np.any(prior) else np.percentile(gradient, 76)
    gray_distance = np.abs(gray.astype(np.float32) - ref_gray)
    candidate = (
        (ab_distance < cfg.thresholds.road_ab_distance)
        | (chroma_distance < cfg.thresholds.road_ab_distance * 0.9)
        | ((gradient < grad_limit) & (gray_distance < cfg.thresholds.road_l_distance))
    )
    candidate = candidate & (l_distance < cfg.thresholds.road_l_distance) & (prior > 0)
    candidate = refine_binary_mask(candidate, cfg.morphology.open_size, cfg.morphology.close_size)
    candidate = connected_to_seed(candidate, seed_mask)
    if not np.any(candidate):
        candidate = connected_to_seed(prior, seed_mask)
    road_mask = cv2.bitwise_and(morph_dilate(candidate, 3), prior)
    road_mask = filter_components_by_area(road_mask, cfg.morphology.min_component_area)
    road_mask = keep_largest_component(road_mask)
    road_mask = to_mask_uint8(road_mask)
    road_mask[hood_cut:, :] = 0
    return road_mask, trapezoid, vanishing_point
