"""Classical sky detection and removal."""

from __future__ import annotations

import cv2
import numpy as np

from config import ProjectConfig
from methods.color_spaces import blue_dominance, compute_chroma, get_color_spaces
from methods.horizon_detection import estimate_horizon_row
from methods.morphology_ops import refine_binary_mask
from methods.texture_features import compute_gradient_magnitude
from utils.image_utils import apply_binary_mask
from utils.mask_utils import border_connected, filter_components_by_area, to_mask_uint8



def detect_sky(image: np.ndarray, cfg: ProjectConfig) -> tuple[np.ndarray, int]:
    """Detect sky using top-region color and texture heuristics."""

    spaces = get_color_spaces(image)
    hsv = spaces['hsv']
    lab = spaces['lab']
    gray = spaces['gray']
    h, w = gray.shape
    horizon_row = estimate_horizon_row(image)
    blue_score = blue_dominance(image)
    chroma = compute_chroma(lab)
    gradient = compute_gradient_magnitude(gray)
    top_limit = min(int(h * cfg.thresholds.sky_top_ratio), horizon_row + int(0.12 * h))
    y_coords = np.arange(h).reshape(-1, 1)
    top_region = y_coords <= max(top_limit, int(0.26 * h))
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    l = lab[:, :, 0].astype(np.float32)
    low_texture_thr = np.percentile(gradient[: max(1, top_limit), :], cfg.thresholds.sky_low_texture_percentile)
    blue_sky = (blue_score > cfg.thresholds.sky_blue_margin) & (v > 90) & (s < 190)
    bright_overcast = (l > cfg.thresholds.sky_min_lightness) & (chroma < cfg.thresholds.sky_low_chroma) & (v > 95)
    low_texture = gradient < low_texture_thr
    candidate = top_region & ((blue_sky & low_texture) | (bright_overcast & low_texture) | blue_sky)
    candidate = border_connected(candidate, 'top')
    candidate = candidate & ~(y_coords > (horizon_row + int(0.10 * h)))
    sky_mask = refine_binary_mask(candidate, cfg.morphology.open_size, cfg.morphology.close_size)
    sky_mask = filter_components_by_area(sky_mask, cfg.morphology.min_component_area)
    sky_mask = to_mask_uint8(sky_mask)
    return sky_mask, int(horizon_row)



def remove_sky(image: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    """Zero out the sky region."""

    return apply_binary_mask(image, cv2.bitwise_not(sky_mask), fill_value=0)
