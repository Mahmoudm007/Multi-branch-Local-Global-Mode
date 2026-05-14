"""Snow candidate extraction inside the road ROI."""

from __future__ import annotations

import numpy as np

from config import ProjectConfig
from methods.color_spaces import compute_chroma, get_color_spaces
from methods.morphology_ops import refine_binary_mask
from methods.texture_features import compute_gradient_magnitude, compute_local_variance
from utils.mask_utils import filter_components_by_area, to_mask_uint8



def detect_snow_candidates(
    image: np.ndarray,
    road_mask: np.ndarray,
    sky_mask: np.ndarray,
    lane_mask: np.ndarray,
    glare_mask: np.ndarray,
    cfg: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect raw and refined snow candidates inside the road ROI."""

    spaces = get_color_spaces(image)
    hsv = spaces['hsv']
    lab = spaces['lab']
    gray = spaces['gray']
    chroma = compute_chroma(lab)
    gradient = compute_gradient_magnitude(gray)
    variance = compute_local_variance(gray)
    road_base = (road_mask > 0) & (sky_mask == 0) & (lane_mask == 0) & (glare_mask == 0)
    if np.any(road_base):
        l_threshold = max(cfg.thresholds.snow_l_min, float(np.percentile(lab[:, :, 0][road_base], 60)))
        v_threshold = max(cfg.thresholds.snow_v_min, float(np.percentile(hsv[:, :, 2][road_base], 60)))
        grad_threshold = float(np.percentile(gradient[road_base], cfg.thresholds.snow_texture_percentile))
        var_threshold = float(np.percentile(variance[road_base], cfg.thresholds.snow_texture_percentile))
    else:
        l_threshold = cfg.thresholds.snow_l_min
        v_threshold = cfg.thresholds.snow_v_min
        grad_threshold = float(np.percentile(gradient, cfg.thresholds.snow_texture_percentile))
        var_threshold = float(np.percentile(variance, cfg.thresholds.snow_texture_percentile))
    raw = road_base & (
        ((lab[:, :, 0] > l_threshold) | (hsv[:, :, 2] > v_threshold))
        & (hsv[:, :, 1] < cfg.thresholds.snow_s_max)
        & (chroma < cfg.thresholds.snow_chroma_max + 8.0)
    )
    refined = raw & ((gradient < grad_threshold) | (variance < var_threshold))
    raw_mask = to_mask_uint8(refine_binary_mask(raw, cfg.morphology.open_size, cfg.morphology.close_size))
    refined_mask = refine_binary_mask(refined, cfg.morphology.open_size, cfg.morphology.close_size)
    refined_mask = filter_components_by_area(refined_mask, cfg.morphology.min_snow_area)
    return raw_mask, to_mask_uint8(refined_mask)
