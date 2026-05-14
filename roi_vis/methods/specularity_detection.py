"""Glare and specularity detection."""

from __future__ import annotations

import cv2
import numpy as np

from config import ProjectConfig
from methods.color_spaces import compute_chroma, get_color_spaces
from methods.morphology_ops import refine_binary_mask
from utils.mask_utils import filter_components_by_area, to_mask_uint8



def detect_glare_mask(
    image: np.ndarray,
    road_mask: np.ndarray,
    sky_mask: np.ndarray,
    cfg: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect clustered specular highlights inside the road ROI."""

    spaces = get_color_spaces(image)
    hsv = spaces['hsv']
    lab = spaces['lab']
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    chroma = compute_chroma(lab)
    base = np.where(road_mask > 0, 255, 0).astype(np.uint8)
    base[sky_mask > 0] = 0
    if np.any(base):
        v_threshold = max(cfg.thresholds.glare_v_min, float(np.percentile(v[base > 0], 92)) + 10.0)
    else:
        v_threshold = cfg.thresholds.glare_v_min
    local_bg = cv2.GaussianBlur(v, (0, 0), sigmaX=11)
    local_contrast = v - local_bg
    candidate = (
        (base > 0)
        & (v > v_threshold)
        & (s < cfg.thresholds.glare_s_max)
        & (chroma < 26.0)
        & (local_contrast > cfg.thresholds.glare_local_contrast)
    )
    glare_mask = refine_binary_mask(candidate, cfg.morphology.open_size, cfg.morphology.close_size)
    road_area = int(max(1, (base > 0).sum()))
    glare_mask = filter_components_by_area(
        glare_mask,
        cfg.morphology.min_glare_area,
        max_area=max(cfg.morphology.min_glare_area, int(road_area * 0.05)),
    )
    glare_mask = to_mask_uint8(glare_mask)
    if np.any(glare_mask):
        suppressed = cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)
    else:
        suppressed = image.copy()
    return glare_mask, suppressed
