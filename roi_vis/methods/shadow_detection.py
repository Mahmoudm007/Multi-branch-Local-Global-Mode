"""Shadow detection and suppression."""

from __future__ import annotations

import cv2
import numpy as np

from config import ProjectConfig
from methods.color_spaces import compute_chroma, get_color_spaces
from methods.morphology_ops import refine_binary_mask
from utils.image_utils import masked_median
from utils.mask_utils import filter_components_by_area, to_mask_uint8



def detect_shadow_mask(
    image: np.ndarray,
    road_mask: np.ndarray,
    cfg: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect chromatically consistent dark regions inside the road ROI."""

    spaces = get_color_spaces(image)
    hsv = spaces['hsv']
    lab = spaces['lab']
    v = hsv[:, :, 2].astype(np.float32)
    illumination = cv2.GaussianBlur(v, (0, 0), sigmaX=21)
    chroma = compute_chroma(lab)
    base = road_mask > 0
    ref_a = masked_median(lab[:, :, 1], road_mask)
    ref_b = masked_median(lab[:, :, 2], road_mask)
    ab_distance = np.sqrt((lab[:, :, 1].astype(np.float32) - ref_a) ** 2 + (lab[:, :, 2].astype(np.float32) - ref_b) ** 2)
    if np.any(base):
        chroma_limit = float(np.percentile(chroma[base], 80))
    else:
        chroma_limit = 30.0
    candidate = base & (v < illumination * cfg.thresholds.shadow_ratio) & ((ab_distance < cfg.thresholds.shadow_ab_distance) | (chroma < chroma_limit))
    shadow_mask = refine_binary_mask(candidate, cfg.morphology.open_size, cfg.morphology.close_size)
    shadow_mask = filter_components_by_area(shadow_mask, cfg.morphology.min_shadow_area)
    shadow_mask = to_mask_uint8(shadow_mask)
    corrected = image.astype(np.float32).copy()
    if np.any(shadow_mask):
        gain = np.clip(illumination / (v + 1.0), 1.0, 1.35)
        for channel in range(3):
            corrected[:, :, channel][shadow_mask > 0] *= gain[shadow_mask > 0]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return shadow_mask, corrected
