"""Road-only snow coverage estimation using classical region analysis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from methods.color_spaces import compute_chroma, get_color_spaces
from methods.texture_features import compute_gradient_magnitude, compute_local_variance
from utils.image_utils import normalize_to_uint8
from utils.mask_utils import filter_components_by_area, keep_largest_component, morph_close, morph_open, to_mask_uint8
from utils.vis_utils import draw_mask_contours, overlay_mask


@dataclass
class CoverageResult:
    """Outputs from the road-only snow coverage estimator."""

    coverage_percent: float
    roi_pixels: int
    snow_pixels: int
    region_rows: int
    region_cols: int
    roi_mask: np.ndarray
    analysis_mask: np.ndarray
    raw_snow_mask: np.ndarray
    snow_mask: np.ndarray
    support_mask: np.ndarray
    overlay: np.ndarray



def otsu_threshold_from_values(values: np.ndarray) -> float:
    """Compute an Otsu threshold from a 1-D value array."""

    values_u8 = np.clip(normalize_to_uint8(values), 0, 255)
    hist = np.bincount(values_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(np.median(values))
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b))
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-6:
        return lo
    return lo + (hi - lo) * (idx / 255.0)



def estimate_hood_cut_row(gray: np.ndarray) -> int:
    """Estimate the top of the hood/dash from lower horizontal edges."""

    h, w = gray.shape
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    row_signal = grad_y[:, w // 8 : 7 * w // 8].mean(axis=1)
    smooth = np.convolve(row_signal, np.ones(21, dtype=np.float32) / 21.0, mode='same')
    lo = int(0.60 * h)
    hi = int(0.94 * h)
    fallback = int(0.78 * h)
    band = smooth[lo:hi]
    if band.size == 0:
        return fallback
    threshold = float(np.percentile(band, 88))
    strong = np.where(band >= threshold)[0]
    if strong.size == 0:
        return fallback
    hood_row = lo + int(strong[0]) - 10
    return int(np.clip(hood_row, int(0.68 * h), fallback))



def build_roi_mask(image: np.ndarray) -> np.ndarray:
    """Build the non-black road ROI mask from a ROAD_ONLY image."""

    roi_mask = (image.max(axis=2) > 12).astype(np.uint8) * 255
    roi_mask = morph_close(roi_mask, 5)
    roi_mask = keep_largest_component(roi_mask)
    roi_mask = filter_components_by_area(roi_mask, 500)
    return to_mask_uint8(roi_mask)



def build_analysis_mask(roi_mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Trim the ROI so coverage focuses on the visible inner road surface."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    hood_cut = estimate_hood_cut_row(gray)
    trimmed = roi_mask.copy()
    trimmed[max(hood_cut - 24, 0) :, :] = 0

    core = np.zeros_like(trimmed)
    for y in range(h):
        xs = np.where(trimmed[y] > 0)[0]
        if xs.size < 12:
            continue
        left = int(xs[0])
        right = int(xs[-1])
        width = right - left + 1
        trim_ratio = 0.10 + 0.08 * (y / max(1, h - 1))
        margin = max(2, int(width * trim_ratio))
        if width <= 2 * margin + 4:
            continue
        core[y, left + margin : right - margin + 1] = 255

    core = morph_open(core, 3)
    core = morph_close(core, 5)
    core = filter_components_by_area(core, 300)
    core = to_mask_uint8(core)
    if np.count_nonzero(core) < 0.15 * max(1, np.count_nonzero(trimmed)):
        return to_mask_uint8(trimmed)
    return core



def compute_snow_masks(
    image: np.ndarray,
    analysis_mask: np.ndarray,
    region_rows: int,
    region_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pixel-level and region-supported snow masks."""

    spaces = get_color_spaces(image)
    hsv = spaces['hsv']
    lab = spaces['lab']
    gray = spaces['gray']

    valid = analysis_mask > 0
    if not np.any(valid):
        empty = np.zeros_like(analysis_mask)
        return empty, empty, empty

    chroma = compute_chroma(lab)
    gradient = compute_gradient_magnitude(gray)
    variance = compute_local_variance(gray)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    lightness = lab[:, :, 0].astype(np.float32)
    whiteness = lightness - 0.90 * chroma - 0.32 * saturation

    whiteness_values = whiteness[valid]
    saturation_values = saturation[valid]
    chroma_values = chroma[valid]
    gradient_values = gradient[valid]
    variance_values = variance[valid]
    value_values = value[valid]

    whiteness_otsu = otsu_threshold_from_values(whiteness_values)
    whiteness_thr = max(float(np.percentile(whiteness_values, 55)), whiteness_otsu)
    bright_thr = max(132.0, float(np.percentile(lightness[valid], 58)))
    sat_thr = min(88.0, float(np.percentile(saturation_values, 60)) + 10.0)
    chroma_thr = min(30.0, float(np.percentile(chroma_values, 62)) + 5.0)
    grad_thr = float(np.percentile(gradient_values, 78))
    var_thr = float(np.percentile(variance_values, 78))
    glare_value_thr = float(np.percentile(value_values, 94))

    local_bg = cv2.GaussianBlur(value, (0, 0), sigmaX=9)
    glare_mask = valid & (value > glare_value_thr) & (saturation < sat_thr) & ((value - local_bg) > 14.0)

    raw_snow = (
        valid
        & (whiteness > whiteness_thr)
        & (lightness > bright_thr)
        & (saturation < sat_thr)
        & (chroma < chroma_thr)
        & ~glare_mask
    )

    y_idx, x_idx = np.where(valid)
    y_min, y_max = int(y_idx.min()), int(y_idx.max()) + 1
    x_min, x_max = int(x_idx.min()), int(x_idx.max()) + 1
    tile_h = max(20, int(np.ceil((y_max - y_min) / max(1, region_rows))))
    tile_w = max(20, int(np.ceil((x_max - x_min) / max(1, region_cols))))

    support = np.zeros_like(analysis_mask)
    for y0 in range(y_min, y_max, tile_h):
        y1 = min(y_max, y0 + tile_h)
        for x0 in range(x_min, x_max, tile_w):
            x1 = min(x_max, x0 + tile_w)
            tile_valid = valid[y0:y1, x0:x1]
            valid_count = int(np.count_nonzero(tile_valid))
            if valid_count < 40 or valid_count < 0.22 * tile_valid.size:
                continue
            tile_raw = raw_snow[y0:y1, x0:x1][tile_valid]
            tile_snow_fraction = float(tile_raw.mean()) if tile_raw.size else 0.0
            tile_whiteness = float(np.mean(whiteness[y0:y1, x0:x1][tile_valid]))
            tile_saturation = float(np.mean(saturation[y0:y1, x0:x1][tile_valid]))
            tile_gradient = float(np.mean(gradient[y0:y1, x0:x1][tile_valid]))
            tile_variance = float(np.mean(variance[y0:y1, x0:x1][tile_valid]))

            snow_tile = False
            if tile_snow_fraction >= 0.48:
                snow_tile = True
            elif (
                tile_snow_fraction >= 0.18
                and tile_whiteness > (whiteness_thr + 2.0)
                and tile_saturation < (sat_thr + 12.0)
                and tile_gradient < grad_thr * 1.25
                and tile_variance < var_thr * 1.20
            ):
                snow_tile = True
            elif (
                tile_whiteness > (whiteness_thr + 8.0)
                and tile_saturation < (sat_thr + 8.0)
                and tile_gradient < grad_thr * 1.35
            ):
                snow_tile = True

            if snow_tile:
                support[y0:y1, x0:x1][tile_valid] = 255

    snow_mask = raw_snow & (support > 0)
    snow_mask = morph_open(snow_mask, 3)
    snow_mask = morph_close(snow_mask, 5)
    snow_mask = filter_components_by_area(snow_mask, 30)

    return to_mask_uint8(raw_snow), to_mask_uint8(support), to_mask_uint8(snow_mask)



def build_overlay(image: np.ndarray, analysis_mask: np.ndarray, snow_mask: np.ndarray) -> np.ndarray:
    """Build an audit overlay for the estimated snow coverage."""

    overlay = image.copy()
    overlay = draw_mask_contours(overlay, analysis_mask, (0, 255, 0), thickness=2)
    overlay = overlay_mask(overlay, snow_mask, (255, 255, 0), alpha=0.55)
    return overlay



def estimate_road_snow_coverage(
    image: np.ndarray,
    region_rows: int = 18,
    region_cols: int = 24,
) -> CoverageResult:
    """Estimate snow coverage on the road surface of a ROAD_ONLY image."""

    roi_mask = build_roi_mask(image)
    analysis_mask = build_analysis_mask(roi_mask, image)
    raw_snow_mask, support_mask, snow_mask = compute_snow_masks(image, analysis_mask, region_rows, region_cols)
    roi_pixels = int(np.count_nonzero(analysis_mask))
    snow_pixels = int(np.count_nonzero(snow_mask))
    coverage_percent = 0.0 if roi_pixels == 0 else 100.0 * snow_pixels / roi_pixels
    overlay = build_overlay(image, analysis_mask, snow_mask)
    return CoverageResult(
        coverage_percent=coverage_percent,
        roi_pixels=roi_pixels,
        snow_pixels=snow_pixels,
        region_rows=region_rows,
        region_cols=region_cols,
        roi_mask=roi_mask,
        analysis_mask=analysis_mask,
        raw_snow_mask=raw_snow_mask,
        snow_mask=snow_mask,
        support_mask=support_mask,
        overlay=overlay,
    )


