"""Classical region partitioning with an optional SLIC fallback."""

from __future__ import annotations

import cv2
import numpy as np

from utils.mask_utils import to_mask_uint8
from utils.vis_utils import draw_mask_contours, overlay_mask

try:
    from skimage.segmentation import slic
except Exception:
    slic = None



def watershed_grid_superpixels(image: np.ndarray, desired_regions: int = 400) -> np.ndarray:
    """Fallback oversegmentation using watershed seeded on a grid."""

    h, w = image.shape[:2]
    step = max(14, int(np.sqrt((h * w) / float(max(1, desired_regions)))))
    markers = np.zeros((h, w), dtype=np.int32)
    label = 1
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            markers[max(0, y - 1) : min(h, y + 2), max(0, x - 1) : min(w, x + 2)] = label
            label += 1
    ws_input = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.watershed(ws_input, markers)
    markers[markers < 1] = 0
    return markers



def generate_superpixels(image: np.ndarray, desired_regions: int = 400) -> np.ndarray:
    """Generate region labels with SLIC when available, otherwise watershed."""

    if slic is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = slic(rgb, n_segments=desired_regions, compactness=10, start_label=1)
        return labels.astype(np.int32)
    return watershed_grid_superpixels(image, desired_regions=desired_regions)



def boundaries_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return a binary mask for label boundaries."""

    boundary = np.zeros(labels.shape, dtype=np.uint8)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return to_mask_uint8(boundary)



def filter_regions_by_rules(labels: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """Keep regions that mostly overlap the road corridor."""

    h, w = labels.shape
    center_x = 0.5 * w
    region_mask = np.zeros_like(road_mask)
    max_label = int(labels.max())
    for label in range(1, max_label + 1):
        region = labels == label
        area = int(region.sum())
        if area < 50:
            continue
        overlap = float((road_mask[region] > 0).mean())
        ys, xs = np.where(region)
        if ys.size == 0:
            continue
        y_ratio = float(ys.mean() / h)
        x_deviation = abs(xs.mean() - center_x) / center_x
        if overlap > 0.55 and y_ratio > 0.22 and x_deviation < 0.90:
            region_mask[region] = 255
    return region_mask



def visualize_superpixels(
    image: np.ndarray,
    labels: np.ndarray,
    region_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a superpixel boundary overlay and a boundary mask."""

    boundary = boundaries_from_labels(labels)
    overlay = overlay_mask(image, boundary, (0, 255, 255), alpha=0.55)
    if region_mask is not None:
        overlay = draw_mask_contours(overlay, region_mask, (0, 255, 0), thickness=2)
    return overlay, boundary
