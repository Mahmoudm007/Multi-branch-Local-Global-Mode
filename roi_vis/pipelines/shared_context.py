"""Shared scene-context construction for pipeline reuse."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import ProjectConfig
from methods.ipm import warp_trapezoid_patch
from methods.lane_detection import detect_lane_mask
from methods.region_filters import suppress_nonroad_objects
from methods.road_region_detection import detect_road_roi
from methods.shadow_detection import detect_shadow_mask
from methods.sky_detection import detect_sky
from methods.snow_detection import detect_snow_candidates
from methods.specularity_detection import detect_glare_mask
from methods.superpixel_ops import filter_regions_by_rules, generate_superpixels, visualize_superpixels
from methods.texture_features import compute_texture_bundle
from utils.image_utils import apply_binary_mask


@dataclass
class SceneContext:
    """Shared reusable analysis state for one image."""

    image: np.ndarray
    sky_mask: np.ndarray | None = None
    horizon_row: int | None = None
    road_mask: np.ndarray | None = None
    trapezoid: np.ndarray | None = None
    vanishing_point: tuple[int, int] | None = None
    lane_mask: np.ndarray | None = None
    glare_mask: np.ndarray | None = None
    glare_suppressed: np.ndarray | None = None
    shadow_mask: np.ndarray | None = None
    shadow_suppressed: np.ndarray | None = None
    snow_raw: np.ndarray | None = None
    snow_refined: np.ndarray | None = None
    nonroad_mask: np.ndarray | None = None
    nonroad_image: np.ndarray | None = None
    texture_map: np.ndarray | None = None
    entropy_map: np.ndarray | None = None
    gradient_map: np.ndarray | None = None
    superpixel_overlay: np.ndarray | None = None
    region_mask: np.ndarray | None = None
    road_patch: np.ndarray | None = None
    best_image: np.ndarray | None = None



def build_base_context(image: np.ndarray, cfg: ProjectConfig, ctx: SceneContext | None = None) -> SceneContext:
    """Build the base sky and road context."""

    ctx = SceneContext(image=image) if ctx is None else ctx
    if ctx.sky_mask is None or ctx.horizon_row is None:
        ctx.sky_mask, ctx.horizon_row = detect_sky(image, cfg)
    if ctx.road_mask is None or ctx.trapezoid is None or ctx.vanishing_point is None:
        ctx.road_mask, ctx.trapezoid, ctx.vanishing_point = detect_road_roi(
            image,
            ctx.sky_mask,
            ctx.horizon_row,
            cfg,
        )
    return ctx



def ensure_lane_context(ctx: SceneContext, cfg: ProjectConfig) -> SceneContext:
    """Populate the lane mask."""

    if ctx.lane_mask is None:
        ctx.lane_mask = detect_lane_mask(ctx.image, ctx.road_mask, cfg)
    return ctx



def ensure_glare_context(ctx: SceneContext, cfg: ProjectConfig) -> SceneContext:
    """Populate the glare mask."""

    if ctx.glare_mask is None or ctx.glare_suppressed is None:
        ctx.glare_mask, ctx.glare_suppressed = detect_glare_mask(
            ctx.image,
            ctx.road_mask,
            ctx.sky_mask,
            cfg,
        )
    return ctx



def ensure_shadow_context(ctx: SceneContext, cfg: ProjectConfig) -> SceneContext:
    """Populate the shadow mask."""

    if ctx.shadow_mask is None or ctx.shadow_suppressed is None:
        ctx.shadow_mask, ctx.shadow_suppressed = detect_shadow_mask(ctx.image, ctx.road_mask, cfg)
    return ctx



def ensure_snow_context(ctx: SceneContext, cfg: ProjectConfig) -> SceneContext:
    """Populate snow masks."""

    ctx = ensure_lane_context(ctx, cfg)
    ctx = ensure_glare_context(ctx, cfg)
    if ctx.snow_raw is None or ctx.snow_refined is None:
        ctx.snow_raw, ctx.snow_refined = detect_snow_candidates(
            ctx.image,
            ctx.road_mask,
            ctx.sky_mask,
            ctx.lane_mask,
            ctx.glare_mask,
            cfg,
        )
    return ctx



def ensure_texture_context(ctx: SceneContext, cfg: ProjectConfig) -> SceneContext:
    """Populate texture feature maps."""

    if ctx.texture_map is None or ctx.entropy_map is None or ctx.gradient_map is None:
        ctx.texture_map, ctx.entropy_map, ctx.gradient_map = compute_texture_bundle(ctx.image)
    return ctx



def ensure_nonroad_context(ctx: SceneContext, cfg: ProjectConfig) -> SceneContext:
    """Populate non-road suppression outputs."""

    ctx = ensure_lane_context(ctx, cfg)
    ctx = ensure_glare_context(ctx, cfg)
    if ctx.nonroad_mask is None or ctx.nonroad_image is None:
        ctx.nonroad_mask, ctx.nonroad_image = suppress_nonroad_objects(
            ctx.image,
            ctx.road_mask,
            ctx.sky_mask,
            ctx.horizon_row,
            ctx.vanishing_point,
            cfg,
            lane_mask=ctx.lane_mask,
            glare_mask=ctx.glare_mask,
        )
    return ctx



def ensure_superpixel_context(ctx: SceneContext, cfg: ProjectConfig) -> SceneContext:
    """Populate superpixel visualization and refined region mask."""

    if ctx.superpixel_overlay is None or ctx.region_mask is None:
        labels = generate_superpixels(ctx.image)
        ctx.region_mask = filter_regions_by_rules(labels, ctx.road_mask)
        ctx.superpixel_overlay, _ = visualize_superpixels(ctx.image, labels, ctx.region_mask)
    return ctx



def ensure_best_context(ctx: SceneContext, cfg: ProjectConfig) -> SceneContext:
    """Populate the final recommended combined representation."""

    ctx = ensure_nonroad_context(ctx, cfg)
    ctx = ensure_snow_context(ctx, cfg)
    if ctx.best_image is None:
        best = ctx.nonroad_image.copy()
        suppress_mask = cv2.bitwise_or(ctx.lane_mask, ctx.glare_mask)
        suppress_mask = cv2.bitwise_and(suppress_mask, ctx.nonroad_mask)
        if np.any(suppress_mask):
            best = cv2.inpaint(best, suppress_mask, 3, cv2.INPAINT_TELEA)
        best = apply_binary_mask(best, ctx.nonroad_mask, fill_value=0)
        ctx.best_image = best
    if ctx.road_patch is None and ctx.trapezoid is not None:
        ctx.road_patch = warp_trapezoid_patch(ctx.best_image, ctx.trapezoid)
    return ctx
