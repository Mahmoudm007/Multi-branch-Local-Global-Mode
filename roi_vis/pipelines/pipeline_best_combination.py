"""Recommended best-combination preprocessing pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context, ensure_best_context
from utils.vis_utils import draw_mask_contours, overlay_mask



def build_combined_overlay(ctx):
    """Build a combined debug overlay from the main masks."""

    overlay = ctx.image.copy()
    overlay = overlay_mask(overlay, ctx.sky_mask, (255, 128, 0), alpha=0.22)
    overlay = draw_mask_contours(overlay, ctx.road_mask, (0, 255, 0), thickness=2)
    overlay = overlay_mask(overlay, ctx.lane_mask, (0, 255, 255), alpha=0.55)
    overlay = overlay_mask(overlay, ctx.glare_mask, (255, 0, 255), alpha=0.55)
    overlay = overlay_mask(overlay, ctx.snow_refined, (255, 255, 0), alpha=0.45)
    return overlay



def run_pipeline(image, cfg, ctx=None):
    """Run the recommended combined preprocessing pipeline."""

    ctx = ensure_best_context(build_base_context(image, cfg, ctx), cfg)
    result = PipelineResult(name='best_combination')
    result.add_artifact('best_combined', ctx.best_image)
    result.add_artifact('road_roi_mask', ctx.road_mask)
    result.add_artifact('snow_candidate', ctx.snow_raw)
    result.add_artifact('snow_refined', ctx.snow_refined)
    result.add_artifact('lane_mask', ctx.lane_mask)
    result.add_artifact('glare_mask', ctx.glare_mask)
    result.add_artifact('combined_overlay', build_combined_overlay(ctx))
    result.add_artifact('road_crop', ctx.road_patch)
    return result, ctx
