"""Road-ROI pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context
from utils.image_utils import apply_binary_mask
from utils.vis_utils import draw_mask_contours, draw_point, draw_polygon



def run_pipeline(image, cfg, ctx=None):
    """Run the road-ROI pipeline."""

    ctx = build_base_context(image, cfg, ctx)
    overlay = draw_mask_contours(ctx.image, ctx.road_mask, (0, 255, 0), thickness=2)
    overlay = draw_polygon(overlay, ctx.trapezoid, (255, 0, 0), thickness=2)
    overlay = draw_point(overlay, ctx.vanishing_point, (0, 255, 255), radius=5)
    result = PipelineResult(name='road_roi')
    result.add_artifact('road_roi_mask', ctx.road_mask)
    result.add_artifact('road_only', apply_binary_mask(ctx.image, ctx.road_mask, fill_value=0))
    result.add_artifact('road_overlay', overlay)
    return result, ctx
