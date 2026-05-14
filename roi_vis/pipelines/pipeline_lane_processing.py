"""Lane-processing pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context, ensure_lane_context
from utils.vis_utils import overlay_mask



def run_pipeline(image, cfg, ctx=None):
    """Run the lane-processing pipeline."""

    ctx = ensure_lane_context(build_base_context(image, cfg, ctx), cfg)
    result = PipelineResult(name='lane_processing')
    result.add_artifact('lane_mask', ctx.lane_mask)
    result.add_artifact('lane_overlay', overlay_mask(ctx.image, ctx.lane_mask, (0, 255, 255), alpha=0.55))
    return result, ctx
