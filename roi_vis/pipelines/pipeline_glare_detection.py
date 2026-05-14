"""Glare-detection pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context, ensure_glare_context
from utils.vis_utils import overlay_mask



def run_pipeline(image, cfg, ctx=None):
    """Run the glare-detection pipeline."""

    ctx = ensure_glare_context(build_base_context(image, cfg, ctx), cfg)
    result = PipelineResult(name='glare_detection')
    result.add_artifact('glare_mask', ctx.glare_mask)
    result.add_artifact('glare_suppressed', ctx.glare_suppressed)
    result.add_artifact('glare_overlay', overlay_mask(ctx.image, ctx.glare_mask, (255, 0, 255), alpha=0.50))
    return result, ctx
