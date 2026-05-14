"""Shadow-detection pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context, ensure_shadow_context
from utils.vis_utils import overlay_mask



def run_pipeline(image, cfg, ctx=None):
    """Run the shadow-detection pipeline."""

    ctx = ensure_shadow_context(build_base_context(image, cfg, ctx), cfg)
    result = PipelineResult(name='shadow_detection')
    result.add_artifact('shadow_mask', ctx.shadow_mask)
    result.add_artifact('shadow_suppressed', ctx.shadow_suppressed)
    result.add_artifact('shadow_overlay', overlay_mask(ctx.image, ctx.shadow_mask, (120, 0, 255), alpha=0.45))
    return result, ctx
