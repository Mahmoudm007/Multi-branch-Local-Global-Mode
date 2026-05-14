"""Snow-candidate pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context, ensure_snow_context
from utils.vis_utils import overlay_mask



def run_pipeline(image, cfg, ctx=None):
    """Run the snow-candidate pipeline."""

    ctx = ensure_snow_context(build_base_context(image, cfg, ctx), cfg)
    overlay = overlay_mask(ctx.image, ctx.snow_refined, (255, 255, 0), alpha=0.50)
    result = PipelineResult(name='snow_candidates')
    result.add_artifact('snow_candidate', ctx.snow_raw)
    result.add_artifact('snow_refined', ctx.snow_refined)
    result.add_artifact('snow_overlay', overlay)
    return result, ctx
