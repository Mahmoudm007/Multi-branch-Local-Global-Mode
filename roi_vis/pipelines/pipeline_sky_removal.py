"""Sky-removal pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context
from utils.vis_utils import draw_horizon, overlay_mask
from methods.sky_detection import remove_sky



def run_pipeline(image, cfg, ctx=None):
    """Run the sky-removal pipeline."""

    ctx = build_base_context(image, cfg, ctx)
    result = PipelineResult(name='sky_removal')
    result.add_artifact('sky_mask', ctx.sky_mask)
    result.add_artifact('sky_removed', remove_sky(ctx.image, ctx.sky_mask))
    result.add_artifact('horizon_overlay', draw_horizon(overlay_mask(ctx.image, ctx.sky_mask, (255, 120, 0)), ctx.horizon_row))
    return result, ctx
