"""Superpixel and region-partitioning pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context, ensure_superpixel_context



def run_pipeline(image, cfg, ctx=None):
    """Run the superpixel pipeline."""

    ctx = ensure_superpixel_context(build_base_context(image, cfg, ctx), cfg)
    result = PipelineResult(name='superpixels')
    result.add_artifact('superpixels', ctx.superpixel_overlay)
    result.add_artifact('region_mask', ctx.region_mask)
    return result, ctx
