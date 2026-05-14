"""Texture-map pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context, ensure_texture_context



def run_pipeline(image, cfg, ctx=None):
    """Run the texture-map pipeline."""

    ctx = ensure_texture_context(build_base_context(image, cfg, ctx), cfg)
    result = PipelineResult(name='texture_maps')
    result.add_artifact('texture_map', ctx.texture_map)
    result.add_artifact('entropy_map', ctx.entropy_map)
    result.add_artifact('gradient_map', ctx.gradient_map)
    return result, ctx
