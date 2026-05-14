"""Non-road object suppression pipeline."""

from __future__ import annotations

from pipelines.base import PipelineResult
from pipelines.shared_context import build_base_context, ensure_nonroad_context



def run_pipeline(image, cfg, ctx=None):
    """Run the non-road suppression pipeline."""

    ctx = ensure_nonroad_context(build_base_context(image, cfg, ctx), cfg)
    result = PipelineResult(name='object_suppression')
    result.add_artifact('nonroad_suppressed', ctx.nonroad_image)
    return result, ctx
