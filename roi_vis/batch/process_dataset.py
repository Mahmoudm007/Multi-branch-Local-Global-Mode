"""Dataset batch processor."""

from __future__ import annotations

from batch.dataset_scanner import scan_dataset
from batch.generate_summary_csv import append_summary_row, read_summary_rows, write_summary_csv
from batch.output_writer import write_artifact
from config import ProjectConfig
from pipelines import (
    pipeline_best_combination,
    pipeline_glare_detection,
    pipeline_lane_processing,
    pipeline_object_suppression,
    pipeline_road_roi,
    pipeline_shadow_detection,
    pipeline_sky_removal,
    pipeline_snow_candidates,
    pipeline_superpixels,
    pipeline_texture_maps,
)
from pipelines.shared_context import build_base_context, ensure_best_context
from utils.image_utils import resize_and_normalize
from utils.io_utils import read_image
from utils.logging_utils import get_logger
from utils.mask_utils import mask_ratio
from utils.path_utils import ensure_dir


PIPELINES = {
    'sky_removal': pipeline_sky_removal.run_pipeline,
    'road_roi': pipeline_road_roi.run_pipeline,
    'lane_processing': pipeline_lane_processing.run_pipeline,
    'shadow_detection': pipeline_shadow_detection.run_pipeline,
    'glare_detection': pipeline_glare_detection.run_pipeline,
    'texture_maps': pipeline_texture_maps.run_pipeline,
    'snow_candidates': pipeline_snow_candidates.run_pipeline,
    'object_suppression': pipeline_object_suppression.run_pipeline,
    'superpixels': pipeline_superpixels.run_pipeline,
    'best_combination': pipeline_best_combination.run_pipeline,
}

ALL_PIPELINES = [
    'sky_removal',
    'road_roi',
    'lane_processing',
    'shadow_detection',
    'glare_detection',
    'texture_maps',
    'snow_candidates',
    'object_suppression',
    'superpixels',
    'best_combination',
]



def _build_warning_list(ctx) -> list[str]:
    warnings = []
    if mask_ratio(ctx.sky_mask) < 0.01:
        warnings.append('low_sky_mask')
    if mask_ratio(ctx.road_mask) < 0.04:
        warnings.append('low_road_mask')
    if mask_ratio(ctx.road_mask) > 0.80:
        warnings.append('large_road_mask')
    if mask_ratio(ctx.lane_mask, ctx.road_mask) < 0.0005:
        warnings.append('sparse_lane_mask')
    if mask_ratio(ctx.glare_mask, ctx.road_mask) > 0.08:
        warnings.append('large_glare_mask')
    if mask_ratio(ctx.snow_refined, ctx.road_mask) > 0.92:
        warnings.append('very_large_snow_mask')
    return warnings



def _build_summary_row(item, ctx) -> dict:
    warnings = _build_warning_list(ctx)
    return {
        'image_path': str(item.input_path),
        'split': item.split,
        'class': item.class_name,
        'sky_ratio': round(mask_ratio(ctx.sky_mask), 6),
        'road_ratio': round(mask_ratio(ctx.road_mask), 6),
        'lane_ratio': round(mask_ratio(ctx.lane_mask, ctx.road_mask), 6),
        'glare_ratio': round(mask_ratio(ctx.glare_mask, ctx.road_mask), 6),
        'snow_ratio_within_road': round(mask_ratio(ctx.snow_refined, ctx.road_mask), 6),
        'warnings': ';'.join(warnings),
    }



def _write_pipeline_result(cfg: ProjectConfig, item, result) -> None:
    for artifact in result.artifacts:
        if artifact.data is None:
            continue
        write_artifact(cfg, item.relative_parent, item.input_path, artifact.key, artifact.data)



def _report_paths(cfg: ProjectConfig, pipeline_name: str):
    reports_dir = ensure_dir(cfg.dataset.output_root / 'REPORTS')
    progress_path = reports_dir / f'progress_{pipeline_name}.csv'
    pipeline_summary_path = reports_dir / f'summary_{pipeline_name}.csv'
    latest_summary_path = reports_dir / cfg.report_name
    return progress_path, pipeline_summary_path, latest_summary_path



def process_dataset(cfg: ProjectConfig, pipeline_name: str) -> None:
    """Process the configured dataset with the chosen pipeline."""

    ensure_dir(cfg.dataset.output_root)
    logger = get_logger(
        'classical_road_preprocessing',
        debug=cfg.debug,
        log_path=cfg.dataset.output_root / 'REPORTS' / 'run.log',
    )
    _, items = scan_dataset(cfg, logger)
    if not items:
        logger.warning('No images found for processing.')
        return

    progress_path, pipeline_summary_path, latest_summary_path = _report_paths(cfg, pipeline_name)
    completed_rows = read_summary_rows(progress_path)
    completed_paths = {row['image_path'] for row in completed_rows}
    pending_items = [item for item in items if str(item.input_path) not in completed_paths]

    selected = ALL_PIPELINES if pipeline_name == 'all' else [pipeline_name]
    total = len(items)
    pending_total = len(pending_items)
    logger.info('Processing %d images with pipeline=%s', total, pipeline_name)
    if completed_rows:
        logger.info(
            'Resume detected: %d images already complete, %d remaining.',
            len(completed_rows),
            pending_total,
        )
    if pending_total == 0:
        write_summary_csv(pipeline_summary_path, completed_rows)
        write_summary_csv(latest_summary_path, completed_rows)
        logger.info('Nothing left to process. Summary already complete: %s', pipeline_summary_path)
        return

    summary_rows = list(completed_rows)
    for index, item in enumerate(pending_items, start=1):
        image = read_image(item.input_path)
        if image is None:
            logger.warning('Failed to read image: %s', item.input_path)
            continue
        working = resize_and_normalize(image, cfg.resize_shape)
        ctx = None
        for selected_name in selected:
            runner = PIPELINES[selected_name]
            result, ctx = runner(working, cfg, ctx)
            _write_pipeline_result(cfg, item, result)
        ctx = ensure_best_context(build_base_context(working, cfg, ctx), cfg)
        row = _build_summary_row(item, ctx)
        append_summary_row(progress_path, row)
        summary_rows.append(row)
        if index % 25 == 0 or index == pending_total:
            logger.info('Processed %d/%d remaining images', index, pending_total)

    write_summary_csv(pipeline_summary_path, summary_rows)
    write_summary_csv(latest_summary_path, summary_rows)
    logger.info('Wrote summary CSV: %s', pipeline_summary_path)
