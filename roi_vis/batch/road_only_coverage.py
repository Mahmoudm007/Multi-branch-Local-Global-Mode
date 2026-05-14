"""Batch processing for snow coverage from ROAD_ONLY images."""

from __future__ import annotations

import csv
from pathlib import Path

from methods.road_snow_coverage import estimate_road_snow_coverage
from utils.io_utils import is_image_file, read_image, write_image
from utils.logging_utils import get_logger
from utils.path_utils import ensure_dir


REPORT_FIELDS = [
    'road_only_path',
    'image_name',
    'snow_coverage_percent',
    'road_roi_pixels',
    'snow_pixels',
    'region_rows',
    'region_cols',
]



def _normalize_row(row: dict) -> dict:
    return {field: row.get(field, '') for field in REPORT_FIELDS}



def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get('road_only_path'):
                rows.append(_normalize_row(row))
    return rows



def append_row(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(_normalize_row(row))



def write_rows(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_normalize_row(row))



def iter_road_only_images(road_only_root: Path):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    for path in sorted(road_only_root.rglob('*')):
        if is_image_file(path, exts):
            yield path



def output_paths(output_root: Path):
    reports_dir = ensure_dir(output_root / 'REPORTS')
    return {
        'mask_root': ensure_dir(output_root / 'ROAD_SNOW_MASK'),
        'overlay_root': ensure_dir(output_root / 'ROAD_SNOW_OVERLAY'),
        'progress_csv': reports_dir / 'progress_road_snow_coverage.csv',
        'report_csv': reports_dir / 'road_snow_coverage.csv',
    }



def save_mask_and_overlay(paths: dict, road_only_root: Path, image_path: Path, result) -> None:
    relative_parent = image_path.relative_to(road_only_root).parent
    mask_dir = ensure_dir(paths['mask_root'] / relative_parent)
    overlay_dir = ensure_dir(paths['overlay_root'] / relative_parent)
    stem = image_path.stem
    write_image(mask_dir / f'SNOWCOVMASK_{stem}.png', result.snow_mask)
    write_image(overlay_dir / f'SNOWCOVOVL_{stem}.jpg', result.overlay)



def process_road_only_coverage(
    road_only_root: Path,
    output_root: Path,
    region_rows: int = 18,
    region_cols: int = 24,
    max_images: int | None = None,
    save_debug_outputs: bool = True,
) -> None:
    """Process ROAD_ONLY images and write snow coverage outputs."""

    paths = output_paths(output_root)
    logger = get_logger(
        'road_snow_coverage',
        debug=False,
        log_path=output_root / 'REPORTS' / 'road_snow_coverage.log',
    )
    all_images = list(iter_road_only_images(road_only_root))
    if max_images is not None:
        all_images = all_images[:max_images]
    if not all_images:
        logger.warning('No ROAD_ONLY images found under: %s', road_only_root)
        return

    completed_rows = read_rows(paths['progress_csv'])
    completed_paths = {row['road_only_path'] for row in completed_rows}
    pending_images = [path for path in all_images if str(path) not in completed_paths]

    logger.info('Found %d ROAD_ONLY images.', len(all_images))
    if completed_rows:
        logger.info('Resume detected: %d images already complete, %d remaining.', len(completed_rows), len(pending_images))

    rows = list(completed_rows)
    if not pending_images:
        write_rows(paths['report_csv'], rows)
        logger.info('Nothing left to process. Report is already complete: %s', paths['report_csv'])
        return

    total_pending = len(pending_images)
    for index, image_path in enumerate(pending_images, start=1):
        image = read_image(image_path)
        if image is None:
            logger.warning('Failed to read image: %s', image_path)
            continue
        result = estimate_road_snow_coverage(image, region_rows=region_rows, region_cols=region_cols)
        if save_debug_outputs:
            save_mask_and_overlay(paths, road_only_root, image_path, result)
        row = {
            'road_only_path': str(image_path),
            'image_name': image_path.name,
            'snow_coverage_percent': round(result.coverage_percent, 6),
            'road_roi_pixels': result.roi_pixels,
            'snow_pixels': result.snow_pixels,
            'region_rows': result.region_rows,
            'region_cols': result.region_cols,
        }
        append_row(paths['progress_csv'], row)
        rows.append(row)
        if index % 25 == 0 or index == total_pending:
            logger.info('Processed %d/%d ROAD_ONLY images', index, total_pending)

    write_rows(paths['report_csv'], rows)
    logger.info('Wrote snow coverage report: %s', paths['report_csv'])

