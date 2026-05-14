"""Run every copied classical ROI pipeline on one isolated image.

This script is intentionally self-contained inside this folder. It imports the
copied project modules next to it and writes outputs only under ./outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from batch.prefix_naming import get_artifact_spec
from config import create_default_config
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
from utils.io_utils import read_image, write_image
from utils.mask_utils import mask_ratio
from utils.path_utils import ensure_dir


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_NAME = "capture_log_20251218_110045_123_IMG_2025-12-18_11-17-44.webp"
INPUT_IMAGE = SCRIPT_DIR / "input" / IMAGE_NAME
OUTPUT_ROOT = SCRIPT_DIR / "outputs"

PIPELINES = [
    ("sky_removal", pipeline_sky_removal.run_pipeline),
    ("road_roi", pipeline_road_roi.run_pipeline),
    ("lane_processing", pipeline_lane_processing.run_pipeline),
    ("shadow_detection", pipeline_shadow_detection.run_pipeline),
    ("glare_detection", pipeline_glare_detection.run_pipeline),
    ("texture_maps", pipeline_texture_maps.run_pipeline),
    ("snow_candidates", pipeline_snow_candidates.run_pipeline),
    ("object_suppression", pipeline_object_suppression.run_pipeline),
    ("superpixels", pipeline_superpixels.run_pipeline),
    ("best_combination", pipeline_best_combination.run_pipeline),
]


def artifact_extension(file_type: str, default_ext: str) -> str:
    """Choose stable visualization extensions for the isolated run."""

    return ".png" if file_type == "mask" else default_ext


def artifact_path(key: str, stem: str) -> Path:
    """Build the output path for one artifact key."""

    spec = get_artifact_spec(key)
    ext = artifact_extension(spec.file_type, spec.default_ext)
    return OUTPUT_ROOT / spec.folder / f"{spec.prefix}_{stem}{ext}"


def write_artifact_once(key: str, data: np.ndarray, stem: str, written_keys: set[str]) -> Path | None:
    """Write the first artifact for each key and skip duplicate keys from later pipelines."""

    if data is None or key in written_keys:
        return None
    out_path = artifact_path(key, stem)
    write_image(out_path, data)
    written_keys.add(key)
    return out_path


def as_bgr(image: np.ndarray) -> np.ndarray:
    """Convert masks/maps/images into displayable BGR tiles."""

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def labeled_tile(image: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    """Resize one image and draw a compact label strip."""

    width, height = size
    tile = cv2.resize(as_bgr(image), (width, height), interpolation=cv2.INTER_AREA)
    tile = tile.copy()
    cv2.rectangle(tile, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def build_visual_grid(working: np.ndarray, artifacts: dict[str, np.ndarray]) -> np.ndarray:
    """Create a compact visual comparison grid for the ROI extraction run."""

    tile_size = (320, 240)
    cells = [
        ("normalized_input", working),
        ("sky_mask", artifacts["sky_mask"]),
        ("road_overlay", artifacts["road_overlay"]),
        ("road_only", artifacts["road_only"]),
        ("lane_overlay", artifacts["lane_overlay"]),
        ("glare_overlay", artifacts["glare_overlay"]),
        ("snow_overlay", artifacts["snow_overlay"]),
        ("nonroad_suppressed", artifacts["nonroad_suppressed"]),
        ("superpixels", artifacts["superpixels"]),
        ("combined_overlay", artifacts["combined_overlay"]),
        ("best_combined", artifacts["best_combined"]),
        ("road_crop", artifacts["road_crop"]),
    ]
    tiles = [labeled_tile(image, label, tile_size) for label, image in cells]
    rows = []
    cols = 4
    for start in range(0, len(tiles), cols):
        rows.append(np.hstack(tiles[start : start + cols]))
    return np.vstack(rows)


def write_summary(stem: str, ctx, artifact_paths: dict[str, str]) -> None:
    """Write JSON, CSV, and text summaries for quick inspection."""

    reports_dir = ensure_dir(OUTPUT_ROOT / "REPORTS")
    ratios = {
        "sky_ratio": round(mask_ratio(ctx.sky_mask), 6),
        "road_ratio": round(mask_ratio(ctx.road_mask), 6),
        "lane_ratio_within_road": round(mask_ratio(ctx.lane_mask, ctx.road_mask), 6),
        "glare_ratio_within_road": round(mask_ratio(ctx.glare_mask, ctx.road_mask), 6),
        "shadow_ratio_within_road": round(mask_ratio(ctx.shadow_mask, ctx.road_mask), 6),
        "snow_ratio_within_road": round(mask_ratio(ctx.snow_refined, ctx.road_mask), 6),
    }
    summary = {
        "input_image": str(INPUT_IMAGE),
        "output_root": str(OUTPUT_ROOT),
        "resize_shape": {"width": 960, "height": 720},
        "pipeline_order": [name for name, _ in PIPELINES],
        "mask_ratios": ratios,
        "artifact_paths": artifact_paths,
    }

    (reports_dir / f"summary_{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (reports_dir / f"summary_{stem}.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in ratios.items():
            writer.writerow({"metric": key, "value": value})

    lines = [
        "Single-image ROI visualization run",
        f"Input: {INPUT_IMAGE}",
        f"Output root: {OUTPUT_ROOT}",
        "",
        "Pipeline order:",
        *[f"- {name}" for name, _ in PIPELINES],
        "",
        "Mask ratios:",
        *[f"- {key}: {value}" for key, value in ratios.items()],
    ]
    (reports_dir / f"run_report_{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the isolated single-image visualization workflow."""

    cfg = create_default_config()
    cfg.dataset.output_root = OUTPUT_ROOT
    cfg.resize_width = 960
    cfg.resize_height = 720

    image = read_image(INPUT_IMAGE)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {INPUT_IMAGE}")

    stem = INPUT_IMAGE.stem
    working = resize_and_normalize(image, cfg.resize_shape)

    ensure_dir(OUTPUT_ROOT / "INPUT")
    write_image(OUTPUT_ROOT / "INPUT" / f"NORMALIZED_{stem}.jpg", working)

    ctx = None
    written_keys: set[str] = set()
    artifacts: dict[str, np.ndarray] = {}
    artifact_paths: dict[str, str] = {}

    for pipeline_name, runner in PIPELINES:
        result, ctx = runner(working, cfg, ctx)
        for artifact in result.artifacts:
            artifacts.setdefault(artifact.key, artifact.data)
            out_path = write_artifact_once(artifact.key, artifact.data, stem, written_keys)
            if out_path is not None:
                artifact_paths[artifact.key] = str(out_path)

    ctx = ensure_best_context(build_base_context(working, cfg, ctx), cfg)
    grid = build_visual_grid(working, artifacts)
    grid_path = OUTPUT_ROOT / "VISUAL_SUMMARY" / f"ROI_METHOD_GRID_{stem}.jpg"
    write_image(grid_path, grid)
    artifact_paths["visual_grid"] = str(grid_path)

    write_summary(stem, ctx, artifact_paths)
    print(f"Wrote isolated ROI visualizations to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
