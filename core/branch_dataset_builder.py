from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

from .branch_asset_manifest import (
    ImageRecord,
    output_path_for_record,
    scan_defined_dataset,
    summarize_records,
    validate_image_file,
)
from .experiment_registry import BRANCH_DIRS, CROPPED, SEGMENTED, THERMAL
from .progress_tracker import CSVProgressTracker, atomic_write_json, ensure_dir


GENERATED_IMAGE_BRANCHES = (CROPPED, THERMAL, SEGMENTED)

PROGRESS_FIELDS = (
    "key",
    "branch",
    "split",
    "class_folder",
    "source_path",
    "output_path",
    "status",
    "seconds",
    "warnings",
    "error",
    "output_bytes",
    "completed_at",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _save_image_atomic(image: Image.Image, path: Path) -> None:
    ensure_dir(path.parent)
    suffix = path.suffix if path.suffix else ".png"
    tmp = path.with_name(path.name + ".tmp" + suffix)
    save_kwargs = {}
    if suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs.update({"quality": 95, "subsampling": 0})
    elif suffix.lower() == ".webp":
        save_kwargs.update({"quality": 95, "method": 4})
    image.save(tmp, **save_kwargs)
    tmp.replace(path)


def crop_local_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    top = int(math.floor(0.25 * height))
    bottom = int(math.ceil(0.70 * height))
    left = int(math.floor(0.10 * width))
    right = int(math.ceil(0.90 * width))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    return image.crop((left, top, right, bottom))


def crop_box_for_image(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    return (
        int(math.floor(0.10 * width)),
        int(math.floor(0.25 * height)),
        int(math.ceil(0.90 * width)),
        int(math.ceil(0.70 * height)),
    )


def thermal_clahe_inferno(image: Image.Image) -> Image.Image:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the CLAHE_Inferno branch")
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored)


def _gray_world_normalize_bgr(bgr: np.ndarray) -> np.ndarray:
    arr = bgr.astype(np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    global_mean = float(means.mean())
    scale = global_mean / np.maximum(means, 1.0)
    arr *= scale.reshape(1, 1, 3)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = 1 + int(np.argmax(areas))
    return np.where(labels == label, 255, 0).astype(np.uint8)


def _refine_mask(mask: np.ndarray, open_size: int = 5, close_size: int = 13) -> np.ndarray:
    if cv2 is None:
        return mask
    mask = (mask > 0).astype(np.uint8) * 255
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _estimate_horizon(gray: np.ndarray) -> int:
    if cv2 is None:
        return int(gray.shape[0] * 0.45)
    height, width = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(40, int(0.10 * width)),
        maxLineGap=20,
    )
    rows: list[tuple[float, float]] = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = max(abs(x2 - x1), 1)
            slope = abs((y2 - y1) / dx)
            midpoint = 0.5 * (y1 + y2)
            length = math.hypot(x2 - x1, y2 - y1)
            if slope < 0.18 and midpoint < 0.72 * height and length > 0.10 * width:
                rows.append((midpoint, length))
    if rows:
        rows.sort(key=lambda item: item[0])
        total = sum(weight for _, weight in rows)
        acc = 0.0
        for row, weight in rows:
            acc += weight
            if acc >= total / 2:
                return int(np.clip(row, 0.18 * height, 0.68 * height))

    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    signal = np.abs(sobel_y[:, width // 5 : 4 * width // 5]).mean(axis=1)
    kernel = np.ones(31, dtype=np.float32) / 31.0
    smooth = np.convolve(signal, kernel, mode="same")
    lo, hi = int(0.18 * height), int(0.68 * height)
    return int(lo + np.argmax(smooth[lo:hi]))


def _estimate_vanishing_point(gray: np.ndarray, horizon: int) -> tuple[int, int]:
    if cv2 is None:
        h, w = gray.shape
        return w // 2, horizon
    height, width = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 170)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=max(35, int(0.08 * width)),
        maxLineGap=20,
    )
    left: list[tuple[float, float, float]] = []
    right: list[tuple[float, float, float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 2:
                continue
            slope = dy / dx
            if abs(slope) < 0.25 or abs(slope) > 4.0:
                continue
            midpoint_y = 0.5 * (y1 + y2)
            if midpoint_y < 0.25 * height or midpoint_y > 0.95 * height:
                continue
            intercept = y1 - slope * x1
            length = math.hypot(dx, dy)
            if slope < 0:
                left.append((slope, intercept, length))
            else:
                right.append((slope, intercept, length))
    left = sorted(left, key=lambda item: item[2], reverse=True)[:8]
    right = sorted(right, key=lambda item: item[2], reverse=True)[:8]
    points: list[tuple[float, float]] = []
    for m1, b1, _ in left:
        for m2, b2, _ in right:
            denom = m1 - m2
            if abs(denom) < 1e-4:
                continue
            x = (b2 - b1) / denom
            y = m1 * x + b1
            if -0.25 * width <= x <= 1.25 * width and 0 <= y <= 0.85 * height:
                points.append((x, y))
    if not points:
        return width // 2, horizon
    xs, ys = np.array(points).T
    return int(np.clip(np.median(xs), 0.35 * width, 0.65 * width)), int(np.clip(np.median(ys), 0.18 * height, 0.68 * height))


def _road_trapezoid(shape: tuple[int, int], horizon: int, vp: tuple[int, int]) -> np.ndarray:
    height, width = shape
    bottom_y = int(0.92 * height)
    top_y = int(np.clip(horizon + 0.05 * height, 0.20 * height, bottom_y - 40))
    center_x = int(np.clip(vp[0], 0.35 * width, 0.65 * width))
    bottom_half = int(0.46 * width)
    top_half = int(0.11 * width)
    return np.array(
        [
            [max(0, center_x - bottom_half), bottom_y],
            [max(0, center_x - top_half), top_y],
            [min(width - 1, center_x + top_half), top_y],
            [min(width - 1, center_x + bottom_half), bottom_y],
        ],
        dtype=np.int32,
    )


def _detect_sky_mask(bgr: np.ndarray, gray: np.ndarray, horizon: int) -> np.ndarray:
    if cv2 is None:
        return np.zeros(gray.shape, dtype=np.uint8)
    height, _ = gray.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    blue, green, red = cv2.split(bgr)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    top_limit = int(min(max(0.26 * height, horizon + 0.12 * height), 0.74 * height))
    top_region = np.zeros_like(gray, dtype=np.uint8)
    top_region[:top_limit, :] = 1
    low_texture_threshold = np.percentile(gradient[: max(1, top_limit), :], 72)
    blue_sky = (
        (blue.astype(np.int16) > green.astype(np.int16) + 8)
        & (blue.astype(np.int16) > red.astype(np.int16) + 8)
        & (hsv[:, :, 2] > 95)
        & (hsv[:, :, 1] > 20)
    )
    overcast = (lab[:, :, 0] > 145) & (hsv[:, :, 1] < 75) & (gradient < low_texture_threshold)
    mask = ((blue_sky | overcast) & (top_region > 0)).astype(np.uint8) * 255
    mask = _refine_mask(mask, 7, 19)
    return mask


def _detect_road_mask(bgr: np.ndarray, gray: np.ndarray, sky_mask: np.ndarray, trapezoid: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return np.ones(gray.shape, dtype=np.uint8) * 255
    height, width = gray.shape
    prior = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillPoly(prior, [trapezoid], 255)
    prior[sky_mask > 0] = 0
    prior[int(0.94 * height) :, :] = 0

    seed = np.zeros_like(gray, dtype=np.uint8)
    x0, x1 = int(0.44 * width), int(0.56 * width)
    y0, y1 = int(0.60 * height), int(0.80 * height)
    seed[y0:y1, x0:x1] = 255
    seed = cv2.bitwise_and(seed, prior)
    if int(seed.sum()) == 0:
        seed[int(0.65 * height) : int(0.82 * height), int(0.45 * width) : int(0.55 * width)] = 255
        seed = cv2.bitwise_and(seed, prior)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    seed_pixels = seed > 0
    if not np.any(seed_pixels):
        return prior
    ref = lab[seed_pixels].reshape(-1, 3).mean(axis=0)
    ref_gray = float(gray[seed_pixels].mean())
    ab_dist = np.linalg.norm(lab[:, :, 1:].astype(np.float32) - ref[1:].reshape(1, 1, 2), axis=2)
    l_dist = np.abs(lab[:, :, 0].astype(np.float32) - float(ref[0]))
    gray_dist = np.abs(gray.astype(np.float32) - ref_gray)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    grad_threshold = np.percentile(gradient[prior > 0], 82) if np.any(prior > 0) else np.percentile(gradient, 82)
    candidate = (
        (prior > 0)
        & (ab_dist < 34)
        & (l_dist < 58)
        & (gray_dist < 72)
        & (gradient < grad_threshold)
    ).astype(np.uint8) * 255
    candidate = _refine_mask(candidate, 9, 25)

    # Keep candidate components connected to the lower-center seed.
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), 8)
    if num_labels > 1:
        seed_labels = set(int(v) for v in np.unique(labels[seed > 0]) if int(v) != 0)
        connected = np.isin(labels, list(seed_labels)).astype(np.uint8) * 255 if seed_labels else candidate
    else:
        connected = candidate
    connected = cv2.dilate(connected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    connected = cv2.bitwise_and(connected, prior)
    connected = _largest_component(connected)
    return connected


def _detect_lane_mask(bgr: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return np.zeros(road_mask.shape, dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    white = (hls[:, :, 1] > 175) & (hls[:, :, 2] < 95)
    yellow = (hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 95)
    edges = cv2.Canny(gray, 80, 180) > 0
    mask = ((white | yellow | edges) & (road_mask > 0)).astype(np.uint8) * 255
    mask = _refine_mask(mask, 3, 9)
    return mask


def _detect_glare_mask(bgr: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return np.zeros(road_mask.shape, dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    local_contrast = gray.astype(np.int16) - blur.astype(np.int16)
    mask = (
        (road_mask > 0)
        & (hsv[:, :, 2] > 205)
        & (hsv[:, :, 1] < 70)
        & (local_contrast > 12)
    ).astype(np.uint8) * 255
    return _refine_mask(mask, 3, 11)


def segmented_best_combined(image: Image.Image, resize_shape: tuple[int, int] = (960, 720)) -> tuple[Image.Image, list[str]]:
    """Classical best-combination road-focused representation.

    The original repository modules documented in SEG.md are not present in this
    workspace, so this implementation preserves the same deterministic classical-CV
    flow: resize/white-balance, sky and horizon cues, conservative trapezoid road ROI,
    lane/glare suppression, inpainting, and black non-road suppression. Only this
    final BEST_COMBINED image is returned for classifier training.
    """

    if cv2 is None:
        raise RuntimeError("OpenCV is required for segmented_best_combined")
    warnings: list[str] = []
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, resize_shape, interpolation=cv2.INTER_AREA)
    bgr = _gray_world_normalize_bgr(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    horizon = _estimate_horizon(gray)
    sky = _detect_sky_mask(bgr, gray, horizon)
    vp = _estimate_vanishing_point(gray, horizon)
    trapezoid = _road_trapezoid(gray.shape, horizon, vp)
    road = _detect_road_mask(bgr, gray, sky, trapezoid)
    road_ratio = float((road > 0).mean())
    if road_ratio < 0.04:
        warnings.append(f"small_road_roi:{road_ratio:.4f}")
    if road_ratio > 0.85:
        warnings.append(f"large_road_roi:{road_ratio:.4f}")
    lane = _detect_lane_mask(bgr, road)
    glare = _detect_glare_mask(bgr, road)
    inpaint_mask = cv2.bitwise_or(lane, glare)
    cleaned = bgr.copy()
    if np.any(inpaint_mask > 0):
        cleaned = cv2.inpaint(cleaned, inpaint_mask, 3, cv2.INPAINT_TELEA)
    best = np.zeros_like(cleaned)
    best[road > 0] = cleaned[road > 0]
    best_rgb = cv2.cvtColor(best, cv2.COLOR_BGR2RGB)
    return Image.fromarray(best_rgb), warnings


TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image | tuple[Image.Image, list[str]]]] = {
    CROPPED: crop_local_image,
    THERMAL: thermal_clahe_inferno,
    SEGMENTED: segmented_best_combined,
}


class BranchDatasetBuilder:
    def __init__(
        self,
        dataset_root: Path,
        asset_root: Path,
        defined_folder: str = "1 Defined",
        seed: int = 42,
    ) -> None:
        self.dataset_root = dataset_root
        self.asset_root = asset_root
        self.defined_folder = defined_folder
        self.seed = seed
        self.manifest_dir = ensure_dir(asset_root / "manifests")
        self.report_dir = ensure_dir(asset_root / "reports")
        self.preview_dir = ensure_dir(asset_root / "previews")

    def scan(self, splits: tuple[str, ...], max_images_per_class: int | None = None) -> tuple[list[ImageRecord], dict[str, int], dict[int, str]]:
        return scan_defined_dataset(
            self.dataset_root,
            defined_folder=self.defined_folder,
            splits=splits,
            max_images_per_class=max_images_per_class,
        )

    def write_source_manifest(self, records: list[ImageRecord], class_to_index: dict[str, int], index_to_class: dict[int, str]) -> None:
        manifest_path = self.manifest_dir / "source_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "sample_id",
                "split",
                "class_folder",
                "class_name",
                "label",
                "relative_path",
                "source_path",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "sample_id": record.sample_id,
                        "split": record.split,
                        "class_folder": record.class_folder,
                        "class_name": record.class_name,
                        "label": record.label,
                        "relative_path": record.relative_path.as_posix(),
                        "source_path": str(record.source_path),
                    }
                )
        atomic_write_json(
            self.manifest_dir / "source_manifest.json",
            {
                "dataset_root": str(self.dataset_root),
                "defined_folder": self.defined_folder,
                "total_images": len(records),
                "class_to_index": class_to_index,
                "index_to_class": {str(k): v for k, v in index_to_class.items()},
                "counts": summarize_records(records),
                "created_at": _utc_now(),
            },
        )

    def branch_defined_root(self, branch: str) -> Path:
        return self.asset_root / BRANCH_DIRS[branch] / self.defined_folder

    def generate(
        self,
        branches: tuple[str, ...] = GENERATED_IMAGE_BRANCHES,
        splits: tuple[str, ...] = ("train", "val"),
        skip_completed: bool = True,
        overwrite: bool = False,
        max_images_per_class: int | None = None,
        validate: bool = True,
        previews: bool = True,
    ) -> dict[str, dict[str, object]]:
        records, class_to_index, index_to_class = self.scan(splits, max_images_per_class)
        self.write_source_manifest(records, class_to_index, index_to_class)
        summary: dict[str, dict[str, object]] = {}
        for branch in branches:
            summary[branch] = self.generate_branch(
                branch=branch,
                records=records,
                skip_completed=skip_completed,
                overwrite=overwrite,
            )
            if validate:
                summary[branch]["validation"] = self.validate_branch(branch, records)
            if previews:
                self.write_previews(branch, records)
        atomic_write_json(
            self.manifest_dir / "branch_generation_summary.json",
            {
                "asset_root": str(self.asset_root),
                "branches": summary,
                "class_to_index": class_to_index,
                "created_at": _utc_now(),
            },
        )
        return summary

    def generate_branch(
        self,
        branch: str,
        records: list[ImageRecord],
        skip_completed: bool = True,
        overwrite: bool = False,
    ) -> dict[str, object]:
        if branch not in TRANSFORMS:
            raise KeyError(f"Unsupported generated branch: {branch}")
        branch_root = self.branch_defined_root(branch)
        tracker = CSVProgressTracker(self.manifest_dir / f"progress_{branch}.csv", PROGRESS_FIELDS)
        completed = tracker.completed_keys(self.asset_root) if skip_completed and not overwrite else set()
        stats = {"ok": 0, "skipped": 0, "error": 0, "warnings": 0}
        start_all = time.perf_counter()
        for record in records:
            output_path = output_path_for_record(branch_root, record, branch)
            key = record.sample_id
            if not overwrite and (key in completed or (skip_completed and validate_image_file(output_path)[0])):
                stats["skipped"] += 1
                continue
            start = time.perf_counter()
            warnings: list[str] = []
            error = ""
            status = "ok"
            try:
                image = _read_rgb(record.source_path)
                result = TRANSFORMS[branch](image)
                if isinstance(result, tuple):
                    out_image, warnings = result
                else:
                    out_image = result
                _save_image_atomic(out_image.convert("RGB"), output_path)
                valid, message, _, _ = validate_image_file(output_path)
                if not valid:
                    raise RuntimeError(f"written image failed validation: {message}")
                if warnings:
                    stats["warnings"] += 1
            except Exception as exc:  # noqa: BLE001 - progress must capture all failures
                status = "error"
                error = str(exc)
                stats["error"] += 1
            else:
                stats["ok"] += 1
            tracker.append(
                {
                    "key": key,
                    "branch": branch,
                    "split": record.split,
                    "class_folder": record.class_folder,
                    "source_path": str(record.source_path),
                    "output_path": output_path.relative_to(self.asset_root).as_posix(),
                    "status": status,
                    "seconds": f"{time.perf_counter() - start:.4f}",
                    "warnings": "|".join(warnings),
                    "error": error,
                    "output_bytes": output_path.stat().st_size if output_path.exists() else 0,
                    "completed_at": _utc_now(),
                }
            )
        stats["seconds_total"] = round(time.perf_counter() - start_all, 4)
        stats["asset_root"] = str(branch_root)
        atomic_write_json(self.report_dir / f"summary_{branch}.json", stats)
        return stats

    def validate_branch(self, branch: str, records: list[ImageRecord]) -> dict[str, object]:
        branch_root = self.branch_defined_root(branch)
        report_path = self.report_dir / f"validation_{branch}.csv"
        rows: list[dict[str, object]] = []
        ok = missing = corrupt = 0
        for record in records:
            output_path = output_path_for_record(branch_root, record, branch)
            valid, message, size, mode = validate_image_file(output_path)
            if valid:
                ok += 1
            elif message == "missing":
                missing += 1
            else:
                corrupt += 1
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "split": record.split,
                    "class_folder": record.class_folder,
                    "output_path": str(output_path),
                    "status": "ok" if valid else message,
                    "width": size[0] if size else "",
                    "height": size[1] if size else "",
                    "mode": mode or "",
                }
            )
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "branch": branch,
            "expected": len(records),
            "ok": ok,
            "missing": missing,
            "corrupt": corrupt,
            "report": str(report_path),
            "created_at": _utc_now(),
        }
        atomic_write_json(self.report_dir / f"validation_{branch}.json", summary)
        return summary

    def write_previews(self, branch: str, records: list[ImageRecord], samples_per_class: int = 3) -> None:
        rng = random.Random(self.seed)
        branch_root = self.branch_defined_root(branch)
        for split in sorted({record.split for record in records}):
            selected: list[ImageRecord] = []
            class_names = sorted({record.class_folder for record in records if record.split == split})
            for class_name in class_names:
                candidates = [record for record in records if record.split == split and record.class_folder == class_name]
                rng.shuffle(candidates)
                selected.extend(candidates[:samples_per_class])
            if not selected:
                continue
            self._write_comparison_grid(branch, split, selected, branch_root)
            if branch == CROPPED:
                self._write_crop_audit_grid(split, selected)

    def _write_comparison_grid(self, branch: str, split: str, records: list[ImageRecord], branch_root: Path) -> None:
        thumb_w, thumb_h = 220, 150
        label_h = 42
        cols = 2
        rows = len(records)
        canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for row, record in enumerate(records):
            y = row * (thumb_h + label_h)
            original = ImageOps.contain(_read_rgb(record.source_path), (thumb_w, thumb_h))
            branch_path = output_path_for_record(branch_root, record, branch)
            branch_img = ImageOps.contain(_read_rgb(branch_path), (thumb_w, thumb_h)) if branch_path.exists() else Image.new("RGB", (thumb_w, thumb_h), "black")
            canvas.paste(original, (0, y))
            canvas.paste(branch_img, (thumb_w, y))
            label = f"{record.class_folder} | {Path(record.sample_id).name}"
            draw.text((4, y + thumb_h + 4), "original", fill=(0, 0, 0), font=font)
            draw.text((thumb_w + 4, y + thumb_h + 4), branch, fill=(0, 0, 0), font=font)
            draw.text((4, y + thumb_h + 20), label[:75], fill=(80, 80, 80), font=font)
        _save_image_atomic(canvas, self.preview_dir / f"{branch}_{split}_preview.png")

    def _write_crop_audit_grid(self, split: str, records: list[ImageRecord]) -> None:
        thumb_w, thumb_h = 240, 160
        label_h = 38
        cols = 2
        rows = len(records)
        canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for row, record in enumerate(records):
            y = row * (thumb_h + label_h)
            original = _read_rgb(record.source_path)
            overlay = original.copy()
            odraw = ImageDraw.Draw(overlay)
            odraw.rectangle(crop_box_for_image(original), outline=(255, 32, 32), width=max(2, original.width // 240))
            cropped = crop_local_image(original)
            overlay = ImageOps.contain(overlay, (thumb_w, thumb_h))
            cropped = ImageOps.contain(cropped, (thumb_w, thumb_h))
            canvas.paste(overlay, (0, y))
            canvas.paste(cropped, (thumb_w, y))
            draw.text((4, y + thumb_h + 4), "crop rectangle", fill=(0, 0, 0), font=font)
            draw.text((thumb_w + 4, y + thumb_h + 4), "cropped before resize", fill=(0, 0, 0), font=font)
            draw.text((4, y + thumb_h + 20), f"{record.class_folder} | {Path(record.sample_id).name}"[:75], fill=(80, 80, 80), font=font)
        _save_image_atomic(canvas, self.preview_dir / f"cropped_local_{split}_crop_audit.png")
