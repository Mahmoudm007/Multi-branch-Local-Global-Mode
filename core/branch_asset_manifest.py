from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

from .experiment_registry import build_class_mapping, discover_class_folders


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BRANCH_FILENAME_SUFFIXES = {
    "cropped": "CROP",
    "thermal": "THERMAL",
    "segmented": "SEG",
}

CROP_TOP_REMOVED = 0.25
CROP_BOTTOM_REMOVED = 0.30
CROP_LEFT_REMOVED = 0.10
CROP_RIGHT_REMOVED = 0.10
SEGMENTED_OUTPUT_SIZE = (960, 720)


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    relative_path: Path
    split: str
    class_folder: str
    class_name: str
    label: int
    sample_id: str

    def to_row(self) -> dict[str, str | int]:
        row = asdict(self)
        row["source_path"] = str(self.source_path)
        row["relative_path"] = self.relative_path.as_posix()
        return row


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def scan_defined_dataset(
    dataset_root: Path,
    defined_folder: str = "1 Defined",
    splits: Iterable[str] = ("train", "val"),
    max_images_per_class: int | None = None,
) -> tuple[list[ImageRecord], dict[str, int], dict[int, str]]:
    defined_root = dataset_root / defined_folder
    if not defined_root.exists():
        raise FileNotFoundError(f"Defined dataset folder does not exist: {defined_root}")
    class_to_index, index_to_class, folder_to_class = build_class_mapping(defined_root)
    records: list[ImageRecord] = []
    for split in splits:
        split_root = defined_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Missing split folder: {split_root}")
        for class_dir in discover_class_folders(split_root):
            class_name = folder_to_class.get(class_dir.name)
            if class_name is None:
                normalized = class_dir.name.lower()
                try:
                    from .experiment_registry import normalize_dataset_class_name

                    normalized = normalize_dataset_class_name(class_dir.name)
                except Exception:
                    pass
                class_name = normalized if normalized in class_to_index else next(
                    (name for name in class_to_index if name in normalized),
                    normalized,
                )
            if class_name not in class_to_index:
                raise KeyError(f"Class folder '{class_dir.name}' normalized to '{class_name}', which is not in the training class map")
            label = class_to_index[class_name]
            image_paths = sorted(p for p in class_dir.rglob("*") if is_image_file(p))
            if max_images_per_class is not None:
                image_paths = image_paths[: max(0, max_images_per_class)]
            for image_path in image_paths:
                rel = image_path.relative_to(defined_root)
                sample_id = rel.as_posix()
                records.append(
                    ImageRecord(
                        source_path=image_path,
                        relative_path=rel,
                        split=split,
                        class_folder=class_dir.name,
                        class_name=class_name,
                        label=label,
                        sample_id=sample_id,
                    )
                )
    return records, class_to_index, index_to_class


def generated_relative_path_for_record(record: ImageRecord, branch: str) -> Path:
    suffix = BRANCH_FILENAME_SUFFIXES.get(branch)
    if not suffix:
        return record.relative_path
    return record.relative_path.with_name(f"{record.relative_path.stem}_{suffix}{record.relative_path.suffix}")


def output_path_for_record(branch_defined_root: Path, record: ImageRecord, branch: str | None = None) -> Path:
    relative_path = generated_relative_path_for_record(record, branch) if branch else record.relative_path
    return branch_defined_root / relative_path


def crop_box_for_size(width: int, height: int) -> tuple[int, int, int, int]:
    left = int(math.floor(CROP_LEFT_REMOVED * width))
    top = int(math.floor(CROP_TOP_REMOVED * height))
    right = int(math.ceil((1.0 - CROP_RIGHT_REMOVED) * width))
    bottom = int(math.ceil((1.0 - CROP_BOTTOM_REMOVED) * height))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    return left, top, right, bottom


def expected_output_size_for_record(record: ImageRecord, branch: str) -> tuple[int, int] | None:
    if branch in {"cropped", "thermal"}:
        with Image.open(record.source_path) as image:
            width, height = ImageOps.exif_transpose(image).size
        left, top, right, bottom = crop_box_for_size(width, height)
        return right - left, bottom - top
    if branch == "segmented":
        return SEGMENTED_OUTPUT_SIZE
    return None


def validate_image_file(path: Path) -> tuple[bool, str, tuple[int, int] | None, str | None]:
    try:
        if not path.exists():
            return False, "missing", None, None
        if path.stat().st_size <= 0:
            return False, "empty", None, None
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
        if size[0] <= 0 or size[1] <= 0:
            return False, "invalid_size", size, mode
        if mode not in {"RGB", "RGBA", "L", "P"}:
            return False, f"unexpected_mode:{mode}", size, mode
        return True, "ok", size, mode
    except Exception as exc:  # noqa: BLE001 - validation must report all image failures
        return False, f"unreadable:{exc}", None, None


def validate_generated_image_file(path: Path, record: ImageRecord, branch: str) -> tuple[bool, str, tuple[int, int] | None, str | None]:
    valid, message, size, mode = validate_image_file(path)
    if not valid:
        return valid, message, size, mode
    expected_size = expected_output_size_for_record(record, branch)
    if expected_size is not None and size != expected_size:
        return (
            False,
            f"wrong_size:{size[0]}x{size[1]}_expected_{expected_size[0]}x{expected_size[1]}",
            size,
            mode,
        )
    return valid, message, size, mode


def summarize_records(records: Iterable[ImageRecord]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        split_summary = summary.setdefault(record.split, {})
        split_summary[record.class_folder] = split_summary.get(record.class_folder, 0) + 1
    return summary
