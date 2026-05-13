from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset

from .branch_asset_manifest import ImageRecord, output_path_for_record, scan_defined_dataset, validate_image_file
from .experiment_registry import (
    AUX_TEXT,
    BRANCH_DIRS,
    ORIGINAL,
    VISUAL_BRANCHES,
    ExperimentSpec,
)
from .progress_tracker import atomic_write_json, ensure_dir
from .tfidf_text import TfidfSettings, build_class_description_matrix


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


@dataclass
class DataBundle:
    train_records: list[ImageRecord]
    val_records: list[ImageRecord]
    class_to_index: dict[str, int]
    index_to_class: dict[int, str]
    class_description_matrix: np.ndarray | None
    aux_feature_dim: int
    class_description_rows: list[dict[str, object]]
    asset_filter_summary: dict[str, Any]
    asset_filter_rows: list[dict[str, object]]


class MultiBranchRoadDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        dataset_defined_root: Path,
        asset_root: Path,
        defined_folder: str,
        experiment: ExperimentSpec,
        image_size: int,
        class_description_matrix: np.ndarray | None = None,
    ) -> None:
        self.records = records
        self.dataset_defined_root = dataset_defined_root
        self.asset_root = asset_root
        self.defined_folder = defined_folder
        self.experiment = experiment
        self.image_size = image_size
        self.class_description_matrix = class_description_matrix

    def __len__(self) -> int:
        return len(self.records)

    def _path_for_branch(self, record: ImageRecord, branch: str) -> Path:
        if branch == ORIGINAL:
            return record.source_path
        if branch not in BRANCH_DIRS:
            raise KeyError(f"Unsupported image branch: {branch}")
        branch_root = self.asset_root / BRANCH_DIRS[branch] / self.defined_folder
        return output_path_for_record(branch_root, record, branch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        images: dict[str, torch.Tensor] = {}
        for branch in self.experiment.visual_branches:
            images[branch] = image_to_tensor(self._path_for_branch(record, branch), self.image_size)
        item: dict[str, Any] = {
            "images": images,
            "label": torch.tensor(record.label, dtype=torch.long),
            "sample_id": record.sample_id,
            "image_path": str(record.source_path),
            "relative_path": record.relative_path.as_posix(),
            "label_name": record.class_name,
            "class_folder": record.class_folder,
        }
        if self.experiment.uses_aux_text and self.class_description_matrix is not None:
            item["aux_features"] = torch.from_numpy(self.class_description_matrix[record.label].astype(np.float32, copy=False))
        else:
            item["aux_features"] = torch.zeros(0, dtype=torch.float32)
        return item


def collate_multibranch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    labels = torch.stack([item["label"] for item in batch])
    branch_names = batch[0]["images"].keys()
    images = {branch: torch.stack([item["images"][branch] for item in batch]) for branch in branch_names}
    aux = torch.stack([item["aux_features"] for item in batch])
    return {
        "images": images,
        "labels": labels,
        "aux_features": aux,
        "sample_ids": [item["sample_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "relative_paths": [item["relative_path"] for item in batch],
        "label_names": [item["label_name"] for item in batch],
        "class_folders": [item["class_folder"] for item in batch],
    }


def limit_records(records: list[ImageRecord], max_samples: int | None, seed: int) -> list[ImageRecord]:
    if max_samples is None or max_samples <= 0 or len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    by_label: dict[int, list[ImageRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    selected: list[ImageRecord] = []
    per_class = max(1, max_samples // max(1, len(by_label)))
    for label_records in by_label.values():
        copied = list(label_records)
        rng.shuffle(copied)
        selected.extend(copied[:per_class])
    if len(selected) < max_samples:
        remaining = [record for record in records if record not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])
    return sorted(selected[:max_samples], key=lambda r: r.sample_id)


def validate_experiment_assets(
    records: list[ImageRecord],
    asset_root: Path,
    defined_folder: str,
    experiment: ExperimentSpec,
) -> list[str]:
    errors: list[str] = []
    for branch in experiment.visual_branches:
        if branch == ORIGINAL:
            continue
        branch_root = asset_root / BRANCH_DIRS[branch] / defined_folder
        for record in records:
            path = output_path_for_record(branch_root, record, branch)
            valid, message, _, _ = validate_image_file(path)
            if not valid:
                errors.append(f"{branch}:{record.sample_id}:{message}")
                if len(errors) >= 25:
                    errors.append("asset validation stopped after 25 failures")
                    return errors
    return errors


def filter_records_for_available_assets(
    records: list[ImageRecord],
    asset_root: Path,
    defined_folder: str,
    experiment: ExperimentSpec,
) -> tuple[list[ImageRecord], list[dict[str, object]], dict[str, Any]]:
    generated_branches = [branch for branch in experiment.visual_branches if branch != ORIGINAL]
    if not generated_branches:
        return (
            records,
            [],
            {
                "mode": "no_generated_branches",
                "input_records": len(records),
                "kept_records": len(records),
                "excluded_records": 0,
                "branch_issue_counts": {},
                "split_counts_before": _count_records_by_split(records),
                "split_counts_after": _count_records_by_split(records),
            },
        )

    kept: list[ImageRecord] = []
    rows: list[dict[str, object]] = []
    branch_issue_counts = {branch: 0 for branch in generated_branches}
    excluded_sample_ids: set[str] = set()
    for record in records:
        record_ok = True
        for branch in generated_branches:
            branch_root = asset_root / BRANCH_DIRS[branch] / defined_folder
            path = output_path_for_record(branch_root, record, branch)
            valid, message, size, mode = validate_image_file(path)
            if valid:
                continue
            record_ok = False
            branch_issue_counts[branch] += 1
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "split": record.split,
                    "class_folder": record.class_folder,
                    "class_name": record.class_name,
                    "label": record.label,
                    "branch": branch,
                    "status": message,
                    "expected_path": str(path),
                    "source_path": str(record.source_path),
                    "width": size[0] if size else "",
                    "height": size[1] if size else "",
                    "mode": mode or "",
                }
            )
        if record_ok:
            kept.append(record)
        else:
            excluded_sample_ids.add(record.sample_id)
    summary = {
        "mode": "filter_missing_generated_assets",
        "input_records": len(records),
        "kept_records": len(kept),
        "excluded_records": len(excluded_sample_ids),
        "branch_issue_counts": branch_issue_counts,
        "split_counts_before": _count_records_by_split(records),
        "split_counts_after": _count_records_by_split(kept),
    }
    return kept, rows, summary


def _count_records_by_split(records: list[ImageRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    return counts


def write_asset_filter_report(metadata_dir: Path, rows: list[dict[str, object]], summary: dict[str, Any]) -> None:
    ensure_dir(metadata_dir)
    atomic_write_json(metadata_dir / "asset_filter_summary.json", summary)
    report_path = metadata_dir / "asset_exclusions.csv"
    fieldnames = [
        "sample_id",
        "split",
        "class_folder",
        "class_name",
        "label",
        "branch",
        "status",
        "expected_path",
        "source_path",
        "width",
        "height",
        "mode",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_data_bundle(
    dataset_root: Path,
    asset_root: Path,
    defined_folder: str,
    experiment: ExperimentSpec,
    tfidf_settings: TfidfSettings,
    metadata_dir: Path | None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    seed: int = 42,
    allow_missing_generated_assets: bool = True,
) -> DataBundle:
    records, class_to_index, index_to_class = scan_defined_dataset(dataset_root, defined_folder, ("train", "val"))
    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    train = limit_records(train, max_train_samples, seed)
    val = limit_records(val, max_val_samples, seed + 1)
    selected_records = train + val
    asset_filter_rows: list[dict[str, object]] = []
    asset_filter_summary: dict[str, Any] = {
        "mode": "strict_validation",
        "input_records": len(selected_records),
        "kept_records": len(selected_records),
        "excluded_records": 0,
        "branch_issue_counts": {},
        "split_counts_before": _count_records_by_split(selected_records),
        "split_counts_after": _count_records_by_split(selected_records),
    }
    if allow_missing_generated_assets:
        filtered, asset_filter_rows, asset_filter_summary = filter_records_for_available_assets(
            selected_records,
            asset_root,
            defined_folder,
            experiment,
        )
        train = [record for record in filtered if record.split == "train"]
        val = [record for record in filtered if record.split == "val"]
        if metadata_dir is not None:
            write_asset_filter_report(metadata_dir, asset_filter_rows, asset_filter_summary)
        if not train or not val:
            raise FileNotFoundError(
                f"No usable training data remains for {experiment.name} after filtering missing generated assets. "
                f"Summary: {asset_filter_summary}"
            )
    else:
        asset_errors = validate_experiment_assets(selected_records, asset_root, defined_folder, experiment)
        if asset_errors:
            joined = "\n".join(asset_errors[:25])
            raise FileNotFoundError(f"Missing or corrupt generated assets for {experiment.name}:\n{joined}")
        if metadata_dir is not None:
            write_asset_filter_report(metadata_dir, asset_filter_rows, asset_filter_summary)
    class_description_matrix = None
    aux_feature_dim = 0
    rows: list[dict[str, object]] = []
    if experiment.uses_aux_text:
        class_description_matrix, _, rows = build_class_description_matrix(index_to_class, tfidf_settings, metadata_dir)
        aux_feature_dim = int(class_description_matrix.shape[1])
    return DataBundle(
        train,
        val,
        class_to_index,
        index_to_class,
        class_description_matrix,
        aux_feature_dim,
        rows,
        asset_filter_summary,
        asset_filter_rows,
    )


def build_dataloaders(
    bundle: DataBundle,
    dataset_root: Path,
    asset_root: Path,
    defined_folder: str,
    experiment: ExperimentSpec,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> tuple[DataLoader, DataLoader]:
    defined_root = dataset_root / defined_folder
    train_dataset = MultiBranchRoadDataset(
        bundle.train_records,
        defined_root,
        asset_root,
        defined_folder,
        experiment,
        image_size,
        bundle.class_description_matrix,
    )
    val_dataset = MultiBranchRoadDataset(
        bundle.val_records,
        defined_root,
        asset_root,
        defined_folder,
        experiment,
        image_size,
        bundle.class_description_matrix,
    )
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_multibranch,
    }
    if num_workers > 0:
        common["persistent_workers"] = persistent_workers
        common["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def compute_class_weights(
    records: list[ImageRecord],
    index_to_class: dict[int, str],
    boosts: dict[str, float] | None = None,
) -> torch.Tensor:
    labels = [record.label for record in records]
    num_classes = len(index_to_class)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    boosts = boosts or {}
    for idx, name in index_to_class.items():
        boost = boosts.get(name, boosts.get(name.replace(" ", "_"), 1.0))
        weights[idx] *= float(boost)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
