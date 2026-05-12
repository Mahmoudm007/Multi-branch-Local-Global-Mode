from __future__ import annotations

import argparse
import logging
from pathlib import Path

from core.branch_dataset_builder import GENERATED_IMAGE_BRANCHES, BranchDatasetBuilder


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cropped, CLAHE-Inferno thermal, and classical BEST_COMBINED branch assets."
    )
    parser.add_argument("--dataset-root", default="Dataset_classes", help="Root containing the supervised dataset.")
    parser.add_argument("--defined-folder", default="1 Defined", help="Defined dataset subfolder to process.")
    parser.add_argument("--asset-root", default="Generated_Branches", help="Root for generated branch datasets.")
    parser.add_argument(
        "--branches",
        nargs="+",
        default=list(GENERATED_IMAGE_BRANCHES),
        choices=list(GENERATED_IMAGE_BRANCHES),
        help="Generated branches to build.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to process.")
    parser.add_argument("--max-images-per-class", type=int, default=None, help="Optional smoke-test limit per split/class.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Reserved worker count for branch generation; default 0 keeps processing sequential and resumable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate assets even when valid outputs already exist.")
    parser.add_argument(
        "--skip-completed",
        nargs="?",
        const=True,
        type=parse_bool,
        default=True,
        help="Skip valid or progress-marked completed outputs.",
    )
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate generated image counts and readability.",
    )
    parser.add_argument(
        "--previews",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write random visual audit preview grids.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Preview sampling seed.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    builder = BranchDatasetBuilder(
        dataset_root=Path(args.dataset_root),
        asset_root=Path(args.asset_root),
        defined_folder=args.defined_folder,
        seed=args.seed,
    )
    if int(args.num_workers) != 0:
        logging.warning("Branch generation is currently sequential; --num-workers=%s will be ignored.", args.num_workers)
    logging.info("Generating branches %s from %s/%s with num_workers=0", ", ".join(args.branches), args.dataset_root, args.defined_folder)
    summary = builder.generate(
        branches=tuple(args.branches),
        splits=tuple(args.splits),
        skip_completed=bool(args.skip_completed),
        overwrite=bool(args.overwrite),
        max_images_per_class=args.max_images_per_class,
        validate=bool(args.validate),
        previews=bool(args.previews),
    )
    for branch, stats in summary.items():
        validation = stats.get("validation", {}) if isinstance(stats, dict) else {}
        logging.info(
            "%s: ok=%s skipped=%s errors=%s validation_ok=%s missing=%s corrupt=%s",
            branch,
            stats.get("ok") if isinstance(stats, dict) else "?",
            stats.get("skipped") if isinstance(stats, dict) else "?",
            stats.get("error") if isinstance(stats, dict) else "?",
            validation.get("ok") if isinstance(validation, dict) else "?",
            validation.get("missing") if isinstance(validation, dict) else "?",
            validation.get("corrupt") if isinstance(validation, dict) else "?",
        )


if __name__ == "__main__":
    main()
