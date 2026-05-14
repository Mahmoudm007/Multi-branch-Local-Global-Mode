from __future__ import annotations

import argparse
import logging
from pathlib import Path

from core.experiment_registry import MODEL_FAMILIES, build_experiments
from core.experiment_runner import ExperimentRunner, RunnerArgs


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
    experiments = [spec.name for spec in build_experiments(include_original_only=True)]
    parser = argparse.ArgumentParser(description="Run the five-branch winter road benchmark sequentially.")
    parser.add_argument("--config", default="configs/multibranch_default.yaml", help="YAML config path.")
    parser.add_argument("--output-root", default="Output_v3", help="Benchmark output root.")
    parser.add_argument("--dataset-root", default="Dataset_classes_v1", help="Override supervised dataset root.")
    parser.add_argument("--asset-root", default=None, help="Override generated branch asset root.")
    parser.add_argument("--defined-folder", default=None, help="Override defined dataset folder.")
    parser.add_argument("--model", choices=list(MODEL_FAMILIES), default=None, help="Run one model family across selected experiments.")
    parser.add_argument("--experiment", choices=experiments, default=None, help="Run one experiment across selected models.")
    parser.add_argument("--fusion", choices=["concat", "gated", "film"], default=None, help="Fusion backend.")
    parser.add_argument("--resume", nargs="?", const=True, type=parse_bool, default=True, help="Resume interrupted benchmark.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--skip-completed", nargs="?", const=True, type=parse_bool, default=True, help="Skip robustly completed runs.")
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    parser.add_argument(
        "--include-original-only",
        action="store_true",
        default=True,
        help="Include exp_original_only baseline. Enabled by default.",
    )
    parser.add_argument("--exclude-original-only", dest="include_original_only", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Write the planned run manifest without training.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Smoke-test train sample cap.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Smoke-test val sample cap.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(args.config) if args.config else None
    if config_path is not None and not config_path.exists():
        logging.warning("Config %s does not exist; using built-in defaults.", config_path)
        config_path = None
    runner = ExperimentRunner(
        RunnerArgs(
            output_root=Path(args.output_root),
            config_path=config_path,
            model=args.model,
            experiment=args.experiment,
            fusion=args.fusion,
            resume=bool(args.resume),
            skip_completed=bool(args.skip_completed),
            include_original_only=bool(args.include_original_only),
            dry_run=bool(args.dry_run),
            dataset_root=Path(args.dataset_root) if args.dataset_root else None,
            asset_root=Path(args.asset_root) if args.asset_root else None,
            defined_folder=args.defined_folder,
            device=args.device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
        )
    )
    runner.run()


if __name__ == "__main__":
    main()
