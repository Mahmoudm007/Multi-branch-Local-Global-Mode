from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress falls back to plain prints
    tqdm = None

from .data_loading import build_dataloaders, compute_class_weights, prepare_data_bundle
from .evaluation_artifacts import (
    build_prediction_rows,
    compute_metrics,
    ensure_run_layout,
    save_classification_report_text,
    save_confusion,
    save_embedding_and_advanced_outputs,
    save_epoch_high_loss_samples,
    save_grad_saliency_panels,
    save_patch_occlusion_and_faithfulness,
    save_predictions,
    save_sample_panels,
    save_standard_plots,
    save_csv,
)
from .experiment_registry import (
    MODEL_FAMILIES,
    ExperimentSpec,
    build_experiments,
    get_experiment,
)
from .multibranch_model import FiveBranchExperimentModel, build_timm_backbone, resolve_timm_model_name
from .progress_tracker import CSVProgressTracker, atomic_write_json, atomic_write_text, ensure_dir
from .tfidf_text import settings_from_config


PROGRESS_FIELDS = (
    "key",
    "model",
    "experiment",
    "fusion",
    "status",
    "started_at",
    "completed_at",
    "seconds",
    "run_dir",
    "message",
)


REQUIRED_COMPLETION_ARTIFACTS = (
    "checkpoints/best.pt",
    "checkpoints/last.pt",
    "logs/history.csv",
    "high_loss_samples/high_loss_epoch_manifest.csv",
    "metrics/metrics.json",
    "predictions/predictions.csv",
    "confusion/confusion_matrix_raw.csv",
    "confusion/confusion_matrix_normalized.png",
    "gradcam_saliency/gradcam_saliency_manifest.csv",
    "gradcam_heatmap/gradcam_heatmap_manifest.csv",
    "true_vs_pred/true_vs_pred_image_manifest.csv",
    "metadata/run_summary.json",
)

CURRENT_ARTIFACT_SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "dataset_root": "Dataset_classes_v1",
        "defined_folder": "1 Defined",
        "asset_root": "Generated_Branches",
        "image_size": 224,
        "allow_missing_generated_assets": True,
    },
    "training": {
        "seed": 42,
        "deterministic": True,
        "device": "auto",
        "epochs": 30,
        "batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "mixed_precision": True,
        "gradient_accumulation_steps": 1,
        "early_stopping_patience": 8,
        "monitor_metric": "val_accuracy",
        "checkpoint_metric": "val_accuracy",
        "progress_bar": True,
        "progress_update_steps": 1,
        "epoch_progress_log": True,
        "epoch_progress_to_stderr": True,
        "stop_on_run_failure": True,
        "class_weights": {
            "enabled": True,
            "boosts": {"fully": 1.25, "one track partly": 2.0},
        },
    },
    "pretrained": {
        "requested": True,
        "allow_random_init_if_missing": False,
        "allow_remote_download": False,
        "train_random_init_if_pretrained_fails": True,
        "fallback_random_epochs": 35,
    },
    "tfidf": {
        "max_features": 4000,
        "ngram_min": 1,
        "ngram_max": 2,
        "min_df": 1,
        "stop_words": None,
        "use_svd": True,
        "svd_components": 128,
        "svd_normalize": True,
    },
    "fusion": {
        "mode": "gated",
        "hidden_dim": 512,
        "aux_hidden_dim": 512,
        "aux_align_dim": 256,
        "dropout": 0.2,
        "auxiliary_loss_weight": 0.15,
        "separate_branch_backbones": False,
    },
    "evaluation": {"top_k_samples": 20, "high_loss_top_k_per_epoch": 50},
    "explainability": {
        "splits": ["val"],
        "comparison_samples_per_class": 2,
        "save_gradcam_all_validation": True,
        "gradcam_saliency_layout": "gradcam_saliency/<TrueClass>/<original_image_stem>.png",
        "gradcam_heatmap_layout": "gradcam_heatmap/<TrueClass>/<original_image_stem>.png",
    },
    "advanced_analysis": {
        "enable_embedding_analysis": True,
        "enable_cka": True,
        "enable_cca": True,
        "enable_transformer_attribution": True,
        "enable_tcav": True,
        "enable_retrieval_boards": True,
        "enable_selective_classification": True,
        "enable_conformal": True,
        "enable_trust_score": True,
        "enable_faithfulness_tests": True,
        "enable_data_cartography": True,
        "enable_data_attribution": True,
        "enable_regional_shap": False,
    },
    "experiment_overrides": {
        "exp_original_thermal_segmented_cropped_auxtext": {
            "epochs": 70,
            "early_stopping_patience": 25,
        }
    },
}


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs")
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = deep_update(config, loaded)
    return config


def set_deterministic(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def is_complete(run_dir: Path) -> bool:
    for rel in REQUIRED_COMPLETION_ARTIFACTS:
        path = run_dir / rel
        if not path.exists() or path.stat().st_size <= 0:
            return False
    try:
        summary = json.loads((run_dir / "metadata" / "run_summary.json").read_text(encoding="utf-8"))
        return summary.get("status") == "complete" and int(summary.get("artifact_schema_version", 0)) >= CURRENT_ARTIFACT_SCHEMA_VERSION
    except Exception:
        return False


def metric_mode(metric_name: str) -> str:
    lowered = metric_name.lower()
    if any(token in lowered for token in ("loss", "error", "ece", "brier")):
        return "min"
    return "max"


def initial_metric_value(mode: str) -> float:
    return float("inf") if mode == "min" else float("-inf")


def metric_improved(value: float, best_value: float, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    if mode == "min":
        return value < best_value
    return value > best_value


def resolve_metric_value(
    metric_name: str,
    row: dict[str, Any],
    train_stats: dict[str, Any],
    val_stats: dict[str, Any],
) -> float:
    candidates: list[Any] = []
    if metric_name in row:
        candidates.append(row[metric_name])
    if metric_name.startswith("val_"):
        candidates.append(val_stats.get(metric_name.removeprefix("val_")))
    elif metric_name.startswith("train_"):
        candidates.append(train_stats.get(metric_name))
    else:
        candidates.extend((val_stats.get(metric_name), train_stats.get(metric_name)))

    for value in candidates:
        if value is not None:
            return float(value)

    available = ", ".join(sorted(row))
    raise KeyError(f"Metric {metric_name!r} is not available. Available metrics: {available}")


@dataclass
class RunnerArgs:
    output_root: Path
    config_path: Path | None = None
    model: str | None = None
    experiment: str | None = None
    fusion: str | None = None
    resume: bool = True
    skip_completed: bool = True
    include_original_only: bool = True
    dry_run: bool = False
    dataset_root: Path | None = None
    asset_root: Path | None = None
    defined_folder: str | None = None
    device: str | None = None
    batch_size: int | None = None
    epochs: int | None = None
    num_workers: int | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None


class ExperimentRunner:
    def __init__(self, args: RunnerArgs) -> None:
        self.args = args
        self.config = load_config(args.config_path)
        self._apply_cli_overrides()
        self.output_root = ensure_dir(args.output_root)
        self.global_root = ensure_dir(self.output_root / "_global_comparison_per_combination")
        self.global_metadata = ensure_dir(self.global_root / "metadata")
        self.progress = CSVProgressTracker(self.global_metadata / "progress_runs.csv", PROGRESS_FIELDS)
        training = self.config["training"]
        set_deterministic(int(training["seed"]), bool(training.get("deterministic", True)))
        self.device = resolve_device(str(training.get("device", "auto")))
        self.capability_manifest: dict[str, Any] = {}

    def _apply_cli_overrides(self) -> None:
        if self.args.dataset_root is not None:
            self.config["data"]["dataset_root"] = str(self.args.dataset_root)
        if self.args.asset_root is not None:
            self.config["data"]["asset_root"] = str(self.args.asset_root)
        if self.args.defined_folder is not None:
            self.config["data"]["defined_folder"] = self.args.defined_folder
        if self.args.device is not None:
            self.config["training"]["device"] = self.args.device
        if self.args.batch_size is not None:
            self.config["training"]["batch_size"] = self.args.batch_size
        if self.args.epochs is not None:
            self.config["training"]["epochs"] = self.args.epochs
        if self.args.num_workers is not None:
            self.config["training"]["num_workers"] = self.args.num_workers
        if self.args.fusion is not None:
            self.config["fusion"]["mode"] = self.args.fusion

    def selected_models(self) -> list[str]:
        if self.args.model:
            if self.args.model not in MODEL_FAMILIES:
                raise KeyError(f"Unknown model family '{self.args.model}'. Valid: {', '.join(MODEL_FAMILIES)}")
            return [self.args.model]
        return list(MODEL_FAMILIES)

    def selected_experiments(self) -> list[ExperimentSpec]:
        if self.args.experiment:
            return [get_experiment(self.args.experiment, include_original_only=True)]
        return build_experiments(include_original_only=self.args.include_original_only)

    def write_experiment_manifest(self, experiments: list[ExperimentSpec]) -> None:
        atomic_write_json(
            self.global_metadata / "experiment_manifest.json",
            {
                "experiments": [
                    {
                        "name": spec.name,
                        "branches": list(spec.branches),
                        "visual_branches": list(spec.visual_branches),
                        "uses_auxiliary_text": spec.uses_aux_text,
                    }
                    for spec in experiments
                ],
                "default_excludes_original_only": not self.args.include_original_only,
                "created_at": utc_now(),
            },
        )

    def probe_model(self, model_name: str) -> dict[str, Any]:
        if model_name in self.capability_manifest:
            return self.capability_manifest[model_name]
        pretrained = self.config["pretrained"]
        build = build_timm_backbone(
            model_name,
            pretrained_requested=bool(pretrained.get("requested", True)),
            allow_random_fallback=bool(pretrained.get("train_random_init_if_pretrained_fails", True)),
            allow_remote_download=bool(pretrained.get("allow_remote_download", False)),
        )
        # Do not keep the probe model in memory.
        if build.model is not None:
            del build.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        info = {
            "model_family": model_name,
            "timm_model_name": build.timm_model_name or resolve_timm_model_name(model_name),
            "status": build.status,
            "available": build.status.startswith("available"),
            "pretrained": build.pretrained,
            "visual_dim": build.visual_dim,
            "error": build.error,
        }
        self.capability_manifest[model_name] = info
        atomic_write_json(self.global_metadata / "model_capability_manifest.json", self.capability_manifest)
        return info

    def run(self) -> None:
        models = self.selected_models()
        experiments = self.selected_experiments()
        self.write_experiment_manifest(experiments)
        if self.args.dry_run:
            self._write_dry_run(models, experiments)
            return
        for model_name in models:
            capability = self.probe_model(model_name)
            if not capability.get("available"):
                self._record_model_skip(model_name, experiments, capability)
                continue
            for experiment in experiments:
                self.run_one(model_name, experiment)
            self.write_model_comparison(model_name)
        self.write_global_comparisons(experiments)
        atomic_write_json(self.global_metadata / "model_capability_manifest.json", self.capability_manifest)

    def _write_dry_run(self, models: list[str], experiments: list[ExperimentSpec]) -> None:
        rows = []
        for model_name in models:
            capability = self.probe_model(model_name)
            for experiment in experiments:
                rows.append(
                    {
                        "model": model_name,
                        "experiment": experiment.name,
                        "branches": " + ".join(experiment.branches),
                        "fusion": self.config["fusion"]["mode"],
                        "model_available": capability.get("available"),
                        "timm_model_name": capability.get("timm_model_name"),
                    }
                )
        save_csv(self.global_metadata / "dry_run_plan.csv", rows)

    def _record_model_skip(self, model_name: str, experiments: list[ExperimentSpec], capability: dict[str, Any]) -> None:
        for experiment in experiments:
            self.progress.append(
                {
                    "key": f"{model_name}/{experiment.name}/{self.config['fusion']['mode']}",
                    "model": model_name,
                    "experiment": experiment.name,
                    "fusion": self.config["fusion"]["mode"],
                    "status": "skipped_model_unavailable",
                    "started_at": utc_now(),
                    "completed_at": utc_now(),
                    "seconds": 0,
                    "run_dir": "",
                    "message": capability.get("error", "model unavailable"),
                }
            )

    def run_one(self, model_name: str, experiment: ExperimentSpec) -> None:
        fusion_mode = str(self.config["fusion"].get("mode", "gated"))
        run_dir = self.output_root / model_name / experiment.name
        ensure_run_layout(run_dir)
        key = f"{model_name}/{experiment.name}/{fusion_mode}"
        if self.args.skip_completed and is_complete(run_dir):
            self.progress.append(
                {
                    "key": key,
                    "model": model_name,
                    "experiment": experiment.name,
                    "fusion": fusion_mode,
                    "status": "skipped_complete",
                    "started_at": utc_now(),
                    "completed_at": utc_now(),
                    "seconds": 0,
                    "run_dir": str(run_dir),
                    "message": "completion artifacts present",
                }
            )
            return

        started = utc_now()
        start_time = time.perf_counter()
        self.progress.append(
            {
                "key": key,
                "model": model_name,
                "experiment": experiment.name,
                "fusion": fusion_mode,
                "status": "started",
                "started_at": started,
                "completed_at": "",
                "seconds": "",
                "run_dir": str(run_dir),
                "message": "",
            }
        )
        try:
            summary = self._train_and_evaluate(model_name, experiment, run_dir)
        except Exception as exc:  # noqa: BLE001
            atomic_write_json(
                run_dir / "metadata" / "run_summary.json",
                {
                    "status": "failed",
                    "model": model_name,
                    "experiment": experiment.name,
                    "fusion": fusion_mode,
                    "error": str(exc),
                    "started_at": started,
                    "completed_at": utc_now(),
                },
            )
            self.progress.append(
                {
                    "key": key,
                    "model": model_name,
                    "experiment": experiment.name,
                    "fusion": fusion_mode,
                    "status": "failed",
                    "started_at": started,
                    "completed_at": utc_now(),
                    "seconds": f"{time.perf_counter() - start_time:.2f}",
                    "run_dir": str(run_dir),
                    "message": str(exc),
                }
            )
            print(
                f"[RUN FAILED] model={model_name} experiment={experiment.name} error={exc}",
                flush=True,
            )
            if bool(self.config.get("training", {}).get("stop_on_run_failure", True)):
                raise
            return

        self.progress.append(
            {
                "key": key,
                "model": model_name,
                "experiment": experiment.name,
                "fusion": fusion_mode,
                "status": "complete",
                "started_at": started,
                "completed_at": utc_now(),
                "seconds": f"{time.perf_counter() - start_time:.2f}",
                "run_dir": str(run_dir),
                "message": f"best_epoch={summary.get('best_epoch')}",
            }
        )

    def _train_and_evaluate(self, model_name: str, experiment: ExperimentSpec, run_dir: Path) -> dict[str, Any]:
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]
        fusion_cfg = self.config["fusion"]
        pretrained_cfg = self.config["pretrained"]
        experiment_train_cfg = self.config.get("experiment_overrides", {}).get(experiment.name, {})
        configured_epochs = (
            int(train_cfg.get("epochs", 25))
            if self.args.epochs is not None
            else int(experiment_train_cfg.get("epochs", train_cfg.get("epochs", 25)))
        )
        configured_patience = int(
            experiment_train_cfg.get(
                "early_stopping_patience",
                train_cfg.get("early_stopping_patience", 8),
            )
        )
        dataset_root = Path(data_cfg["dataset_root"])
        asset_root = Path(data_cfg["asset_root"])
        defined_folder = str(data_cfg.get("defined_folder", "1 Defined"))
        metadata_dir = run_dir / "metadata"
        tfidf_settings = settings_from_config(self.config)
        bundle = prepare_data_bundle(
            dataset_root=dataset_root,
            asset_root=asset_root,
            defined_folder=defined_folder,
            experiment=experiment,
            tfidf_settings=tfidf_settings,
            metadata_dir=metadata_dir,
            max_train_samples=self.args.max_train_samples,
            max_val_samples=self.args.max_val_samples,
            seed=int(train_cfg.get("seed", 42)),
            allow_missing_generated_assets=bool(data_cfg.get("allow_missing_generated_assets", True)),
        )
        excluded_assets = int(bundle.asset_filter_summary.get("excluded_records", 0))
        if excluded_assets:
            print(
                (
                    f"[DATA FILTER] experiment={experiment.name} excluded_records={excluded_assets} "
                    f"kept_records={bundle.asset_filter_summary.get('kept_records')} "
                    f"branch_issue_counts={bundle.asset_filter_summary.get('branch_issue_counts')}"
                ),
                flush=True,
            )
        class_names = [bundle.index_to_class[idx] for idx in sorted(bundle.index_to_class)]
        train_loader, val_loader = build_dataloaders(
            bundle=bundle,
            dataset_root=dataset_root,
            asset_root=asset_root,
            defined_folder=defined_folder,
            experiment=experiment,
            image_size=int(data_cfg.get("image_size", 224)),
            batch_size=int(train_cfg.get("batch_size", 32)),
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            persistent_workers=bool(train_cfg.get("persistent_workers", True)),
            prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        )
        print(
            (
                f"\n[RUN] model={model_name} experiment={experiment.name} "
                f"fusion={fusion_cfg.get('mode', 'gated')} device={self.device} "
                f"epochs={configured_epochs} early_stop_patience={configured_patience} "
                f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
            ),
            flush=True,
        )
        model = FiveBranchExperimentModel(
            backbone_family=model_name,
            experiment=experiment,
            num_classes=len(class_names),
            hidden_dim=int(fusion_cfg.get("hidden_dim", 512)),
            fusion_mode=str(fusion_cfg.get("mode", "gated")),
            dropout=float(fusion_cfg.get("dropout", 0.2)),
            aux_feature_dim=bundle.aux_feature_dim,
            aux_hidden_dim=int(fusion_cfg.get("aux_hidden_dim", 512)),
            aux_align_dim=int(fusion_cfg.get("aux_align_dim", 256)),
            separate_branch_backbones=bool(fusion_cfg.get("separate_branch_backbones", False)),
            pretrained_requested=bool(pretrained_cfg.get("requested", True)),
            allow_random_fallback=bool(pretrained_cfg.get("train_random_init_if_pretrained_fails", True)),
            allow_remote_download=bool(pretrained_cfg.get("allow_remote_download", False)),
        ).to(self.device)
        parameter_counts = count_model_parameters(model)
        print(
            (
                f"[MODEL PARAMS] model={model_name} experiment={experiment.name} "
                f"timm={model.backbone_build.timm_model_name} "
                f"total={parameter_counts['total']:,} ({parameter_counts['total_millions']:.2f}M) "
                f"trainable={parameter_counts['trainable']:,} ({parameter_counts['trainable_millions']:.2f}M) "
                f"frozen={parameter_counts['frozen']:,} ({parameter_counts['frozen_millions']:.2f}M)"
            ),
            flush=True,
        )
        self.capability_manifest[model_name] = {
            **self.capability_manifest.get(model_name, {}),
            **model.metadata(),
            "parameter_counts": parameter_counts,
            "available": True,
        }
        atomic_write_json(self.global_metadata / "model_capability_manifest.json", self.capability_manifest)

        boosts = train_cfg.get("class_weights", {}).get("boosts", {}) if train_cfg.get("class_weights", {}).get("enabled", True) else {}
        weights = compute_class_weights(bundle.train_records, bundle.index_to_class, boosts).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        epochs = configured_epochs
        if (
            self.args.epochs is None
            and not model.backbone_build.pretrained
            and bool(pretrained_cfg.get("train_random_init_if_pretrained_fails", True))
        ):
            epochs = max(epochs, int(pretrained_cfg.get("fallback_random_epochs", epochs)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        scaler = make_grad_scaler(enabled=bool(train_cfg.get("mixed_precision", True)) and self.device.type == "cuda")
        grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
        aux_weight = float(fusion_cfg.get("auxiliary_loss_weight", 0.05)) if experiment.uses_aux_text else 0.0

        history: list[dict[str, Any]] = []
        cartography_rows: list[dict[str, Any]] = []
        high_loss_epoch_rows: list[dict[str, Any]] = []
        high_loss_top_k = int(self.config.get("evaluation", {}).get("high_loss_top_k_per_epoch", 50))
        monitor_metric = str(train_cfg.get("monitor_metric", "val_accuracy"))
        checkpoint_metric = str(train_cfg.get("checkpoint_metric", monitor_metric))
        monitor_mode = metric_mode(monitor_metric)
        checkpoint_mode = metric_mode(checkpoint_metric)
        best_monitor_metric = initial_metric_value(monitor_mode)
        best_checkpoint_metric = initial_metric_value(checkpoint_mode)
        best_epoch = 0
        patience = configured_patience
        stale = 0
        for epoch in range(1, epochs + 1):
            print(f"[EPOCH {epoch}/{epochs}] train start: {model_name}/{experiment.name}", flush=True)
            train_stats, train_cart = self._train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                grad_accum,
                aux_weight,
                epoch,
                epochs,
                model_name,
                experiment.name,
            )
            cartography_rows.extend(train_cart)
            print(f"[EPOCH {epoch}/{epochs}] val start: {model_name}/{experiment.name}", flush=True)
            val_stats, epoch_y_true, epoch_probs, epoch_losses, epoch_payload = self._evaluate(
                model,
                val_loader,
                criterion,
                aux_weight,
                collect_embeddings=False,
                progress_desc=f"{model_name}/{experiment.name} epoch {epoch}/{epochs} val",
            )
            epoch_prediction_rows = build_prediction_rows(
                epoch_payload["sample_ids"],
                epoch_payload["image_paths"],
                epoch_y_true,
                epoch_probs,
                epoch_losses,
                class_names,
            )
            high_loss_epoch_rows.extend(
                save_epoch_high_loss_samples(run_dir, epoch_prediction_rows, epoch, top_k=high_loss_top_k)
            )
            save_csv(run_dir / "high_loss_samples" / "high_loss_epoch_manifest.csv", high_loss_epoch_rows)
            scheduler.step()
            row = {"epoch": epoch, **train_stats, **{f"val_{k}": v for k, v in val_stats.items()}}
            history.append(row)
            save_csv(run_dir / "logs" / "history.csv", history)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "config": self.config,
                    "metadata": model.metadata(),
                    "monitor_metric": monitor_metric,
                    "monitor_value": resolve_metric_value(monitor_metric, row, train_stats, val_stats),
                    "checkpoint_metric": checkpoint_metric,
                    "checkpoint_value": resolve_metric_value(checkpoint_metric, row, train_stats, val_stats),
                },
                run_dir / "checkpoints" / "last.pt",
            )
            monitor_value = resolve_metric_value(monitor_metric, row, train_stats, val_stats)
            checkpoint_value = resolve_metric_value(checkpoint_metric, row, train_stats, val_stats)
            should_stop = False
            if metric_improved(monitor_value, best_monitor_metric, monitor_mode):
                best_monitor_metric = monitor_value
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    should_stop = True
            if metric_improved(checkpoint_value, best_checkpoint_metric, checkpoint_mode):
                best_checkpoint_metric = checkpoint_value
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "config": self.config,
                        "metadata": model.metadata(),
                        "monitor_metric": monitor_metric,
                        "monitor_value": monitor_value,
                        "checkpoint_metric": checkpoint_metric,
                        "checkpoint_value": checkpoint_value,
                    },
                    run_dir / "checkpoints" / "best.pt",
                )
            epoch_line = (
                f"[EPOCH {epoch}/{epochs}] done: "
                f"train_loss={train_stats['train_loss']:.4f} "
                f"train_acc={train_stats['train_accuracy']:.4f} "
                f"train_f1={train_stats['train_macro_f1']:.4f} "
                f"train_precision={train_stats['train_macro_precision']:.4f} "
                f"train_recall={train_stats['train_macro_recall']:.4f} "
                f"train_top2={train_stats['train_top2_accuracy']:.4f} "
                f"train_top3={train_stats['train_top3_accuracy']:.4f} "
                f"val_loss={val_stats['loss']:.4f} "
                f"val_acc={val_stats['accuracy']:.4f} "
                f"val_f1={val_stats['macro_f1']:.4f} "
                f"val_precision={val_stats['macro_precision']:.4f} "
                f"val_recall={val_stats['macro_recall']:.4f} "
                f"val_bal_acc={val_stats['balanced_accuracy']:.4f} "
                f"val_top2={val_stats['top2_accuracy']:.4f} "
                f"val_top3={val_stats['top3_accuracy']:.4f} "
                f"val_aux={val_stats['aux_loss']:.4f} "
                f"{monitor_metric}={monitor_value:.4f} "
                f"best_{checkpoint_metric}={best_checkpoint_metric:.4f} "
                f"best_epoch={best_epoch}"
            )
            self._emit_epoch_progress(run_dir, epoch_line)
            if should_stop:
                self._emit_epoch_progress(
                    run_dir,
                    f"[EARLY STOP] model={model_name} experiment={experiment.name} epoch={epoch} stale_epochs={stale}",
                )
                break

        if (run_dir / "checkpoints" / "best.pt").exists():
            checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=self.device)
            model.load_state_dict(checkpoint["model"])

        print(f"[FINAL EVAL] model={model_name} experiment={experiment.name}", flush=True)
        val_stats, y_true, probs, losses, payload = self._evaluate(
            model,
            val_loader,
            criterion,
            aux_weight,
            collect_embeddings=True,
            progress_desc=f"{model_name}/{experiment.name} final val",
        )
        prediction_rows = save_predictions(run_dir, payload["sample_ids"], payload["image_paths"], y_true, probs, losses, class_names)
        metrics, calibration_rows = compute_metrics(y_true, probs, class_names)
        metrics.update(
            {
                "val_loss": float(val_stats["loss"]),
                "val_accuracy": float(val_stats["accuracy"]),
                "monitor_metric": monitor_metric,
                "best_monitor_metric": float(best_monitor_metric),
                "checkpoint_metric": checkpoint_metric,
                "best_checkpoint_metric": float(best_checkpoint_metric),
                "best_epoch": best_epoch,
            }
        )
        atomic_write_json(run_dir / "metrics" / "metrics.json", metrics)
        save_csv(run_dir / "metrics" / "metrics.csv", [flatten_metrics(metrics)])
        save_csv(run_dir / "metrics" / "calibration_table.csv", calibration_rows)
        save_classification_report_text(run_dir, metrics)
        save_confusion(run_dir, y_true, probs, class_names)
        save_standard_plots(run_dir, y_true, probs, class_names, calibration_rows)
        save_sample_panels(run_dir, prediction_rows, int(self.config.get("evaluation", {}).get("top_k_samples", 20)))
        save_grad_saliency_panels(
            model,
            val_loader,
            run_dir,
            class_names,
            self.device,
            max_per_class=int(self.config.get("explainability", {}).get("comparison_samples_per_class", 2)),
        )
        if self.config.get("advanced_analysis", {}).get("enable_embedding_analysis", True):
            save_embedding_and_advanced_outputs(run_dir, payload, class_names, experiment.uses_aux_text)
        if self.config.get("advanced_analysis", {}).get("enable_transformer_attribution", True) or self.config.get("advanced_analysis", {}).get("enable_faithfulness_tests", True):
            save_patch_occlusion_and_faithfulness(
                model,
                val_loader,
                run_dir,
                class_names,
                self.device,
                max_samples=3,
                grid_size=4,
            )
        if cartography_rows and self.config.get("advanced_analysis", {}).get("enable_data_cartography", True):
            self._write_cartography(run_dir, cartography_rows)
        self._write_auxiliary_diagnostics(run_dir, payload, y_true, probs, experiment.uses_aux_text, model, val_loader, bundle)
        self._write_metadata(run_dir, model, experiment, bundle, class_names, best_epoch, metrics)
        return {"best_epoch": best_epoch, "metrics": metrics}

    def _train_epoch(
        self,
        model: FiveBranchExperimentModel,
        loader: Any,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scaler: Any,
        grad_accum: int,
        aux_weight: float,
        epoch: int,
        total_epochs: int,
        model_name: str,
        experiment_name: str,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_aux = 0.0
        total = 0
        correct = 0
        labels_epoch: list[np.ndarray] = []
        preds_epoch: list[np.ndarray] = []
        probs_epoch: list[np.ndarray] = []
        iteration_store = _empty_iteration_store()
        cartography_rows: list[dict[str, Any]] = []
        optimizer.zero_grad(set_to_none=True)
        progress = self._progress_bar(
            loader,
            total=len(loader),
            desc=f"{model_name}/{experiment_name} epoch {epoch}/{total_epochs} train",
        )
        update_steps = max(1, int(self.config.get("training", {}).get("progress_update_steps", 5)))
        for step, batch in enumerate(progress, start=1):
            images = {branch: tensor.to(self.device, non_blocking=True) for branch, tensor in batch["images"].items()}
            labels = batch["labels"].to(self.device, non_blocking=True)
            aux = batch["aux_features"].to(self.device, non_blocking=True)
            aux_mask = batch["aux_mask"].to(self.device, non_blocking=True)
            with autocast_context(enabled=scaler.is_enabled(), device_type=self.device.type):
                features = model.extract_analysis_features(images, aux, aux_mask)
                logits = features["logits"]
                ce = criterion(logits, labels).mean()
                aux_loss = model.auxiliary_alignment_loss(features) if aux_weight > 0 else logits.new_tensor(0.0)
                loss = ce + aux_weight * aux_loss
                scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            if step % grad_accum == 0 or step == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                batch_size = int(labels.shape[0])
                total += batch_size
                total_loss += float(loss.detach().item()) * batch_size
                total_ce += float(ce.detach().item()) * batch_size
                total_aux += float(aux_loss.detach().item()) * batch_size
                true_probs = probs.gather(1, labels.view(-1, 1)).squeeze(1).detach().cpu().numpy()
                pred_np = preds.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                labels_epoch.append(labels_np)
                preds_epoch.append(pred_np)
                probs_epoch.append(probs.detach().cpu().numpy())
                _append_iteration_arrays(
                    iteration_store,
                    _compute_iteration_arrays(model, features, labels, probs, preds),
                )
                for i, sample_id in enumerate(batch["sample_ids"]):
                    cartography_rows.append(
                        {
                            "epoch": epoch,
                            "sample_id": sample_id,
                            "true_index": int(labels_np[i]),
                            "pred_index": int(pred_np[i]),
                            "true_probability": float(true_probs[i]),
                            "correct": bool(pred_np[i] == labels_np[i]),
                        }
                    )
            if step % update_steps == 0 or step == len(loader):
                running_denom = max(1, total)
                labels_running = np.concatenate(labels_epoch) if labels_epoch else np.zeros(0, dtype=int)
                preds_running = np.concatenate(preds_epoch) if preds_epoch else np.zeros(0, dtype=int)
                probs_running = np.concatenate(probs_epoch) if probs_epoch else None
                running_class_stats = _classification_stats_from_arrays(labels_running, preds_running, probs_running)
                running_loss_metrics = _loss_metric_values(total_loss, total_ce, total_aux, running_denom)
                running_extra = _summarize_iteration_store(iteration_store)
                iteration_line = _format_iteration_metrics(
                    "TRAIN",
                    epoch,
                    total_epochs,
                    step,
                    len(loader),
                    running_class_stats,
                    running_loss_metrics,
                    running_extra,
                )
                print(iteration_line, flush=True)
                self._set_progress_postfix(
                    progress,
                    {
                        "loss": f"{total_loss / running_denom:.4f}",
                        "acc": f"{correct / running_denom:.3f}",
                        "f1": f"{running_class_stats['macro_f1']:.3f}",
                        "ce": f"{total_ce / running_denom:.4f}",
                    },
                )
        denom = max(1, total)
        epoch_metrics = _classification_stats_from_arrays(
            np.concatenate(labels_epoch) if labels_epoch else np.zeros(0, dtype=int),
            np.concatenate(preds_epoch) if preds_epoch else np.zeros(0, dtype=int),
            np.concatenate(probs_epoch) if probs_epoch else None,
        )
        epoch_loss_metrics = _loss_metric_values(total_loss, total_ce, total_aux, denom)
        epoch_extra_metrics = _summarize_iteration_store(iteration_store)
        train_stats = {
            "train_loss": total_loss / denom,
            "train_ce_loss": total_ce / denom,
            "train_aux_loss": total_aux / denom,
            "train_accuracy": correct / denom,
            "train_balanced_accuracy": epoch_metrics["balanced_accuracy"],
            "train_macro_precision": epoch_metrics["macro_precision"],
            "train_macro_recall": epoch_metrics["macro_recall"],
            "train_macro_f1": epoch_metrics["macro_f1"],
            "train_weighted_precision": epoch_metrics["weighted_precision"],
            "train_weighted_recall": epoch_metrics["weighted_recall"],
            "train_weighted_f1": epoch_metrics["weighted_f1"],
            "train_top2_accuracy": epoch_metrics["top2_accuracy"],
            "train_top3_accuracy": epoch_metrics["top3_accuracy"],
        }
        train_stats.update({f"train_{key}": value for key, value in epoch_loss_metrics.items()})
        train_stats.update({f"train_{key}": value for key, value in epoch_extra_metrics.items()})
        return (
            train_stats,
            cartography_rows,
        )

    def _evaluate(
        self,
        model: FiveBranchExperimentModel,
        loader: Any,
        criterion: nn.Module,
        aux_weight: float,
        collect_embeddings: bool,
        progress_desc: str | None = None,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        model.eval()
        total_loss = 0.0
        total_ce = 0.0
        total_aux = 0.0
        total = 0
        labels_all: list[np.ndarray] = []
        probs_all: list[np.ndarray] = []
        losses_all: list[np.ndarray] = []
        preds_all: list[np.ndarray] = []
        iteration_store = _empty_iteration_store()
        payload: dict[str, Any] = {"sample_ids": [], "image_paths": []}
        embedding_store: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            progress = self._progress_bar(loader, total=len(loader), desc=progress_desc or "validation")
            update_steps = max(1, int(self.config.get("training", {}).get("progress_update_steps", 5)))
            for step, batch in enumerate(progress, start=1):
                images = {branch: tensor.to(self.device, non_blocking=True) for branch, tensor in batch["images"].items()}
                labels = batch["labels"].to(self.device, non_blocking=True)
                aux = batch["aux_features"].to(self.device, non_blocking=True)
                aux_mask = batch["aux_mask"].to(self.device, non_blocking=True)
                features = model.extract_analysis_features(images, aux, aux_mask)
                logits = features["logits"]
                ce_each = criterion(logits, labels)
                ce = ce_each.mean()
                aux_loss = model.auxiliary_alignment_loss(features) if aux_weight > 0 else logits.new_tensor(0.0)
                loss = ce + aux_weight * aux_loss
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                batch_size = int(labels.shape[0])
                total += batch_size
                total_loss += float(loss.item()) * batch_size
                total_ce += float(ce.item()) * batch_size
                total_aux += float(aux_loss.item()) * batch_size
                labels_all.append(labels.cpu().numpy())
                probs_all.append(probs.cpu().numpy())
                preds_all.append(preds.cpu().numpy())
                losses_all.append(ce_each.detach().cpu().numpy())
                _append_iteration_arrays(
                    iteration_store,
                    _compute_iteration_arrays(model, features, labels, probs, preds),
                )
                payload["sample_ids"].extend(batch["sample_ids"])
                payload["image_paths"].extend(batch["image_paths"])
                if collect_embeddings:
                    for key, value in features.items():
                        if isinstance(value, torch.Tensor) and (
                            key.endswith("_embedding") or key.endswith("_projected_embedding") or key in {"fused_visual_embedding", "probabilities", "aux_input", "aux_mask"}
                        ):
                            embedding_store.setdefault(key, []).append(value.detach().cpu().numpy())
                if step % update_steps == 0 or step == len(loader):
                    denom_running = max(1, total)
                    labels_running = np.concatenate(labels_all) if labels_all else np.zeros(0, dtype=int)
                    preds_running = np.concatenate(preds_all) if preds_all else np.zeros(0, dtype=int)
                    probs_running = np.concatenate(probs_all) if probs_all else None
                    running_class_stats = _classification_stats_from_arrays(labels_running, preds_running, probs_running)
                    running_loss_metrics = _loss_metric_values(total_loss, total_ce, total_aux, denom_running)
                    running_extra = _summarize_iteration_store(iteration_store)
                    iteration_line = _format_iteration_metrics(
                        "VAL",
                        None,
                        None,
                        step,
                        len(loader),
                        running_class_stats,
                        running_loss_metrics,
                        running_extra,
                    )
                    print(iteration_line, flush=True)
                    self._set_progress_postfix(
                        progress,
                        {
                            "loss": f"{total_loss / denom_running:.4f}",
                            "acc": f"{running_class_stats['accuracy']:.3f}",
                            "f1": f"{running_class_stats['macro_f1']:.3f}",
                            "ce": f"{total_ce / denom_running:.4f}",
                        },
                    )
        labels_np = np.concatenate(labels_all) if labels_all else np.zeros(0, dtype=int)
        probs_np = np.concatenate(probs_all) if probs_all else np.zeros((0, 0), dtype=np.float32)
        losses_np = np.concatenate(losses_all) if losses_all else np.zeros(0, dtype=np.float32)
        for key, chunks in embedding_store.items():
            payload[key] = np.concatenate(chunks, axis=0)
        payload["labels"] = labels_np
        payload["probabilities"] = probs_np
        denom = max(1, total)
        pred_np = probs_np.argmax(axis=1) if probs_np.size else np.zeros(0, dtype=int)
        class_stats = _classification_stats_from_arrays(labels_np, pred_np, probs_np if probs_np.size else None)
        loss_metrics = _loss_metric_values(total_loss, total_ce, total_aux, denom)
        extra_metrics = _summarize_iteration_store(iteration_store)
        stats = {
            "loss": total_loss / denom,
            "ce_loss": total_ce / denom,
            "aux_loss": total_aux / denom,
            **class_stats,
            **loss_metrics,
            **extra_metrics,
        }
        return stats, labels_np, probs_np, losses_np, payload

    def _progress_bar(self, iterable: Any, total: int, desc: str) -> Any:
        enabled = bool(self.config.get("training", {}).get("progress_bar", True))
        if not enabled or tqdm is None:
            print(f"[PROGRESS] {desc}: {total} batches", flush=True)
            return iterable
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            mininterval=1.0,
            ascii=True,
        )

    @staticmethod
    def _set_progress_postfix(progress: Any, values: dict[str, str]) -> None:
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(values, refresh=False)

    def _emit_epoch_progress(self, run_dir: Path, line: str) -> None:
        print(line, flush=True)
        train_cfg = self.config.get("training", {})
        if bool(train_cfg.get("epoch_progress_log", True)):
            log_path = run_dir / "logs" / "epoch_progress.log"
            ensure_dir(log_path.parent)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{utc_now()} {line}\n")
                handle.flush()
        if bool(train_cfg.get("epoch_progress_to_stderr", True)):
            print(line, file=sys.stderr, flush=True)

    def _write_cartography(self, run_dir: Path, rows: list[dict[str, Any]]) -> None:
        save_csv(run_dir / "cartography" / "training_dynamics.csv", rows)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["sample_id"]), []).append(row)
        summary = []
        for sample_id, sample_rows in grouped.items():
            summary.append(
                {
                    "sample_id": sample_id,
                    "mean_true_probability": float(np.mean([r["true_probability"] for r in sample_rows])),
                    "correctness": float(np.mean([1.0 if r["correct"] else 0.0 for r in sample_rows])),
                    "variability": float(np.std([r["true_probability"] for r in sample_rows])),
                }
            )
        save_csv(run_dir / "cartography" / "cartography_summary.csv", summary)

    def _write_auxiliary_diagnostics(
        self,
        run_dir: Path,
        payload: dict[str, Any],
        y_true: np.ndarray,
        probs: np.ndarray,
        aux_active: bool,
        model: FiveBranchExperimentModel,
        val_loader: Any,
        bundle: Any,
    ) -> None:
        if not aux_active or "aux_embedding" not in payload:
            atomic_write_json(run_dir / "metadata" / "auxiliary_diagnostics.json", {"active": False})
            return
        visual = torch.tensor(payload["fused_visual_embedding"])
        aux = torch.tensor(payload["aux_embedding"])
        cosine = F.cosine_similarity(F.normalize(visual, dim=1), F.normalize(aux, dim=1), dim=1).numpy()
        pred = probs.argmax(axis=1)
        rows = []
        for idx, sample_id in enumerate(payload["sample_ids"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "true_index": int(y_true[idx]),
                    "pred_index": int(pred[idx]),
                    "correct": bool(pred[idx] == y_true[idx]),
                    "image_aux_cosine": float(cosine[idx]),
                }
            )
        save_csv(run_dir / "metadata" / "image_aux_cosine_by_sample.csv", rows)
        zero_probs = self._eval_probs_with_aux_override(model, val_loader, mode="zero")
        shuffle_probs = self._eval_probs_with_aux_override(model, val_loader, mode="shuffle")
        zero_delta = float(np.max(np.abs(probs - zero_probs))) if zero_probs.shape == probs.shape else None
        shuffle_delta = float(np.max(np.abs(probs - shuffle_probs))) if shuffle_probs.shape == probs.shape else None
        text_probe = self._text_only_probe(bundle)
        atomic_write_json(
            run_dir / "metadata" / "auxiliary_diagnostics.json",
            {
                "active": True,
                "mean_cosine": float(np.mean(cosine)),
                "correct_mean_cosine": float(np.mean(cosine[pred == y_true])) if np.any(pred == y_true) else None,
                "incorrect_mean_cosine": float(np.mean(cosine[pred != y_true])) if np.any(pred != y_true) else None,
                "zero_aux_max_probability_delta": zero_delta,
                "shuffle_aux_max_probability_delta": shuffle_delta,
                "text_only_probe": text_probe,
                "leakage_rule": "TF-IDF class descriptions are label-derived; final logits are computed from fused visual embeddings only.",
            },
        )

    def _eval_probs_with_aux_override(self, model: FiveBranchExperimentModel, loader: Any, mode: str) -> np.ndarray:
        model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                images = {branch: tensor.to(self.device, non_blocking=True) for branch, tensor in batch["images"].items()}
                aux = batch["aux_features"].to(self.device, non_blocking=True)
                aux_mask = batch["aux_mask"].to(self.device, non_blocking=True)
                if mode == "zero":
                    aux = torch.zeros_like(aux)
                elif mode == "shuffle" and aux.shape[0] > 1:
                    aux = aux[torch.randperm(aux.shape[0], device=aux.device)]
                logits = model(images, aux, aux_mask)
                chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)

    def _text_only_probe(self, bundle: Any) -> dict[str, Any]:
        if bundle.class_description_matrix is None:
            return {"active": False}
        train_x = np.stack([bundle.class_description_matrix[record.label] for record in bundle.train_records])
        train_y = np.array([record.label for record in bundle.train_records])
        val_x = np.stack([bundle.class_description_matrix[record.label] for record in bundle.val_records])
        val_y = np.array([record.label for record in bundle.val_records])
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            clf = LogisticRegression(max_iter=1000, multi_class="auto")
            clf.fit(train_x, train_y)
            pred = clf.predict(val_x)
            return {
                "active": True,
                "method": "logistic_regression_on_tfidf_only",
                "val_accuracy": float(accuracy_score(val_y, pred)),
                "interpretation": "High accuracy demonstrates why direct TF-IDF-to-logit use would leak labels.",
            }
        except Exception as exc:  # noqa: BLE001
            return {"active": True, "method": "logistic_regression_on_tfidf_only", "error": str(exc)}

    def _write_metadata(
        self,
        run_dir: Path,
        model: FiveBranchExperimentModel,
        experiment: ExperimentSpec,
        bundle: Any,
        class_names: list[str],
        best_epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        atomic_write_json(run_dir / "metadata" / "config_snapshot.json", self.config)
        atomic_write_json(
            run_dir / "metadata" / "environment.json",
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "cwd": os.getcwd(),
            },
        )
        atomic_write_json(
            run_dir / "metadata" / "run_summary.json",
            {
                "status": "complete",
                "artifact_schema_version": CURRENT_ARTIFACT_SCHEMA_VERSION,
                "model": model.backbone_family,
                "experiment": experiment.name,
                "branches": list(experiment.branches),
                "fusion": self.config["fusion"]["mode"],
                "best_epoch": best_epoch,
                "class_to_index": bundle.class_to_index,
                "index_to_class": {str(k): v for k, v in bundle.index_to_class.items()},
                "class_names": class_names,
                "asset_filter_summary": bundle.asset_filter_summary,
                "model_metadata": {**model.metadata(), "parameter_counts": count_model_parameters(model)},
                "primary_metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
                "shap_enabled": False,
                "completed_at": utc_now(),
            },
        )

    def write_model_comparison(self, model_name: str) -> None:
        comparison_root = ensure_dir(self.output_root / model_name / "_comparison_across_experiments")
        for subdir in ("tables", "plots", "explainability", "confusion", "gradcam", "metrics", "metadata"):
            ensure_dir(comparison_root / subdir)
        rows = []
        for run_dir in sorted((self.output_root / model_name).iterdir()) if (self.output_root / model_name).exists() else []:
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            metrics_path = run_dir / "metrics" / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {"model": model_name, "experiment": run_dir.name}
            row.update(flatten_metrics(metrics))
            rows.append(row)
        if rows:
            save_csv(comparison_root / "tables" / "metrics_summary.csv", rows)
            atomic_write_json(comparison_root / "metadata" / "comparison_summary.json", {"runs": len(rows), "model": model_name})

    def write_global_comparisons(self, experiments: list[ExperimentSpec]) -> None:
        best_by_experiment: dict[str, Any] = {}
        for experiment in experiments:
            root = ensure_dir(self.global_root / experiment.name)
            for subdir in (
                "data_reports",
                "explainability",
                "tables",
                "plots",
                "embeddings",
                "cka",
                "cca_alignment",
                "selective_classification",
                "conformal",
                "trust_score",
                "cartography",
                "metadata",
            ):
                ensure_dir(root / subdir)
            rows = []
            for model_name in MODEL_FAMILIES:
                metrics_path = self.output_root / model_name / experiment.name / "metrics" / "metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                row = {"model": model_name, "experiment": experiment.name}
                row.update(flatten_metrics(metrics))
                rows.append(row)
            if rows:
                save_csv(root / "tables" / "metrics_summary.csv", rows)
                best = max(rows, key=lambda row: float(row.get("macro_f1", row.get("accuracy", 0.0))))
                best_by_experiment[experiment.name] = best
                atomic_write_text(
                    root / "benchmark_summary_extended.md",
                    f"# {experiment.name}\n\nRuns: {len(rows)}\n\nBest by macro F1: {best.get('model')}\n",
                )
                atomic_write_text(
                    root / "final_benchmark_summary.md",
                    f"# Final Summary: {experiment.name}\n\nBest model: {best.get('model')}\n",
                )
        atomic_write_json(self.global_metadata / "best_model_selection.json", best_by_experiment)


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            flat[key] = value
    return flat


def make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool, device_type: str) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type if device_type in {"cuda", "cpu"} else "cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def module_parameter_dtype(module: nn.Module) -> torch.dtype:
    for parameter in module.parameters(recurse=True):
        return parameter.dtype
    return torch.float32


ITERATION_METRIC_ORDER = (
    "acc",
    "bal_acc",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "top_2_acc",
    "top_3_acc",
    "loss",
    "combined_cls_loss",
    "visual_cls_loss",
    "description_cls_loss",
    "enhanced_visual_loss",
    "align_loss",
    "consistency_loss",
    "contrastive_loss",
    "prototype_separation_loss",
    "fusion_gate",
    "visual_acc",
    "description_acc",
    "branch_agreement",
    "fused_visual_agreement",
    "fused_text_agreement",
    "enhancement_agreement",
    "base_visual_acc",
    "enhanced_visual_acc",
    "true_visual_prob",
    "true_description_prob",
    "true_fused_prob",
    "true_text_similarity",
    "top_text_similarity",
    "avg_fusion_gate",
    "gate_correct_gap",
)


ITERATION_ARRAY_KEYS = (
    "fusion_gate",
    "avg_fusion_gate",
    "branch_agreement",
    "fused_visual_agreement",
    "fused_text_agreement",
    "base_visual_correct",
    "enhanced_visual_correct",
    "true_visual_prob",
    "true_fused_prob",
    "true_text_similarity",
    "top_text_similarity",
)


def _nan() -> float:
    return float("nan")


def _loss_metric_values(total_loss: float, total_ce: float, total_aux: float, denom: int) -> dict[str, float]:
    denom = max(1, denom)
    return {
        "loss": total_loss / denom,
        "combined_cls_loss": total_ce / denom,
        "visual_cls_loss": total_ce / denom,
        "description_cls_loss": _nan(),
        "enhanced_visual_loss": _nan(),
        "align_loss": total_aux / denom,
        "consistency_loss": _nan(),
        "contrastive_loss": _nan(),
        "prototype_separation_loss": _nan(),
    }


def _empty_iteration_store() -> dict[str, list[np.ndarray]]:
    return {key: [] for key in ITERATION_ARRAY_KEYS}


def _append_iteration_arrays(store: dict[str, list[np.ndarray]], arrays: dict[str, np.ndarray]) -> None:
    for key, value in arrays.items():
        if key in store and value.size:
            store[key].append(value.astype(np.float32, copy=False))


def _compute_iteration_arrays(
    model: FiveBranchExperimentModel,
    features: dict[str, Any],
    labels: torch.Tensor,
    probs: torch.Tensor,
    preds: torch.Tensor,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    labels_col = labels.view(-1, 1)
    correct = (preds == labels).detach().float().cpu().numpy()
    arrays["enhanced_visual_correct"] = correct
    arrays["true_fused_prob"] = probs.gather(1, labels_col).squeeze(1).detach().float().cpu().numpy()

    original_projected = features.get("original_projected_embedding")
    fused_visual = features.get("fused_visual_embedding")
    if isinstance(original_projected, torch.Tensor):
        classifier_input = original_projected.to(dtype=module_parameter_dtype(model.classifier))
        base_logits = model.classifier(classifier_input)
        base_probs = torch.softmax(base_logits, dim=1)
        base_preds = base_probs.argmax(dim=1)
        arrays["base_visual_correct"] = (base_preds == labels).detach().float().cpu().numpy()
        arrays["true_visual_prob"] = base_probs.gather(1, labels_col).squeeze(1).detach().float().cpu().numpy()
    if isinstance(fused_visual, torch.Tensor) and isinstance(original_projected, torch.Tensor):
        arrays["fused_visual_agreement"] = _cosine_numpy(fused_visual, original_projected)

    agreement_tensors: list[torch.Tensor] = []
    if isinstance(original_projected, torch.Tensor):
        for key, value in features.items():
            if key.endswith("_projected_embedding") and key != "original_projected_embedding" and isinstance(value, torch.Tensor):
                agreement_tensors.append(F.cosine_similarity(F.normalize(original_projected.float(), dim=1), F.normalize(value.float(), dim=1), dim=1))
    if agreement_tensors:
        arrays["branch_agreement"] = torch.stack(agreement_tensors, dim=1).mean(dim=1).detach().float().cpu().numpy()

    gates = features.get("branch_gates")
    if isinstance(gates, dict) and gates:
        gate_tensors = [value.float().mean(dim=1) for value in gates.values() if isinstance(value, torch.Tensor)]
        if gate_tensors:
            per_sample_gate = torch.stack(gate_tensors, dim=1).mean(dim=1).detach().float().cpu().numpy()
            arrays["fusion_gate"] = per_sample_gate
            arrays["avg_fusion_gate"] = per_sample_gate

    aux_embedding = features.get("aux_embedding")
    if isinstance(fused_visual, torch.Tensor) and isinstance(aux_embedding, torch.Tensor):
        text_similarity = _cosine_numpy(fused_visual, aux_embedding)
        arrays["fused_text_agreement"] = text_similarity
        arrays["true_text_similarity"] = text_similarity
        fused_norm = F.normalize(fused_visual.float(), dim=1)
        aux_norm = F.normalize(aux_embedding.float(), dim=1)
        arrays["top_text_similarity"] = torch.matmul(fused_norm, aux_norm.T).max(dim=1).values.detach().float().cpu().numpy()
    return arrays


def _cosine_numpy(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    return F.cosine_similarity(F.normalize(left.float(), dim=1), F.normalize(right.float(), dim=1), dim=1).detach().float().cpu().numpy()


def _summarize_iteration_store(store: dict[str, list[np.ndarray]]) -> dict[str, float]:
    def mean_of(key: str) -> float:
        chunks = store.get(key, [])
        if not chunks:
            return _nan()
        values = np.concatenate(chunks)
        if values.size == 0:
            return _nan()
        return float(np.nanmean(values))

    summary = {
        "fusion_gate": mean_of("fusion_gate"),
        "visual_acc": mean_of("enhanced_visual_correct"),
        "description_acc": _nan(),
        "branch_agreement": mean_of("branch_agreement"),
        "fused_visual_agreement": mean_of("fused_visual_agreement"),
        "fused_text_agreement": mean_of("fused_text_agreement"),
        "enhancement_agreement": _nan(),
        "base_visual_acc": mean_of("base_visual_correct"),
        "enhanced_visual_acc": mean_of("enhanced_visual_correct"),
        "true_visual_prob": mean_of("true_visual_prob"),
        "true_description_prob": _nan(),
        "true_fused_prob": mean_of("true_fused_prob"),
        "true_text_similarity": mean_of("true_text_similarity"),
        "top_text_similarity": mean_of("top_text_similarity"),
        "avg_fusion_gate": mean_of("avg_fusion_gate"),
        "gate_correct_gap": _nan(),
    }
    gate_chunks = store.get("avg_fusion_gate", [])
    correct_chunks = store.get("enhanced_visual_correct", [])
    if gate_chunks and correct_chunks:
        gates = np.concatenate(gate_chunks)
        correct = np.concatenate(correct_chunks).astype(bool)
        if np.any(correct) and np.any(~correct):
            summary["gate_correct_gap"] = float(np.nanmean(gates[correct]) - np.nanmean(gates[~correct]))
    return summary


def _format_iteration_metrics(
    stage: str,
    epoch: int | None,
    total_epochs: int | None,
    step: int,
    total_steps: int,
    class_stats: dict[str, float],
    loss_metrics: dict[str, float],
    extra_metrics: dict[str, float],
) -> str:
    metrics = {
        "acc": class_stats.get("accuracy", _nan()),
        "bal_acc": class_stats.get("balanced_accuracy", _nan()),
        "macro_precision": class_stats.get("macro_precision", _nan()),
        "macro_recall": class_stats.get("macro_recall", _nan()),
        "macro_f1": class_stats.get("macro_f1", _nan()),
        "weighted_f1": class_stats.get("weighted_f1", _nan()),
        "top_2_acc": class_stats.get("top2_accuracy", _nan()),
        "top_3_acc": class_stats.get("top3_accuracy", _nan()),
        **loss_metrics,
        **extra_metrics,
    }
    prefix = f"{stage} | iter={step}/{total_steps}"
    if epoch is not None and total_epochs is not None:
        prefix = f"{stage} | epoch={epoch}/{total_epochs} | iter={step}/{total_steps}"
    parts = [prefix]
    parts.extend(f"{key}={_format_metric_value(metrics.get(key, _nan()))}" for key in ITERATION_METRIC_ORDER)
    return " | ".join(parts)


def _format_metric_value(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.4f}"


def count_model_parameters(model: nn.Module) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
        "total_millions": total / 1_000_000.0,
        "trainable_millions": trainable / 1_000_000.0,
        "frozen_millions": frozen / 1_000_000.0,
    }


def _classification_stats_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "top2_accuracy": 0.0,
            "top3_accuracy": 0.0,
        }
    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
            weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
            accuracy = float(accuracy_score(y_true, y_pred))
            balanced = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        accuracy = float(np.mean(y_true == y_pred))
        balanced = accuracy
        macro_p = macro_r = macro_f1 = weighted_p = weighted_r = weighted_f1 = 0.0
    top2 = 0.0
    top3 = 0.0
    if probs is not None and probs.size:
        top2 = _topk_accuracy(y_true, probs, 2)
        top3 = _topk_accuracy(y_true, probs, 3)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "top2_accuracy": top2,
        "top3_accuracy": top3,
    }


def _topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    if y_true.size == 0 or probs.size == 0:
        return 0.0
    k = max(1, min(k, probs.shape[1]))
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))
