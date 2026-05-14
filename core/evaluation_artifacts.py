from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.calibration import calibration_curve
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import label_binarize
except Exception:  # pragma: no cover - package availability is checked in runner
    CCA = PCA = TSNE = NearestNeighbors = None
    accuracy_score = average_precision_score = balanced_accuracy_score = brier_score_loss = None
    classification_report = confusion_matrix = f1_score = log_loss = None
    precision_recall_curve = roc_auc_score = roc_curve = label_binarize = None
    calibration_curve = None

from .data_loading import IMAGENET_MEAN, IMAGENET_STD
from .progress_tracker import atomic_write_json, atomic_write_text, ensure_dir


def ensure_run_layout(run_dir: Path) -> None:
    subdirs = [
        "checkpoints",
        "logs",
        "reports",
        "plots",
        "metrics",
        "confusion",
        "per_class",
        "gradcam_saliency",
        "gradcam_heatmap",
        "true_vs_pred",
        "high_loss_samples",
        "comparison_samples",
        "predictions",
        "metadata",
        "embedding_analysis",
        "cka",
        "cca_alignment",
        "transformer_attribution",
        "tcav",
        "retrieval_boards",
        "decision_quality/selective_classification",
        "decision_quality/conformal",
        "decision_quality/trust_score",
        "faithfulness",
        "cartography",
        "data_attribution",
    ]
    for subdir in subdirs:
        ensure_dir(run_dir / subdir)


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict[str, float]]]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows: list[dict[str, float]] = []
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        mask = (confidences >= lo) & (confidences < hi if idx < n_bins - 1 else confidences <= hi)
        if not np.any(mask):
            rows.append({"bin_low": lo, "bin_high": hi, "count": 0, "accuracy": math.nan, "confidence": math.nan})
            continue
        acc = float(correct[mask].mean())
        conf = float(confidences[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(acc - conf)
        rows.append({"bin_low": lo, "bin_high": hi, "count": int(mask.sum()), "accuracy": acc, "confidence": conf})
    return float(ece), rows


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> tuple[dict[str, Any], list[dict[str, float]]]:
    y_pred = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": float(np.mean([y_true[i] in np.argsort(probs[i])[-2:] for i in range(len(y_true))])),
        "mean_confidence": float(confidences.mean()),
        "mean_entropy": float(entropy(probs).mean()),
    }
    labels = list(range(len(class_names)))
    try:
        metrics["log_loss"] = float(log_loss(y_true, probs, labels=labels))
    except Exception as exc:  # noqa: BLE001
        metrics["log_loss_error"] = str(exc)
    try:
        y_bin = label_binarize(y_true, classes=labels)
        metrics["roc_auc_ovr"] = float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"))
    except Exception as exc:  # noqa: BLE001
        metrics["roc_auc_ovr_error"] = str(exc)
    try:
        y_bin = label_binarize(y_true, classes=labels)
        metrics["average_precision_macro"] = float(average_precision_score(y_bin, probs, average="macro"))
    except Exception as exc:  # noqa: BLE001
        metrics["average_precision_error"] = str(exc)
    try:
        briers = []
        for cls_idx in labels:
            briers.append(brier_score_loss((y_true == cls_idx).astype(int), probs[:, cls_idx]))
        metrics["brier_score_macro"] = float(np.mean(briers))
    except Exception as exc:  # noqa: BLE001
        metrics["brier_error"] = str(exc)
    ece, calibration_rows = expected_calibration_error(y_true, probs)
    metrics["ece"] = ece
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    metrics["classification_report"] = report
    return metrics, calibration_rows


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_save_figure(fig: plt.Figure, path: Path, pad_inches: float = 0.18) -> None:
    """Save Matplotlib figures with enough padding for labels, legends, and colorbars."""
    ensure_dir(path.parent)
    try:
        fig.tight_layout(pad=1.25)
    except Exception:
        pass
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)


def save_predictions(
    run_dir: Path,
    sample_ids: list[str],
    image_paths: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    losses: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    rows = build_prediction_rows(sample_ids, image_paths, y_true, probs, losses, class_names)
    save_csv(run_dir / "predictions" / "predictions.csv", rows)
    return rows


def build_prediction_rows(
    sample_ids: list[str],
    image_paths: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    losses: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    y_pred = probs.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for i, sample_id in enumerate(sample_ids):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": image_paths[i],
            "true_index": int(y_true[i]),
            "true_label": class_names[int(y_true[i])],
            "pred_index": int(y_pred[i]),
            "pred_label": class_names[int(y_pred[i])],
            "confidence": float(probs[i, y_pred[i]]),
            "loss": float(losses[i]),
            "entropy": float(entropy(probs[i : i + 1])[0]),
            "correct": bool(y_pred[i] == y_true[i]),
        }
        for cls_idx, cls_name in enumerate(class_names):
            row[f"prob_{cls_name.replace(' ', '_')}"] = float(probs[i, cls_idx])
        rows.append(row)
    return rows


def save_confusion(run_dir: Path, y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> None:
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.savetxt(run_dir / "confusion" / "confusion_matrix_raw.csv", cm, delimiter=",", fmt="%d")
    np.savetxt(run_dir / "confusion" / "confusion_matrix_normalized.csv", norm, delimiter=",", fmt="%.6f")
    _plot_confusion(cm, class_names, run_dir / "confusion" / "confusion_matrix_raw.png", "Raw confusion matrix")
    _plot_confusion(norm, class_names, run_dir / "confusion" / "confusion_matrix_normalized.png", "Normalized confusion matrix")
    rows: list[dict[str, Any]] = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                rows.append({"true_label": true_name, "pred_label": pred_name, "count": int(cm[i, j]), "rate": float(norm[i, j])})
    rows.sort(key=lambda row: row["count"], reverse=True)
    save_csv(run_dir / "confusion" / "hardest_class_pairs.csv", rows, ["true_label", "pred_label", "count", "rate"])


def _plot_confusion(matrix: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=140)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    fmt = ".2f" if matrix.dtype.kind == "f" else "d"
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _safe_save_figure(fig, path)


def save_standard_plots(run_dir: Path, y_true: np.ndarray, probs: np.ndarray, class_names: list[str], calibration_rows: list[dict[str, float]]) -> None:
    y_pred = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    ent = entropy(probs)
    _bar_per_class(run_dir / "per_class" / "per_class_f1.png", y_true, y_pred, class_names)
    _hist(run_dir / "plots" / "entropy_histogram.png", ent, "Entropy", "Validation entropy")
    _hist(run_dir / "plots" / "confidence_histogram.png", confidences, "Confidence", "Validation confidence")
    _reliability(run_dir / "plots" / "reliability_diagram.png", calibration_rows)
    _roc_pr_curves(run_dir, y_true, probs, class_names)


def _bar_per_class(path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
    scores = [report[name]["f1-score"] for name in class_names]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.bar(class_names, scores, color="#4C78A8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1")
    ax.tick_params(axis="x", rotation=25)
    _safe_save_figure(fig, path)


def _hist(path: Path, values: np.ndarray, xlabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.hist(values, bins=30, color="#72B7B2", edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    _safe_save_figure(fig, path)


def _reliability(path: Path, rows: list[dict[str, float]]) -> None:
    x = [row["confidence"] for row in rows if row["count"]]
    y = [row["accuracy"] for row in rows if row["count"]]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.plot(x, y, marker="o", color="#F58518")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _safe_save_figure(fig, path)


def _roc_pr_curves(run_dir: Path, y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> None:
    labels = list(range(len(class_names)))
    y_bin = label_binarize(y_true, classes=labels)
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5), dpi=140)
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5), dpi=140)
    for cls_idx, cls_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], probs[:, cls_idx])
            precision, recall, _ = precision_recall_curve(y_bin[:, cls_idx], probs[:, cls_idx])
        except Exception:
            continue
        ax_roc.plot(fpr, tpr, label=cls_name)
        ax_pr.plot(recall, precision, label=cls_name)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curves")
    ax_roc.legend(fontsize=8)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall curves")
    ax_pr.legend(fontsize=8)
    for fig, path in [(fig_roc, run_dir / "plots" / "roc_curves.png"), (fig_pr, run_dir / "plots" / "pr_curves.png")]:
        _safe_save_figure(fig, path)


def unnormalize_tensor(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    tensor = tensor.clamp(0, 1)
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_sample_panels(run_dir: Path, prediction_rows: list[dict[str, Any]], max_samples: int = 20) -> None:
    hardest = sorted(prediction_rows, key=lambda row: float(row["loss"]), reverse=True)[:max_samples]
    confident = sorted(prediction_rows, key=lambda row: float(row["confidence"]), reverse=True)[:max_samples]
    _write_true_vs_pred_individuals(prediction_rows, run_dir / "true_vs_pred")
    _write_prediction_grid(hardest, run_dir / "high_loss_samples" / "high_loss_samples.png", "High-loss samples")
    _write_prediction_grid(confident, run_dir / "true_vs_pred" / "true_vs_pred_samples.png", "True vs predicted samples")


def save_epoch_high_loss_samples(
    run_dir: Path,
    prediction_rows: list[dict[str, Any]],
    epoch: int,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    top_rows = sorted(prediction_rows, key=lambda row: float(row["loss"]), reverse=True)[: max(0, top_k)]
    epoch_dir = ensure_dir(run_dir / "high_loss_samples" / f"epoch_{epoch:03d}")
    used_names: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(top_rows, start=1):
        source_path = Path(str(row["image_path"]))
        output_name = _ranked_prediction_filename(rank, source_path, used_names)
        output_path = epoch_dir / output_name
        caption = (
            f"epoch={epoch} | rank={rank} | true={row['true_label']} | pred={row['pred_label']} | "
            f"confidence={float(row['confidence']):.3f} | loss={float(row['loss']):.4f} | correct={row['correct']}"
        )
        _write_captioned_prediction_image(source_path, output_path, caption)
        manifest_rows.append(
            {
                "epoch": epoch,
                "rank": rank,
                "sample_id": row["sample_id"],
                "source_image": source_path.name,
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "confidence": row["confidence"],
                "loss": row["loss"],
                "correct": row["correct"],
                "output_path": str(output_path),
            }
        )
    save_csv(epoch_dir / "high_loss_manifest.csv", manifest_rows)
    _write_prediction_grid(top_rows, epoch_dir / "high_loss_grid.png", f"Epoch {epoch} high-loss samples")
    return manifest_rows


def _write_true_vs_pred_individuals(rows: list[dict[str, Any]], output_root: Path) -> None:
    ensure_dir(output_root)
    used_names_by_label: dict[str, set[str]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        source_path = Path(str(row["image_path"]))
        label_folder = _safe_label_folder(str(row["true_label"]))
        class_dir = ensure_dir(output_root / label_folder)
        used_names = used_names_by_label.setdefault(label_folder, set())
        output_name = _unique_panel_filename(source_path, used_names)
        output_path = class_dir / output_name
        caption = (
            f"true={row['true_label']} | pred={row['pred_label']} | "
            f"confidence={float(row['confidence']):.3f} | loss={float(row['loss']):.4f} | correct={row['correct']}"
        )
        _write_captioned_prediction_image(source_path, output_path, caption)
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "source_image": source_path.name,
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "confidence": row["confidence"],
                "loss": row["loss"],
                "correct": row["correct"],
                "output_path": str(output_path),
            }
        )
    save_csv(output_root / "true_vs_pred_image_manifest.csv", manifest_rows)


def _write_captioned_prediction_image(source_path: Path, output_path: Path, caption: str) -> None:
    ensure_dir(output_path.parent)
    font = ImageFont.load_default()
    try:
        image = ImageOps.contain(Image.open(source_path).convert("RGB"), (640, 420))
    except Exception:
        image = Image.new("RGB", (640, 420), "black")
    lines = _wrap_caption(caption, width=104, max_lines=3)
    caption_h = max(62, 16 * len(lines) + 18)
    canvas = Image.new("RGB", (image.width, image.height + caption_h), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((8, image.height + 10), "\n".join(lines), fill=(0, 0, 0), font=font, spacing=3)
    canvas.save(output_path)


def _ranked_prediction_filename(rank: int, source_path: Path, used_names: set[str]) -> str:
    stem = _safe_filename_stem(source_path)
    filename = f"rank_{rank:03d}_{stem}.png"
    if filename not in used_names:
        used_names.add(filename)
        return filename
    counter = 1
    while True:
        candidate = f"rank_{rank:03d}_{stem}_{counter}.png"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def _safe_label_folder(label: str) -> str:
    normalized = str(label).strip().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    safe = "".join(char if char.isalnum() or char in {" ", "-", "_"} else "_" for char in normalized)
    return safe.strip(" ._") or "unknown"


def _write_prediction_grid(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    if not rows:
        return
    thumb = (190, 130)
    label_h = 52
    cols = min(5, len(rows))
    grid_rows = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cols * thumb[0], grid_rows * (thumb[1] + label_h) + 26), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, 6), title, fill=(0, 0, 0), font=font)
    for idx, row in enumerate(rows):
        x = (idx % cols) * thumb[0]
        y = 26 + (idx // cols) * (thumb[1] + label_h)
        try:
            image = ImageOps.contain(Image.open(row["image_path"]).convert("RGB"), thumb)
        except Exception:
            image = Image.new("RGB", thumb, "black")
        canvas.paste(image, (x, y))
        correct = "ok" if row["correct"] else "wrong"
        text = f"T:{row['true_label']} P:{row['pred_label']}\n{correct} conf={float(row['confidence']):.2f} loss={float(row['loss']):.2f}"
        draw.multiline_text((x + 4, y + thumb[1] + 3), text, fill=(0, 0, 0), font=font, spacing=2)
    ensure_dir(path.parent)
    canvas.save(path)


def save_grad_saliency_panels(
    model: torch.nn.Module,
    loader: Any,
    run_dir: Path,
    class_names: list[str],
    device: torch.device,
    max_per_class: int = 2,
) -> None:
    model.eval()
    selected_for_summary: dict[int, int] = {idx: 0 for idx in range(len(class_names))}
    selected_heatmap_summary: dict[int, int] = {idx: 0 for idx in range(len(class_names))}
    saved_by_class: dict[int, int] = {idx: 0 for idx in range(len(class_names))}
    heatmap_saved_by_class: dict[int, int] = {idx: 0 for idx in range(len(class_names))}
    saliency_used_names: dict[int, set[str]] = {idx: set() for idx in range(len(class_names))}
    heatmap_used_names: dict[int, set[str]] = {idx: set() for idx in range(len(class_names))}
    summary_panels: list[Image.Image] = []
    heatmap_summary_panels: list[Image.Image] = []
    manifest_rows: list[dict[str, Any]] = []
    heatmap_manifest_rows: list[dict[str, Any]] = []
    font = ImageFont.load_default()
    for class_name in class_names:
        ensure_dir(run_dir / "gradcam_saliency" / _class_gradcam_folder(class_name))
        ensure_dir(run_dir / "gradcam_heatmap" / _class_gradcam_folder(class_name))
    target_layer = _find_gradcam_target_layer(model)
    hook_handles: list[Any] = []
    activations: list[torch.Tensor] = []
    gradients: dict[int, torch.Tensor] = {}

    def forward_hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor) else output
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return
        hook_index = len(activations)
        activations.append(tensor)
        tensor.register_hook(lambda grad, idx=hook_index: gradients.__setitem__(idx, grad))

    if target_layer is not None:
        hook_handles.append(target_layer.register_forward_hook(forward_hook))
    for batch in loader:
        activations.clear()
        gradients.clear()
        labels = batch["labels"].to(device)
        images = {branch: tensor.to(device) for branch, tensor in batch["images"].items()}
        first_branch = next(iter(images))
        images[first_branch] = images[first_branch].detach().clone().requires_grad_(True)
        aux = batch["aux_features"].to(device)
        aux_mask = batch["aux_mask"].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(images, aux, aux_mask)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        selected_scores = logits.gather(1, preds.view(-1, 1)).sum()
        selected_scores.backward()
        grads = images[first_branch].grad.detach().abs().max(dim=1)[0]
        gradcam_heat = _batch_gradcam_heatmaps(activations, gradients, target_index=0)
        for idx in range(labels.shape[0]):
            cls = int(labels[idx].item())
            pred = int(preds[idx].item())
            confidence = float(probs[idx, pred].detach().item())
            grad = grads[idx]
            grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
            original = unnormalize_tensor(images[first_branch][idx])
            caption = (
                f"mode={run_dir.name}\n"
                f"true={class_names[cls]} pred={class_names[pred]} conf={confidence:.2f}\n"
                "method=gradient_saliency_fallback"
            )
            panel = _saliency_panel(
                original,
                grad.cpu().numpy(),
                caption,
                font,
            )
            class_folder = _class_gradcam_folder(class_names[cls])
            output_filename = _unique_panel_filename(Path(batch["image_paths"][idx]), saliency_used_names[cls])
            output_path = run_dir / "gradcam_saliency" / class_folder / output_filename
            panel.save(output_path)
            saved_by_class[cls] += 1
            manifest_rows.append(
                {
                    "sample_id": batch["sample_ids"][idx],
                    "true_label": class_names[cls],
                    "pred_label": class_names[pred],
                    "confidence": confidence,
                    "method": "gradient_saliency_fallback",
                    "branch": first_branch,
                    "output_path": str(output_path.relative_to(run_dir)),
                }
            )
            if selected_for_summary[cls] < max_per_class:
                summary_panels.append(panel)
                selected_for_summary[cls] += 1
            heat = gradcam_heat[idx] if gradcam_heat is not None and idx < gradcam_heat.shape[0] else grad.cpu().numpy()
            heatmap_panel = _heatmap_panel(
                original,
                heat,
                (
                    f"{run_dir.name} | true={class_names[cls]} | "
                    f"pred={class_names[pred]} | conf={confidence:.3f} | "
                    f"{'standard_gradcam' if gradcam_heat is not None else 'gradcam_input_gradient_fallback'}"
                ),
                font,
            )
            heatmap_output_filename = _unique_panel_filename(Path(batch["image_paths"][idx]), heatmap_used_names[cls])
            heatmap_output_path = run_dir / "gradcam_heatmap" / class_folder / heatmap_output_filename
            heatmap_panel.save(heatmap_output_path)
            heatmap_saved_by_class[cls] += 1
            heatmap_manifest_rows.append(
                {
                    "sample_id": batch["sample_ids"][idx],
                    "true_label": class_names[cls],
                    "pred_label": class_names[pred],
                    "confidence": confidence,
                    "method": "standard_gradcam" if gradcam_heat is not None else "gradcam_input_gradient_fallback",
                    "branch": first_branch,
                    "target_layer": _module_name_for(model, target_layer) if target_layer is not None else "",
                    "output_path": str(heatmap_output_path.relative_to(run_dir)),
                }
            )
            if selected_heatmap_summary[cls] < max_per_class:
                heatmap_summary_panels.append(heatmap_panel)
                selected_heatmap_summary[cls] += 1
    for handle in hook_handles:
        handle.remove()
    if summary_panels:
        _tile_panels(summary_panels, run_dir / "gradcam_saliency" / "saliency_panels" / "gradcam_saliency_panels.png")
    if heatmap_summary_panels:
        _tile_panels(heatmap_summary_panels, run_dir / "gradcam_heatmap" / "heatmap_panels" / "gradcam_heatmap_panels.png")
    save_csv(
        run_dir / "gradcam_saliency" / "gradcam_saliency_manifest.csv",
        manifest_rows,
        ["sample_id", "true_label", "pred_label", "confidence", "method", "branch", "output_path"],
    )
    save_csv(
        run_dir / "gradcam_heatmap" / "gradcam_heatmap_manifest.csv",
        heatmap_manifest_rows,
        ["sample_id", "true_label", "pred_label", "confidence", "method", "branch", "target_layer", "output_path"],
    )
    atomic_write_json(
        run_dir / "gradcam_saliency" / "gradcam_saliency_method_manifest.json",
        {
            "primary_method": "gradient_saliency_fallback",
            "reason": "Generic gradient-based fallback used for broad timm backbone compatibility.",
            "scope": "all_validation_samples",
            "layout": "gradcam_saliency/<TrueClass>/<original_image_stem>.png",
            "max_per_class": max_per_class,
            "saved_by_class": {_class_gradcam_folder(class_names[idx]): count for idx, count in saved_by_class.items()},
            "summary_panel_path": "gradcam_saliency/saliency_panels/gradcam_saliency_panels.png",
        },
    )
    atomic_write_json(
        run_dir / "gradcam_heatmap" / "gradcam_heatmap_method_manifest.json",
        {
            "primary_method": "standard_gradcam" if target_layer is not None else "gradcam_input_gradient_fallback",
            "target_layer": _module_name_for(model, target_layer) if target_layer is not None else None,
            "scope": "all_validation_samples",
            "layout": "gradcam_heatmap/<TrueClass>/<original_image_stem>.png",
            "max_per_class": max_per_class,
            "saved_by_class": {_class_gradcam_folder(class_names[idx]): count for idx, count in heatmap_saved_by_class.items()},
            "summary_panel_path": "gradcam_heatmap/heatmap_panels/gradcam_heatmap_panels.png",
        },
    )


def save_patch_occlusion_and_faithfulness(
    model: torch.nn.Module,
    loader: Any,
    run_dir: Path,
    class_names: list[str],
    device: torch.device,
    max_samples: int = 3,
    grid_size: int = 4,
) -> None:
    model.eval()
    rows: list[dict[str, Any]] = []
    faithfulness_rows: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    processed = 0
    font = ImageFont.load_default()
    for batch in loader:
        images = {branch: tensor.to(device) for branch, tensor in batch["images"].items()}
        aux = batch["aux_features"].to(device)
        aux_mask = batch["aux_mask"].to(device)
        labels = batch["labels"].to(device)
        branch = next(iter(images))
        with torch.no_grad():
            logits = model(images, aux, aux_mask)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        for idx in range(labels.shape[0]):
            if processed >= max_samples:
                break
            target = int(preds[idx].item())
            base_prob = float(probs[idx, target].item())
            image_tensor = images[branch][idx : idx + 1]
            _, _, height, width = image_tensor.shape
            patch_h = height // grid_size
            patch_w = width // grid_size
            drops = np.zeros((grid_size, grid_size), dtype=np.float32)
            patch_scores: list[tuple[float, int, int]] = []
            for gy in range(grid_size):
                for gx in range(grid_size):
                    masked_images = {name: value[idx : idx + 1].clone() for name, value in images.items()}
                    y0 = gy * patch_h
                    y1 = height if gy == grid_size - 1 else (gy + 1) * patch_h
                    x0 = gx * patch_w
                    x1 = width if gx == grid_size - 1 else (gx + 1) * patch_w
                    masked_images[branch][:, :, y0:y1, x0:x1] = 0.0
                    with torch.no_grad():
                        masked_prob = torch.softmax(model(masked_images, aux[idx : idx + 1], aux_mask[idx : idx + 1]), dim=1)[0, target].item()
                    drop = base_prob - float(masked_prob)
                    drops[gy, gx] = drop
                    patch_scores.append((drop, gy, gx))
                    rows.append(
                        {
                            "sample_id": batch["sample_ids"][idx],
                            "branch": branch,
                            "target_label": class_names[target],
                            "patch_y": gy,
                            "patch_x": gx,
                            "base_probability": base_prob,
                            "masked_probability": float(masked_prob),
                            "probability_drop": float(drop),
                        }
                    )
            patch_scores.sort(reverse=True)
            deletion_images = {name: value[idx : idx + 1].clone() for name, value in images.items()}
            for step, (_, gy, gx) in enumerate(patch_scores, start=1):
                y0 = gy * patch_h
                y1 = height if gy == grid_size - 1 else (gy + 1) * patch_h
                x0 = gx * patch_w
                x1 = width if gx == grid_size - 1 else (gx + 1) * patch_w
                deletion_images[branch][:, :, y0:y1, x0:x1] = 0.0
                with torch.no_grad():
                    deletion_prob = torch.softmax(model(deletion_images, aux[idx : idx + 1], aux_mask[idx : idx + 1]), dim=1)[0, target].item()
                faithfulness_rows.append(
                    {
                        "sample_id": batch["sample_ids"][idx],
                        "curve": "deletion_by_occlusion_importance",
                        "step": step,
                        "fraction_removed": float(step / len(patch_scores)),
                        "target_probability": float(deletion_prob),
                    }
                )
            original = unnormalize_tensor(images[branch][idx])
            heat = drops
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
            heat = np.kron(heat, np.ones((height // grid_size + 1, width // grid_size + 1)))[:height, :width]
            panels.append(
                _saliency_panel(
                    original,
                    heat,
                    f"patch occlusion\ntrue={class_names[int(labels[idx])]} pred={class_names[target]} base={base_prob:.2f}",
                    font,
                )
            )
            processed += 1
        if processed >= max_samples:
            break
    save_csv(run_dir / "transformer_attribution" / "patch_occlusion.csv", rows)
    save_csv(run_dir / "faithfulness" / "occlusion_deletion_curves.csv", faithfulness_rows)
    if panels:
        _tile_panels(panels, run_dir / "transformer_attribution" / "patch_occlusion_panels.png")
    atomic_write_json(
        run_dir / "transformer_attribution" / "patch_occlusion_manifest.json",
        {"status": "complete", "branch": "first_active_visual_branch", "grid_size": grid_size, "samples": processed},
    )
    atomic_write_json(
        run_dir / "faithfulness" / "faithfulness_manifest.json",
        {"status": "complete", "method": "patch occlusion deletion curve", "samples": processed},
    )


def _saliency_panel(original: np.ndarray, heat: np.ndarray, caption: str, font: ImageFont.ImageFont) -> Image.Image:
    base = Image.fromarray(original).resize((180, 180))
    heat_img = Image.fromarray(np.uint8(plt.get_cmap("inferno")(heat)[:, :, :3] * 255)).resize((180, 180))
    overlay = Image.blend(base, heat_img, alpha=0.45)
    caption_lines = _wrap_caption(caption, width=86, max_lines=5)
    caption_h = max(68, 14 * len(caption_lines) + 10)
    panel = Image.new("RGB", (540, 180 + caption_h), "white")
    panel.paste(base, (0, 0))
    panel.paste(heat_img, (180, 0))
    panel.paste(overlay, (360, 0))
    draw = ImageDraw.Draw(panel)
    draw.multiline_text((4, 184), "\n".join(caption_lines), fill=(0, 0, 0), font=font, spacing=2)
    return panel


def _heatmap_panel(original: np.ndarray, heat: np.ndarray, caption: str, font: ImageFont.ImageFont) -> Image.Image:
    tile = 224
    caption_lines = _wrap_caption(caption, width=108, max_lines=3)
    caption_h = max(76, 16 * len(caption_lines) + 18)
    base = Image.fromarray(original).resize((tile, tile), Image.BILINEAR)
    heat = np.asarray(heat, dtype=np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    heat = (heat - float(heat.min())) / (float(heat.max() - heat.min()) + 1e-8)
    heat_img = Image.fromarray(np.uint8(plt.get_cmap("jet")(heat)[:, :, :3] * 255)).resize((tile, tile), Image.BILINEAR)
    overlay = Image.blend(base, heat_img, alpha=0.46)
    panel = Image.new("RGB", (tile * 3, tile + caption_h), "white")
    panel.paste(base, (0, 0))
    panel.paste(heat_img, (tile, 0))
    panel.paste(overlay, (tile * 2, 0))
    draw = ImageDraw.Draw(panel)
    draw.multiline_text((8, tile + 12), "\n".join(caption_lines), fill=(0, 0, 0), font=font, spacing=3)
    return panel


def _find_gradcam_target_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    scan_model: torch.nn.Module = model
    if hasattr(model, "shared_backbone") and isinstance(getattr(model, "shared_backbone"), torch.nn.Module):
        scan_model = getattr(model, "shared_backbone")
    elif hasattr(model, "branch_backbones") and len(getattr(model, "branch_backbones")):
        first_key = next(iter(getattr(model, "branch_backbones").keys()))
        scan_model = getattr(model, "branch_backbones")[first_key]
    target = None
    for module in scan_model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target = module
    return target


def _module_name_for(model: torch.nn.Module, target: torch.nn.Module | None) -> str:
    if target is None:
        return ""
    for name, module in model.named_modules():
        if module is target:
            return name
    return target.__class__.__name__


def _batch_gradcam_heatmaps(
    activations: list[torch.Tensor],
    gradients: dict[int, torch.Tensor],
    target_index: int = 0,
) -> np.ndarray | None:
    if target_index >= len(activations) or target_index not in gradients:
        return None
    activation = activations[target_index].detach()
    gradient = gradients[target_index].detach()
    if activation.ndim != 4 or gradient.ndim != 4:
        return None
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False).squeeze(1)
    cam_min = cam.amin(dim=(1, 2), keepdim=True)
    cam_max = cam.amax(dim=(1, 2), keepdim=True)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam.cpu().numpy()


def _wrap_caption(caption: str, width: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    for raw_line in str(caption).splitlines():
        wrapped = textwrap.wrap(raw_line, width=width) or [""]
        lines.extend(wrapped)
    if len(lines) <= max_lines:
        return lines
    trimmed = lines[:max_lines]
    trimmed[-1] = trimmed[-1].rstrip() + "..."
    return trimmed


def _class_gradcam_folder(class_name: str) -> str:
    cleaned = str(class_name).replace("_", " ").replace("-", " ")
    cleaned = " ".join(part for part in cleaned.split() if not part.isdigit())
    return " ".join(word.capitalize() for word in cleaned.split())


def _safe_filename_stem(path: Path) -> str:
    stem = path.stem.strip() or "image"
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in stem)
    return safe.strip("._") or "image"


def _unique_panel_filename(source_path: Path, used_names: set[str]) -> str:
    stem = _safe_filename_stem(source_path)
    filename = f"{stem}.png"
    if filename not in used_names:
        used_names.add(filename)
        return filename
    counter = 1
    while True:
        candidate = f"{stem}_{counter}.png"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def _tile_panels(panels: list[Image.Image], path: Path, cols: int = 1) -> None:
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    rows = math.ceil(len(panels) / cols)
    canvas = Image.new("RGB", (cols * width, rows * height), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, ((idx % cols) * width, (idx // cols) * height))
    ensure_dir(path.parent)
    canvas.save(path)


def save_embedding_and_advanced_outputs(
    run_dir: Path,
    payload: dict[str, Any],
    class_names: list[str],
    aux_active: bool,
) -> None:
    embeddings = payload.get("fused_visual_embedding")
    labels = np.asarray(payload.get("labels", []), dtype=int)
    probs = np.asarray(payload.get("probabilities", []), dtype=np.float32)
    sample_ids = payload.get("sample_ids", [])
    if embeddings is None or len(labels) == 0:
        return
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.savez_compressed(
        run_dir / "embedding_analysis" / "embeddings.npz",
        fused_visual_embedding=embeddings,
        labels=labels,
        probabilities=probs,
        sample_ids=np.array(sample_ids, dtype=object),
    )
    _save_projection_atlas(run_dir, embeddings, labels, class_names)
    _save_cka(run_dir, payload, class_names)
    _save_cca(run_dir, payload, labels, class_names, aux_active)
    _save_retrieval_boards(run_dir, embeddings, labels, probs, sample_ids, payload.get("image_paths", []), class_names)
    _save_decision_quality(run_dir, embeddings, labels, probs, class_names)
    _save_tcav_stub(run_dir)
    _save_transformer_attribution_stub(run_dir)
    _save_faithfulness_stub(run_dir)
    _save_data_attribution(run_dir, embeddings, labels, probs, sample_ids, class_names)


def _save_projection_atlas(run_dir: Path, embeddings: np.ndarray, labels: np.ndarray, class_names: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    for idx, coord in enumerate(coords):
        rows.append({"sample_index": idx, "x": float(coord[0]), "y": float(coord[1]), "label": class_names[int(labels[idx])], "method": "pca"})
    save_csv(run_dir / "embedding_analysis" / "projection_pca.csv", rows)
    _scatter_projection(run_dir / "embedding_analysis" / "projection_pca.png", coords, labels, class_names, "PCA projection")
    manifest = {"primary": "pca", "pacmap": "not_available_or_not_requested", "trimap": "not_available_or_not_requested"}
    if embeddings.shape[0] >= 8:
        try:
            perplexity = max(2, min(30, embeddings.shape[0] // 3))
            tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
            tsne_coords = tsne.fit_transform(embeddings)
            rows = [
                {"sample_index": idx, "x": float(coord[0]), "y": float(coord[1]), "label": class_names[int(labels[idx])], "method": "tsne"}
                for idx, coord in enumerate(tsne_coords)
            ]
            save_csv(run_dir / "embedding_analysis" / "projection_tsne_fallback.csv", rows)
            _scatter_projection(run_dir / "embedding_analysis" / "projection_tsne_fallback.png", tsne_coords, labels, class_names, "t-SNE fallback")
            manifest["fallback"] = "tsne"
        except Exception as exc:  # noqa: BLE001
            manifest["tsne_error"] = str(exc)
    atomic_write_json(run_dir / "embedding_analysis" / "projection_manifest.json", manifest)


def _scatter_projection(path: Path, coords: np.ndarray, labels: np.ndarray, class_names: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    for idx, name in enumerate(class_names):
        mask = labels == idx
        ax.scatter(coords[mask, 0], coords[mask, 1], label=name, s=14, alpha=0.8)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _safe_save_figure(fig, path)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    dot = np.linalg.norm(x.T @ y, ord="fro") ** 2
    norm = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(dot / (norm + 1e-12))


def _save_cka(run_dir: Path, payload: dict[str, Any], class_names: list[str]) -> None:
    keys = [key for key, value in payload.items() if key.endswith("_embedding") and isinstance(value, np.ndarray)]
    rows = []
    matrix = np.zeros((len(keys), len(keys)), dtype=np.float32)
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            value = linear_cka(payload[key_i], payload[key_j])
            matrix[i, j] = value
            rows.append({"embedding_a": key_i, "embedding_b": key_j, "linear_cka": value})
    save_csv(run_dir / "cka" / "linear_cka.csv", rows, ["embedding_a", "embedding_b", "linear_cka"])
    if keys:
        fig, ax = plt.subplots(figsize=(max(5, len(keys) * 1.2), max(4, len(keys) * 1.0)), dpi=140)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right")
        ax.set_yticks(range(len(keys)), keys)
        ax.set_title("Linear CKA")
        fig.colorbar(im, ax=ax)
        _safe_save_figure(fig, run_dir / "cka" / "linear_cka_heatmap.png")


def _save_cca(run_dir: Path, payload: dict[str, Any], labels: np.ndarray, class_names: list[str], aux_active: bool) -> None:
    pairs: list[tuple[str, str]] = []
    if aux_active and isinstance(payload.get("aux_embedding"), np.ndarray):
        pairs.append(("fused_visual_embedding", "aux_embedding"))
    embedding_keys = [key for key in payload if key.endswith("_projected_embedding")]
    for idx, key_a in enumerate(embedding_keys):
        for key_b in embedding_keys[idx + 1 :]:
            pairs.append((key_a, key_b))
    rows: list[dict[str, Any]] = []
    for key_a, key_b in pairs:
        a = np.asarray(payload.get(key_a), dtype=np.float32)
        b = np.asarray(payload.get(key_b), dtype=np.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 4 or b.shape[0] < 4:
            continue
        n_components = max(1, min(5, a.shape[1], b.shape[1], a.shape[0] - 1))
        try:
            cca = CCA(n_components=n_components, max_iter=500)
            za, zb = cca.fit_transform(a, b)
            corrs = [float(np.corrcoef(za[:, i], zb[:, i])[0, 1]) for i in range(n_components)]
            rows.append({"embedding_a": key_a, "embedding_b": key_b, "mean_abs_corr": float(np.nanmean(np.abs(corrs))), "components": json.dumps(corrs)})
            for cls_idx, cls_name in enumerate(class_names):
                mask = labels == cls_idx
                if mask.sum() >= 4:
                    rows.append(
                        {
                            "embedding_a": key_a,
                            "embedding_b": key_b,
                            "class_name": cls_name,
                            "mean_abs_corr": float(np.nanmean(np.abs([np.corrcoef(za[mask, i], zb[mask, i])[0, 1] for i in range(n_components)]))),
                            "components": "per_class",
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            rows.append({"embedding_a": key_a, "embedding_b": key_b, "error": str(exc)})
    save_csv(run_dir / "cca_alignment" / "cca_alignment.csv", rows)


def _save_retrieval_boards(
    run_dir: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    sample_ids: list[str],
    image_paths: list[str],
    class_names: list[str],
) -> None:
    if embeddings.shape[0] < 3 or NearestNeighbors is None:
        return
    nn = NearestNeighbors(n_neighbors=min(6, embeddings.shape[0]), metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    pred = probs.argmax(axis=1)
    hard = np.where(pred != labels)[0]
    if hard.size == 0:
        hard = np.argsort(probs.max(axis=1))[: min(5, len(labels))]
    rows: list[dict[str, Any]] = []
    for query_idx in hard[:10]:
        for rank, neighbor_idx in enumerate(indices[query_idx]):
            rows.append(
                {
                    "query_sample_id": sample_ids[query_idx],
                    "neighbor_rank": int(rank),
                    "neighbor_sample_id": sample_ids[int(neighbor_idx)],
                    "distance": float(distances[query_idx, rank]),
                    "query_label": class_names[int(labels[query_idx])],
                    "neighbor_label": class_names[int(labels[int(neighbor_idx)])],
                }
            )
    save_csv(run_dir / "retrieval_boards" / "prototype_retrieval.csv", rows)
    atomic_write_json(run_dir / "retrieval_boards" / "retrieval_manifest.json", {"method": "cosine nearest neighbors", "queries": int(min(10, len(hard)))})


def _save_decision_quality(run_dir: Path, embeddings: np.ndarray, labels: np.ndarray, probs: np.ndarray, class_names: list[str]) -> None:
    confidences = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    rows = []
    for coverage in np.linspace(0.1, 1.0, 10):
        k = max(1, int(round(coverage * len(confidences))))
        keep = np.argsort(confidences)[-k:]
        rows.append({"coverage": float(k / len(confidences)), "risk": float(1.0 - correct[keep].mean()), "kept": int(k)})
    save_csv(run_dir / "decision_quality" / "selective_classification" / "risk_coverage.csv", rows)
    conformal_rows = []
    for alpha in (0.05, 0.10, 0.20):
        threshold = 1.0 - alpha
        set_sizes = (np.cumsum(np.sort(probs, axis=1)[:, ::-1], axis=1) < threshold).sum(axis=1) + 1
        conformal_rows.append({"alpha": alpha, "mean_set_size": float(set_sizes.mean()), "max_set_size": int(set_sizes.max())})
    save_csv(run_dir / "decision_quality" / "conformal" / "prediction_set_summary.csv", conformal_rows)
    trust_rows = []
    centroids = []
    for cls_idx in range(len(class_names)):
        mask = labels == cls_idx
        centroids.append(embeddings[mask].mean(axis=0) if mask.any() else np.zeros(embeddings.shape[1], dtype=np.float32))
    centroids = np.stack(centroids)
    dists = ((embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2) ** 0.5
    for idx in range(len(labels)):
        own = dists[idx, pred[idx]]
        other = np.min(np.delete(dists[idx], pred[idx])) if len(class_names) > 1 else own
        trust_rows.append({"sample_index": idx, "trust_score": float(other / (own + 1e-8)), "pred_label": class_names[int(pred[idx])]})
    save_csv(run_dir / "decision_quality" / "trust_score" / "trust_scores.csv", trust_rows)


def _save_tcav_stub(run_dir: Path) -> None:
    atomic_write_json(
        run_dir / "tcav" / "tcav_manifest.json",
        {
            "status": "skipped",
            "reason": "No user-curated concept manifest was found for this run.",
            "expected_input": "Provide concept image manifests to enable TCAV-style concept vectors.",
        },
    )


def _save_transformer_attribution_stub(run_dir: Path) -> None:
    atomic_write_json(
        run_dir / "transformer_attribution" / "patch_occlusion_manifest.json",
        {
            "status": "available_as_configurable_lightweight_analysis",
            "method": "patch occlusion should be run on selected samples to control runtime.",
        },
    )


def _save_faithfulness_stub(run_dir: Path) -> None:
    atomic_write_json(
        run_dir / "faithfulness" / "faithfulness_manifest.json",
        {
            "status": "lightweight_summary_written",
            "methods": ["occlusion", "insertion_deletion"],
            "note": "Full insertion/deletion curves are intentionally configurable because they are expensive across 195 runs.",
        },
    )


def _save_data_attribution(run_dir: Path, embeddings: np.ndarray, labels: np.ndarray, probs: np.ndarray, sample_ids: list[str], class_names: list[str]) -> None:
    rows = []
    pred = probs.argmax(axis=1)
    for cls_idx, cls_name in enumerate(class_names):
        mask = (labels == cls_idx) & (pred == labels)
        if not mask.any():
            continue
        centroid = embeddings[mask].mean(axis=0)
        distances = ((embeddings - centroid.reshape(1, -1)) ** 2).sum(axis=1) ** 0.5
        for idx in np.argsort(distances)[:5]:
            rows.append({"class_name": cls_name, "sample_id": sample_ids[int(idx)], "centroid_distance": float(distances[int(idx)])})
    save_csv(run_dir / "data_attribution" / "embedding_attribution_approximation.csv", rows)


def save_classification_report_text(run_dir: Path, metrics: dict[str, Any]) -> None:
    report = metrics.get("classification_report", {})
    atomic_write_text(run_dir / "reports" / "classification_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
