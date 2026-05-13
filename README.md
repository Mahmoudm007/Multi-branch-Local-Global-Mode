# Five-Branch Winter Road-Surface Benchmark

This repository now contains a self-contained benchmark project for five-branch winter road-surface-condition classification.

The default supervised dataset is:

```text
Dataset_classes/
  1 Defined/
    train/
    val/
```

The five classes are discovered deterministically from the training folders and normalized through `class_descriptions.py`:

1. `bare`
2. `centre partly`
3. `two track partly`
4. `one track partly`
5. `fully`

## Branches

Every experiment includes the original RGB image branch.

Generated image branches are written under `Generated_Branches/`:

```text
Generated_Branches/
  cropped_local/1 Defined/train|val/<class>/<image>_CROP.<ext>
  thermal_clahe_inferno/1 Defined/train|val/<class>/<image>_THERMAL.<ext>
  segmented_best_combined/1 Defined/train|val/<class>/<image>_SEG.<ext>
  manifests/
  reports/
  previews/
```

Branch definitions:

- `original`: source RGB image only.
- `cropped`: deterministic local crop keeping rows `floor(0.25H):ceil(0.70H)` and columns `floor(0.10W):ceil(0.90W)`.
- `thermal`: exact `CLAHE_Inferno` transform from `CLACH.md`: RGB to grayscale, OpenCV CLAHE `clipLimit=2.0`, `tileGridSize=(8,8)`, Inferno colormap, BGR back to RGB.
- `segmented`: deterministic classical CV best-combination branch. Only the final `BEST_COMBINED` road-focused image is saved as classifier input.
- `auxiliary_text`: fixed class descriptions from `class_descriptions.py`, encoded with TF-IDF and used only for auxiliary alignment/diagnostics. Final logits remain image-driven.

## Generate Branch Assets

Full generation:

```bash
python run_all_branches.py
```

Resume or repair a previous generation:

```bash
python run_all_branches.py --skip-completed true
```

The resume path validates the current expected suffixed output file before skipping. This means stale progress CSV rows cannot hide missing `_CROP`, `_THERMAL`, or `_SEG` files.

Smoke test:

```bash
python run_all_branches.py --max-images-per-class 2
```

Useful outputs:

- `Generated_Branches/manifests/source_manifest.csv`
- `Generated_Branches/manifests/progress_<branch>.csv`
- `Generated_Branches/reports/validation_<branch>.csv`
- `Generated_Branches/previews/*_preview.png`
- `Generated_Branches/previews/cropped_local_*_crop_audit.png`

During training, any remaining missing or corrupt generated branch assets are filtered out of that run instead of crashing the full benchmark. The exclusion details are written per run to `Output/<model>/<experiment>/metadata/asset_exclusions.csv` and summarized in `asset_filter_summary.json`; training still fails if filtering would leave the train or validation split empty.

## Experiment Registry

Default experiments are the 15 required combinations:

```text
exp_original_thermal_segmented_cropped_auxtext
exp_original_thermal_cropped_auxtext
exp_original_thermal
exp_original_segmented
exp_original_cropped
exp_original_auxtext
exp_original_thermal_segmented
exp_original_thermal_cropped
exp_original_thermal_auxtext
exp_original_segmented_cropped
exp_original_segmented_auxtext
exp_original_cropped_auxtext
exp_original_thermal_segmented_cropped
exp_original_thermal_segmented_auxtext
exp_original_segmented_cropped_auxtext
```

Optional non-default baseline:

```text
exp_original_only
```

Model families run in this order:

```text
convnext, convnext_v2, maxvit, deit_base, seresnext50, inception_v3,
xception, beit_base, pvt_v2, mambaout, coatnet, focalnet, davit
```

Unavailable local timm backbones are logged in:

```text
Output/_global_comparison_per_combination/metadata/model_capability_manifest.json
```

## Run Training

Full benchmark:

```bash
python run_training_exp.py
```

Single model across all default experiments:

```bash
python run_training_exp.py --model convnext
```

Single experiment across all models:

```bash
python run_training_exp.py --experiment exp_original_thermal_segmented
```

One model and one experiment:

```bash
python run_training_exp.py --model convnext --experiment exp_original_thermal
```

Fusion backend override:

```bash
python run_training_exp.py --fusion concat
python run_training_exp.py --fusion gated
python run_training_exp.py --fusion film
```

Dry-run manifest without training:

```bash
python run_training_exp.py --dry-run
```

Small smoke run:

```bash
python run_training_exp.py --model convnext --experiment exp_original_cropped --epochs 1 --batch-size 2 --num-workers 0 --max-train-samples 10 --max-val-samples 10
```

## Output Layout

Each run writes:

```text
Output/<ModelName>/<ExperimentName>/
  checkpoints/
  logs/
  reports/
  plots/
  metrics/
  confusion/
  per_class/
  gradcam_saliency/
    Bare/<original_image_stem>.png
    Centre Partly/<original_image_stem>.png
    Two Track Partly/<original_image_stem>.png
    One Track Partly/<original_image_stem>.png
    Fully/<original_image_stem>.png
    saliency_panels/gradcam_saliency_panels.png
  gradcam_heatmap/
    Bare/<original_image_stem>.png
    Centre Partly/<original_image_stem>.png
    Two Track Partly/<original_image_stem>.png
    One Track Partly/<original_image_stem>.png
    Fully/<original_image_stem>.png
    heatmap_panels/gradcam_heatmap_panels.png
  true_vs_pred/
  high_loss_samples/
  comparison_samples/
  predictions/
  metadata/
  embedding_analysis/
  cka/
  cca_alignment/
  transformer_attribution/
  tcav/
  retrieval_boards/
  decision_quality/selective_classification/
  decision_quality/conformal/
  decision_quality/trust_score/
  faithfulness/
  cartography/
  data_attribution/
```

Completion requires all of these artifacts:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `logs/history.csv`
- `metrics/metrics.json`
- `predictions/predictions.csv`
- raw and normalized confusion matrices
- `gradcam_saliency/gradcam_saliency_manifest.csv`
- `gradcam_heatmap/gradcam_heatmap_manifest.csv`
- `true_vs_pred/true_vs_pred_image_manifest.csv`
- `metadata/run_summary.json` with status `complete`

Per-model comparisons are written under:

```text
Output/<ModelName>/_comparison_across_experiments/
```

Global per-experiment comparisons are written under:

```text
Output/_global_comparison_per_combination/<ExperimentName>/
```

## Notes

- No augmentation is used.
- SHAP and regional SHAP are disabled.
- GradCAM output uses a generic gradient-saliency fallback for broad backbone compatibility when a stable GradCAM target layer is not selected.
- The segmented branch is classical CV preprocessing, not learned segmentation.
- Pretrained weights are requested in config, but remote downloads are disabled. If cached pretrained weights are not available through the local environment, the runner logs random initialization fallback.
