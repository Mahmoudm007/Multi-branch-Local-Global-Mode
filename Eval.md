# Road Surface Condition Benchmark

## Problem Setup

The default task is 5-class road-surface-condition classification:

1. `Bare`
2. `Centre_Partly`
3. `TwoTrack_Partly`
4. `OneTrack_Partly`
5. `Fully`


## Model Inventory

### Existing baseline backbones

These were already present before this extension:

- `convnext`
- `convnext_v2`
- `maxvit`
- `deit_base`
- `seresnext50`
- `inception_v3`
- `xception`
- `beit_base`
- `pvt_v2`
- `mambaout` "local Mamba-style integration using timm `MambaOut`"
- `coatnet`
- `focalnet`
- `davit`

## Training Behavior

Current training changes include:

- best checkpointing based on `validation loss`
- class-weight boost for:
  fully: 1.25
  one track partly: 2.0

## Outputs needed
for each model
- checkpoints (best and last)
- history CSV
- predictions CSV
- predictions images (true vs. predicted labels, with confidence)
- metrics JSON/CSV
- classification reports
- confusion matrices
- per-class plots
- high-loss samples
- explainability panels
- config snapshots
- environment metadata
- run summary JSON
- global comparison tables, workbook, and standard plots
- top-2 accuracy
- log-loss
- ROC AUC
- average precision
- Brier score
- ECE
- entropy
- calibration tables
- confidence summaries
- ROC curves
- PR curves
- reliability diagrams
- entropy histograms
- class-confidence plots
- family summaries
- mode summaries
- warning-count tracking
- generic comparison scatter plots
- metric-rank heatmap
- capability manifest for available vs unavailable requested model families
- multimodal fusion mode support for `concat`, `gated`, `film`
- embedding projection atlas with PaCMAP/TriMAP-if-available and logged t-SNE fallback
- per-model and global CKA outputs
- multimodal CCA alignment diagnostics
- transformer-oriented patch-occlusion attribution
- TCAV-style concept analysis from user-curated concept manifests
- prototype and counter-prototype retrieval boards
- selective classification risk-coverage diagnostics
- multiclass conformal prediction-set outputs
- trust-score diagnostics
- faithfulness tests using occlusion plus insertion/deletion curves
- dataset cartography from per-epoch training dynamics
- embedding-based training-data attribution approximation for hard cases
- best-model-only regional SHAP with region metrics and band summaries

## Output Layout
Per model:

```text
Output_v2/<ModelName>/
├── checkpoints/
├── logs/
├── reports/
├── plots/
├── metrics/
├── confusion/
├── per_class/
├── gradcam/
├── true_vs_pred/
├── high_loss_samples/
├── comparison_samples/
├── predictions/
├── metadata/
├── embedding_analysis/
├── cka/
├── cca_alignment/
├── transformer_attribution/
├── tcav/
├── retrieval_boards/
├── decision_quality/
│   ├── selective_classification/
│   ├── conformal/
│   └── trust_score/
├── faithfulness/
├── cartography/
├── data_attribution/
└── shap_regional/        # best overall model only
```

Global outputs:

```text
Output_v2/_global_comparison/
├── data_reports/
├── explainability/
├── tables/
├── plots/
├── embeddings/
├── cka/
├── cca_alignment/
├── selective_classification/
├── conformal/
├── trust_score/
├── cartography/
├── metadata/
├── model_capability_manifest.json
├── best_model_selection.json
├── benchmark_summary_extended.md
└── final_benchmark_summary.md
```