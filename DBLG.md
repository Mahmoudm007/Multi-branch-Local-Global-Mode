# Dual Branch Local-Global Mode (DBLG)

This document explains the `dual_branch_local_global` mode in this project from both
an engineering and research perspective. It covers what the mode is trying to do,
where it is implemented, how the image is routed into the local branch, how the
model is trained and evaluated, how it relates to the literature, and what caveats
matter when interpreting results.

The short version: DBLG is an image-only advanced mode that sends the full image
through a global visual branch and a deterministic lower-image crop through a local
visual branch. The local branch is intended to focus on the near-road surface and
tire-path evidence, while the global branch keeps broader scene context. Their
features are projected, gated, concatenated, and classified.

## 1. What This Mode Is

The mode name in the project is:

```text
dual_branch_local_global
```

The core implementation is:

```text
core/advanced_modes.py
  class DualBranchLocalGlobalClassifier
```

The implementation has three main ideas:

1. Use the full image to preserve global context.
2. Use a fixed lower crop to emphasize the near-road region.
3. Fuse the two feature vectors with a learned gate before classification.

In road-surface condition classification, this is useful because the class may depend
on both:

- global scene cues: lane layout, road shoulder, horizon, snowbanks, traffic context,
  illumination, camera viewpoint, and visible pavement extent.
- local surface cues: tire paths, thin snow film, exposed asphalt, slush texture,
  compacted snow, wet sheen, glare, and surface material near the vehicle path.

The model is not a text-fusion model and does not use TF-IDF class descriptions in
the final DBLG prediction path. The builder reuses an auxiliary-model API to obtain
the image backbone, but the DBLG forward pass itself ignores `aux_features`.

## 2. Non-Technical Explanation

Think of DBLG as asking two visual questions about the same road image:

1. What does the entire scene suggest?
2. What does the lower road surface directly in front of the vehicle suggest?

The whole-scene view can detect whether the road is broadly winter-like, whether the
camera is on a highway or urban street, and whether the surroundings imply snow
coverage. The lower crop is a closer look at the part of the image that usually
contains the lane surface and tire tracks. The model then learns how much of the
local evidence to trust for each image by producing a gate vector.

The local branch is currently fixed to the bottom part of the image. It does not
detect the road automatically. It assumes that, after resizing and augmentation, the
road surface of interest is usually in the lower part of the image.

## 3. Repository Map

The most relevant files are:

```text
core/advanced_modes.py
  DualBranchLocalGlobalClassifier
  build_advanced_model
  AdvancedModeController

core/mode_registry.py
  NEW_MODES
  MODE_SPECS["dual_branch_local_global"]
  mode_uses_aux
  mode_config

configs/default_config.yaml
  benchmark.modes
  mode_defaults.dual_branch_local_global

configs/sdre_config.yaml
  same mode defaults for SDRE runs

core/trainer.py
  model construction
  optimizer, loss, checkpoints
  evaluation and advanced-analysis calls

core/evaluator.py
  prediction pass and standard evaluation artifacts

core/advanced_analysis.py
  embedding payload collection and optional analyses

models/*.py
  backbone modules that can be wrapped by DBLG
```

The mode is documented at a high level in:

```text
README.md
Explanation.md
```

## 4. Exact Implemented Architecture

The current DBLG architecture is:

```text
input image tensor
    |
    +--> full image ------------------> global_backbone --> global_raw
    |
    +--> bottom crop, resized to input -> local_backbone  --> local_raw

global_raw -> global_proj -> global_hidden
local_raw  -> local_proj  -> local_hidden

concat(global_hidden, local_hidden) -> gate -> branch_gate
local_hidden * branch_gate -> fused_local

concat(global_hidden, fused_local) -> classifier -> logits
```

In code, the central class is:

```python
class DualBranchLocalGlobalClassifier(nn.Module):
    ...
```

Constructor inputs:

```python
global_backbone: nn.Module
local_backbone: nn.Module
visual_dim: int
num_classes: int
hidden_dim: int = 512
crop_ratio: float = 0.55
dropout: float = 0.2
```

Internal modules:

```python
self.global_proj = nn.Linear(visual_dim, hidden_dim)
self.local_proj = nn.Linear(visual_dim, hidden_dim)
self.gate = nn.Sequential(
    nn.LayerNorm(hidden_dim * 2),
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.Sigmoid(),
)
self.classifier = nn.Sequential(
    nn.LayerNorm(hidden_dim * 2),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, num_classes),
)
```

The classifier output shape is:

```text
[batch_size, num_classes]
```

For the default dataset config, `num_classes = 5`:

```text
bare
centre partly
two track partly
one track partly
fully
```

## 5. How The Image Is Fixed To The Local Branch

The local image is created by `_local_crop()` in `core/advanced_modes.py`:

```python
def _local_crop(self, images: torch.Tensor) -> torch.Tensor:
    _, _, height, width = images.shape
    start = int(height * max(0.0, min(0.95, 1.0 - self.crop_ratio)))
    crop = images[:, :, start:, :]
    return F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False)
```

Input tensor shape:

```text
images: [B, C, H, W]
```

The local crop keeps:

```text
rows start through H - 1
all columns
all channels
```

Then it resizes that crop back to:

```text
[B, C, H, W]
```

This resize is important because most backbones expect a fixed input size such as
`224 x 224`.

### 5.1 Crop Formula

The crop start row is:

```text
start = int(H * clamp(1 - crop_ratio, minimum=0.0, maximum=0.95))
```

Where:

```text
crop_ratio = fraction of image height to keep from the bottom
```

Examples for `H = 224`:

| crop_ratio | start row | kept rows | meaning |
|---:|---:|---:|---|
| 0.25 | 168 | 56 | bottom 25 percent |
| 0.40 | 134 | 90 | bottom 40 percent |
| 0.55 | 100 | 124 | bottom about 55 percent |
| 0.70 | 67 | 157 | bottom 70 percent |
| 1.00 | 0 | 224 | full image |
| 0.00 | 212 | 12 | clamped to bottom 5 percent |

The default is:

```yaml
crop_ratio: 0.55
```

So with the default image size of `224`, the local branch sees roughly rows 100
through 223, resized back to `224 x 224`.

### 5.2 Why Resize The Crop Back To Full Size?

The crop is resized because most project backbones are configured for a common input
resolution. Resizing allows the local branch to use the same backbone API as the
global branch. It also makes the local evidence occupy the full visual field of the
backbone, increasing the effective resolution of the bottom road region.

The tradeoff is that resizing changes the scale of road texture. Fine details in the
bottom crop are enlarged. This can help texture recognition, but it also means the
local branch does not see geometry at the original global scale.

### 5.3 What "Fixed Local Branch" Means Here

In this project, "fixed local branch" means:

- the crop location is deterministic.
- the crop is not predicted by the model.
- the crop is always the bottom part of the image.
- the crop ratio comes from config.
- no bounding boxes, masks, segmentation, or attention maps are used to define the
  crop.

This is simpler and more reproducible than a learned crop, but it depends on a camera
view assumption: the road surface must usually appear in the bottom part of the image.

## 6. Forward Pass In Detail

The implemented feature extraction method is:

```python
def extract_analysis_features(self, images, aux_features=None):
    global_raw = self._encode(self.global_backbone, images)
    local_raw = self._encode(self.local_backbone, self._local_crop(images))
    global_hidden = self.global_proj(global_raw)
    local_hidden = self.local_proj(local_raw)
    gate = self.gate(torch.cat([global_hidden, local_hidden], dim=1))
    fused_local = local_hidden * gate
    fused = torch.cat([global_hidden, fused_local], dim=1)
    logits = self.classifier(fused)
    return {
        "image_embedding": global_raw,
        "local_embedding": local_raw,
        "fused_embedding": fused,
        "branch_gate": gate,
        "logits": logits,
    }
```

The model returns these analysis tensors:

| key | shape | meaning |
|---|---:|---|
| `image_embedding` | `[B, visual_dim]` | raw full-image/global backbone feature |
| `local_embedding` | `[B, visual_dim]` | raw bottom-crop/local backbone feature |
| `fused_embedding` | `[B, 2 * hidden_dim]` | concatenated global plus gated local feature |
| `branch_gate` | `[B, hidden_dim]` | learned vector gate applied to local hidden feature |
| `logits` | `[B, num_classes]` | final class scores |

The final `forward()` simply returns:

```python
return self.extract_analysis_features(images, aux_features)["logits"]
```

`aux_features` is accepted only for API compatibility. It is not used in DBLG.

## 7. Encoder Behavior

The `_encode()` helper is:

```python
def _encode(self, backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = backbone(images)
    if isinstance(output, (list, tuple)):
        output = [item for item in output if isinstance(item, torch.Tensor)][-1]
    if output.ndim > 2:
        output = torch.flatten(output, 1)
    return output
```

This means DBLG can handle backbones that return:

- one feature tensor.
- a list or tuple of tensors.
- feature maps with more than two dimensions.

If a backbone returns a spatial feature map such as `[B, C, Hf, Wf]`, DBLG flattens it
to `[B, C * Hf * Wf]`. In practice, the project's backbone builder usually creates
models whose output has a feature-vector shape compatible with `visual_dim`.

## 8. Shared Backbone Versus Separate Local Backbone

The builder in `build_advanced_model()` constructs DBLG as follows:

```python
global_backbone, visual_dim, metadata = _build_feature_backbone(...)
local_backbone = global_backbone
if bool(cfg.get("separate_local_backbone", False)):
    local_backbone, _, _ = _build_feature_backbone(...)
model = DualBranchLocalGlobalClassifier(...)
```

Default config:

```yaml
separate_local_backbone: false
```

Therefore, by default:

```text
global_backbone and local_backbone are the same Python module object
```

The same backbone is called twice:

1. once with the full image.
2. once with the cropped/resized image.

Because the weights are shared, gradients from both branches update the same backbone.
This makes the mode less parameter-heavy and encourages one feature extractor to learn
both full-scene and bottom-road evidence.

If you set:

```yaml
separate_local_backbone: true
```

then the builder creates a second backbone for the local branch. That doubles the
backbone parameter cost and compute memory for the visual backbone, but gives the local
branch independent parameters.

### 8.1 Practical Consequences

Shared backbone:

- lower memory use than two independent backbones.
- less risk of overfitting on small data.
- easier comparison to single-backbone baselines.
- global and local branches cannot specialize completely independently.
- the backbone is executed twice per batch, so compute is still roughly doubled even
  though parameters are shared.

Separate backbone:

- more capacity.
- possible specialization: one backbone for scene context, one for surface texture.
- higher memory use.
- more parameters.
- more overfitting risk.
- harder to train when data is limited.

### 8.2 Metadata Caveat

Current metadata sets:

```python
"branch_type": "shared_backbone_bottom_crop"
```

whenever `mode == "dual_branch_local_global"`.

This remains true even if `separate_local_backbone: true`. If you rely on metadata for
analysis, this field should be updated to distinguish:

```text
shared_backbone_bottom_crop
separate_backbone_bottom_crop
```

## 9. Feature Fusion Mathematics

Let:

```text
x = input image
x_local = bottom_crop_resize(x)
Bg = global backbone
Bl = local backbone
Pg = global projection
Pl = local projection
G = gate network
C = classifier
```

The model computes:

```text
g_raw = Bg(x)
l_raw = Bl(x_local)

g = Pg(g_raw)
l = Pl(l_raw)

a = sigmoid(Linear(LayerNorm(concat(g, l))))
l_gated = a * l

z = concat(g, l_gated)
logits = C(z)
```

Where:

```text
g_raw, l_raw: [B, visual_dim]
g, l, a:      [B, hidden_dim]
z:            [B, 2 * hidden_dim]
logits:       [B, num_classes]
```

The gate is a vector, not a scalar. Each hidden feature dimension can be suppressed or
passed through differently.

The gate has values between 0 and 1 because of `Sigmoid()`.

Important: the gate only scales the local branch. The global branch is always retained
in the fused representation.

## 10. Why Gate Only The Local Branch?

The implemented design treats global context as the stable anchor and local evidence
as a potentially helpful but sometimes noisy supplement.

This is reasonable for road imagery because the lower crop can fail in predictable
cases:

- camera is mounted unusually high or low.
- road is not centered.
- vehicle hood or dashboard blocks the lower image.
- lane markings, glare, windshield artifacts, or shadows dominate the lower image.
- road evidence appears higher in the image due to hills, curves, or camera pitch.
- bottom crop contains mostly bumper, snowbank, shoulder, or non-road surface.

By gating the local feature, the model can learn to reduce local influence when the
local crop is not useful. The global branch remains available for context.

## 11. What This Mode Does Not Do

DBLG does not currently:

- learn the crop location.
- detect the road region.
- use semantic segmentation.
- use lane detection.
- use bounding boxes.
- use class-description TF-IDF in final prediction.
- add a branch-specific loss.
- directly supervise the local branch.
- save branch gates to prediction CSVs.
- save local embeddings in the current advanced-analysis payload.
- create local-branch-specific GradCAM by default.

The only training objective for this mode is classification loss through the final
logits, unless project-level class weighting or other global training policies are
enabled.

## 12. Training Behavior

DBLG is a non-legacy advanced mode. In `core/trainer.py`, non-legacy modes are built
through:

```python
build_advanced_model(...)
```

and trained with:

```python
AdvancedModeController.compute_loss(...)
```

For DBLG:

- `mode_uses_aux("dual_branch_local_global")` is `False`.
- no auxiliary TF-IDF loss is applied.
- there is no DBLG-specific loss branch inside `compute_loss()`.
- the default loss is cross entropy from `criterion_none(logits, labels).mean()`.

The training pipeline still uses project-level settings:

```yaml
training:
  learning_rate: 0.0003
  weight_decay: 0.0001
  mixed_precision: true
  gradient_accumulation_steps: 1
  early_stopping_patience: 15
  monitor_metric: val_accuracy
  checkpoint_metric: val_accuracy
```

The optimizer is:

```python
torch.optim.AdamW
```

The scheduler is:

```python
torch.optim.lr_scheduler.CosineAnnealingLR
```

The criterion is:

```python
nn.CrossEntropyLoss(weight=class_weights)
```

and the per-sample version is:

```python
nn.CrossEntropyLoss(weight=class_weights, reduction="none")
```

Class weights come from `compute_class_weights()`. In the default config, class weights
are enabled and include boosts:

```yaml
class_weights:
  enabled: true
  boosts:
    fully: 1.25
    one track partly: 2.0
```

## 13. Data And Augmentation Interaction

The local crop is taken after the dataloader has already produced the tensor. That
means the crop is taken from the transformed image, not the raw original file.

If training augmentations are enabled, both branches receive consistent views:

```text
global branch: augmented full tensor
local branch: bottom crop from that same augmented tensor
```

This matters for horizontal flips, brightness/contrast changes, fog, rain, shadow,
blur, and noise. The local crop is not independently augmented after cropping.

Default image size:

```yaml
image_size: 224
```

Default training augmentations include:

- horizontal flip.
- illumination changes.
- fog.
- rain.
- shadow.
- blur.
- noise.
- minority-only balancing/oversampling.

## 14. Configuration

DBLG is included in the benchmark modes:

```yaml
benchmark:
  modes:
    - dual_branch_local_global
```

The default mode settings are:

```yaml
mode_defaults:
  dual_branch_local_global:
    mode_family: local_global
    hidden_dim: 512
    crop_ratio: 0.55
    separate_local_backbone: false
```

The same block exists in:

```text
configs/default_config.yaml
configs/sdre_config.yaml
```

### 14.1 `hidden_dim`

Controls the projected feature size for each branch:

```text
global_hidden: [B, hidden_dim]
local_hidden:  [B, hidden_dim]
fused:         [B, 2 * hidden_dim]
```

Default:

```yaml
hidden_dim: 512
```

Larger values increase fusion capacity and classifier parameters. Smaller values reduce
memory and may regularize the model.

### 14.2 `crop_ratio`

Controls how much of the bottom image is sent to the local branch.

Default:

```yaml
crop_ratio: 0.55
```

Recommended ranges to test:

```yaml
crop_ratio: 0.40
crop_ratio: 0.50
crop_ratio: 0.55
crop_ratio: 0.65
crop_ratio: 0.75
```

Interpretation:

- lower values focus more tightly on the near-road surface.
- higher values include more lane and context in the local branch.
- too low may miss useful road evidence.
- too high makes the local branch too similar to the global branch.

### 14.3 `separate_local_backbone`

Controls whether the local branch has independent backbone weights.

Default:

```yaml
separate_local_backbone: false
```

To train separate global and local backbones:

```yaml
separate_local_backbone: true
```

This is a large capacity increase. It should usually be tested only after the shared
backbone version has a clear baseline.

### 14.4 `dropout`

DBLG receives `dropout` from the mode config if present, otherwise from:

```yaml
fusion:
  dropout: 0.2
```

You can explicitly add it:

```yaml
dual_branch_local_global:
  mode_family: local_global
  hidden_dim: 512
  crop_ratio: 0.55
  separate_local_backbone: false
  dropout: 0.2
```

## 15. Backbone Compatibility

DBLG is not tied to one model family. It wraps whatever backbone module is being
benchmarked.

For example, if the benchmark is running DaViT:

```text
models/davit_model.py
```

DBLG uses the DaViT model module to create a feature backbone, then wraps it inside
`DualBranchLocalGlobalClassifier`.

The backbone modules follow a common pattern:

```python
def build_model_with_auxiliary(**kwargs):
    return build_model_with_auxiliary_from_spec(SPEC, **kwargs)
```

The advanced-mode builder calls `_build_feature_backbone()`, which calls
`build_model_with_auxiliary()` and extracts:

```python
aux_model.image_backbone
aux_model.visual_dim
```

This is why the function receives `aux_feature_dim` even for DBLG. It is a builder API
reuse detail, not a sign that DBLG uses auxiliary text features.

## 16. Output Artifacts

DBLG uses the same output structure as other nested advanced modes.

For a model named `DaViT`, outputs will be under a path similar to:

```text
Output/<model_name>/dual_branch_local_global/
```

or, for variant runs:

```text
Output/<model_name>/<variant>/dual_branch_local_global/
```

The exact output root is created by the path utilities in `core/path_utils.py`.

Typical outputs include:

- checkpoints.
- logs.
- training history CSV.
- training curve plot.
- predictions CSV.
- true-vs-pred CSV.
- overall metrics JSON/CSV.
- per-class metrics.
- classification report.
- confusion matrices.
- normalized confusion matrix.
- confidence histograms.
- entropy histograms.
- ROC curves.
- precision-recall curves.
- reliability diagram.
- hardest samples.
- most confident wrong samples.
- least confident correct samples.
- high-loss samples per epoch.
- explainability outputs, when enabled and available.
- advanced-analysis outputs, when enabled.
- mode-specific config JSON.
- run summary JSON.

The advanced mode controller always writes:

```text
metadata/mode_specific/mode_config.json
```

For DBLG, mode-specific training curves are only created if the training history has
extra `train_extra_*` columns. Because DBLG has no mode-specific extra loss metrics by
default, this plot may not exist.

## 17. Important Artifact Caveat: Local Branch Values Are Returned But Not Fully Saved

The model returns:

```text
local_embedding
branch_gate
```

from `extract_analysis_features()`.

However, the current `core/advanced_analysis.py` payload collector only persists a
fixed set of arrays:

```text
image_embedding
aux_embedding
aux_input
fused_embedding
logits
probabilities
```

So DBLG exposes local information internally, but the default advanced-analysis
payload does not currently save `local_embedding` or `branch_gate`.

If you want to study whether the local branch is helping, this is one of the most
useful follow-up changes:

```text
Add local_embedding and branch_gate to the collected arrays in core/advanced_analysis.py.
Add summaries of branch_gate mean/std by class, split, correctness, and confidence.
```

## 18. How To Run This Mode

To run all configured models and modes:

```powershell
python run_all_models.py --config configs/default_config.yaml
```

To run using the SDRE config:

```powershell
python run_all_models.py --config configs/sdre_config.yaml
```

If the runner supports selecting modes through config only, edit:

```yaml
benchmark:
  modes:
    - dual_branch_local_global
```

Then run the benchmark normally.

If you want to test DBLG with a smaller comparison set, use only:

```yaml
benchmark:
  modes:
    - image_only
    - dual_branch_local_global
```

This gives the cleanest comparison:

```text
single full-image branch versus full-image plus bottom-crop branch
```

## 19. Best Ablation Plan

To understand DBLG properly, run these ablations:

### 19.1 Baseline

```yaml
modes:
  - image_only
  - dual_branch_local_global
```

Purpose:

```text
Does adding the local branch improve validation/test metrics?
```

### 19.2 Crop Ratio Sweep

Try:

```yaml
crop_ratio: 0.35
crop_ratio: 0.45
crop_ratio: 0.55
crop_ratio: 0.65
crop_ratio: 0.75
```

Purpose:

```text
Find the best crop size for the camera geometry and dataset.
```

Expected pattern:

- too small: misses road context.
- middle: focuses tire paths and near-road surface.
- too large: becomes close to full-image branch.

### 19.3 Shared Versus Separate Backbones

Compare:

```yaml
separate_local_backbone: false
```

against:

```yaml
separate_local_backbone: true
```

Purpose:

```text
Determine whether the data size supports independent local specialization.
```

### 19.4 Gate Diagnostics

After adding gate export, compare:

- gate mean by true class.
- gate mean by predicted class.
- gate mean for correct versus incorrect samples.
- gate mean for high-loss samples.
- gate mean across crop ratios.
- gate variance across backbones.

Purpose:

```text
Check whether the model actually learns class- or sample-dependent local reliance.
```

### 19.5 Local-Only Control

This is not implemented as a named mode yet, but it would be highly informative:

```text
bottom crop -> backbone -> classifier
```

Purpose:

```text
Separate the value of local evidence from the value of fusion.
```

## 20. Interpreting Metrics

DBLG should not be judged only by overall accuracy. For road-surface classification,
per-class behavior matters.

Look especially at:

- `two track partly`.
- `one track partly`.
- `centre partly`.
- `fully`.

These classes can depend strongly on local tire-path evidence. DBLG is most useful if
it improves those ambiguous partial-coverage classes without hurting `bare`.

Recommended metrics:

- accuracy.
- balanced accuracy.
- macro F1.
- weighted F1.
- per-class recall.
- per-class precision.
- confusion matrix.
- top-2 accuracy.
- calibration error.
- high-confidence wrong samples.
- hardest-sample panels.

Useful interpretation questions:

- Does DBLG reduce confusion between `one track partly` and `two track partly`?
- Does DBLG reduce confusion between `centre partly` and `two track partly`?
- Does DBLG improve `fully` recall?
- Does DBLG over-predict snowier classes because the crop emphasizes snow texture?
- Does DBLG fail on images where the lower crop is not road?
- Does DBLG help more for some backbones than others?

## 21. Relationship To Literature

This implementation is best understood as a simple deterministic local-global
dual-branch classifier. It is not a direct reproduction of a single published DBLG
paper. Instead, it draws on several established research themes:

- global context plus local discriminative evidence.
- part-based and region-based fine-grained recognition.
- two-stream or multi-branch neural networks.
- attention/gating for adaptive feature weighting.
- winter road-surface classification using deep CNNs.

The literature below is a curated map of the most relevant work. It is not literally
every paper ever published on local-global vision models, but it covers the main
technical lineage needed to understand this implementation.

### 21.1 Road-Surface Condition Classification Literature

1. Pan, Fu, Yu, and Muresan, "Winter Road Surface Condition Recognition Using A
   Pretrained Deep Convolutional Network", 2018.
   Link: https://arxiv.org/abs/1812.06858

   Relevance:

   - Uses pretrained CNNs for winter road-surface condition recognition.
   - Motivates transfer learning for small or specialized road datasets.
   - Highlights image noise issues such as glare and residual salts.
   - DBLG builds on the same broad idea that image backbones can learn road-surface
     state from visual evidence.

2. Zhang, Nateghinia, Miranda-Moreno, and Sun, "Winter road surface condition
   classification using convolutional neural network (CNN): visible light and thermal
   image fusion", Canadian Journal of Civil Engineering, 2021.
   Link: https://doi.org/10.1139/cjce-2020-0613

   Relevance:

   - Develops single-stream visible, single-stream thermal, and dual-stream visible
     plus thermal CNNs.
   - The dual-stream model is a road-condition example of branch fusion improving
     classification.
   - DBLG is not multi-modal, but it is similar in spirit: two visual streams provide
     complementary evidence.

3. Xie and Kwon, "Development of a Highly Transferable Urban Winter Road Surface
   Classification Model: A Deep Learning Approach", Transportation Research Record,
   2022.
   Link: https://doi.org/10.1177/03611981221090235

   Relevance:

   - Focuses on urban winter road-surface classification and transfer learning.
   - The project's classes and Edmonton/Alberta-style winter-road context align with
     the motivation for robust urban road-surface classifiers.
   - Supports the importance of classifying road images without manual inspection.

4. Wu, Kwon, and Huynh, "Winter Road Surface Condition Recognition Using Semantic
   Segmentation and the Generative Adversarial Network: A Case Study of Iowa, U.S.A.",
   Transportation Research Record, 2023/2024.
   Link: https://doi.org/10.1177/03611981231188370

   Relevance:

   - Represents a more explicit road-region modeling direction.
   - Contrasts with DBLG: DBLG does not segment the road; it uses a fixed crop.
   - Useful as a reference for future DBLG improvements using segmentation masks.

5. "Integrating convolutional neural networks and explainable AI for enhanced winter
   road surface conditions classification using stationary RWIS imagery", Cold Regions
   Science and Technology, 2026.
   Link: https://doi.org/10.1016/j.coldregions.2026.104832

   Relevance:

   - Emphasizes full stationary RWIS imagery, camera angle, explainability, and image
     resolution.
   - Directly relates to the DBLG assumption that camera geometry affects where the
     useful road evidence appears.
   - Supports the need to validate crop settings and explainability outputs.

### 21.2 Fine-Grained Recognition And Local Evidence

1. Zhang, Donahue, Girshick, and Darrell, "Part-based R-CNNs for Fine-grained
   Category Detection", ECCV 2014.
   Link: https://arxiv.org/abs/1407.3867

   Relevance:

   - Shows that part localization can improve fine-grained categorization.
   - DBLG's lower crop is a simple fixed-region version of the broader idea that
     localized evidence can reveal subtle class differences.

2. Jaderberg, Simonyan, Zisserman, and Kavukcuoglu, "Spatial Transformer Networks",
   2015.
   Link: https://arxiv.org/abs/1506.02025

   Relevance:

   - Introduces learnable differentiable spatial transformations.
   - DBLG uses a fixed crop, not a learned transformer.
   - STN is the natural literature reference if DBLG is later upgraded to learn where
     to crop.

3. Fu, Zheng, and Mei, "Look Closer to See Better: Recurrent Attention Convolutional
   Neural Network for Fine-grained Image Recognition", CVPR 2017.
   Link: https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf

   Relevance:

   - Uses coarse-to-fine attention to look at discriminative regions.
   - DBLG's local crop is not recurrent or learned, but the motivation is similar:
     subtle distinctions often require looking closer.

4. Hu, Qi, Huang, and Lu, "See Better Before Looking Closer: Weakly Supervised Data
   Augmentation Network for Fine-Grained Visual Classification", 2019.
   Link: https://arxiv.org/abs/1901.09891

   Relevance:

   - Uses attention maps for attention crop and attention drop.
   - Provides a clear contrast to DBLG: learned attention crop versus deterministic
     lower-road crop.

5. He et al., "TransFG: A Transformer Architecture for Fine-grained Recognition",
   2021.
   Link: https://arxiv.org/abs/2103.07976

   Relevance:

   - Uses transformer attention to select discriminative patches.
   - Useful for understanding alternatives where local regions come from token
     importance rather than a hand-fixed crop.

6. Rios, Hu, and Lai, "Global-Local Similarity for Efficient Fine-Grained Image
   Recognition with Vision Transformers", 2024.
   Link: https://arxiv.org/abs/2407.12891

   Relevance:

   - Selects discriminative crops by comparing global CLS-token representation with
     local patch representations.
   - Very close in concept to global-local classification, but the crop selection is
     learned/derived from transformer token similarity, not fixed to the bottom image.

### 21.3 Global-Local Two-Pathway Models

1. He, Grant, and Ou, "Global-Local Transformer for Brain Age Estimation", 2021.
   Link: https://arxiv.org/abs/2109.01663

   Relevance:

   - Uses a global pathway for whole-image context and a local pathway for fine-grained
     patch details.
   - This is conceptually close to DBLG, although the domain and fusion method differ.

2. Multi-scale and local-global transformer families.

   Relevance:

   - Many modern vision models combine local windows/convolutions with global
     attention.
   - This supports the general principle that local texture and global layout are
     complementary.
   - DBLG implements this principle at the model-wrapper level rather than inside the
     backbone architecture.

### 21.4 Gating And Attention Literature

1. Hu, Shen, Albanie, Sun, and Wu, "Squeeze-and-Excitation Networks", 2017/2018.
   Link: https://arxiv.org/abs/1709.01507

   Relevance:

   - Introduces channel-wise feature recalibration with learned sigmoid gates.
   - DBLG's branch gate is not an SE block, but it uses the same general idea:
     learn a sigmoid vector that scales feature dimensions.

2. General attention and gated fusion methods.

   Relevance:

   - The DBLG gate is a lightweight fusion mechanism.
   - It avoids blindly concatenating raw local and global features.
   - It lets the model reduce the local contribution when the bottom crop is noisy.

## 22. How DBLG Differs From The Literature

DBLG is intentionally simple compared with many research models.

| Topic | Many literature models | This DBLG implementation |
|---|---|---|
| Local region | learned attention, proposals, segmentation, token selection | fixed bottom crop |
| Branches | often independent or specialized | shared backbone by default |
| Fusion | attention, bilinear pooling, cross-attention, transformer fusion | vector gate on local feature plus concat |
| Supervision | sometimes region/part losses or ranking losses | standard cross entropy only |
| Interpretability | sometimes region attention maps | standard project explainability; gate not yet exported |
| Road prior | often generic object parts | hard-coded lower-road prior |

The advantage is engineering simplicity and reproducibility. The disadvantage is that
the local crop is only as good as the camera geometry assumption.

## 23. Why This Implementation Makes Sense For This Project

This project benchmarks many backbone families and modes. DBLG is useful because it
adds a local/global hypothesis without requiring:

- new annotations.
- road masks.
- manual bounding boxes.
- lane detection.
- separate preprocessing.
- extra text features.
- a custom dataset format.

It can wrap the same backbone modules used elsewhere, so it can be compared against:

- `image_only`.
- `image_plus_tfidf_aux`.
- `aux_cross_attention`.
- `aux_film`.
- `metric_learning_hybrid`.
- `attention_consistency`.
- other advanced modes.

This makes DBLG a practical experimental mode rather than a completely separate model
pipeline.

## 24. Strengths

DBLG has several practical strengths:

- It tests a domain-specific hypothesis: bottom-road surface evidence matters.
- It keeps the full scene available.
- It is compatible with all project backbones that expose a feature vector.
- It needs no new labels.
- It is deterministic and easy to reproduce.
- It can be trained with the existing pipeline.
- It has a simple interpretation: global context plus gated local crop.
- It can improve classes where tire-path evidence is decisive.
- It can reduce reliance on sky, trees, roadside background, or snowbanks if those
  cues are misleading.

## 25. Weaknesses And Failure Modes

DBLG can fail when:

- the road is not in the bottom 55 percent of the image.
- the camera is tilted or mounted differently.
- the bottom crop contains the vehicle hood.
- the bottom crop contains shoulder, curb, parked cars, or non-road pixels.
- the important evidence is farther ahead in the lane.
- snow texture near the vehicle is not representative of the full road class.
- resizing the crop distorts local scale too much.
- the local crop creates overconfidence in ambiguous scenes.
- the shared backbone cannot specialize for both views.
- the separate backbone overfits when enabled.

It can also amplify dataset bias. If some classes are associated with a particular
camera position or road geometry, the lower crop may learn that shortcut.

## 26. Debugging Checklist

If DBLG performs worse than `image_only`, check:

1. Are input images actually road-forward images with road in the lower half?
2. Are there many images where the lower crop is non-road?
3. Is `crop_ratio` too small?
4. Is `crop_ratio` too large, making the mode redundant?
5. Is the shared backbone underfitting because it is called on two different views?
6. Is `separate_local_backbone: true` overfitting?
7. Are augmentations changing geometry in a way that makes the lower crop unreliable?
8. Are the misclassified samples mostly from unusual camera angles?
9. Are partial classes still confused because the crop misses the tire paths?
10. Are class weights and oversampling causing the model to over-predict minority
    classes?

## 27. Recommended Visual Inspection

Before trusting DBLG, inspect the local crops.

A simple debugging visualization should show, for each sample:

```text
original image
bottom crop before resize
bottom crop after resize
true label
prediction
confidence
correct/incorrect
```

Recommended samples:

- random train samples.
- random validation samples.
- high-loss validation samples.
- most confident wrong validation samples.
- one panel per class.
- examples from each camera/source if metadata exists.

This directly tests the key assumption:

```text
The bottom crop contains the road-surface evidence needed for classification.
```

## 28. Suggested Code Improvements

### 28.1 Save Local Embeddings And Branch Gates

Extend `core/advanced_analysis.py` to collect:

```text
local_embedding
branch_gate
```

Then save summaries:

```text
gate_mean_by_class.csv
gate_mean_by_correctness.csv
gate_mean_by_prediction.csv
gate_mean_by_confidence_bin.csv
```

This would make the learned local reliance measurable.

### 28.2 Add Local Crop Visualization

Add an artifact under:

```text
plots/dual_branch_local_global_local_crop_examples.png
```

This should show the exact tensor region used by the local branch.

### 28.3 Fix Branch Metadata

Change metadata logic from:

```python
"branch_type": "shared_backbone_bottom_crop"
```

to something like:

```python
"branch_type": (
    "separate_backbone_bottom_crop"
    if mode == "dual_branch_local_global" and bool(cfg.get("separate_local_backbone", False))
    else "shared_backbone_bottom_crop"
)
```

### 28.4 Add A Local-Only Mode

Add a mode:

```text
local_crop_only
```

This would isolate whether the crop alone is useful.

### 28.5 Add A Multi-Crop Local Branch

Instead of one lower crop, test:

```text
bottom 40 percent
bottom 55 percent
center road band
left tire-path crop
right tire-path crop
```

Fuse multiple local features. This may help with `one track partly` versus
`two track partly`, because left/right tire-path asymmetry is class-relevant.

### 28.6 Add A Learned Crop Option

Possible approaches:

- Spatial Transformer Network.
- attention-map crop.
- transformer token selection.
- segmentation-guided road mask crop.
- saliency-guided top-k region crop.

This would reduce reliance on fixed camera geometry, but adds complexity and possible
instability.

### 28.7 Add Road-Mask-Guided Local Branch

If road segmentation masks are available or can be generated, the local branch could
receive:

```text
image * road_mask
```

or:

```text
bounding box around road mask
```

This would make the local branch more semantically grounded than a bottom crop.

## 29. Comparison To Auxiliary Text Modes

DBLG is different from modes such as:

```text
aux_cross_attention
aux_film
aux_text_fusion_variants
aux_gated_fusion
conditional_moe
```

Those modes use class-description TF-IDF features as auxiliary context or regularizers.
DBLG does not.

For DBLG:

```text
prediction input = image only
auxiliary text = not used
local branch = deterministic crop from image tensor
```

This makes DBLG useful as a clean visual architecture comparison.

## 30. Important Interpretation For Class Descriptions

The word "dual" also appears in `core/class_descriptions.py` for the class concept of
dual tire tracks:

```text
two track partly
```

That is unrelated to the `dual_branch_local_global` architecture name.

There are two different uses of "dual":

```text
dual branch = two image pathways in the model
dual wheel paths = visual road condition in a class description
```

Do not confuse the two.

## 31. Expected Class-Level Effects

DBLG should be most helpful when local road-surface evidence determines the class.

Potentially helped:

- `two track partly`: both tire paths visible, snow between them.
- `one track partly`: asymmetric tire-path clearing.
- `centre partly`: center condition may require looking at lane center and tire paths.
- `fully`: bottom-road texture can confirm continuous snow coverage.

Potentially less helped:

- `bare`: full-image context may already be enough, and a crop may overemphasize small
  snow patches or glare.

Potential risks:

- If the bottom crop includes only a small road patch, the model may miss broader
  coverage patterns.
- If tire tracks appear higher in the image, the local branch may miss the decisive
  evidence.
- If classes are visually subtle, a single bottom crop may not capture left/right
  asymmetry well enough.

## 32. The Best Way To Explain DBLG In A Paper Or Thesis

A concise technical description:

```text
The dual_branch_local_global mode is a two-view visual classifier. For each input
image, the model encodes both the full frame and a deterministic lower-frame crop. The
full frame captures global scene context, while the lower crop emphasizes near-road
surface and tire-path evidence. Both views are passed through a shared backbone by
default, projected to a common hidden dimension, and fused by a learned sigmoid gate
that modulates the local representation before concatenation and classification. The
model is trained end-to-end with the same cross-entropy objective as the image-only
baseline.
```

A concise non-technical description:

```text
The model looks at the whole road scene and also takes a closer look at the lower
road surface. It then learns how much the close-up road evidence should influence the
final road-condition decision.
```

## 33. Suggested Reporting Table

When reporting DBLG results, include:

| setting | value |
|---|---|
| mode | `dual_branch_local_global` |
| backbone | e.g. `DaViT`, `ConvNeXt`, `Swin`, etc. |
| image size | e.g. `224` |
| crop ratio | e.g. `0.55` |
| crop region | bottom `55%` of tensor height |
| resize after crop | yes, bilinear to original input size |
| shared backbone | yes/no |
| hidden dim | e.g. `512` |
| gate type | vector sigmoid gate on local hidden feature |
| fusion | concatenate global hidden and gated local hidden |
| loss | class-weighted cross entropy |
| auxiliary text | not used |
| optimizer | AdamW |
| scheduler | cosine annealing |

## 34. Minimal Pseudocode

```python
def forward(images):
    local_images = resize(bottom_crop(images, crop_ratio), images.shape[-2:])

    global_raw = backbone(images)
    local_raw = backbone(local_images)  # same backbone by default

    global_hidden = linear_global(global_raw)
    local_hidden = linear_local(local_raw)

    gate = sigmoid(linear(layernorm(concat(global_hidden, local_hidden))))
    local_hidden = local_hidden * gate

    fused = concat(global_hidden, local_hidden)
    logits = classifier(fused)
    return logits
```

## 35. Minimal Mathematical Summary

```text
x_l = resize(bottom_crop(x, r), H, W)
h_g = P_g(B_g(x))
h_l = P_l(B_l(x_l))
a = sigmoid(W_a(LN([h_g; h_l])))
z = [h_g; a * h_l]
y_hat = C(z)
loss = CE(y_hat, y)
```

Where:

```text
r = crop_ratio
CE = cross entropy
```

## 36. Current Implementation Verdict

DBLG is a pragmatic, domain-informed local-global model. It is not the most advanced
possible local-global architecture, but it is a strong experimental mode because it is:

- easy to compare against the image-only baseline.
- easy to tune through `crop_ratio`.
- compatible with many backbones.
- free of extra annotation requirements.
- aligned with the road-condition intuition that near-road tire-path evidence matters.

The main thing to improve is observability. The model already computes the local
embedding and branch gate, but the current analysis pipeline does not persist them by
default. Exporting and analyzing those tensors would make DBLG much easier to explain
and defend.

## 37. Source Links

Project files:

- `core/advanced_modes.py`
- `core/mode_registry.py`
- `configs/default_config.yaml`
- `configs/sdre_config.yaml`
- `core/trainer.py`
- `core/evaluator.py`
- `core/advanced_analysis.py`
- `models/common.py`
- `core/fusion_modules.py`

External literature:

- Pan et al. 2018, winter RSC pretrained CNN: https://arxiv.org/abs/1812.06858
- Zhang et al. 2021, visible/thermal dual-stream winter RSC CNN: https://doi.org/10.1139/cjce-2020-0613
- Xie and Kwon 2022, urban winter RSC transfer learning: https://doi.org/10.1177/03611981221090235
- Wu et al. 2023/2024, segmentation/GAN winter RSC: https://doi.org/10.1177/03611981231188370
- Cold Regions Science and Technology 2026 RWIS CNN/XAI paper: https://doi.org/10.1016/j.coldregions.2026.104832
- Part-based R-CNNs for fine-grained category detection: https://arxiv.org/abs/1407.3867
- Spatial Transformer Networks: https://arxiv.org/abs/1506.02025
- RA-CNN, "Look Closer to See Better": https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Look_Closer_to_CVPR_2017_paper.pdf
- Squeeze-and-Excitation Networks: https://arxiv.org/abs/1709.01507
- WS-DAN: https://arxiv.org/abs/1901.09891
- TransFG: https://arxiv.org/abs/2103.07976
- Global-Local Transformer for Brain Age Estimation: https://arxiv.org/abs/2109.01663
- Global-Local Similarity for FGVC with ViTs: https://arxiv.org/abs/2407.12891
