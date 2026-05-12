# Auxiliary Class-Description Learning With TF-IDF

Note: the requested filename `AUX.md` cannot be created normally on Windows because
`AUX` is a reserved device name, even with an extension. This document is therefore
saved as `AUXILIARY.md`.

This document explains the auxiliary-learning approach in this project: how fixed
class descriptions are converted into TF-IDF vectors, how those vectors are attached
to image samples, how the auxiliary branch is trained, how this affects performance
and GradCAM interpretation, and how the approach should be implemented safely.

The short version: each road-surface class has a fixed textual description in
`core/class_descriptions.py`. During data preparation, every image receives the
description associated with its label. A TF-IDF encoder converts that description into
a numeric vector. The model encodes the image and the text vector into comparable
representations, then uses an auxiliary alignment loss to encourage the visual
embedding to reflect the semantic class definition. The final classifier remains
image-driven so the model cannot simply read the class description and bypass visual
learning.

## 1. Why This Exists

Road-surface labels are not purely visual names. The difference between classes such
as `centre partly`, `two track partly`, and `one track partly` depends on semantic
definitions:

- where snow is located.
- whether tire paths are exposed.
- whether one or both tracks are visible.
- whether the immediate drivable path is snow-covered.
- whether surrounding snow should or should not affect the label.

The auxiliary approach injects those definitions into training. It does not replace
image classification. It acts as a regularizer that encourages image features to
organize themselves in a way that is consistent with the class descriptions.

## 2. Non-Technical Explanation

The image model learns from pictures. The auxiliary branch gives the model a written
description of what each class means during training.

For example, the description of `two track partly` says that both tire paths are
visible while snow remains between them. The model still has to classify the image
from pixels, but the auxiliary loss nudges the image representation toward the text
representation for that class.

This is like telling the model:

```text
When you learn the visual pattern for this class, make the internal feature resemble
the class definition, but do not use the definition itself as the final answer.
```

That last part is essential. Because the text is assigned from the true label, direct
classification from the text vector would be label leakage.

## 3. Repository Map

Relevant files:

```text
core/class_descriptions.py
  fixed class descriptions and class-name normalization

core/dataset_builder.py
  assigns class descriptions to samples
  fits TF-IDF on training text
  creates per-sample auxiliary feature maps
  saves class-description embedding metadata

core/tfidf_encoder.py
  TfidfSettings
  TfidfAuxiliaryEncoder

core/data_utils.py
  RoadSurfaceDataset returns aux_features in each batch

core/fusion_modules.py
  TfidfFusionClassifier for image_plus_tfidf_aux
  auxiliary branch and cosine alignment loss

core/advanced_modes.py
  AuxResearchFusionClassifier
  advanced auxiliary modes
  advanced-mode auxiliary loss

core/trainer.py
  training loop
  auxiliary loss weighting
  optimizer, scheduler, checkpoints

core/evaluator.py
  predictions and metrics

core/explainability.py
  GradCAM / gradient saliency for image-only and auxiliary modes

core/advanced_analysis.py
  embedding payloads, CCA, retrieval, CKA, decision-quality analyses

configs/default_config.yaml
configs/sdre_config.yaml
  TF-IDF and auxiliary-mode configuration
```

## 4. The Class Descriptions

The class descriptions live in:

```text
core/class_descriptions.py
```

The project defines five descriptions:

```text
bare
centre partly
two track partly
one track partly
fully
```

Each description is a paragraph that encodes the operational meaning of the class.
For example:

- `bare`: tire-path region is effectively free of snow.
- `centre partly`: tire paths are largely exposed but snow remains around the lane
  center or adjacent context.
- `two track partly`: both tire paths are exposed while snow remains between them.
- `one track partly`: one tire path is exposed while the other remains snow-covered.
- `fully`: both intended wheel paths are snow-covered.

The file also includes:

```python
normalize_class_name(...)
description_for_class(...)
```

These handle variants such as:

```text
center partly -> centre partly
twotrack partly -> two track partly
onetrack partly -> one track partly
```

## 5. Data Preparation Path

The data path is:

```text
dataset records
  -> assign class description text
  -> fit TF-IDF encoder on training text
  -> transform train/val/test text
  -> store per-sample aux vectors
  -> dataloader returns aux_features tensor
```

The key function is:

```python
prepare_data_bundle(...)
```

It checks:

```yaml
data:
  context_source: fixed_class_descriptions
```

When that value is active, it calls:

```python
apply_fixed_class_descriptions(...)
```

That function sets:

```python
record.context_text = description_for_class(record.label_name)
record.context_match_type = "fixed_class_description"
record.context_json_key = normalize_class_name(record.label_name)
```

Then `fit_tfidf_features(...)` creates the numeric auxiliary vectors.

## 6. Important Leakage Principle

The class-description text is assigned from the true label. That means it contains
label information.

If the final classifier receives the TF-IDF vector directly, the model can learn:

```text
label = classifier(TF-IDF class description)
```

instead of:

```text
label = classifier(image evidence)
```

That would produce misleadingly high performance and unreliable GradCAM. The model
could predict correctly because the auxiliary vector identifies the class, not because
the image branch learned road-surface evidence.

The safe design is:

```text
final logits = classifier(image representation)
auxiliary text = side branch used for training regularization and diagnostics
```

This project follows that principle.

## 7. TF-IDF Encoder Implementation

The encoder lives in:

```text
core/tfidf_encoder.py
```

The settings are:

```python
@dataclass
class TfidfSettings:
    max_features: int = 2048
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 1
    stop_words: str | None = "english"
    use_svd: bool = False
    svd_components: int = 256
    random_state: int = 42
```

The project uses scikit-learn:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
```

The vectorizer is constructed as:

```python
self.vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(settings.ngram_min, settings.ngram_max),
    max_features=settings.max_features,
    min_df=settings.min_df,
    stop_words=settings.stop_words,
    dtype=np.float32,
)
```

Default config:

```yaml
tfidf:
  max_features: 2048
  ngram_min: 1
  ngram_max: 2
  min_df: 1
  stop_words: english
  use_svd: false
  svd_components: 256
```

## 8. What TF-IDF Means

TF-IDF means term frequency-inverse document frequency.

At a high level:

```text
TF-IDF(term, document) = term frequency in this document * inverse document frequency
```

Terms that appear often in one document but not in every document receive higher
weight. Terms that appear everywhere receive lower weight.

In this project:

- each class description is a document.
- unigrams and bigrams are used by default.
- common English stop words are removed.
- the resulting vector is a sparse bag-of-words style semantic representation.
- it is converted to a dense `float32` NumPy array.

Because there are only five class descriptions, the TF-IDF vocabulary is small in
practice even though `max_features` is 2048.

## 9. Optional SVD

The encoder optionally supports:

```yaml
use_svd: true
svd_components: 256
```

When enabled, `TruncatedSVD` reduces the TF-IDF vector dimension. The code only applies
SVD if the sparse matrix has enough rank:

```python
max_rank = min(sparse.shape[0] - 1, sparse.shape[1] - 1)
if max_rank >= 2:
    n_components = min(settings.svd_components, max_rank)
```

With only five class descriptions, the maximum useful rank is limited. SVD can be
useful for larger context corpora, but for fixed five-class descriptions it is usually
not necessary.

## 10. Per-Sample Auxiliary Feature Assignment

After fitting the encoder on training descriptions, the project creates:

```python
feature_maps: dict[str, dict[str, np.ndarray]]
```

The structure is:

```text
feature_maps[split][sample_id] = TF-IDF vector
```

For the training split:

```python
train_features = encoder.fit_transform(train_texts)
feature_maps["train"] = {
    record.sample_id: train_features[i]
    for i, record in enumerate(train_records)
}
```

For validation/test:

```python
features = encoder.transform(texts)
feature_maps[split] = {
    record.sample_id: features[i]
    for i, record in enumerate(split_records)
}
```

The encoder is fitted only on training text, then reused for the other splits. That is
the normal train/test discipline for text preprocessing.

## 11. Saved Metadata

The project saves auxiliary metadata under:

```text
<output_dir>/_global_comparison/metadata/
```

Important files:

```text
tfidf_encoder.pkl
class_description_embedding_cache.csv
class_description_embedding_cache.npz
class_description_embedding_assignments.csv
```

Meaning:

- `tfidf_encoder.pkl`: pickled `TfidfAuxiliaryEncoder`.
- `class_description_embedding_cache.csv`: class name, safe class name, text length,
  embedding dimension, and description text.
- `class_description_embedding_cache.npz`: per-class dense vectors.
- `class_description_embedding_assignments.csv`: sample-to-class-embedding mapping.

These files are useful for auditing whether each image received the expected auxiliary
description.

## 12. Dataloader Behavior

The dataset class is:

```text
core/data_utils.py
  RoadSurfaceDataset
```

Each item returns:

```python
{
    "image": image_tensor,
    "label": torch.tensor(record.label, dtype=torch.long),
    "aux_features": aux_tensor,
    "sample_id": record.sample_id,
    "image_path": str(record.image_path),
    "relative_path": record.relative_path,
    "label_name": record.label_name,
}
```

If the mode uses auxiliary features:

```python
aux_dim = bundle.tfidf_feature_dim
```

If the mode does not use auxiliary features:

```python
aux_dim = 0
```

When `aux_dim > 0`, the dataset retrieves:

```python
aux = self.aux_features.get(aux_key)
```

and returns it as:

```python
torch.from_numpy(aux.astype(np.float32, copy=False))
```

If no vector is found, it returns zeros of the correct dimension.

## 13. Which Modes Use Auxiliary Features

Auxiliary modes are defined in:

```text
core/mode_registry.py
```

The baseline auxiliary mode is:

```text
image_plus_tfidf_aux
```

It is marked:

```python
uses_aux=True
legacy=True
```

Advanced auxiliary modes include:

```text
aux_cross_attention
aux_film
aux_text_fusion_variants
conditional_moe
aux_gated_fusion
```

They are all marked with:

```python
uses_aux=True
```

Modes such as `image_only`, `dual_branch_local_global`, `metric_learning_hybrid`, and
`uncertainty_aware` do not use auxiliary text vectors.

## 14. Baseline Auxiliary Model: `image_plus_tfidf_aux`

The baseline auxiliary model is implemented by:

```text
core/fusion_modules.py
  TfidfFusionClassifier
```

The builder is:

```python
build_tfidf_fusion_classifier(...)
```

The model has:

```python
self.image_backbone
self.image_head
self.aux_branch
self.classifier
```

The image path:

```text
image -> image_backbone -> visual -> image_head -> image_context -> classifier -> logits
```

The auxiliary path:

```text
TF-IDF vector -> aux_branch -> aux_embedding
```

The key implementation is:

```python
visual = self.encode_image(images)
image_context = self.image_head(visual)
aux_embedding = self.encode_aux(aux_features, ...)
logits = self.classifier(image_context)
```

Returned features:

```python
{
    "image_embedding": visual,
    "visual_context_embedding": image_context,
    "aux_embedding": aux_embedding,
    "fused_embedding": image_context,
    "logits": logits,
}
```

Important point:

```text
logits are computed from image_context only
```

The auxiliary vector is not concatenated into the classifier input.

## 15. The Auxiliary Branch

The baseline auxiliary branch is:

```python
self.aux_branch = nn.Sequential(
    nn.LayerNorm(aux_dim),
    nn.Linear(aux_dim, aux_hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(aux_hidden_dim, self.context_dim),
    nn.GELU(),
)
```

Default:

```yaml
fusion:
  aux_hidden_dim: 256
  dropout: 0.2
```

This maps the TF-IDF vector into the same dimensional space as the image context
embedding. That makes cosine alignment possible.

## 16. The Image Head

The image head is:

```python
self.image_head = nn.Sequential(
    nn.LayerNorm(visual_dim),
    nn.Dropout(dropout),
    nn.Linear(visual_dim, self.context_dim),
    nn.GELU(),
    nn.Dropout(dropout),
)
```

Then:

```python
self.classifier = nn.Sequential(
    nn.LayerNorm(self.context_dim),
    nn.Dropout(dropout),
    nn.Linear(self.context_dim, num_classes),
)
```

So the image embedding is projected into a context dimension and classified.

The default context dimension is:

```python
context_dim = fusion_hidden_dim or max(visual_dim // 2, aux_hidden_dim)
```

## 17. Baseline Auxiliary Loss

The auxiliary loss is:

```python
return (
    1.0
    - F.cosine_similarity(
        F.normalize(visual, dim=1, eps=1e-8),
        F.normalize(aux, dim=1, eps=1e-8),
        dim=1,
    )
).mean()
```

Mathematically:

```text
L_aux = mean_i [1 - cosine(v_i, e_i)]
```

Where:

```text
v_i = projected visual embedding for sample i
e_i = encoded TF-IDF class-description embedding for sample i
```

The total baseline auxiliary loss is:

```text
L_total = L_ce + lambda_aux * L_aux
```

Default:

```yaml
fusion:
  auxiliary_loss_weight: 0.05
```

## 18. Training Loop For `image_plus_tfidf_aux`

In `core/trainer.py`, legacy auxiliary training does this:

```python
features = model.extract_analysis_features(images, aux)
logits = features["logits"]
raw_loss = criterion(logits, labels)
aux_loss = model.auxiliary_context_loss(features, aux, labels)
raw_loss = raw_loss + auxiliary_loss_weight * aux_loss
```

The criterion is:

```python
nn.CrossEntropyLoss(weight=class_weights)
```

The optimizer is:

```python
torch.optim.AdamW
```

The scheduler is:

```python
torch.optim.lr_scheduler.CosineAnnealingLR
```

The same augmentation, class weighting, checkpointing, and evaluation paths are used as
the image-only model.

## 19. Advanced Auxiliary Modes

The advanced auxiliary modes are implemented by:

```text
core/advanced_modes.py
  AuxResearchFusionClassifier
```

These modes use a shared pattern:

```text
image -> backbone -> visual_raw -> visual_proj -> visual
TF-IDF -> aux_encoder -> aux
aux interaction creates aux_conditioned diagnostics
final logits = classifier(visual)
auxiliary loss aligns visual and aux representations
```

Again:

```text
final logits remain image-only
```

The advanced auxiliary loss is:

```python
alignment = 1.0 - F.cosine_similarity(
    normalize(visual),
    normalize(aux),
    dim=1,
)
loss = alignment.mean()
conditioned = features.get("aux_conditioned_embedding")
if isinstance(conditioned, torch.Tensor):
    loss = loss + 0.1 * F.mse_loss(
        normalize(conditioned),
        normalize(visual.detach()),
    )
```

The total loss is:

```text
L_total = L_ce + lambda_aux * L_aux + optional mode-specific terms
```

Default advanced auxiliary weight:

```yaml
auxiliary_loss_weight: 0.05
```

## 20. `aux_cross_attention`

This mode creates a visual token and an auxiliary token:

```python
visual_token = visual.unsqueeze(1)
aux_token = aux.unsqueeze(1)
```

Then it applies multi-head attention:

```python
update, weights = attn(
    query=attended,
    key=aux_token,
    value=aux_token,
    need_weights=True,
)
attended = norm(attended + update)
```

Conceptually:

```text
Q = visual token
K = auxiliary token
V = auxiliary token
```

The attention-conditioned output is used for diagnostics and auxiliary loss. The final
classifier still receives the unconditioned visual representation.

Config:

```yaml
aux_cross_attention:
  hidden_dim: 512
  aux_hidden_dim: 256
  dropout: 0.2
  auxiliary_loss_weight: 0.05
  cross_attention_depth: 2
  cross_attention_heads: 4
```

## 21. `aux_film`

FiLM means feature-wise linear modulation. In this project:

```python
gamma, beta = torch.chunk(self.film(aux), 2, dim=1)
gamma = 0.1 * torch.tanh(gamma)
beta = 0.1 * torch.tanh(beta)
aux_conditioned = visual * (1.0 + gamma) + beta
```

This is bounded modulation. The `0.1 * tanh(...)` scaling prevents the text vector
from dominating the visual representation.

Config:

```yaml
aux_film:
  hidden_dim: 512
  aux_hidden_dim: 256
  dropout: 0.2
  auxiliary_loss_weight: 0.05
```

## 22. `aux_gated_fusion`

This mode computes a gate from the concatenated visual and auxiliary embeddings:

```python
gate = self.gate(torch.cat([visual, aux], dim=1))
if gate.shape[1] == 1:
    gate = gate.expand_as(visual)
aux_conditioned = visual + 0.1 * gate * aux
```

The gate can be scalar or vector depending on config:

```yaml
gate_type: vector
```

The final logits still use:

```python
logits = self.residual_classifier(visual)
```

not `aux_conditioned`.

## 23. `aux_text_fusion_variants`

This mode lets one config entry select among:

```text
residual
gated
film
cross_attention
```

Config:

```yaml
aux_text_fusion_variants:
  hidden_dim: 512
  aux_hidden_dim: 256
  fusion_variant: residual
  dropout: 0.2
  auxiliary_loss_weight: 0.05
```

The variant changes how the auxiliary-conditioned diagnostic embedding is computed.
It does not change the final image-only prediction path.

## 24. `conditional_moe`

This mode creates a router from visual and auxiliary embeddings:

```python
router_logits = self.router(torch.cat([visual, aux], dim=1))
router_probs = torch.softmax(router_logits, dim=1)
expert_stack = torch.stack([expert(visual) for expert in self.experts], dim=1)
```

It adds optional expert/load-balancing terms:

```text
L_balance = MSE(mean router usage, uniform usage)
L_expert = cross entropy over expert logits
```

The final prediction returned by `features["logits"]` is still:

```python
logits = self.residual_classifier(visual)
```

The router and expert outputs are diagnostics and regularizers.

## 25. Why The Current Design Avoids Direct Concatenation

Earlier multimodal systems often concatenate:

```text
[image_features; text_features] -> classifier
```

That is valid when text is available independently at inference and is not derived from
the true label.

Here, the text is a fixed class description assigned from the true label. Therefore,
direct concatenation would be unsafe. A classifier could learn a near-perfect mapping:

```text
TF-IDF vector for "bare" -> bare
TF-IDF vector for "fully" -> fully
...
```

That would inflate metrics and weaken visual explanations.

The correct use in this project is:

```text
TF-IDF text vector = training-time auxiliary semantic target
image representation = prediction source
```

## 26. Expected Effect On Performance

Auxiliary class-description learning can improve performance when:

- labels encode subtle semantic distinctions.
- classes are visually adjacent.
- the dataset is not huge.
- image-only training overfits to background or camera shortcuts.
- class descriptions accurately describe the visual decision rules.

Possible improvements:

- higher macro F1.
- higher balanced accuracy.
- improved recall for ambiguous minority classes.
- better confusion structure between adjacent snow states.
- cleaner embedding clusters.
- more semantically meaningful GradCAM regions.
- lower overconfidence on ambiguous images.

Performance can worsen when:

- class descriptions are too similar.
- descriptions contain ambiguous or contradictory wording.
- the auxiliary loss weight is too high.
- labels are noisy.
- images do not visually match the written definitions.
- the model is forced to align with text that is not visually observable.

The auxiliary branch should be judged against `image_only`, not in isolation.

## 27. What To Compare

Recommended baseline comparison:

```yaml
benchmark:
  modes:
    - image_only
    - image_plus_tfidf_aux
```

Recommended advanced comparison:

```yaml
benchmark:
  modes:
    - image_only
    - image_plus_tfidf_aux
    - aux_cross_attention
    - aux_film
    - aux_gated_fusion
    - aux_text_fusion_variants
```

Metrics to inspect:

- validation accuracy.
- test accuracy.
- balanced accuracy.
- macro F1.
- per-class precision.
- per-class recall.
- confusion matrix.
- calibration error.
- high-loss samples.
- most confident wrong samples.
- embedding cluster quality.
- CCA alignment between image and auxiliary spaces.

## 28. Class-Level Expectations

The biggest expected gains are likely in:

```text
centre partly
two track partly
one track partly
fully
```

Why:

- these classes have nuanced tire-path and snow-placement definitions.
- the class descriptions encode those distinctions explicitly.
- image-only models can confuse adjacent partial-snow states.

The `bare` class may already be easier visually, so the gain may be smaller.

Important confusion pairs:

```text
centre partly <-> two track partly
two track partly <-> one track partly
one track partly <-> fully
bare <-> centre partly
```

## 29. GradCAM Behavior

GradCAM is implemented in:

```text
core/explainability.py
```

The auxiliary path affects GradCAM indirectly by changing the learned image
representation during training.

At explanation time, the code does:

```python
if mode_uses_aux(mode):
    logits = model(image_tensor, aux_tensor)
else:
    logits = model(image_tensor)
```

But in the current auxiliary implementations:

```text
logits are image-only
```

So the GradCAM heatmap is driven by the image path. The auxiliary vector is passed for
API compatibility and to support models that accept auxiliary inputs, but it should not
directly control the prediction.

## 30. Why Auxiliary Learning Can Improve GradCAM

If the auxiliary loss works, it can make the visual representation more aligned with
the semantic class definition. That may cause GradCAM to emphasize more relevant road
regions:

- tire paths.
- lane center.
- snow between wheel tracks.
- immediate drivable surface.
- exposed pavement.
- compacted snow or snow film.

This is an indirect effect:

```text
auxiliary text changes training gradients
training changes image features
image features change GradCAM
```

It is not:

```text
text vector directly tells GradCAM where to look
```

## 31. What Bad GradCAM Would Indicate

Bad auxiliary GradCAM patterns include:

- heat concentrated on sky or trees.
- heat concentrated on snowbanks outside the lane.
- heat concentrated on image borders or camera artifacts.
- heat concentrated on text-like watermarks if any exist.
- heat not changing from image-only despite performance claims.
- heat becoming less road-focused while metrics improve suspiciously.

If metrics improve but GradCAM becomes less image-grounded, check for leakage or
dataset shortcuts.

## 32. Paired GradCAM Comparison

For `image_plus_tfidf_aux`, the project creates paired panels:

```text
gradcam/paired_before_vs_after/
```

and global explainability panels can aggregate those under:

```text
_global_comparison/explainability/
```

The all-mode GradCAM comparison path is:

```text
gradcam/all_mode_comparison/val/
```

A useful review workflow:

1. Compare `image_only` and `image_plus_tfidf_aux` overlays on the same validation
   samples.
2. Look at cases where auxiliary mode fixes an image-only error.
3. Look at cases where auxiliary mode introduces a new error.
4. Check whether heat shifts toward tire-path and road-surface evidence.
5. Check whether heat shifts toward irrelevant context.

## 33. Current GradCAM Implementation Details

The code first tries standard GradCAM if `pytorch-grad-cam` is installed:

```python
from pytorch_grad_cam import GradCAM
```

It scans for the last convolution layer in:

```python
scan_model = model.image_backbone if hasattr(model, "image_backbone") else model
```

If no usable GradCAM layer exists, it falls back to gradient saliency:

```python
score.backward()
grad = image_tensor.grad.detach().abs().max(dim=1)[0]
```

Saved panel columns:

```text
original image
heatmap
overlay
```

Caption includes:

```text
mode
true label
predicted label
confidence
method
```

## 34. Advanced Analysis For Auxiliary Modes

`core/advanced_analysis.py` collects:

```text
image_embedding
aux_embedding
aux_input
fused_embedding
logits
probabilities
```

For auxiliary modes, CCA alignment can run:

```text
image_vs_aux
image_vs_fused
aux_vs_fused
```

This helps answer:

- Are image embeddings aligned with auxiliary class descriptions?
- Does the auxiliary branch change the geometry of the representation?
- Are some classes better aligned than others?
- Does better alignment correlate with accuracy?

The retrieval-board logic also uses available embedding spaces. For auxiliary modes,
it can compare examples in image/fused spaces and expose confusion neighborhoods.

## 35. Recommended Auxiliary Diagnostics

Add or inspect:

```text
auxiliary_context_loss over epochs
image-vs-aux cosine similarity by class
image-vs-aux cosine similarity for correct vs incorrect samples
image-vs-aux cosine similarity for high-loss samples
CCA alignment by class
embedding projections colored by class and correctness
GradCAM before/after paired panels
confusion matrix changes
per-class recall changes
```

The training loop already logs extra metrics from auxiliary losses in history rows.
Mode-specific curves are created when `train_extra_*` columns exist.

## 36. The Right Implementation Pattern

For label-derived class descriptions, use this pattern:

```text
image -> image encoder -> visual embedding -> classifier -> logits
class description -> TF-IDF encoder -> aux encoder -> aux embedding
loss = cross_entropy(logits, label) + lambda * alignment_loss(visual, aux)
```

Do not use:

```text
[visual embedding; class-description TF-IDF] -> classifier -> logits
```

unless the text is independently available at inference and not derived from the
ground-truth label.

## 37. Minimal Pseudocode

```python
for batch in loader:
    images = batch["image"]
    labels = batch["label"]
    aux = batch["aux_features"]

    visual_raw = image_backbone(images)
    visual = image_head(visual_raw)
    aux_embedding = aux_branch(aux)

    logits = classifier(visual)

    ce_loss = cross_entropy(logits, labels)
    aux_loss = 1.0 - cosine(normalize(visual), normalize(aux_embedding))
    loss = ce_loss + lambda_aux * aux_loss.mean()

    loss.backward()
    optimizer.step()
```

## 38. Minimal Mathematical Form

Let:

```text
x_i = image
y_i = class label
t_i = class-description text assigned from y_i
E_img = image encoder
P_img = image projection head
E_txt = TF-IDF vectorizer plus auxiliary MLP
C = classifier
```

Then:

```text
v_i = P_img(E_img(x_i))
e_i = E_txt(t_i)
z_i = C(v_i)
L_ce = CE(z_i, y_i)
L_aux = 1 - cosine(normalize(v_i), normalize(e_i))
L_total = L_ce + lambda_aux * L_aux
```

The prediction is:

```text
argmax(C(v_i))
```

not:

```text
argmax(C([v_i; e_i]))
```

## 39. Configuration Guide

Core config:

```yaml
data:
  context_source: fixed_class_descriptions

tfidf:
  max_features: 2048
  ngram_min: 1
  ngram_max: 2
  min_df: 1
  stop_words: english
  use_svd: false
  svd_components: 256

fusion:
  mode: concat
  aux_hidden_dim: 256
  dropout: 0.2
  auxiliary_loss_weight: 0.05
```

For baseline auxiliary:

```yaml
benchmark:
  modes:
    - image_only
    - image_plus_tfidf_aux
```

For advanced auxiliary:

```yaml
benchmark:
  modes:
    - aux_cross_attention
    - aux_film
    - aux_text_fusion_variants
    - aux_gated_fusion
```

## 40. Tuning Recommendations

Start with:

```yaml
auxiliary_loss_weight: 0.05
```

Then test:

```yaml
auxiliary_loss_weight: 0.01
auxiliary_loss_weight: 0.03
auxiliary_loss_weight: 0.05
auxiliary_loss_weight: 0.10
auxiliary_loss_weight: 0.20
```

Expected behavior:

- too low: little effect on representation.
- moderate: may improve embedding structure and ambiguous classes.
- too high: can force visual features to chase text structure at the expense of pixel
  evidence.

For TF-IDF:

```yaml
ngram_min: 1
ngram_max: 2
```

is a good default because bigrams preserve phrases such as:

```text
tire paths
snow covered
wheel paths
road surface
fully covered
partly clear
```

Avoid aggressive SVD unless you have more text documents or a larger context corpus.

## 41. What To Watch For

Warning signs:

- auxiliary mode reaches near-perfect validation accuracy immediately.
- validation loss becomes unrealistically tiny.
- changing the auxiliary vector changes the prediction even though the image is fixed.
- GradCAM is weak or irrelevant while metrics improve.
- performance improves only on validation but not test.
- confusion matrix becomes too clean for visually ambiguous labels.

Positive signs:

- moderate performance gain over image-only.
- better macro F1 or balanced accuracy.
- improved recall for partial-snow classes.
- GradCAM remains road-focused or becomes more road-focused.
- embedding analysis shows cleaner but not trivial class structure.
- CCA/image-aux alignment improves without collapse.

## 42. How To Test For Leakage

Run these checks:

### 42.1 Aux Shuffle Test

At evaluation time, shuffle auxiliary vectors across samples.

Expected safe behavior:

```text
predictions should not materially change
```

Because logits are image-only.

If predictions change strongly, the auxiliary vector is influencing inference logits.

### 42.2 Zero Aux Test

Replace auxiliary vectors with zeros at evaluation time.

Expected safe behavior:

```text
predictions should be nearly identical
```

The current implementation should satisfy this because logits do not use aux.

### 42.3 Text-Only Probe

Train a small classifier on TF-IDF vectors alone.

Expected behavior:

```text
near-perfect class prediction
```

This is not a good model; it demonstrates why direct text-to-logit prediction would
be leakage.

### 42.4 GradCAM Sanity Check

Compare GradCAM overlays for:

```text
correct aux vector
zero aux vector
shuffled aux vector
```

Expected safe behavior:

```text
overlays should remain nearly unchanged
```

## 43. How This Should Adjust Performance

The auxiliary approach should improve performance only through better learned image
features. It should not make inference easier by giving the answer through text.

The expected mechanism is:

```text
class description -> semantic auxiliary embedding
semantic embedding -> alignment pressure during training
alignment pressure -> visual features cluster around class semantics
better visual features -> better image-only logits
```

Performance changes should be gradual and plausible:

- training loss may decrease slightly faster.
- validation macro F1 may improve.
- hard classes may improve more than easy classes.
- GradCAM may become more focused on road evidence.
- embedding projections may show cleaner class separation.

Performance should not jump to perfect accuracy unless the dataset is extremely easy
or there is leakage.

## 44. Class Description Quality Guidelines

Good class descriptions should:

- describe visual evidence, not just label names.
- mention what should and should not count.
- distinguish adjacent classes.
- avoid circular definitions such as "bare means bare".
- avoid overly long irrelevant context.
- use consistent vocabulary across classes.
- include tire-path logic for road-surface classes.
- avoid mentioning dataset-specific camera/source shortcuts.

Bad descriptions can hurt training. If a description emphasizes a cue that is not
visible in the image, the auxiliary loss pushes the visual embedding toward a target
that the image cannot justify.

## 45. Suggested Description Improvements

For this dataset, useful terms include:

```text
tire path
wheel path
exposed pavement
snow between tracks
snow-covered lane
immediate drivable path
lane center
partial clearing
asymmetric clearing
surrounding shoulder snow
```

Each class description should answer:

1. What exact road region matters?
2. What snow pattern defines the class?
3. What visual evidence should not override the class?
4. Which neighboring class is most easily confused with it?

## 46. Literature Context

This implementation combines older information-retrieval ideas with modern auxiliary
representation learning and visual explanation methods.

### 46.1 TF-IDF And Text Encoding

1. Salton and Buckley, "Term-weighting approaches in automatic text retrieval", 1988.

   Relevance:

   - Classical foundation for TF-IDF-style term weighting.
   - Explains why discriminative words receive more weight than common words.

2. scikit-learn `TfidfVectorizer` documentation.
   Link: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

   Relevance:

   - This project uses scikit-learn's `TfidfVectorizer`.
   - The documented parameters map directly to `max_features`, `ngram_range`,
     `min_df`, `stop_words`, and `dtype`.

### 46.2 Auxiliary And Privileged Information

1. Caruana, "Multitask Learning", Machine Learning, 1997.
   Link: https://doi.org/10.1023/A:1007379606734

   Relevance:

   - Auxiliary objectives can improve representation learning by sharing inductive
     bias across related tasks.
   - This project uses a related idea: an auxiliary semantic alignment objective shapes
     the visual representation.

2. Vapnik and Vashist, "A new learning paradigm: Learning using privileged
   information", Neural Networks, 2009.
   Link: https://doi.org/10.1016/j.neunet.2009.06.042

   Relevance:

   - Privileged information is available during training but not necessarily used at
     inference.
   - Class descriptions in this project are best treated as privileged training
     information because they are label-derived.

3. Lopez-Paz et al., "Unifying distillation and privileged information", ICLR 2016.
   Link: https://arxiv.org/abs/1511.03643

   Relevance:

   - Connects privileged information and representation transfer.
   - Supports the idea of using extra information during training without relying on it
     at inference.

### 46.3 Conditional And Multimodal Fusion

1. Vaswani et al., "Attention Is All You Need", 2017.
   Link: https://arxiv.org/abs/1706.03762

   Relevance:

   - Foundation for scaled dot-product attention and multi-head attention.
   - `aux_cross_attention` uses PyTorch multi-head attention with visual query and
     auxiliary key/value tokens.

2. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", 2017.
   Link: https://arxiv.org/abs/1709.07871

   Relevance:

   - Foundation for feature-wise linear modulation.
   - `aux_film` uses a bounded FiLM-like transformation for auxiliary diagnostics.

3. Hu et al., "Squeeze-and-Excitation Networks", 2017/2018.
   Link: https://arxiv.org/abs/1709.01507

   Relevance:

   - Establishes learned sigmoid feature recalibration.
   - `aux_gated_fusion` uses a learned gate to control auxiliary-conditioned feature
     adjustments.

### 46.4 Explainability

1. Zhou et al., "Learning Deep Features for Discriminative Localization", CVPR 2016.
   Link: https://arxiv.org/abs/1512.04150

   Relevance:

   - Introduces class activation mapping ideas for localization.
   - GradCAM builds on this family of methods.

2. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization", 2016/2017.
   Link: https://arxiv.org/abs/1610.02391

   Relevance:

   - The project uses GradCAM when possible.
   - Auxiliary modes should improve GradCAM only by improving image representations,
     not by letting text drive the output.

### 46.5 Road-Surface Classification

1. Pan et al., "Winter Road Surface Condition Recognition Using A Pretrained Deep
   Convolutional Network", 2018.
   Link: https://arxiv.org/abs/1812.06858

   Relevance:

   - Demonstrates pretrained CNN use for winter road-surface recognition.
   - Supports image-based road-condition classification as the core task.

2. Zhang et al., "Winter road surface condition classification using convolutional
   neural network (CNN): visible light and thermal image fusion", 2021.
   Link: https://doi.org/10.1139/cjce-2020-0613

   Relevance:

   - Shows that auxiliary or additional streams can help road-surface classification.
   - The project's text auxiliary stream is not thermal fusion, but both approaches
     use extra information to improve representation quality.

## 47. Relationship To DBLG

`DBLG.md` explains the dual-branch local-global image mode. `AUXILIARY.md` explains
the class-description auxiliary text mode.

The key difference:

```text
DBLG auxiliary source = another image view, bottom crop
AUX auxiliary source = class-description text vector
```

DBLG remains entirely visual. AUX uses text during training, but predictions remain
image-driven.

Both approaches are designed to improve visual representation quality without
requiring direct non-image input for the final classifier.

## 48. Current Implementation Verdict

The current auxiliary implementation is conservative and appropriate for
label-derived class descriptions.

Strong points:

- class descriptions are centralized and auditable.
- TF-IDF encoder is reproducible and saved.
- final logits are image-only.
- auxiliary loss is simple and interpretable.
- advanced modes support controlled cross-attention, FiLM, gating, and MoE diagnostics.
- GradCAM remains image-grounded.
- metadata supports auditing class-description assignments.

Main limitations:

- TF-IDF is shallow and does not understand synonymy deeply.
- fixed class descriptions create only five unique text vectors.
- auxiliary alignment can be too weak if `lambda_aux` is low.
- auxiliary alignment can over-regularize if `lambda_aux` is high.
- current `fusion.mode` names can be confusing because baseline logits do not directly
  fuse text and image.
- advanced auxiliary diagnostics are not all exported into prediction CSVs.

Best next improvements:

- add explicit leakage tests.
- save image-vs-aux cosine similarity by sample.
- add GradCAM comparison summaries for improved/worsened samples.
- log auxiliary alignment by class.
- run `image_only` versus `image_plus_tfidf_aux` on identical splits and seeds.
- tune `auxiliary_loss_weight`.
- consider a stronger text encoder only if leakage safeguards remain intact.

## 49. Source Links

Project files:

- `core/class_descriptions.py`
- `core/dataset_builder.py`
- `core/tfidf_encoder.py`
- `core/data_utils.py`
- `core/fusion_modules.py`
- `core/advanced_modes.py`
- `core/trainer.py`
- `core/evaluator.py`
- `core/explainability.py`
- `core/advanced_analysis.py`
- `core/mode_registry.py`
- `configs/default_config.yaml`
- `configs/sdre_config.yaml`

External references:

- scikit-learn `TfidfVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Multitask Learning: https://doi.org/10.1023/A:1007379606734
- Learning using privileged information: https://doi.org/10.1016/j.neunet.2009.06.042
- Unifying distillation and privileged information: https://arxiv.org/abs/1511.03643
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- FiLM: https://arxiv.org/abs/1709.07871
- Squeeze-and-Excitation Networks: https://arxiv.org/abs/1709.01507
- CAM: https://arxiv.org/abs/1512.04150
- Grad-CAM: https://arxiv.org/abs/1610.02391
- Winter RSC pretrained CNN: https://arxiv.org/abs/1812.06858
- Winter RSC visible/thermal fusion CNN: https://doi.org/10.1139/cjce-2020-0613
