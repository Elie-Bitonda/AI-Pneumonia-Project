# Proposed preprocessing decisions — not executed

These recommendations use the canonical 5,824-image audit. No modeling pixels,
augmentation, model, Grad-CAM or folds were created. Hash representations and
contact-sheet thumbnails are temporary audit views, not model-ready datasets.
The strategy is sufficiently specified for a Phase 1 report, with unresolved
review explicitly separated from final modeling approval.

## 1. Image size

Propose **224x224** as a practical initial input canvas. Native sizes vary from
384–2,916 pixels wide and 127–2,713 high, so batching requires a common shape.
The audit supports standardizing shape, but cannot establish that 224 is optimal
or preserves all relevant fine detail. Some thin images will require upsampling
and many large images will lose detail. Treat 224 as an engineering proposal;
any later resolution comparison must use training/inner-validation data only.

## 2. Aspect ratio

Prefer **aspect-preserving resize plus symmetric padding**, without cropping.
Use scale `min(224/width, 224/height)`, round dimensions to nearest integer
(minimum one pixel), apply antialiased bilinear resampling, then center on a
224x224 canvas. Put any odd extra padding pixel on the bottom/right. Initial
padding value: zero in the native 0–255 intensity domain, before normalization.
Bilinear resampling is a simple smoothing choice for downsampling; its impact
requires later validation. Avoid direct square stretching as the primary plan:
aspect ratios range approximately 0.835–3.379, and stretching would change
visible proportions differently by image. Do not crop anatomy to fill a square.

Important limitation: aspect ratio is associated with class (rank effect
0.277), so padding geometry can retain that class cue. Padding is a geometric
preservation choice, **not proof of shortcut removal**. Compare plausible
alternatives only inside later training/inner-validation work, and review
extreme framing manually. Do not choose transformations using held-out scores.

## 3. Channel handling

Use a common grayscale intensity representation. All 283 RGB images have
exactly equal channels; taking one RGB channel is lossless for these verified
contents. L images retain their stored intensity. For a pretrained three-channel
CNN, replicate that channel three times; for a one-channel baseline, retain
one channel. No disease-related color has been established. Grayscale text and
markers can remain and need review; channel standardization does not remove them.

## 4. Pixel scaling and normalization

For a simple CNN baseline, convert to float32 and divide the native 8-bit
intensities by 255, yielding [0,1]. This fixed conversion is not a fitted
dataset statistic. Do not fit a global scaler to all image pixels.

For a pretrained ImageNet backbone, follow the **specific selected weights'
documented input normalization**, normally after [0,1] scaling. Do not assume
every pretrained architecture has the same mean/SD or preprocessing contract.
The final backbone is not selected yet. For the now-named torchvision
ResNet-18 and EfficientNet-B0 `IMAGENET1K_V1` options, specify
`z_c = (x_c/255 - mean_c)/std_c`, with RGB-order mean
`[0.485, 0.456, 0.406]` and SD `[0.229, 0.224, 0.225]` after replication.
These are external weight conventions, not quantities inferred from this audit.
The channels cease to be numerically identical after channel-specific
normalization; this is expected and does not introduce acquired color.
See [ResNet-18 documentation](https://docs.pytorch.org/vision/0.13/models/generated/torchvision.models.resnet18.html)
and [EfficientNet-B0 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0).
Their supplied inference presets resize to 256 and center-crop to 224;
EfficientNet-B0 uses bicubic interpolation. The project's proposed bilinear
aspect-preserving 224 canvas deliberately replaces that geometry to avoid
cropping. Preserve the documented intensity normalization, and record this
geometry departure for later training-only validation rather than claiming
the whole pretrained inference preset is used unchanged.
Fixed external pretrained
constants are not estimated from this dataset. Any alternative dataset-derived
mean/SD must be estimated only on each fold's training portion, including an
explicit decision about padded pixels, and then frozen for its validation data.
Do not combine incompatible baseline and pretrained scaling conventions.

## 5. Planned training-only augmentation

The observed framing and intensity variation supports investigating **small**
perturbations, not claiming benefit before testing. Candidate starting ranges
are rotation within +/-5 degrees and translation up to 2% of canvas size,
with sufficient padding and human inspection to avoid cutting off content.
Consider only modest intensity perturbations after inspecting the reviewed
brightness/contrast examples; do not set aggressive ranges from global extrema.
Record seeds and probabilities if implemented later, and disable all stochastic
augmentation for validation/test. These are hypotheses for future evaluation,
not validated medical invariances or operations performed in Phase 1.

Do not use vertical flips, 90/180-degree rotations, horizontal flips that alter
laterality, aggressive random crops/zoom, aspect-ratio stretching, large
brightness/contrast changes, color jitter, or unvalidated erasing/distortion
that could obscure anatomy or create artificial patterns. No automatic
histogram equalization/CLAHE is justified by this audit.

## 6. Quality flags

Retain all currently flagged images pending documented human review. Pooled
IQR rules flag 54.27% of NORMAL but only 5.36% bacterial and 5.79% viral images,
mostly because NORMAL images are larger and occupy more bytes. A statistical
outlier is not an invalid medical image. Use flags to prioritize review and
record evidence-based decisions; do not Winsorize pixels, impute image values,
or discard images solely for exceeding metadata fences.

## 7. Class imbalance

The canonical counts are 1,579/2,760/1,485; largest/smallest ratio is about 1.859.
Propose weighted multiclass cross-entropy in Phase 2, with training-fold-only
weights `w_c = N_train/(3*n_c_train)`. This gives each class equal aggregate
training weight without synthesizing images. Recompute within each training
fold after its approved inclusion decisions. Compare with an unweighted baseline
using inner validation if needed; report macro-F1, balanced accuracy and class
recall/precision in later evaluation. Do not compute weights from validation
labels or apply SMOTE to raw pixels. No loss or weights were fitted here.

## 8. Split strategy

Preserve **StratifiedKFold(n_splits=5, shuffle=True, random_state=42)** as the
proposed strategy. Five folds meet the assignment, shuffling avoids directory
order, and seed 42 makes allocation reproducible. Current class counts support
five-way label stratification, but do not prove patient independence.
Patient-level grouping cannot currently be justified from the available filename
metadata. Patient-level leakage remains explicit. No final folds are created.

Before implementing folds, adjudicate 223 pHash candidates (48 cross-class,
27 cross-source). Confirmed duplicate copies require a documented representative
selection or a justified change to the split design, not arbitrary separation
across folds. Do not merge candidates or invent patient groups automatically.
If reliable patient identities become available, revisit grouped stratification.

## 9. Leakage control and readiness

Use only the training portion of each fold for normalization statistics,
augmentation, class weights and every learned transformation. Apply fixed
deterministic validation preprocessing using only that training-fitted state.
Hyperparameter decisions require inner validation, not held-out fold feedback.
The full-dataset Phase 1 statistics are descriptive diagnostics, not fitted
preprocessing parameters or evaluation results. Diagnostic metadata are not
CNN predictors. Source train/val/test folders remain metadata only.

The dataset is sufficiently characterized to **write the Phase 1 Strategy
Report**, including limitations and conditional plans. It is not yet cleared
as a final modeling cohort: near-duplicate adjudication, full-resolution quality
review, uncertain patient identity, label provenance, and shortcut-risk
mitigation remain unresolved. Preserve provisional N=5,824 and leave final N
unresolved. The requested PDF remains a placeholder for the later writing step.
