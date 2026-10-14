# Phase 1 preprocessing decisions

These decisions are based on the provisional canonical audit of 5,824 unique
readable image contents. They define the planned preprocessing strategy; no
model training or final fold assignment was performed during Phase 1.

## 1. Image size

Use a **224x224** input canvas as the initial standardized shape. Native widths
range from 384 to 2,916 pixels and heights from 127 to 2,713 pixels, so a common
shape is required for batching. The audit does not establish that 224 preserves
all fine detail; it is an engineering choice to be validated later.

## 2. Aspect ratio and padding

Use aspect-preserving resize followed by symmetric padding. The scale factor is
`min(224/width, 224/height)`; resized dimensions are rounded to the nearest
integer, with a minimum of one pixel. Antialiased bilinear interpolation is
used, and any odd extra padding pixel is placed on the bottom or right.

Direct square stretching is not the primary choice because canonical aspect
ratios range from about 0.835 to 3.379. Cropping is also avoided as a default
because it can remove peripheral anatomy. Padding is applied before
normalization.

Aspect ratio is associated with class (`eta_H^2 = 0.277`), so padding can
preserve a class-correlated geometric cue. This is documented as a limitation;
padding is a geometry-preservation choice, not proof that shortcut information
has been removed.

## 3. Channel handling

Use a common grayscale intensity representation. Of the 5,824 canonical
contents, 5,541 are stored as one-channel `L` images and 283 as RGB. Every RGB
image has identical channels, so taking one channel preserves the verified
intensity information. A three-channel pretrained input can be formed by
replicating that intensity channel three times.

## 4. Pixel scaling and normalization

For a custom image-model baseline, convert pixel values to `float32` and
divide by 255 to obtain values in [0,1]. This operation is fixed and does not
estimate statistics from the dataset.

For pretrained ImageNet weights, replicate grayscale to three channels and
apply the normalization convention required by the selected weights. Any
dataset-derived normalization statistics must be estimated on the active
training subset only and then frozen for validation and evaluation.

## 5. Training-only augmentation

Candidate augmentation is intentionally conservative:
- rotation within +/-5 degrees;
- translation up to 2% of the canvas;
- modest intensity perturbation only after visual inspection.

Do not use vertical flips, 90/180-degree rotations, aggressive cropping,
aspect-ratio stretching, large brightness/contrast shifts, color jitter, or
unvalidated erasing/distortion. All stochastic augmentation is restricted to
training data and disabled for validation/evaluation.

## 6. Missingness and imputation

No pixel imputation is required for the readable canonical images. Structural
blank error fields remain blank. Unknown patient identifiers are not imputed or
fabricated. Metadata diagnostics are not supplied to the CNN as predictors.

## 7. Outlier handling

IQR fences are used to prioritize image review, not to delete images. The final
canonical fences flag 1,070 unique contents. Flagging is strongly
class-dependent, especially for file size, so automatic deletion or
class-specific exclusion thresholds would introduce additional bias.

No pixel Winsorization or clipping is applied because metadata outliers do not
justify altering radiographic pixel values. For mean intensity and intensity
SD, a `|Z| > 3` sensitivity check is reported separately; it also remains a
review indicator only.

## 8. Class imbalance

Canonical class counts are 1,579 Normal, 2,760 Bacterial Pneumonia, and 1,485
Viral Pneumonia. The largest/smallest ratio is 1.8586.

Use training-fold weighted multiclass cross-entropy with

`w_c = N_train / (3 * n_c_train)`.

Weights are recomputed from each active training subset only. SMOTE is not used
on raw pixels because synthetic interpolation between image vectors is not an
appropriate image-generation strategy for this task.

## 9. Split strategy and leakage prevention

Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` as the
primary outer partitioning strategy with the conservative relation groups
documented in `patient_identity_assessment.md`. These groups are leakage-control
units, not verified patient identities.

The intended role proportions are approximately 72% actual training, 8%
internal validation, and 20% outer evaluation. Exact counts can vary because
group integrity takes priority over perfect proportions.

Before folds are frozen, pHash candidate edges are handled consistently:
confirmed relations are merged, rejected edges are omitted, and unresolved
edges are retained only as split constraints. The original train/validation/test
folders remain provenance metadata only.

All fitted preprocessing state, class weights, and stochastic augmentation are
restricted to the active training subset. Internal validation can guide
preprocessing selection; outer evaluation remains untouched until the
configuration is locked.

## 10. Metadata collinearity

Metadata are diagnostic rather than CNN predictors. Raw-scale VIF is retained as
a conventional diagnostic, while rank-based VIF is used as a complementary
sensitivity analysis for monotonic redundancy.

For a separate metadata-only analysis, the deterministic removal rule is:
1. remove a structurally derived collinear feature before its source measures
   (aspect ratio first);
2. recompute rank-VIF;
3. while any rank-VIF exceeds 10, remove the feature with the largest value;
4. if values tie within 0.01, remove the feature with weaker target MI, then
   alphabetically if still tied.

This rule does not remove image pixels or spatial structure from the CNN input.

## 11. Readiness

The dataset is sufficiently characterized to support the Phase 1 preprocessing
strategy, but the canonical count remains provisional. Near-duplicate review,
quality review, identity uncertainty, and source-provenance limitations remain
documented before final modeling membership is frozen.
