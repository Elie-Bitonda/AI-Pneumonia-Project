# Planned partitioning, nested model selection, and leakage controls

This document specifies Phase 2 behavior. No final fold memberships, CNN
training, augmentation, or fitted image preprocessing were produced in Phase 1.

## Primary outer evaluation

After manual adjudication and a frozen modeling manifest, use:

```python
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

The conservative relation groups are defined in
`phase1_strategy/patient_identity_assessment.md`. They prevent repeated
filename-token relations from crossing roles without claiming verified patient
identity. Five folds are feasible because the smallest class has 1,485 images
and 1,216 relation groups.

The original Kaggle train/val/test folders remain provenance only. Temporal
splitting is not applicable because trustworthy acquisition timestamps are not
available.

## Nested architecture and hyperparameter selection

The custom CNN, ResNet-18, and EfficientNet-B0 will not be compared on outer
folds and then ranked using those same evaluation scores. Within each outer
80% development set, architecture, preprocessing ablations, and
hyperparameters are selected using inner three-fold stratified group-aware
validation. The selected configuration is then refit on the outer development
data, with approximately 10% of that development portion reserved group-aware
for early stopping.

The final fit in each outer iteration is therefore approximately:

| Role | Approximate share of cohort |
|---|---:|
| Actual model training | 72% |
| Internal early-stopping validation | 8% |
| Untouched outer evaluation | 20% |

Exact integer counts vary because group constraints take priority over perfect
class ratios. The outer fold is scored once after every decision is locked.

Ordinary `StratifiedKFold(5, shuffle=True, random_state=42)` is retained only
as a secondary leakage-sensitivity analysis. It is not the headline estimate.

## Role-specific leakage controls

| Operation | Permitted data |
|---|---|
| Stochastic augmentation | active training subset only |
| Dataset-derived normalization | active training subset only |
| Class/sampling weights | active training labels/counts only |
| Metadata-only shortcut baseline | fit inside the same group-aware training split |
| Architecture/hyperparameter selection | inner group-aware validation only |
| Early stopping/checkpoint selection | internal group-aware validation only |
| Outer performance estimate | untouched outer evaluation only |

Fixed division by 255 and published ImageNet normalization constants do not
estimate information from this dataset. Any fitted statistic must be recomputed
inside the current training role.

## Planned shortcut checks

Phase 2 will include:

1. a metadata-only multinomial baseline using width, height, aspect ratio,
   file size, mean intensity, and intensity SD;
2. padding/crop/direct-resize sensitivity experiments;
3. a per-image contrast-normalization ablation after visual validation;
4. Grad-CAM and border/marker occlusion checks to detect dependence on
   padding, text, laterality markers, or peripheral non-lung regions; and
5. grouped-versus-ungrouped CV comparison as a leakage-sensitivity estimate.

## Evaluation metrics

Primary: macro-F1.

Also report: per-class recall/sensitivity, per-class precision/F1, balanced
accuracy, macro one-vs-rest AUROC, overall accuracy as a secondary metric, and
confusion matrices.

## Before creating folds

Adjudicate near-duplicate candidates and quality-review cases; record any
evidence-based exclusions; freeze the canonical membership by SHA-256; and
merge only manually confirmed related-image connections into the conservative
group graph. pHash similarity alone never establishes identity.


## Fixed Phase 2 search budget

To keep nested selection feasible, the search is staged rather than taking the
Cartesian product of every architecture and preprocessing option. The split
seed and model initialization seed are both 42, batch size is 32, each candidate
runs for at most 20 epochs, and early stopping patience is 3 epochs.

1. Using the custom CNN only, compare two geometry candidates: aspect-preserving
   padding and the predeclared crop/resize alternative.
2. On the selected geometry, compare contrast normalization off versus on using
   the custom CNN.
3. With geometry and contrast fixed, compare the custom CNN, ResNet-18, and
   EfficientNet-B0.

The fixed learning rates are 1e-3 for the custom CNN and 3e-4 for pretrained
backbones, with weight decay 1e-4. This caps selection at seven candidate
configurations per outer fold, each evaluated in two inner group-aware folds,
followed by one final fit: at most 75 fits across the five outer folds. If
compute must be reduced, the direct-resize diagnostic is removed first, then
the optional contrast ablation; the three required architectures and grouped
outer evaluation are not removed.

## Shortcut-gated preprocessing selection

Preprocessing is not selected by macro-F1 alone. The three-class label entropy
is 1.056 nats, while file size alone has MI 0.452 nats (42.8% of label entropy),
so a strong nuisance baseline is expected. A representation is eligible for
performance comparison only when the training-fold metadata/nuisance probe has
macro-F1 below 0.60, or, where the source metadata cannot itself be altered,
the image model exceeds that probe by at least 0.10 macro-F1 and border/marker
occlusion changes macro-F1 by no more than 0.03. If no candidate satisfies the
gate, Phase 2 reports shortcut-control failure rather than selecting the
highest-scoring representation.

## Adjudication rule and non-blocking default

I am the technical adjudicator for the 223 pHash candidate pairs. A pair is
confirmed as the same examination only when full-resolution review shows the
same projection/anatomy and matching distinctive local structures, with
differences attributable only to encoding, resize, or contrast. pHash alone
never confirms identity and never changes a class label. Cross-class pairs are
not relabeled without independent source or clinical evidence. If manual review
is unfinished when modeling must begin, both images are retained and the
candidate edge is converted into a split-only grouping constraint so that the
pair cannot cross training and evaluation roles.
