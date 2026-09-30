# Exact planned partitioning and leakage controls

This is a specification only. No indices, split memberships, fold files,
augmentation, training or fitted preprocessing have been created.

## Outer evaluation

After manual adjudication and a frozen modeling manifest, propose:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Each iteration uses approximately 80% outer development and 20% untouched outer
evaluation. Source Kaggle train/val/test assignments are metadata only. Patient
identity cannot currently be justified from filenames; this is class-stratified
image-content evaluation, not proven patient-independent evaluation.

## Internal validation within each development portion

Reserve **10% of the 80% development portion**, stratified by Y, for internal
validation. Remaining data train that iteration's model. Planned reproducible
inner implementation: one StratifiedShuffleSplit with `test_size=0.10` and
`random_state=42 + outer_fold_index`, using zero-based outer indices 0–4.
These are settings, not generated splits.

| Role | Share of development | Approximate share of complete cohort |
|---|---:|---:|
| Actual model training | 90% | 72% |
| Internal validation | 10% | 8% |
| Outer evaluation | Not part of development | 20% |

Integer allocation and class stratification cause small deviations. Purely
arithmetically, if provisional N=5,824 were retained, outer evaluation sizes
would be 1,164 or 1,165 and development 4,660 or 4,659. With the proposed
10% inner holdout and upward test-size rounding, internal validation would
have 466 images, leaving 4,194 or 4,193 for training. This is not a membership
assignment or a final-N promise. Recalculate after approved cohort decisions.

## Role-specific controls

| Operation | Permitted data |
|---|---|
| Stochastic augmentation | Actual ~72% training subset only |
| Dataset-derived normalization statistics | Actual training subset only; freeze for internal and outer evaluation |
| Class weights | Training counts only, N_train/(3*n_class_train) |
| Sampling weights | Training labels/counts only |
| Any learned transformation or feature fitting | Training subset only |
| Gradient updates | Training subset only |
| Early stopping/checkpoint selection | Internal validation only |
| Hyperparameter selection | Internal validation only, within development |
| Final score for each outer iteration | Untouched outer evaluation only, after all choices are locked |

Fixed /255 scaling and external pretrained ImageNet constants do not estimate
information from this dataset. Their exact use is documented in preprocessing
decisions. No training-fold-derived statistic may use the internal validation
or outer evaluation images. Fit normalization after the inner split, not on
the entire outer 80% development portion. Even unlabeled held-out pixels must
not enter fitting of dataset-derived statistics.

For this plan, evaluate the checkpoint chosen by internal validation; do not
silently refit on the full 80% development set. Such a refit would require an
explicit separate protocol. Do not use outer scores to choose hyperparameters,
epochs, augmentation ranges, preprocessing settings, or a winning model and
then present the same scores as untouched evaluation. Aggregate final fold
results after the predeclared process, with this selection caveat explicit.

## Before creating actual folds

Adjudicate near-duplicate candidates, especially cross-class/source pairs;
record approved quality inclusions/exclusions; resolve confirmed conflicting
labels; and freeze the cohort with hashes and provenance. Confirmed related
copies must not cross training/internal-validation/outer-evaluation roles.
Either retain a justified representative or revise the split architecture
explicitly if meaningful groups become available. Do not treat pHash matches
or unverified patient numbers as proven groups. If identities stay unavailable,
retain residual patient-leakage risk as a limitation. No original source split
is exempt from pooling or substituted for the planned five-fold evaluation.
