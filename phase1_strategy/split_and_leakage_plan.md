# Partitioning and leakage-prevention plan

This document defines the Phase 1 partitioning strategy and the controls needed
to prevent preprocessing and duplicate-related leakage. No final fold
memberships were generated during Phase 1.

## Primary partitioning strategy

After the modeling manifest is frozen, use:

```python
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

The conservative relation groups are defined in
`phase1_strategy/patient_identity_assessment.md`. They are leakage-control
units derived from filename relations and candidate review; they are not
presented as verified patient IDs.

Five outer folds are feasible because the smallest class contains 1,485 images
and at least 1,216 filename-derived relation groups before any additional
candidate-edge merging.

The original source `train/val/test` folders are retained as provenance only.
A temporal split is not applicable because trustworthy acquisition timestamps
are unavailable.

## Intended role proportions

Each outer fold leaves approximately 80% of the cohort for development and 20%
for outer evaluation. Within the development portion, approximately 10% is
reserved group-aware for internal validation, giving roughly:

| Role | Approximate share |
|---|---:|
| Actual training | 72% |
| Internal validation | 8% |
| Outer evaluation | 20% |

Exact integer counts can differ because preserving group integrity takes
priority over exact percentages.

## Leakage isolation

| Operation | Permitted data |
|---|---|
| Stochastic augmentation | active training subset only |
| Dataset-derived normalization | active training subset only |
| Class/sampling weights | active training labels/counts only |
| Learned preprocessing transforms | active training subset only |
| Preprocessing selection | internal validation only |
| Outer performance estimate | untouched outer evaluation only |

Fixed division by 255 and published pretrained normalization constants do not
estimate information from this dataset. Any statistic learned from the data is
recomputed inside the current training role.

The full-cohort Phase 1 audit is descriptive. It is used to design the strategy,
not to fit a transformation that is then reused across evaluation folds.

## Filename-derived relation groups

The current filename-only base structure contains 3,117 groups across 5,824
canonical contents. Per-class counts overlap because 979 `person<number>`
tokens occur under both pneumonia subtypes:

`1,437 bacterial groups + 1,216 viral groups - 979 overlaps = 1,674 pneumonia groups`.

Adding 1,443 Normal relation groups gives 3,117 groups overall.

Cross-subtype merging is a conservative leakage-control choice, not a claim
that the shared token proves one patient. A subtype-separated alternative is
retained as a sensitivity structure.

## Near-duplicate handling before folds

The pHash screen produced 223 candidate pairs. Similarity alone never confirms
identity or a duplicate examination.

Before partitioning:
- confirmed related-image edges are merged into the relation graph;
- rejected edges are omitted;
- unresolved edges remain as split-only constraints so their endpoints cannot
  cross training and evaluation roles.

Connected components are formed after these decisions and group feasibility is
checked again. Therefore 3,117 is the filename-only base group count, not a
guaranteed final count after candidate review.

## Exact duplicates

Exact SHA-256 duplicates are canonicalized before splitting, so the same byte
content cannot appear in more than one modeling role. Alternative source paths
are preserved in the manifest for provenance but are not treated as independent
observations.

## Final safeguards before partitioning

Before folds are created:
1. complete or record the status of near-duplicate review;
2. record any evidence-based image exclusions;
3. freeze canonical membership by SHA-256;
4. form the final leakage-control components;
5. verify that every retained relation component belongs to one and only one
   outer role;
6. recompute any training-derived preprocessing state independently within each
   fold.

This plan prevents exact-content leakage, reduces leakage from plausibly related
images, and keeps fitted preprocessing isolated from evaluation data while
preserving uncertainty about true patient identity.
