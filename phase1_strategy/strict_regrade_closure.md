# Strict re-grade closure — 2026-10-14

This note records the evidence added after the 88/100 strict re-grade. It is
not a replacement for the Phase 1 report; it makes the remaining decisions
reproducible in the repository.

## Group structure

The conservative cross-subtype scheme has 3,117 groups, 1,183 multi-image
groups and largest group size 31. Per-class counts overlap: 1,437 bacterial
groups + 1,216 viral groups - 979 shared cross-subtype tokens = 1,674 pneumonia
groups; 1,674 + 1,443 Normal groups = 3,117. A subtype-separated sensitivity
scheme has 4,096 groups, 722 multi-image groups and largest group size 30.
Cross-subtype merging is therefore a conservative leakage-control assumption,
not a patient-identity claim.

## Cluster-aware uncertainty

The image-level Kruskal-Wallis and chi-square tests are retained as descriptive
screens, but inferential emphasis is on effect sizes. A 1,000-replicate
group bootstrap over the 3,117 conservative groups (seed 42) gives:

| Diagnostic | Effect | 95% group-bootstrap CI |
|---|---:|---:|
| width | eta_H^2 = 0.3827 | [0.3567, 0.4066] |
| height | eta_H^2 = 0.4174 | [0.3923, 0.4415] |
| aspect ratio | eta_H^2 = 0.2770 | [0.2549, 0.3017] |
| file size | eta_H^2 = 0.5686 | [0.5454, 0.5918] |
| mean intensity | eta_H^2 = 0.00845 | [0.00394, 0.01498] |
| intensity SD | eta_H^2 = 0.1704 | [0.1495, 0.1904] |
| image mode | Cramer's V = 0.1596 | [0.1411, 0.1844] |

## Collinearity

Rank-VIF exposes the monotonic redundancy hidden by raw Pearson VIF:
width 57.45, height 114.41, aspect ratio 21.10, file size 9.70.
The deterministic metadata-only rule removes the derived aspect ratio first,
then repeatedly removes the largest rank-VIF feature. Height is next; the
remaining width/file-size/mean/SD rank-VIFs are 6.19/6.03/1.01/1.38.

## Outlier sensitivity

The assignment's |Z| > 3 rule flags 22 mean-intensity images (0.38%) and
21 intensity-SD images (0.36%). These are review flags only.

## Phase 2 mapping

The assignment-required Scikit-Learn Pipeline/ColumnTransformer and eight
tabular diagnostic figures belong to the metadata-only shortcut branch:
missingness, target distribution, raw-vs-capped boxplots, upper-triangle
correlation, collinear scatterplots, target importance, Raw→Log→MinMax→Standard
grid, and PCA scree/cumulative variance. The image CNN branch remains separate;
metadata are not CNN predictors.

## Reproducibility status

The repository is public. The README exposes a CI badge. The latest clean
GitHub Actions run passes all 19 unit tests plus the strict re-grade diagnostics.
