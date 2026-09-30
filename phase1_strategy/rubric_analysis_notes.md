# Rubric metadata dimensions, outliers and VIF

The canonical metadata table is **5,824 x 33**. The raw audit table is
**17,568 x 25**, including **11,712 readable file paths**. Metadata columns
include Y, provenance and diagnostics, so P=33 is not the number of CNN input
features. Original audit columns are the first 25 canonical columns, in the
same order. `phase1_variable_inventory.csv` documents every column's meaning,
position, semantic assignment type and inferred storage dtype.

CSV itself is untyped text. Reported storage_dtype means pandas 3.0.6's inferred
in-memory dtype under the documented empty-field parsing rule. Pixel counts
can be float64 in CSV loading because missing values existed earlier; their
semantic type remains discrete. All-empty error columns infer float64 but
semantically remain optional text. JSON-array provenance fields are serialized
text, not predictor categories. Intensity extrema are discrete for this observed
8-bit L/RGB dataset, while means, standard deviations and ratios are continuous.

Canonical type counts: text/path/hash/identifier 11; nominal categorical 9;
discrete 7; continuous 3; boolean 3; ordinal categorical 0; datetime 0.
All columns are excluded from direct CNN predictors. `target_label` is Y;
decoded image pixels supply X. Filepath locates X but is not itself a predictor.

## Exact canonical column order

1. filepath
2. filename
3. original_split
4. original_folder
5. target_label
6. patient_id_if_parseable
7. patient_id_status
8. label_parse_status
9. label_parse_reason
10. width
11. height
12. aspect_ratio
13. image_mode
14. channel_count
15. file_size_bytes
16. mean_intensity
17. std_intensity
18. min_intensity
19. max_intensity
20. readable
21. sha256_hash
22. intensity_basis
23. read_error
24. hash_error
25. is_macos_sidecar
26. all_filepaths
27. all_original_splits
28. all_target_labels
29. all_filenames
30. source_file_count
31. alternative_filepaths
32. manifest_status
33. patient_grouping_validated

Native images have varying width/height/channel counts, so their flattened
image X has no common fixed P before standardization. Planned grayscale input
is 224*224*1 = **50,176 intensity values**. Planned replicated three-channel
input is 224*224*3 = **150,528 channel values**, not 150,528 independent features.
Convolutions preserve spatial/channel structure; these counts are not a proposal
to train a tabular model on flattened pixels.

## Canonical-only IQR bounds and shape

`outlier_bounds_and_skewness.csv` recomputes six global summaries using each
canonical hash exactly once. Quartiles use linear interpolation; skewness is
the adjusted Fisher-Pearson sample coefficient (`pandas.Series.skew`). Fences
are Q1-1.5*IQR and Q3+1.5*IQR; comparisons are strict. Negative lower fences
for inherently nonnegative fields are mathematical thresholds, not plausible
image sizes, and are not silently clamped.

IQR is the preferred **review** method for every field. Width has moderate
right skew (0.5842), height 0.9588, aspect ratio 1.2257 and file size 2.6401;
these shapes and class mixtures favor a robust method over mean/SD Z-scores.
Mean intensity (-0.0210) is nearly symmetric, and intensity SD (-0.2296) is
mildly skewed: Z-scores are more plausible sensitivity diagnostics for those
two, but symmetry alone does not establish a single Gaussian distribution.
The earlier class-wise analysis shows unequal class spreads, so IQR is retained
for consistent robust review, not selected by a p-value or skew threshold alone.

New canonical fences flag **1,070 unique contents** (1,259 metric flags):
73 width, 127 height, 98 ratio, 816 bytes, 105 mean and 40 SD. These are
**not the old flag population**. The prior 1,091 unique contents / 1,280 metric
flags used fences fitted on readable paths before all content deduplication.
The canonical file-size upper fence is now **511,717.375** instead of
505,943.125 bytes; aspect-ratio and intensity fences also change slightly.
All prior quality outputs are preserved as historical diagnostics. Neither
the old nor the new flags automatically removes an image.

Metadata IQR rules are REVIEW FLAGS. Raw X-ray pixels are not capped, clipped
or Winsorized. No row/image is removed solely for being statistically unusual.
This avoids discarding potentially meaningful resolution, exposure or anatomical
variation and avoids the strong class-dependent removal bias demonstrated in
the prior quality analysis. No Z-score or IQR rule establishes medical validity.

## VIF protocol

`metadata_vif.csv` calculates standard linear VIF on the six canonical numeric
diagnostics, **not on Spearman ranks**. Each variable is regressed on the other
five with an intercept solely to calculate R-squared, then
`VIF = 1/(1-R_squared)`. Columns are temporarily centered/scaled for numerical
conditioning; that does not alter VIF or change persisted data. NumPy least
squares avoids a SciPy runtime dependency. This auxiliary diagnostic is not
model training, feature fitting for the CNN, or a predictive classifier.
Results were independently checked against the diagonal of the inverse Pearson
correlation matrix. All auxiliary designs have rank six including the intercept.

| Variable | VIF | >10? |
|---|---:|---|
| width | 15.500345 | Yes |
| height | 30.425155 | Yes |
| aspect_ratio | 4.332771 | No |
| file_size_bytes | 3.305654 | No |
| mean_intensity | 1.026090 | No |
| std_intensity | 1.320826 | No |

1. Pairwise diagnostic screening uses |Spearman rho| >=0.85. Existing flagged
   pairs are width-height (0.9296137104), width-bytes (0.9127594176), and
   height-bytes (0.9383150032).
2. VIF >10 indicates severe linear multicollinearity **if these were traditional
   predictive-model variables**. Rank correlation and linear conditional VIF
   measure different relationships, so their thresholds need not agree exactly.
3. In that hypothetical use, iteratively remove the less theoretically relevant
   or more redundant variable, recompute all VIFs, and document the choices.
   Predictor selection would be performed within training data only.
4. Here these are metadata diagnostics, not CNN predictors. No column removal
   is needed for CNN feature selection, and **no image pixels or dimensions
   are deleted because of VIF**. The variables remain available for auditing.
