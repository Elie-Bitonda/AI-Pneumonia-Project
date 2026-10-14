# Phase 1 dataset audit findings

Audit date: 2026-09-28. Dataset root: the existing outer `chest_xray/` folder.
This document records the data audit and preprocessing evidence used to define
the Phase 1 strategy. No raw file was moved, renamed, rewritten, or deleted, and
no final model folds were generated during the audit.

## Inventory and integrity

| Measure | Observed value |
|---|---:|
| Discovered JPEG/JPG/PNG paths | 17,568 |
| Readable image paths | 11,712 |
| AppleDouble image-extension sidecars | 5,856 |
| Unique readable SHA-256 contents | 5,824 |
| Excess readable byte copies | 5,888 |
| Readable hash groups with conflicting labels | 0 |
| Readable hash groups spanning original source splits | 0 |

The two readable source trees each contain 5,856 paths and are byte-identical
path-for-path. A single readable tree still contains within-tree repetition:
5,794 readable hashes occur twice, 28 occur four times, and two occur six
times across the duplicated source structure. This yields 5,888 excess readable
copies overall. SHA-256 therefore defines exact byte duplication, but it does
not identify re-encoded near-duplicates or establish patient independence.

The filesystem contains 17,591 files in total. Of these, 17,568 have image
extensions. The remaining non-image files include macOS metadata and
`.DS_Store` entries. All 5,856 unreadable image-extension paths under
`__MACOSX` were verified as AppleDouble metadata, so they are treated as
filesystem metadata rather than corrupted radiographs.

The original source split counts are train 15,648, validation 48, and test
1,872 across discovered image-extension paths. These split labels are retained
only as provenance metadata and are not reused as the final evaluation split.

## Canonical unique-content cohort

Canonical selection keeps at most one readable, non-sidecar path per
consistent SHA-256 hash. The deterministic path rule chooses the eligible path
with the fewest relative path components, then case-sensitive lexical order.
No preference is given to train, validation, or test. Alternative paths remain
recorded in the manifest.

The resulting provisional canonical manifest contains **5,824 rows x 33
metadata columns**. It is provisional because near-duplicate review, image
quality review, and identity uncertainty remain unresolved.

| Class | Canonical count | Share |
|---|---:|---:|
| NORMAL | 1,579 | 27.1120% |
| BACTERIAL_PNEUMONIA | 2,760 | 47.3901% |
| VIRAL_PNEUMONIA | 1,485 | 25.4979% |
| **Total** | **5,824** | **100%** |

The largest/smallest class ratio is 1.8586, indicating moderate class
imbalance. No synthetic oversampling of raw pixels is planned.

## Metadata, dimensions, and channels

The canonical image dimensions vary substantially: width ranges from 384 to
2,916 pixels, height from 127 to 2,713 pixels, and aspect ratio from about
0.835 to 3.379. This variation requires an explicit geometric standardization
strategy rather than relying on a library default.

Canonical channel inspection identified 5,541 `L` images and 283 RGB images.
Every RGB image has identical values in all three channels. A common grayscale
intensity representation therefore preserves the verified image information.
A one-channel model can use the intensity directly; a pretrained three-channel
model can receive three replicated copies of the same intensity image.

## Missingness

The original 17,568 x 25 audit matrix contains 96,864 missing cells
(22.054645%). The 5,824 x 33 canonical matrix contains 13,228 missing cells
(6.882701%). Literal complete-row counts are zero because successful operations
intentionally leave `read_error` and `hash_error` empty.

Within the canonical manifest:
- `read_error`: 5,824 structural blanks;
- `hash_error`: 5,824 structural blanks;
- `patient_id_if_parseable`: 1,580 missing values;
- image measurements and target labels: complete.

Patient-ID availability is related to filename conventions and is therefore
treated as MAR-like with respect to observed metadata, but the true identity
missingness mechanism is not established. No patient identifier is imputed.
AppleDouble image measurements are structurally undefined because those files
are not radiographs. No conventional missing-pixel mechanism was found among
the readable canonical images.

## Outlier and quality review

Canonical IQR review fences were computed for width, height, aspect ratio,
file size, mean intensity, and intensity standard deviation. They flag 1,070
unique contents. These are review prompts rather than exclusion rules.

| Diagnostic | Skewness | Lower fence | Upper fence | Flagged |
|---|---:|---:|---:|---:|
| Width | 0.5842 | 300 | 2,316 | 73 |
| Height | 0.9588 | -60.5 | 1,935.5 | 127 |
| Aspect ratio | 1.2257 | 0.7760 | 2.0722 | 98 |
| File size | 2.6401 | -217,363.6 | 511,717.4 | 816 |
| Mean intensity | -0.0210 | 77.4500 | 168.4014 | 105 |
| Intensity SD | -0.2296 | 31.2714 | 83.1669 | 40 |

Flagging is strongly class-dependent: 837/1,579 Normal images, 147/2,760
bacterial images, and 86/1,485 viral images have at least one canonical IQR
flag. File size alone flags 813 Normal images but only three pneumonia images.
This demonstrates why statistical outliers cannot be treated as automatically
invalid images. No pixel Winsorization, clipping, or automatic image deletion
was performed.

For the two near-symmetric intensity summaries, a `|Z| > 3` sensitivity
screen flags 22 mean-intensity images and 21 intensity-SD images. These also
remain review indicators only.

## Metadata-class associations

The six numeric diagnostics are three discrete quantitative measures
(width, height, file size) and three continuous measures (aspect ratio, mean
intensity, intensity SD). Kruskal-Wallis tests were used as descriptive
class-association screens because the metadata show skewness, unequal spread,
and class-dependent distributions. Effect sizes are emphasized over p-values.

| Diagnostic | eta_H^2 | MI (nats) |
|---|---:|---:|
| File size | 0.5686 | 0.4517 |
| Height | 0.4174 | 0.2688 |
| Width | 0.3827 | 0.2380 |
| Aspect ratio | 0.2770 | 0.1584 |
| Intensity SD | 0.1704 | 0.1335 |
| Mean intensity | 0.0085 | 0.0248 |

The strong geometry/storage associations indicate possible acquisition or
export shortcuts. They are not interpreted as clinical relationships and none
of these metadata columns is used as a CNN predictor.

Spearman correlations exceeding `|rho| >= 0.85` are width-height 0.9296,
width-file size 0.9128, and height-file size 0.9383. Raw-scale VIF values are
15.50 for width, 30.43 for height, 4.33 for aspect ratio, 3.31 for file size,
1.03 for mean intensity, and 1.32 for intensity SD. A rank-based sensitivity
analysis confirms stronger monotonic redundancy. These analyses apply to
metadata diagnostics only; they do not imply removing image pixels.

## Near-duplicate screening

Exact SHA-256 deduplication was followed by a 64-bit perceptual-hash screen
using Hamming distance <=4. The screen identified **223 candidate pairs**
involving **262 unique images**:

- 175 same-class pairs;
- 48 cross-class pairs;
- 27 pairs crossing original source splits;
- 10 pairs that are both cross-class and cross-source;
- 16 pairs at Hamming distance 2 and 207 at distance 4.

These are similarity candidates, not confirmed duplicate examinations,
patients, or label errors. The review sheets are quality-control evidence only.
Any confirmed relation is handled as a leakage constraint; unresolved candidate
edges are kept from crossing training and evaluation roles until adjudicated.

## Identity evidence and leakage uncertainty

Filename parsing yields 1,674 pneumonia candidate identifiers, but no patient
identity key is independently verified. There are 170 candidate-token overlaps
across original source splits and 979 tokens appearing under both pneumonia
subtypes. Normal filenames use `IM-*` and `NORMAL2-IM-*` conventions whose
patient semantics are also unverified.

For leakage prevention, conservative filename-derived relation groups are used
without presenting them as verified patient identities. The filename-only base
structure contains 3,117 groups across 5,824 images. Cross-subtype merging is
treated as a conservative assumption and a subtype-separated sensitivity
structure is retained for comparison.

## Preprocessing decisions informed by the audit

| Issue | Evidence | Decision | Modeling implication |
|---|---|---|---|
| AppleDouble sidecars | 5,856 verified metadata files | Exclude from modeling discovery | Do not treat as radiographs |
| Exact duplicates | 5,888 excess readable copies | One canonical path per SHA-256 | Prevent exact-content leakage |
| Variable image geometry | wide size/aspect-ratio range | 224x224 aspect-preserving resize plus symmetric padding | Preserve proportions while monitoring padding cues |
| L/RGB storage | 283 RGB images have equal channels | Use common intensity representation | Replicate intensity for three-channel pretrained inputs |
| Missing identifiers | 1,580 canonical ID blanks | No imputation | Preserve uncertainty |
| Quality flags | strong class-dependent flagging | Review only; no automatic deletion | Avoid class-dependent cleaning bias |
| Class imbalance | 1,579 / 2,760 / 1,485 | Training-fold class weighting | No SMOTE on raw pixels |
| Metadata redundancy | high rho/VIF for geometry/storage | Diagnostic only | Metadata excluded from CNN predictors |
| Near duplicates | 223 pHash candidates | Review and constrain split leakage | Do not infer identity from pHash |
| Source split | validation has only 16 canonical images | Do not reuse source split | Use group-aware five-fold evaluation plan |

## Reproducibility

The audit is reproduced from the project root using the commands documented in
the README. The committed source code generates the inventory, integrity
verification, statistical summaries, channel checks, pHash candidates, and
preprocessing summary tables. Unit tests and the clean CI workflow verify the
core preprocessing utilities without modifying raw image bytes.

The canonical count of 5,824 is therefore a **provisional unique-image-content
count**, not a claim of 5,824 independent patients or a finalized modeling
cohort.
