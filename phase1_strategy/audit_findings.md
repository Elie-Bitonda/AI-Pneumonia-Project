# Phase 1 filesystem audit findings

Audit date: 2026-09-28. Dataset root: the existing outer `chest_xray/` folder.
This is an audit snapshot, not the final dataset selection or a clinical
assessment. Reproduce using the commands in the project README. No raw files
were moved, renamed, rewritten or deleted. No folds or models were created.

## Inventory and integrity

| Measure | Observed value |
|---|---:|
| Discovered JPEG/JPG/PNG files | 17,568 |
| Readable files | 11,712 |
| Unreadable files | 5,856 |
| Unreadable files identified by macOS sidecar path/name | 5,856 |
| Unique readable SHA-256 hashes | 5,824 |
| Unique hashes across all files, including sidecars | 7,345 |
| Duplicate-hash groups across all files | 5,825 |
| Files belonging to those duplicate-hash groups | 16,048 |
| Redundant byte copies across all files | 10,223 |
| Duplicate filename groups | 5,856 |
| Label parsing failures | 0 |
| Hash groups containing conflicting parsed labels | 1 |
| Readable hash groups containing conflicting labels | 0 |
| Readable hash groups spanning source splits | 0 |

The single hash-label conflict is an unreadable macOS-sidecar group containing
4,336 paths. It is not evidence of contradictory labels for a decoded X-ray.
The two readable source trees account for 11,712 paths; simply retaining one
5,856-file tree would still retain exact duplicate images, because only 5,824
distinct readable hashes exist. These are diagnostic observations, not an
automatic choice of files to keep. SHA-256 does not detect re-encoded or visually
near-identical images.

All filenames yielded labels, including sidecars. A parsed label does not make
a file readable or its annotation clinically correct.

Original split counts, retained only as metadata: train **15,648**, validation
**48**, test **1,872**. The source split is not the project's final split.

## Three-class balance

Percentages in the first table use all 17,568 discovered files. This denominator
includes sidecars and copied trees and must not be reported as independent
training examples.

| Class | Discovered files | Percentage | Readable files | Ratio to smallest class |
|---|---:|---:|---:|---:|
| NORMAL | 4,749 | 27.0321% | 3,166 | 1.0603 |
| BACTERIAL_PNEUMONIA | 8,340 | 47.4727% | 5,560 | 1.8620 |
| VIRAL_PNEUMONIA | 4,479 | 25.4952% | 2,986 | 1.0000 |

Maximum/minimum ratio = 8,340 / 4,479 = **1.8620**.

Diagnostic sensitivity analysis using one readable path per hash:

| Class | Unique readable hashes | Percentage |
|---|---:|---:|
| NORMAL | 1,579 | 27.1120% |
| BACTERIAL_PNEUMONIA | 2,760 | 47.3901% |
| VIRAL_PNEUMONIA | 1,485 | 25.4979% |

Maximum/minimum ratio = 2,760 / 1,485 = **1.8586**. No records have been
removed from the inventory. This table is not a final inclusion manifest.

## Dimensions, modes and quality

Readable modes: **11,146 L / one-channel files**, **566 RGB / three-channel
files**. Mode and channels are missing for the 5,856 unreadable sidecars.

| Diagnostic | Minimum | Q1 | Median | Q3 | Maximum |
|---|---:|---:|---:|---:|---:|
| Width (pixels) | 384 | 1,056 | 1,281 | 1,560 | 2,916 |
| Height (pixels) | 127 | 688 | 888 | 1,187 | 2,713 |
| Width/height | 0.8354 | 1.2616 | 1.4159 | 1.5858 | 3.3788 |
| File size (bytes) | 5,441 | 56,065 | 95,839 | 236,016.25 | 2,414,342 |
| Mean intensity | 58.7230 | 111.6164 | 122.7533 | 134.3136 | 221.5366 |
| Population intensity SD | 20.4154 | 50.7369 | 57.3283 | 63.6965 | 87.4804 |

IQR-based review flags cover **2,188 readable file paths** with **2,574
metric-specific flag rows**. Copied images can appear twice. The flag CSV
contains paths, metric, value, lower/upper fences, and reason. These statistics
use readable files with copies retained; review clinical content and acquisition
differences before approving exclusions. No Winsorization or deletion occurred.

## Missing metadata

`data/processed/missing_values.csv` gives counts and percentages for **every**
column. Literal all-column completeness is **0 rows**, and **17,568 rows**
have at least one missing value, because successful operations intentionally
leave error-message columns empty. Excluding the two optional error-message
fields, **8,544 rows** are complete. Unparseable patient IDs and unreadable
images explain the other missing measurements; they are not imputed.

## Patient identity and leakage uncertainty

- **1,674 unique filename-derived candidate IDs** occur across **8,544 files**.
- All 1,674 candidates have multiple file paths, partly because of copied trees.
- **1,109 candidates** have multiple distinct readable byte hashes.
- **170 candidates** appear in more than one original source split.
- **979 candidates** occur under both pneumonia subtypes.
- **Zero patient identities are independently confirmed by this audit**.

The full `personNNN_(bacteria|virus)_NNN` pattern supports extracting a candidate
identifier, but it does not prove globally consistent patient identity. Large
cross-subtype overlap could indicate separate identifier namespaces, repeat
patients or another naming convention. Do not resolve it by appending the class
or source split to the ID: that would hide a possible leakage relation. Normal
image `IM-*` and `NORMAL2-IM-*` codes remain unparsed because their patient-level
semantics are unverified. The published combined OCT/chest-X-ray description
does not establish these chest-X-ray filename semantics.

Group-aware allocation is required if these repeated candidate IDs are confirmed
as shared patients, regardless of original split membership. At present, patient
grouping is unresolved. Exact duplicates must be resolved or isolated in a common
group regardless. The absence of readable cross-source byte duplicates does not
prove patient independence. Follow the README's proposed five-fold strategy
only after reviewing this evidence; no final folds have been generated.

## Manual decisions before Phase 2

1. Verify archive provenance/version and licensing; identify packaging sidecars
   for explicit exclusion in a future selection manifest.
2. Adjudicate duplicate groups, retain their provenance, inspect re-encoded and
   near-duplicate images, and document a canonical-file rule without deleting raw.
3. Verify the global meaning of `personNNN`, investigate cross-subtype/source
   overlaps, and obtain or validate normal-image patient information. Do not
   report candidate counts as established patient counts.
4. Review quality-flagged images, mode differences, unusual aspect ratios and
   intensity values. Document decisions; flags alone are not exclusions.
5. Spot-check subtype labels against source documentation and review any clinical
   label uncertainty. Confirm class balance and at least five independent groups
   per class on the approved manifest before applying grouped five-fold CV.
6. Choose and justify image shape, channel handling, interpolation, padding and
   intensity scaling for the later architecture. Fit data-derived operations
   only inside each training fold; never fit across the pooled dataset.

Validation completed: three synthetic tests cover conservative parsing, corrupt
reads, duplicate identification, raw-byte preservation, output-directory safety,
and zero-IQR flags without mutation. The full output was checked against a fresh
filesystem count; paths are unique, every discovered file has a hash, and readable
rows have dimensions and intensity measurements. The plot was visually inspected.

## Dataset Integrity Verification

Follow-up verification (2026-09-28) recomputed all 17,568 inventoried SHA-256
hashes from local raw files and confirmed unchanged membership, bytes and label
parsing. This section supersedes the earlier tentative patient-grouping plan.
No files were deleted, moved or renamed; no folds or models were created.

| Measure | Verified result and scope |
|---|---|
| RAW FILE COUNT | 17,591 total filesystem files; 17,568 have image extensions |
| READABLE FILE COUNT | 11,712 decoded image paths |
| FILESYSTEM SIDECARS | 5,856 image-extension files have verified AppleDouble headers; 15 additional non-image-extension metadata paths exist |
| OTHER FILESYSTEM METADATA | Eight `.DS_Store` files outside the sidecar tree |
| DUPLICATE FILE COUNT | 5,888 excess readable copies; 10,223 excess files when image-extension sidecars are included |
| UNIQUE IMAGE CONTENT COUNT | 5,824 readable SHA-256 hashes; byte uniqueness is not patient independence |
| PROVISIONAL MODELING N | 5,824 consistent-label, readable, non-sidecar contents |
| FINAL MODELING N | Unresolved pending manual quality, near-duplicate and label review |

### Directory-level proof

`directory_inventory.csv` covers every directory recursively, with total file
counts, image-extension counts, readable counts, sidecars, class counts and
hash counts. Rows overlap and must not be added. Class/hash counts apply to
the image-extension inventory; overall file/sidecar counts cover all types.

| Source tree (relative to outer dataset root) | Readable image paths | Unique readable hashes |
|---|---:|---:|
| `train/` + `val/` + `test/` | 5,856 | 5,824 |
| `chest_xray/train/` + `chest_xray/val/` + `chest_xray/test/` | 5,856 | 5,824 |
| `__MACOSX/` | 0 | 0 |

`directory_copy_comparison.csv` contains **5,856 corresponding suffix pairs**,
both full relative paths and both freshly verified hashes. Every pair has the
same hash and label, with no unmatched suffixes. For example,
`test/NORMAL/IM-0001-0001.jpeg` matches
`chest_xray/test/NORMAL/IM-0001-0001.jpeg`; the CSV records the digest itself.
Thus two copies are demonstrated by path-level byte evidence, not inferred
from counts. The archive already contains these trees; why its distributor
packaged them twice is not documented locally.

### Exact duplicate groups and canonical selection

`duplicate_group_audit.csv` contains **5,825 repeated-hash groups** and every
member path. Among readable images there are **5,824 groups**: 5,794 have two
paths, 28 have four, and two have six. This gives **5,888 excess readable
copies**: 5,856 from the second tree plus 32 repeated contents within one
tree. Within either tree, 30 content groups repeat (28 twice and two three
times). Retaining an arbitrary single tree would therefore still retain 32
excess copies.

The only additional repeated hash is a group of **4,336 unreadable sidecars**
(4,335 excess metadata copies). It spans source splits and all three parsed
classes, producing the sole B/C group. It is excluded with
`CONFLICT_REQUIRES_REVIEW`, and its verified classification is filesystem
metadata. It does not represent an identical X-ray with contradictory labels.

- A, exact duplicate with consistent label: **5,824 groups**.
- B, exact bytes spanning source splits: **one metadata-only group**;
  **zero readable-image groups**.
- C, exact bytes with conflicting class labels: **one metadata-only group**;
  **zero readable-image groups**.
- D, duplicate caused by corresponding paths in different source trees:
  **5,824 groups**.

Categories overlap: all readable groups are A/D. Do not sum category counts.
Across all image-extension files there are **10,223 excess byte copies**.

The canonical selection rule chooses the eligible readable non-sidecar path
with the **fewest relative path components**, then **case-sensitive lexical
order**. Selection does not favor train, val or test. Every alternative source
path remains in JSON-array columns, and there is at most one row per SHA-256.
Conflicting or missing group labels block retention and are recorded in
`canonical_manifest_exclusions.csv`. No quality flag automatically excludes
an image. The selected paths currently belong to the outer train/val/test
tree; those original split names still have no role as final folds.

| Class after content deduplication | Provisional count |
|---|---:|
| NORMAL | 1,579 |
| BACTERIAL_PNEUMONIA | 2,760 |
| VIRAL_PNEUMONIA | 1,485 |
| Total | 5,824 |

### Sidecar evidence

All 5,856 unreadable image-extension paths match the `__MACOSX/.../._<name>`
pattern and have magic `00051607`, version `00020000`, a positive AppleDouble
entry count and entry descriptors bounded by file length. Full evidence is
in `sidecar_verification.csv`. Representative paths include:

- `__MACOSX/chest_xray/test/NORMAL/._IM-0001-0001.jpeg`
- `__MACOSX/chest_xray/test/NORMAL/._IM-0003-0001.jpeg`
- `__MACOSX/chest_xray/test/NORMAL/._IM-0005-0001.jpeg`

Classify them as **FILESYSTEM_METADATA**, not corrupted X-rays. Future modeling
discovery rejects a `__MACOSX` path component or `._` filename prefix even if
decoding succeeds. The raw audit retains them and the raw files stay untouched.

### Patient identity and label provenance

Patient-level grouping cannot currently be justified from the available filename metadata.

Local dataset and archive inspection found no source documentation or patient
identity key. The 1,674 candidates, 170 cross-split candidates and 979
cross-subtype candidates are lexical overlaps, not proven shared patients.
The current plan is therefore
**`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`**, with residual
patient-level leakage stated as a dataset limitation. No folds were created.
See `patient_identity_assessment.md` and `patient_id_semantics_audit.csv`.
`patient_filename_examples.csv` provides 20 representative filenames per class.

`label_provenance.csv` documents the NORMAL folder rule and the PNEUMONIA
folder plus whole `bacteria`/`virus` filename-token rule, with actual examples
and canonical counts. All 2,760 bacterial names match the standard person
pattern. Of 1,485 viral names, 1,484 match it; the remaining
`person447_virus_921_1.jpeg` has an extra suffix. Its viral token is unambiguous,
but its patient candidate is deliberately missing. Normal names comprise
666 `IM-*` and 913 `NORMAL2-IM-*` files; neither convention has established
patient semantics. Folder/filename annotation is not clinical verification.

### Quality-review counts without copied-content inflation

The original IQR fences are retained to explain the first audit, and repeated
`(sha256_hash, metric)` flags are counted once. This reduces **2,188 flagged
paths / 2,574 path-metric flags** to **1,091 unique contents / 1,280 distinct
content-metric flags**. There are **951 images with exactly one flag** and
**140 with multiple flags**. No flagged image was deleted or excluded.

| Reason | Unique flagged contents |
|---|---:|
| Width | 73 |
| Height | 127 |
| Aspect ratio | 98 |
| File size | 835 |
| Mean intensity | 107 |
| Intensity standard deviation | 40 |
| Other | 0 |

| Class | Unique flagged contents | Distinct content-metric flags |
|---|---:|---:|
| NORMAL | 857 | 1,022 |
| BACTERIAL_PNEUMONIA | 148 | 165 |
| VIRAL_PNEUMONIA | 86 | 93 |

`quality_top30_unique_images.csv` lists the 30 most heavily flagged distinct
contents, ordered by flag count descending and then hash ascending: one has
four flags and 29 have three. Ties do not imply different clinical severity.
`quality_flags_unique_content.csv` retains metrics, values and fences.

### Decision Log

| Issue | Evidence | Decision | Reason | Phase 2 consequence |
|---|---|---|---|---|
| Two readable directory trees | All 5,856 corresponding suffixes match hashes and labels | Keep one provisional path per content; preserve alternatives | Directory copies are not new observations | Use canonical manifest after review |
| Additional within-tree repetition | 30 groups produce 32 excess copies per tree | Deduplicate by SHA-256, not filename/tree | A single tree still repeats images | Avoid exact-content leakage and inflated sample counts |
| Unreadable sidecars | All 5,856 have valid AppleDouble headers | Classify FILESYSTEM_METADATA; exclude from modeling discovery | These are packaging metadata | Do not report them as corrupted radiographs |
| One cross-label duplicate group | 4,336 identical metadata sidecars; no readable members | Record CONFLICT_REQUIRES_REVIEW and exclude | Never select a label silently; metadata is ineligible anyway | No readable-image label conflicts currently block manifest |
| Unproven patient numbers | Local documentation absent; 979 subtype and 170 split overlaps | Recommend StratifiedKFold for current plan | Shared numeric token does not prove shared identity | Explicit residual patient-leakage limitation |
| Extra viral filename suffix | `person447_virus_921_1.jpeg` | Keep token-derived label; leave patient candidate missing | Suffix meaning is undocumented | Manual provenance review; no invented group |
| Quality flags inflated by copies | 2,188 paths map to 1,091 flagged contents | Review unique contents, retain original thresholds | Flags are diagnostic, not automatic exclusions | Manual quality decisions before final N |
| Final sample count | Provisional 5,824; clinical and near-duplicate review incomplete | Leave final N unresolved | Byte uniqueness does not establish suitability or independence | Freeze inclusion manifest only after review |

Validation: eight automated tests pass, covering sidecar exclusion even when
readable, complete duplicate grouping, deterministic canonical priority,
one-row-per-hash selection, exclusion of conflicting labels, deterministic
label parsing, raw-byte preservation, quality flags and path/hash comparison.

## Statistical Bias and Preprocessing Verification

All results in this section use the **5,824-row canonical manifest** as the
unit of analysis. The full raw inventory's 17,568 SHA-256 hashes were rechecked
before and after analysis; every file remained byte-identical. Fifteen tests
cover prior safeguards plus BK-tree retrieval, channel arithmetic, pHash
determinism, multiple-testing correction, MI, unique quality counts, manifest
uniqueness and thumbnail source-byte preservation.

### Metadata-class associations and correlations

Class-wise summaries, shape diagnostics, normality checks, histograms and Q-Q
plots support conservative Kruskal-Wallis comparisons rather than selecting
tests only from normality p-values. Rank effect eta-squared H is **0.5686 for
file size**, **0.4174 height**, **0.3827 width**, **0.2770 aspect ratio**,
**0.1704 intensity SD**, and **0.0085 mean intensity**. All remain statistically
significant after Holm adjustment, but patient independence is unverified and
these are exploratory associations, not causal acquisition effects. Small
mean-intensity effect despite a very small p-value illustrates the distinction.

Mode and channel-count contingencies have adequate expected counts and
Cramer's V=0.1596. They describe the same L/RGB partition and are redundant.
The binned MI ordering is led by file size (0.4517 nats), height (0.2688),
width (0.2380), and aspect ratio (0.1584); MI is diagnostic, not CNN feature
selection. See `confounder_assessment.md` for assumptions, effect formulas,
permutation bias and p-value-underflow handling.

Three Spearman pairs exceed |rho| >=0.85: width/height **0.9296**,
width/file size **0.9128**, and height/file size **0.9383**. The upper-triangle
heatmap is `figures/fig3_1_correlation_matrix.png`. No correlation is interpreted
as a reason to remove pixels or manually select CNN predictors.

### Channels, candidate pairs and quality-flag bias

There are **5,541 L** and **283 RGB** contents. All RGB channels are exactly
identical pixel-for-pixel (143 bacterial, 140 viral, zero normal). No RGB image
contains chromatic differences, although grayscale annotations/markers remain
possible and visible in sampled review thumbnails. The proposed three-channel
input can replicate a single verified intensity channel.

The deterministic 64-bit pHash/BK-tree screen at radius <=4 finds **223
candidate pairs** involving **262 unique images**: **48 cross class**,
**27 cross original source split**, **10 cross both**. Sixteen pairs have
distance 2; 207 distance 4; none distance 0. These are unconfirmed candidates,
not 223 established duplicates or 48 established label conflicts. Representative
contact sheets show visibly differing images can have similar global layout.
Source split differences remain provenance. No pairs were merged or removed.

Unique quality-flag rates are **857/1,579 =54.27% NORMAL**,
**148/2,760 =5.36% bacterial**, and **86/1,485 =5.79% viral**. The pooled
file-size threshold lies below the normal-class median and flags 832 normal
contents, versus only three pneumonia contents. IQR flags are **STATISTICAL
OUTLIERS**, not proof of **INVALID MEDICAL IMAGES**. No exclusion occurs.

### Decision Log — final Phase 1 additions

| Issue | Evidence | Decision | Reason | Phase 2 consequence |
|---|---|---|---|---|
| Resolution/storage shortcuts | File size and dimensions have the largest metadata-class effects | Document acquisition-bias risk; standardize input shape | Association does not prove causal origin or eliminate pixel traces | Inspect residual shortcuts; do not add metadata as predictors |
| Skew and unequal dispersion | File-size tails, Q-Q departures and differing class spreads | Use rank comparisons with effect sizes and Holm adjustment | Avoid p-only normality gating and unsupported common-variance ANOVA | Treat p-values as exploratory under unknown independence |
| Correlated diagnostics | Three Spearman pairs >=0.85 | Retain as diagnostic metadata | They are not CNN features | No pixel removal or feature pruning |
| RGB encoding | All 283 RGB images have equal channels | Use intensity channel; replicate for three-channel pretrained input | No chromatic content to preserve | Record exact model normalization contract later |
| Aspect distortion versus padding cues | Aspect ratios 0.835–3.379 and class-related ratios | Propose preserved-ratio resize/padding at 224x224 | Preserve proportions while acknowledging padding shortcut risk | Validate resolution/padding choices inside training work |
| Near-duplicate candidates | 223 pairs, 48 cross-class, 27 cross-source | Human review; no automatic merge/exclusion | pHash similarity does not prove same image or patient | Adjudicate before final split design is implemented |
| Biased IQR review flags | 54.27% normal flag rate, mainly bytes/resolution | Retain flagged images for review | Statistical outlier is not invalid medical image | No blanket class-disproportionate exclusion |
| Class imbalance | 1,579/2,760/1,485 | Propose training-fold weighted cross-entropy | Address counts without synthesizing pixels | Compute weights within each training fold only |
| Patient uncertainty | No verified global identity key | Preserve proposed five-fold StratifiedKFold, seed 42 | Group identities cannot be justified | Explicit unresolved patient-leakage limitation |
| Report readiness versus cohort approval | Reproducible diagnostics complete; human review incomplete | Ready to write Phase 1 strategy; final N unresolved | Planning can honestly report pending decisions | Provisional N stays 5,824; no model or final folds yet |

`preprocessing_decisions.md` specifies the proposed resize, interpolation,
padding, channel, scaling, augmentation limits, imbalance and leakage controls.
All dataset-derived normalization statistics, augmentation, class weights and
learned transformations must use only a fold's training portion. The current
audit fits no CNN preprocessing. Contact sheets are visual review aids, not
clinical judgments; representative visual QA does not replace full human
adjudication. The strategy PDF has not yet been written.

### Completion verification on 2026-09-30

See `phase1c_completion_verification.md` for the complete deliverable inventory,
cross-class pair breakdown, contact-sheet observations, consolidated 15-issue
decision log and final readiness distinctions. Existing valid statistical CSVs
and figures were preserved. All 17,568 raw hashes were verified again unchanged.
`preprocessing_decisions.md` now specifies the external normalization constants
for the requested torchvision ResNet-18/EfficientNet-B0 V1 weight options and
explicitly distinguishes the proposed padding geometry from their crop presets.
The current suite rerun passed eight tests but could not import seven Phase 1C
tests because Windows Application Control blocked SciPy DLLs, including on an
outside-sandbox retry. This is one loader error, no assertion failures and no
skips; the earlier 15-test pass is historical, not a fresh result. The completion
note documents this environment limitation and the independent successful
raw-hash/output checks.
# Final rubric gap closure — 2026-09-30

Additional manifest-only evidence is in `rubric_analysis_notes.md`,
`missingness_assessment.md`, `encoding_strategy.md`,
`split_and_leakage_plan.md`, and `rubric_completion_check.md`.

| Finding | Decision | Evidence / limitation |
|---|---|---|
| Canonical metadata is 5,824 x 33 | Distinguish metadata P from image tensor values | Ordered inventory in `data/processed/phase1_variable_inventory.csv` |
| Canonical 13,228 blank cells include optional errors and 1,580 absent candidate IDs | No imputation; no invented patient identity | All canonical image diagnostic measurements are present |
| Newly fitted canonical IQR fences flag 1,070 contents | Keep as review indicators; preserve historical 1,091-flag population | Different threshold-fitting populations; no exclusions |
| Width VIF 15.500345 and height VIF 30.425155 exceed 10 | Keep diagnostic variables; do not remove pixels/dimensions | VIF protocol is hypothetical for traditional predictors |
| Outer 80/20 with inner 10% of development | Approximately 72/8/20 train/validation/evaluation | Training-only fitting means actual 72% subset; no memberships created |
| Four new utility tests passed | Record separately from earlier suite | Current full-suite SciPy DLL policy block remains unresolved |
| Local folder is not a Git repository; PDF remains placeholder | Numerical evidence is ready for report writing | Git/GitHub delivery and final report remain outstanding |
