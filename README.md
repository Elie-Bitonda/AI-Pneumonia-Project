[![Phase 1 tests](https://github.com/Elie-Bitonda/AI-Pneumonia-Project/actions/workflows/phase1-tests.yml/badge.svg)](https://github.com/Elie-Bitonda/AI-Pneumonia-Project/actions/workflows/phase1-tests.yml)

# Explainable Deep Learning for Multiclass Pediatric Pneumonia Classification from Chest X-Ray Images

This research project investigates **Multi-Class Classification** of pediatric
chest radiographs into `NORMAL`, `BACTERIAL_PNEUMONIA`, and `VIRAL_PNEUMONIA`.
The domain problem is distinguishing radiographic patterns associated with the
dataset's normal and pneumonia subtype labels. Dataset labels are research
annotations, not a substitute for microbiological confirmation or clinical
assessment. This is research software, **not a clinical diagnostic tool**.

Phase 1 inventories source files and proposes preprocessing and leakage
controls. It performs no training, augmentation, SMOTE, pixel Winsorization,
Grad-CAM, fitted feature transformation, or final cross-validation assignment.

## Dataset and source organization

Source: Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases
by Image-Based Deep Learning*, Cell (2018),
[original publication](https://doi.org/10.1016/j.cell.2018.02.010).
The distributed dataset is available through
[Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3) and
[Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
The combined Mendeley record also describes OCT data; its four-class OCT
filename description must not be assumed to establish chest-X-ray patient IDs.
Confirm the provenance/version and applicable license of the local archive.

The existing local source is `chest_xray/`, containing `train/`, `val/`, `test/`,
a second nested `chest_xray/` tree, and `__MACOSX/`. Each source split has
`NORMAL/` and `PNEUMONIA/` folders. The original archive is `archive.zip`.
Raw files remain at their original locations; `data/raw/` is the conventional
location for future clean checkouts. **Pass the outermost extracted dataset
root** to include every source tree. The utility scans `.jpeg`, `.jpg`, and
`.png` case-insensitively, including sidecars with those extensions. It does
not extract archives or inspect other file extensions. All source splits are
pooled; `original_split` is metadata, never the final project split.

The completed local audit found **17,568 image-extension files**, of which
**11,712 are readable** and **5,856 are unreadable macOS sidecars**. There are
**5,824 unique readable SHA-256 hashes**; these are distinct byte contents,
not confirmed independent patients or clinically unique images.
Empirical results for this checkout are recorded in
[Phase 1 audit findings](phase1_strategy/audit_findings.md), with authoritative
counts in `data/processed/audit_summary.json`. No expected dataset size is
hardcoded. Rerun after changing the source and refresh the findings note.

## Repository structure

The canonical metadata matrix is **5,824 rows x 33 columns**; the original
filesystem audit is **17,568 x 25**, including 11,712 readable paths.
`target_label` is the nominal target **Y**. Decoded image pixels are **X**;
none of the metadata columns is a CNN predictor. Native images have varying
dimensions, so image X has no common fixed P before standardization. Planned
224 x 224 grayscale input contains **50,176 intensity values**; three-channel
input contains **150,528 channel values**. Replicated grayscale channels are
not independent features. The exact ordered metadata schema and semantic
types are in `data/processed/phase1_variable_inventory.csv`.

```text
project_root/
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── phase1_strategy/
│   ├── Phase1_Strategy_Report.pdf
│   └── audit_findings.md
├── phase2_implementation/
│   ├── notebooks/
│   │   └── pipeline_execution.ipynb
│   └── Phase2_Final_Report.pdf
├── src/
│   ├── __init__.py
│   ├── transformers.py
│   ├── pipeline_builder.py
│   ├── integrity_verification.py
│   ├── statistical_audit.py
│   ├── image_review.py
│   ├── final_phase1_audit.py
│   └── preprocessing_summary.py
├── figures/
└── tests/
    ├── test_audit.py
    ├── test_integrity_verification.py
    ├── test_final_phase1.py
    └── test_preprocessing_summary.py
```

The Phase 1 report is being finalized for publication at the required path
`phase1_strategy/Phase1_Strategy_Report.pdf`; the Phase 2 report and notebook
remain future deliverables. The source folder and archive are local inputs,
ignored by Git along with raw images, virtual environments, and caches.

Repository status updated 2026-10-14: the project is version-controlled and
published at https://github.com/Elie-Bitonda/AI-Pneumonia-Project. The required
`.gitignore`, `requirements.txt`, README, source, tests, processed audit
artifacts, figures, and strategy directory are tracked. Raw X-rays, the archive,
virtual environments, and caches remain excluded. A GitHub Actions workflow
installs the pinned Phase 1 environment in a clean Linux runner and executes the
complete unit-test suite; the current run passes all 19 tests.

## Phase 1 report

The final submission PDF must be available at
[`phase1_strategy/Phase1_Strategy_Report.pdf`](phase1_strategy/Phase1_Strategy_Report.pdf).
This README link is the canonical navigation point for the Phase 1 report.

## Environment and exact reproduction commands

Tested with Python 3.14.7 on Windows. `requirements.txt` pins the installed
audit dependencies and their transitive versions. Pillow decodes images;
NumPy calculates diagnostics; pandas produces inventory tables; Matplotlib
produces the plot. No training framework or scikit-learn installation is
needed for Phase 1.

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.mplconfig'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.pipeline_builder --dataset-root chest_xray
```

For a newly downloaded dataset placed under `data/raw/`, replace only the
last command with:

```powershell
.\.venv\Scripts\python.exe -m src.pipeline_builder --dataset-root data/raw
```

On POSIX systems use `.venv/bin/python` for the interpreter and
`export MPLCONFIGDIR="$PWD/.mplconfig"`. Outputs default to `data/processed/`
and `figures/`; CLI flags `--output-dir` and `--figures-dir` can override them.
Output directories must be disjoint from the raw root. Reruns replace audit
outputs, never source images. Filepaths in CSVs are relative to the supplied
dataset root; combine that root and `filepath` to locate an image.

## Explicit parsing and measurement choices

`src/transformers.py:parse_identity` contains the complete label parser:

1. Collect ancestor folder names equal to `NORMAL` or `PNEUMONIA`, ignoring
   case. Require exactly one class-folder identity.
2. Split the filename stem with `re.split(r"[^a-z0-9]+", stem.lower())`.
3. `NORMAL` with no `bacteria` or `virus` tokens maps to `NORMAL`.
4. `PNEUMONIA` with exactly one of the whole tokens `bacteria` or `virus`
   maps to `BACTERIAL_PNEUMONIA` or `VIRAL_PNEUMONIA`, respectively.
5. Missing/ambiguous tokens or conflicting folders produce a blank label and
   `label_parse_status=FAILED`. Readability and label status are independent.

Patient candidates require the full stem regex
`person([0-9]+)_(bacteria|virus)_([0-9]+)`; the integer part becomes
`personNNN`, without adding the subtype or source split to the identity.
This avoids hiding possible cross-subtype or cross-split overlap. The
`patient_id_status` is always `CANDIDATE_UNVERIFIED` for such IDs until source
documentation validates their global scope. `IM-*` and `NORMAL2-IM-*` codes
are left unparsed: their patient meaning and namespace are uncertain.
Missing IDs are not fabricated or imputed.

Images are verified then fully decoded. Truncated images are not accepted;
Pillow decompression-bomb warnings are treated as failures. SHA-256 is computed
from original bytes, including unreadable files. Width/height reflect stored
orientation, and aspect ratio is width/height. Original mode and number of
bands are recorded. Pixel statistics use float64, native intensity scale,
population standard deviation (`ddof=0`), all color channels, and no alpha.
Palette images are expanded to RGB only in memory for meaningful intensity
statistics. There is no intensity rescaling, EXIF rotation, or raw write.
`intensity_basis` records the measurement basis. Cross-mode intensity summaries
are exploratory; review mode/bit-depth differences before interpreting flags.

All metadata fields are diagnostic variables, **not CNN input features**.
Missing numeric measurements remain missing; no mean/median imputation is
applied to pixels or metadata. Empty error fields indicate successful reads
and hashes. Therefore literal all-column completeness includes intentionally
empty error fields; the summary also reports completeness without those two
optional fields. `missing_values.csv` reports every column without exceptions.

The quality audit uses readable images, linear-interpolated 25th/50th/75th
percentiles, and fences `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR` for width, height,
aspect ratio, byte size, mean, and standard deviation. Strictly outside values
are flagged. When IQR is zero, values differing from the shared quartile are
flagged. Flags are review prompts; no rows or pixels are deleted or clipped.
Each flagged metric produces one CSV row, so an image can have several reasons.
Unreadable files are listed separately. Full-inventory summaries retain the
effect of repeated copies; recompute diagnostics after approved deduplication.

## Generated outputs

- `image_metadata_audit.csv`: one row per discovered image-extension file,
  all requested fields plus error, parser, patient confidence, and sidecar fields.
- `audit_summary.json`: counts, completeness, candidate-patient leakage,
  imbalance ratio, and runtime versions.
- `duplicate_hashes.csv`, `duplicate_filenames.csv`,
  `duplicate_label_conflicts.csv`: all members of relevant groups.
- `label_failures.csv`, `unreadable_files.csv`: unresolved labels and reads.
- `patient_candidate_audit.csv`: file and distinct-hash counts, source split
  overlap, and label conflicts for each candidate ID.
- `missing_values.csv`, `original_split_counts.csv`, `image_mode_channels.csv`.
- `quality_summary.csv`: ranges, quartiles and IQR fences;
  `image_quality_flags.csv`: flagged paths, values and explicit reasons.
- `class_distribution.csv`: counts, percentages of all files and parsed
  labels, readable counts, and each class/smallest-class ratio.
- `class_distribution_unique_readable_hashes.csv`: sensitivity table with
  one lexicographically first path per readable SHA-256. This is a diagnostic
  view, not an approved deduplication or final selection. Resolve hash-label
  conflicts before trusting its class counts.
- `figures/class_distribution.png`: three-class full-inventory plot, clearly
  labeled as including unreadable files and duplicate copies.

The maximum/minimum class count ratio is undefined if any class is absent;
JSON uses null in that case. No random process is used by the audit. Paths
are sorted lexicographically and output ordering is deterministic.

## Proposed preprocessing and five-fold evaluation strategy

No final folds exist yet. First manually adjudicate unreadable cases, subtype
failures, duplicate-label conflicts and quality flags. Establish a versioned
selection manifest that preserves all original paths and exclusion reasons.
Use SHA-256 for identical bytes; it cannot detect re-encoded duplicates,
near-duplicates or all images of the same patient. Assess those separately.
No identical or related image may straddle training and evaluation folds.

**Current leakage-control plan after Phase 1 review:** use conservative
filename-derived relation groups with
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` as the
primary outer evaluation design. Pneumonia files sharing a `person<number>`
token are kept together across subtype labels. Normal files are grouped within
their observed filename family (`IM` versus `NORMAL2-IM`) by the leading
numeric image token. These are **leakage-control groups, not verified patient
identities**. The current canonical set yields 3,117 groups across 5,824
images, including 1,183 multi-image groups; the largest group contains 31
images. Ordinary `StratifiedKFold(5)` is retained only as a secondary
leakage-sensitivity comparison.

Five outer folds are feasible because the smallest class contains 1,485 images
and 1,216 conservative relation groups. Architecture and hyperparameter
selection among the custom CNN, ResNet-18, and EfficientNet-B0 will occur only
inside each outer development set using inner group-aware validation. For the
final fit in each outer iteration, approximately 10% of the 80% development
portion is reserved group-aware for early stopping, giving roughly 72% training
/ 8% internal validation / 20% untouched outer evaluation. No outer fold may
influence model, preprocessing, or hyperparameter selection.

All train-derived normalization, class/sampling weights, learned
transformations, and stochastic augmentation are fitted/applied on the active
training subset only. Phase 2 will also include a metadata-only shortcut
baseline and padding/crop/resize plus contrast-normalization sensitivity
analyses. Temporal splitting is not applicable because trustworthy acquisition
timestamps are unavailable.
## Dataset integrity verification follow-up

After running the initial audit, reproduce the follow-up with:

```powershell
.\.venv\Scripts\python.exe -m src.integrity_verification --dataset-root chest_xray --archive archive.zip
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The verifier compares filesystem membership and recomputes every inventoried
SHA-256 before reusing previous decode measurements. Changed inputs cause a
failure requiring the initial audit to be rerun. It reads local dataset files
and ZIP directory entries; it does not extract, modify, or delete raw data.

New outputs in `data/processed/`:

- `directory_inventory.csv`: recursive directory counts; rows overlap and must
  not be summed. File counts cover every file type; class/hash counts cover only
  inventoried image extensions. Sidecar counts include metadata with other
  extensions, hence 5,871 overall versus 5,856 image-extension sidecars.
- `directory_copy_comparison.csv`: all 5,856 corresponding readable path pairs,
  with both paths, hashes, label comparison and hash comparison.
- `duplicate_group_audit.csv`: one row per repeated hash. A = same label,
  B = across original splits, C = conflicting labels, D = corresponding suffix
  in duplicate source trees. Categories overlap; A/D is expected for copied
  images. All paths, unique splits, labels, and filenames use JSON arrays.
- `canonical_image_manifest.csv`: **5,824 provisional rows**, at most one
  readable non-sidecar path per consistent hash. Choose the fewest path
  components, then case-sensitive lexical order; no source split has priority.
  All alternative paths remain recorded. Only status-OK recognized labels are
  eligible. Any conflicting or missing group label blocks retention.
- `canonical_manifest_exclusions.csv`: rejected hashes and explicit status,
  including `CONFLICT_REQUIRES_REVIEW`. The observed conflict is metadata only.
- `sidecar_verification.csv`: path pattern, AppleDouble magic/version, entry
  count and bounded-descriptor verification for all 5,856 image sidecars.
- `patient_id_semantics_audit.csv`: every canonical filename, class, candidate
  ID and naming convention. `patient_filename_examples.csv` selects 20 names
  per class across naming families, including the unusual viral suffix case.
- `label_provenance.csv`: rules, actual filenames and canonical class counts.
- `quality_flag_summary.csv`, `quality_flags_unique_content.csv`, and
  `quality_top30_unique_images.csv`: original IQR flags deduplicated by
  `(sha256_hash, metric)`. Thresholds are retained to explain the original
  2,188 paths, not refitted. There are **1,091 unique flagged contents**.
  Top 30 ordering is flag count descending, then SHA-256 ascending for ties.
- `integrity_verification_summary.json`: counts and unresolved final N.

Future modeling discovery must reject every path under `__MACOSX` and every
basename beginning `._`, regardless of extension or readability. The shared
`is_filesystem_metadata` predicate enforces this in canonical selection; full
raw auditing intentionally still includes these paths. Verified AppleDouble
files are `FILESYSTEM_METADATA`, not corrupted X-rays. No quality flags trigger
automatic exclusion. The canonical manifest is provisional; final N remains
unresolved pending manual review, near-duplicate assessment and label review.

## Final Phase 1 statistical and preprocessing verification

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.mplconfig'
.\.venv\Scripts\python.exe -m src.final_phase1_audit --dataset-root chest_xray
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This command requires the canonical manifest and previous audit/quality outputs.
It validates unique hashes, checks all 17,568 inventoried source hashes before
and after, computes canonical-only statistics, inspects all canonical image
channels, and performs BK-tree pHash candidate screening. SciPy is pinned for
statistical tests and DCT computation. Randomized MI references use seed 42 and
499 permutations; no models, augmentations or folds are produced. Run from the
project root; output directories must remain disjoint from the raw directory.

Generated CSVs/JSON in `data/processed/`:

- `class_metadata_summary.csv`, `metadata_target_associations.csv` (including
  effect sizes, Holm-adjusted p, MI and log10 p to distinguish underflow).
- `image_mode_contingency.csv`, `channel_count_contingency.csv` and matching
  `_expected_counts.csv` files.
- `metadata_correlation_matrix.csv`, `high_metadata_correlations.csv`.
- `channel_mode_audit.csv`, `channel_class_counts.csv`.
- `near_duplicate_candidates.csv` (223 unconfirmed pairs, not exclusions).
- `quality_flag_class_analysis.csv`, `feature_target_test_matrix.csv`.
- `final_phase1_summary.json` (manifest fingerprint, versions, parameters,
  preservation checks and key counts).

Figures include `fig3_1_correlation_matrix.png`,
`metadata_class_distributions.png`, `metadata_normal_qq.png`, and four
`near_duplicate_review_*.png` pair sheets. `figures/manual_review/` contains
12 contact sheets covering A–H and `review_index.csv` linking thumbnails to
sources. Category G explicitly records that there are no differing RGB channels.
Thumbnails preserve proportions and display native contrast; these audit views
are not saved CNN preprocessing outputs. Thirty-two selected candidate pairs
are shown, not all 223, and no visual inspection establishes patient identity.

The interpretation and planned decisions are in:

- `phase1_strategy/confounder_assessment.md`
- `phase1_strategy/channel_handling_assessment.md`
- `phase1_strategy/near_duplicate_assessment.md`
- `phase1_strategy/preprocessing_decisions.md`
- The “Statistical Bias and Preprocessing Verification” section of
  `phase1_strategy/audit_findings.md`.

The statistics support the Phase 1 strategy report, not finalizing the cohort.
Manual candidate/quality review, patient uncertainty, and acquisition-bias risks
remain explicit. No raw bytes changed.

## Metadata preprocessing summary

The metadata inventory, missingness summary, canonical IQR review bounds, and
diagnostic VIF table can be reproduced from the committed manifests with:

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing_summary --output-dir data/processed
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_preprocessing_summary.py -v
```

This utility reads the existing audit tables and writes
`phase1_variable_inventory.csv`, `phase1_missingness_summary.csv`,
`outlier_bounds_and_skewness.csv`, and `metadata_vif.csv`. It does not
modify raw images or fit a predictive model.

The canonical IQR review fences flag 1,070 unique contents. The earlier
path-weighted audit identified 1,091 contents using different thresholds; the
two counts are retained only to distinguish the two audit populations. Neither
set is an automatic exclusion list.

The planned partition strategy uses approximately 72% training, 8% internal
validation, and 20% outer evaluation within a five-fold group-aware design.
All data-derived normalization, class/sampling weights, learned
transformations, and augmentation are restricted to the active training data.
Internal validation is used for model and preprocessing selection; outer
evaluation remains isolated until the configuration is locked.

Supporting strategy documents:

- [Missingness assessment](phase1_strategy/missingness_assessment.md)
- [Target encoding strategy](phase1_strategy/encoding_strategy.md)
- [Preprocessing decisions](phase1_strategy/preprocessing_decisions.md)
- [Split and leakage plan](phase1_strategy/split_and_leakage_plan.md)

The clean CI workflow installs the pinned environment, runs the unit tests, and
recomputes the preprocessing summary tables on Linux.
