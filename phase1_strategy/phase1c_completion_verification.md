# Phase 1C completion verification — 2026-09-30

The existing outputs were inspected, not regenerated. Their canonical manifest
fingerprint still matches `final_phase1_summary.json`. All seven specifically
requested CSVs, three requested assessment Markdown files, correlation figure
and contact-sheet directory exist and are complete for their stated purpose.
The zero-byte strategy PDF remains intentionally unwritten. No training,
augmentation, Grad-CAM, final folds or raw changes were performed.

## Complete Phase 1C deliverable inventory

Paths below are relative to the project root. Earlier Phase 1A/B source files
and reports are inputs, not newly created Phase 1C deliverables.

| File(s) | Purpose |
|---|---|
| `src/final_phase1_audit.py` | Canonical validation, orchestration, plots and before/after raw hash checks |
| `src/statistical_audit.py` | Descriptive summaries, rank/contingency tests, effect sizes, MI and quality-rate analysis |
| `src/image_review.py` | RGB comparison, pHash/BK-tree candidates and contact sheets |
| `tests/test_final_phase1.py` | Seven new safeguards; full suite has 15 tests |
| `requirements.txt` | Pinned SciPy and existing runtime dependencies |
| `.gitignore` | Permit generated review figures while keeping raw data ignored |
| `README.md` | Reproduction commands and output descriptions |
| `data/processed/class_metadata_summary.csv` | 18 class-variable rows; nine requested descriptive measures plus shape/normality |
| `data/processed/metadata_target_associations.csv` | Eight tests with statistics, effects, MI, Holm correction and underflow-safe log p |
| `data/processed/metadata_correlation_matrix.csv` | Six-by-six Spearman matrix |
| `data/processed/high_metadata_correlations.csv` | Three pairs meeting absolute rho >=0.85 |
| `data/processed/image_mode_contingency.csv` | Observed mode/class counts |
| `data/processed/image_mode_expected_counts.csv` | Null expected counts for Chi-square validity |
| `data/processed/channel_count_contingency.csv` | Observed channel-count/class counts |
| `data/processed/channel_count_expected_counts.csv` | Null expected channel-count frequencies |
| `data/processed/channel_mode_audit.csv` | 5,824 per-content channel inspections and pHashes |
| `data/processed/channel_class_counts.csv` | Aggregated mode/channel-equality counts by class |
| `data/processed/near_duplicate_candidates.csv` | 223 unconfirmed pairs with labels, distance, hashes and source provenance |
| `data/processed/quality_flag_class_analysis.csv` | Three class rows with total, flag rates, multiplicity and metric breakdown |
| `data/processed/feature_target_test_matrix.csv` | Eight diagnostic/test mappings, all CNN-feature flags NO |
| `data/processed/final_phase1_summary.json` | Parameters, versions, manifest checksum and original run verification |
| `phase1_strategy/confounder_assessment.md` | Statistical assumptions, acquisition-risk interpretation and biased quality flags |
| `phase1_strategy/channel_handling_assessment.md` | Exact RGB findings and planned intensity standardization |
| `phase1_strategy/near_duplicate_assessment.md` | pHash method, counts, search limits and manual-review policy |
| `phase1_strategy/preprocessing_decisions.md` | Planned image handling, training-only controls and readiness |
| `phase1_strategy/audit_findings.md` | Appended statistical findings and decision log |
| `figures/fig3_1_correlation_matrix.png` | Upper-triangle correlation heatmap |
| `figures/metadata_class_distributions.png` | Six class-wise density histograms |
| `figures/metadata_normal_qq.png` | Eighteen class-variable normal Q-Q panels |
| `figures/near_duplicate_review_01.png`, `02.png`, `03.png`, `04.png` (same prefix) | Four pair-review sheets |
| `figures/manual_review/A_top30_01.png`, `A_top30_02.png` | Thirty previously flagged contents |
| `figures/manual_review/B_width.png` | Low/high widths within each class |
| `figures/manual_review/C_height.png` | Low/high heights within each class |
| `figures/manual_review/D_aspect_ratio.png` | Low/high aspect ratios within each class |
| `figures/manual_review/E_mean_intensity.png` | Brightness extremes |
| `figures/manual_review/F_std_intensity.png` | Contrast-statistic extremes |
| `figures/manual_review/G_RGB_differences_01.png` | Correctly empty category: no RGB channel differences |
| `figures/manual_review/H_near_duplicates_01.png`, `02.png`, `03.png`, `04.png` (same prefix) | 32 selected candidate pairs, including high-priority examples |
| `figures/manual_review/review_index.csv` | Thumbnail-to-source mapping |

This completion pass changes only the preprocessing Markdown to specify the
now-requested model normalization, adds this verification note, and links the
note from the decision log. Existing valid CSVs, plots and Python utilities
are preserved. Generated caches are not scientific deliverables.

## Verification of reported numbers

CSV checks confirmed 5,824 unique manifest hashes; exact membership of the
channel audit; 18 summary rows; eight associations; six-by-six correlations;
223 distinct pairs with correctly recomputed distances from saved hashes;
and 1,091 unique quality flags. Every PNG successfully passes Pillow integrity
verification. Raw membership and all 17,568 SHA-256 hashes match the original
inventory in the read-only verification on 2026-09-30.

### Current test-run limitation

The earlier Phase 1C session recorded 15 passing tests. On 2026-09-30 the
complete-suite command was attempted twice, including an approved execution
outside the sandbox. **Eight tests passed; zero assertion failures; zero
skips; one module-import error prevented the seven Phase 1C tests from loading.**
Unittest prints “Ran 9 tests” because its failed-import placeholder counts as
one test; this is not nine actual project tests. The project contains 15 tests,
of which seven were not executed in this rerun. Windows Application Control
blocked SciPy `_ccallback_c` initially and `_ansari_swilk_statistics` on retry.
No DLLs, security policies, dependencies or tests were changed to bypass it.
Therefore this completion check cannot claim a fresh 15/15 pass. Resolve the
runtime's application-control issue and rerun before claiming clean-environment
test reproducibility. Raw-hash and saved-output consistency checks succeeded
independently and existing analysis outputs were not overwritten.

Candidate pairs: NORMAL/NORMAL 162; NORMAL/bacterial 14; NORMAL/viral 25;
bacterial/bacterial 6; bacterial/viral 9; viral/viral 7. Same-class total 175,
cross-class 48, cross-source 27, both cross-class/source 10; 262 unique images.
Distances: 16 pairs at 2, 207 at 4, none at 0/1/3. The minimum-distance pairs
include `NORMAL2-IM-1176-0001.jpeg` / `NORMAL2-IM-0892-0001.jpeg`,
`IM-0237-0001.jpeg` / `NORMAL2-IM-1350-0001.jpeg`, and
`IM-0507-0001.jpeg` / `IM-0697-0001.jpeg` (all distance 2, NORMAL).
These are strong hash matches, not confirmed identical images or patients.

## Nonclinical contact-sheet observations

| Category | Observation |
|---|---|
| A, previously flagged | Recognizable radiographs with varied framing and R/L labels; large dimensions/bytes do not by themselves make images unusable |
| B, width | Small-width examples often have tight chest crops; large-width examples show more surrounding field. Thumbnail size is standardized, so native resolution is established by metadata |
| C, height | Very short source images form narrow horizontal chest strips; taller examples include more superior/inferior field. Line-like overlays are visible in some examples |
| D, aspect ratio | Clear differences between tall framing and severe horizontal crops; proportions and positioning vary |
| E, brightness | Dark and very bright renderings remain recognizable; several show circular pads and cable/line-like structures, plus R/L markers |
| F, intensity SD | Low-SD examples look flatter/paler; high-SD examples have stronger light/dark separation. Text and laterality markers remain visible |
| G, RGB differences | Empty by verified equality; no missing images or failed generation |
| H, candidate pairs | Similar global composition can accompany visibly different framing, positioning and markers even at distance 2; no patient or label conclusion is justified |

Examples of apparent equipment/cable overlays include the brightness sheet's
`person501_bacteria_2113.jpeg` and `person14_virus_44.jpeg`. These are visual
descriptions, not device identification or clinical interpretation. Severe crop
examples include high-aspect-ratio `person325_virus_661.jpeg`. Orientation and
positioning vary, but no acquisition projection or pathology is inferred.
Full-resolution review remains necessary where thumbnails are insufficient.

## Consolidated final decision log

| Issue | Evidence | Decision | Reason | Phase 2 consequence |
|---|---|---|---|---|
| macOS sidecars | 5,856 image-extension AppleDouble files | Exclude from modeling discovery; retain raw | Filesystem metadata, not corrupt radiographs | Use validated manifest |
| Duplicate source trees | 5,856 corresponding byte-matched paths | One canonical path per hash | Copied trees add no observations | Preserve alternatives for provenance |
| Within-tree duplicates | 30 groups; 32 excess copies per tree | Deduplicate exact contents | Single-tree selection alone is insufficient | 5,824 provisional contents |
| Class imbalance | 1,579/2,760/1,485 | Training-fold weighted cross-entropy | No synthetic pixel interpolation needed | N_train/(3*n_class_train), never full-dataset weights |
| Quality outliers | 54.27% normal versus about 5–6% pneumonia flagged | Manual review; no Phase 1 removal | Statistical outlier != invalid medical image | Final inclusion decisions must be documented |
| RGB/L differences | 283 RGB files have equal channels | Common intensity; replicate for pretrained models | No chromatic signal lost | One-channel custom baseline; three-channel pretrained input |
| Acquisition bias | Largest rank effects: bytes, height, width, ratio | Document and investigate residual shortcuts | Association is not causal proof | Standardization alone does not establish generalization |
| Metadata multicollinearity | Three rho values >=0.85 | Keep diagnostic interpretation only | CNN consumes spatial pixels, not dimension columns | No dimension/pixel dropping |
| Near duplicates | 223 candidates; 48 cross-class | Adjudicate, no automatic removal | pHash similarity is not patient identity | Resolve confirmed relationships before folds |
| Patient uncertainty | No validated ID key | Non-grouped stratification with explicit limitation | Invented groups would be unsupported | No claims of patient-independent evaluation |
| Kaggle source split | Original train/val/test retained as metadata | Do not reuse as final split | Assignment requires pooled five-fold strategy | StratifiedKFold(5, shuffle=True, random_state=42) |
| Resizing | Widely variable sizes/ratios | 224 square canvas, preserved ratio, zero padding | Avoid geometric stretching/cropping | Validate padding/resolution effects later |
| Normalization | Eight-bit grayscale; named ImageNet options | x/255 baseline; external mean/SD for pretrained | Different input contracts | Training-only estimation for any dataset-derived alternative |
| Augmentation | Framing/intensity variation | Candidate +/-5-degree rotation, <=2% translation | Modest perturbations pending review | Training only; no flipping/aggressive cropping/color jitter |
| Leakage control | Exact copies resolved; identities/candidates uncertain | Freeze selection first; fit only in training fold | Prevent direct and fitted-state leakage | Inner validation for tuning; fixed held-out preprocessing |

## Readiness

Phase 1 report writing can proceed. No further Phase 1 coding is necessary for
the requested scope. N=5,824 is **provisional canonical unique-image-content N**,
not final modeling N. Patient identity, causal acquisition provenance, pHash
screening sensitivity, and external validity can remain explicit report
limitations. Before training, record inclusion decisions for questionable
quality/cropping cases and candidate pairs, resolve confirmed cross-label
duplicates, freeze the modeling manifest, and pin/test the exact preprocessing
and training-only split workflow. Unknown patient identity need not be invented
or treated as solvable; formally accept its limitation if no key is obtainable.

Class-specific IQR thresholds are not approved exclusion rules. They can change
which observations look unusual but do not supply medical validity criteria,
and label-dependent cleaning could create additional bias. Keep the existing
flags for review and exclude only on a separately justified, documented basis.
