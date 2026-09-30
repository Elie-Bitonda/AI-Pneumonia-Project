# Phase 1 rubric evidence coverage

Checked 2026-09-30 against the supplied requirements. COMPLETE means the
requested Phase 1 evidence or written strategy exists, not that Phase 2 has
been executed or that clinical validity is established. Points below are rubric
category labels, not awarded marks. Paths are relative to the project root.

## 1. GitHub Architecture & Documentation — 15 pts

| Requirement | Evidence/file | Status | Any action still required |
|---|---|---|---|
| Project structure and dependency declaration | `.gitignore`, `requirements.txt`, `data/raw/`, `data/processed/`, `phase1_strategy/`, `phase2_implementation/notebooks/`, `src/`, `figures/` | COMPLETE | None for directory coverage |
| README problem, domain, source, empirical N, image P, Y, Multi-Class Classification, setup, reproduction, structure, leakage and nonclinical disclaimer | `README.md` | COMPLETE | Keep synchronized with final report |
| Git version tracking and deliverable status | `git rev-parse`, `git remote -v`, `git status --short` all report not a Git repository | MISSING | Initialize/version required deliverables separately; current files are unversioned, not an enumerable Git untracked list |
| GitHub remote | No Git repository or configured remote | MISSING | Configure required GitHub delivery separately; nothing was pushed or published |
| Current full-suite reproducibility | `tests/`; `phase1_strategy/phase1c_completion_verification.md`; four new rubric tests passed | PARTIAL | Resolve Windows Application Control blocking SciPy DLLs and rerun complete suite; earlier 15 passed, latest existing-suite rerun had eight executable passes and seven unable to load |
| Written Phase 1 Strategy Report | `phase1_strategy/Phase1_Strategy_Report.pdf` is a zero-byte placeholder | MISSING | Write and visually verify the report using completed evidence |

## 2. Data Audit & Metadata Inventory — 20 pts

| Requirement | Evidence/file | Status | Any action still required |
|---|---|---|---|
| Raw, readable and unique-content N | `data/processed/audit_summary.json`, `canonical_image_manifest.csv`, `integrity_verification_summary.json` | COMPLETE | Report 17,568 paths, 11,712 readable paths, provisional 5,824 contents; do not claim patient independence |
| Exact metadata N x P and ordered schema | `data/processed/phase1_variable_inventory.csv`; `phase1_strategy/rubric_analysis_notes.md` | COMPLETE | Use canonical 5,824 x 33; original 17,568 x 25 |
| Meaning, storage dtype, semantic type, Y versus X | Same inventory and notes; `README.md` | COMPLETE | Preserve distinction between 33 metadata columns and planned image tensor values |
| Class counts and source provenance | `data/processed/label_provenance.csv`, `canonical_image_manifest.csv`; `README.md` | COMPLETE | Archive version/license and global patient identity remain unverified source limitations |
| Baseline missingness by column and matrix for both tables | `data/processed/phase1_missingness_summary.csv`; `phase1_strategy/missingness_assessment.md` | COMPLETE | Preserve structural versus substantive interpretation |
| Canonical-only statistical unit | Unique SHA-256 validation in `src/rubric_gap_audit.py` and earlier audit utilities | COMPLETE | Keep source copies as provenance, not additional observations |

## 3. Cleaning, Imputation & Outlier Strategy — 20 pts

| Requirement | Evidence/file | Status | Any action still required |
|---|---|---|---|
| Sidecar and exact-copy handling; raw preservation | `phase1_strategy/audit_findings.md`; `data/processed/sidecar_verification.csv`, `directory_copy_comparison.csv`, `canonical_image_manifest.csv` | COMPLETE | Retain raw; use provisional canonical selection for analysis |
| MCAR/MAR/MNAR versus structural missingness and treatment | `phase1_strategy/missingness_assessment.md` | COMPLETE | No ID invention/imputation; no unsupported mechanism certainty |
| Global skew, means, medians, quartiles, fences and counts | `data/processed/outlier_bounds_and_skewness.csv` | COMPLETE | New canonical fence population is 1,070, distinct from historical 1,091 |
| IQR versus Z-score rationale for each diagnostic | Same CSV; `phase1_strategy/rubric_analysis_notes.md` | COMPLETE | IQR review only; no pixel Winsorization or automatic exclusion |
| Class-dependent quality-flag bias | `data/processed/quality_flag_class_analysis.csv`; `phase1_strategy/confounder_assessment.md` | COMPLETE | Preserve historical flag definition when citing prior rates |
| RGB equivalence and planned channel handling | `data/processed/channel_mode_audit.csv`; `phase1_strategy/channel_handling_assessment.md` | COMPLETE | Implement later; no transformations now |
| Near-duplicate screening and review materials | `data/processed/near_duplicate_candidates.csv`; `figures/manual_review/`; `phase1_strategy/near_duplicate_assessment.md` | COMPLETE | Screening does not prove patient identity |
| Final modeling-cohort adjudication | Review sheets and provisional manifest | PARTIAL | Human review of candidate pairs/quality/labels before freezing modeling cohort; not a prerequisite to documenting Phase 1 strategy |

## 4. Statistical Analysis, Encoding & Scaling Strategy — 25 pts

| Requirement | Evidence/file | Status | Any action still required |
|---|---|---|---|
| Class summaries, shape, conservative test choice, effects and diagnostic MI | `data/processed/class_metadata_summary.csv`, `metadata_target_associations.csv`; `phase1_strategy/confounder_assessment.md` | COMPLETE | Interpret associations as shortcut risks, not causal proof or CNN feature selection |
| Categorical contingencies, validity and Cramer's V | `data/processed/image_mode_contingency.csv`, `channel_count_contingency.csv`, matching expected counts and associations | COMPLETE | Retain categorical treatment for mode/channel association despite discrete channel count |
| Feature-target matching matrix | `data/processed/feature_target_test_matrix.csv` | COMPLETE | All diagnostic columns remain excluded from CNN predictors |
| Spearman heatmap and absolute rho >=0.85 screen | `data/processed/metadata_correlation_matrix.csv`, `high_metadata_correlations.csv`; `figures/fig3_1_correlation_matrix.png` | COMPLETE | No pixel/dimension removal |
| All six VIFs, >10 rule, hypothetical iterative redundancy protocol | `data/processed/metadata_vif.csv`; `phase1_strategy/rubric_analysis_notes.md` | COMPLETE | Apply removal protocol only if variables become actual traditional predictors; no CNN feature removal here |
| Nominal target encoding and integer-index cross-entropy | `phase1_strategy/encoding_strategy.md` | COMPLETE | Preserve NORMAL=0, BACTERIAL=1, VIRAL=2; no ordinal interpretation |
| Image size, aspect preservation, channels and baseline/pretrained scaling | `phase1_strategy/preprocessing_decisions.md` | COMPLETE | Planned 224 square canvas; preserve aspect ratio; one channel baseline or three replicated channels pretrained; baseline /255 versus architecture-specific normalization |
| Augmentation and imbalance strategy | Same preprocessing decisions | COMPLETE | Training-only candidate transforms and class weights; no augmentation executed; no SMOTE on raw pixels |

## 5. Split Strategy & Leakage Prevention Architecture — 20 pts

| Requirement | Evidence/file | Status | Any action still required |
|---|---|---|---|
| Reproducible outer five-fold stratification | `phase1_strategy/split_and_leakage_plan.md` | COMPLETE | Proposed StratifiedKFold(5, shuffle=True, random_state=42); no folds generated |
| Exact intended internal proportions | Same split plan; `README.md` | COMPLETE | Outer 80/20, inner 10% of development => approximately 72/8/20; integer rounding acknowledged |
| Training-only fitted state, weights, sampling and augmentation | Same split plan | COMPLETE | Use actual training subset, not all outer development data |
| Internal-only early stopping and tuning, untouched outer evaluation | Same split plan | COMPLETE | Lock fold choices before final evaluation; do not tune from outer scores |
| Duplicate controls and patient limitation | Same split plan; `phase1_strategy/patient_identity_assessment.md`, `near_duplicate_assessment.md` | COMPLETE | Before folds, adjudicate confirmed relationships; do not fabricate patient groups or claim patient-independent validation |

## Readiness and outstanding work

No additional numerical Phase 1 analysis is required to write the report.
Remaining delivery gaps are Git/GitHub setup, fresh full-suite verification
after the environment block is resolved, and the report itself. Provisional
cohort adjudication and unresolved identity/provenance/generalization issues
must be stated in the report and addressed or explicitly accepted before
modeling. No Phase 2 work, raw modifications, augmentation or folds occurred.
