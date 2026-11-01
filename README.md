# Explainable Deep Learning for Multiclass Pediatric Pneumonia Classification

A reproducible research pipeline for pediatric chest X-rays, focused on **data integrity, leakage-safe preprocessing, diagnostic analysis, and reproducibility** before predictive modeling.

> **Research use only.** Dataset labels are research annotations and this repository does not provide clinical diagnosis or treatment recommendations.

## At a glance

| Item | Verified evidence |
|---|---:|
| Canonical readable image contents | **5,824** |
| Class distribution | **1,579 Normal · 2,760 Bacterial · 1,485 Viral** |
| Leakage-control structure | **3,117 relation groups → 2,926 final components** |
| Unresolved pHash constraints | **223 edges** |
| Evaluation design | **5 group-aware outer folds** |
| Fold-1 diagnostic matrix | **4,191 × 6 → 4,191 × 4** |
| Reproducibility | **16/16 notebook cells · 30/30 Phase 2 tests** |

The canonical cohort represents unique readable **SHA-256 image contents**, not 5,824 independently verified patients.

## Research flow

```mermaid
flowchart LR
    A[Raw chest X-rays] --> B[Integrity + SHA-256 audit]
    B --> C[Canonical cohort<br/>5,824 images]
    C --> D[Relation groups + pHash constraints]
    D --> E[5 group-aware folds]
    E --> F[Training-only preprocessing]
    F --> G[Metadata diagnostics]
    G --> H[Leakage + reproducibility verification]
```

The implementation deliberately separates two branches:

- **Image pipeline:** validated grayscale handling, deterministic resize/padding, and training-only augmentation controls.
- **Metadata diagnostic pipeline:** Scikit-Learn `Pipeline`/`ColumnTransformer` analysis fitted only on training rows. Metadata are used to investigate acquisition shortcuts and are **not CNN predictors**.

## Key Phase 2 findings

| Diagnostic | Result |
|---|---:|
| Retained metadata variables | **4 of 6** |
| Fold-1 maximum rank-VIF | **121.224 → 6.146** |
| Fold-1 file-size skewness | **2.634 → 0.470** with diagnostic `log1p` |
| PCA variance | **51.68%, 24.39%, 17.94%, 5.98%** |
| Components for ≥90% variance | **3** |
| Components for ≥95% variance | **4** |

The Fold-1 pipeline preserves all 4,191 training rows while removing two redundant metadata columns (`height` and `aspect_ratio`). The fitted four-column schema is applied to validation and outer-evaluation roles **without refitting**.

## Visual diagnostics

<p align="center">
  <img src="figures/fig1_2_target_distribution.png" width="47%" alt="Canonical target distribution">
  <img src="figures/fig3_1_correlation_matrix.png" width="47%" alt="Fold-1 metadata correlation matrix">
</p>

<p align="center">
  <img src="figures/fig4_1_transformation_grid.png" width="47%" alt="Transformation comparison">
  <img src="figures/fig5_1_pca_scree_plot.png" width="47%" alt="PCA scree and cumulative variance plot">
</p>

The complete 300-DPI diagnostic set is available in [`figures/`](figures/):

**Missingness · Target distribution · Outliers · Correlation · Collinear scatterplots · Target association · Transformations · PCA**

## Leakage-control design

Each canonical image is assigned to outer evaluation exactly once. Preprocessing statistics and diagnostic transformations are learned from the active training subset only.

| Fold | Train | Validation | Outer evaluation |
|---:|---:|---:|---:|
| 1 | 4,191 | 468 | 1,165 |
| 2 | 4,192 | 466 | 1,166 |
| 3 | 4,195 | 466 | 1,163 |
| 4 | 4,192 | 466 | 1,166 |
| 5 | 4,194 | 466 | 1,164 |

Verified safeguards include SHA-256 isolation, relation-group isolation, final-component isolation, pHash-edge isolation, exactly-once outer-evaluation coverage, and exclusion of targets/identifiers from fitted metadata.

## Repository

```text
AI-Pneumonia-Project/
├── data/processed/          # fold, grouping, audit and diagnostic evidence
├── figures/                 # publication-quality Phase 2 diagnostics
├── phase1_strategy/         # preserved Phase 1 strategy
├── phase2_implementation/
│   ├── notebooks/           # executed pipeline notebook
│   ├── execute_notebook.py
│   └── phase2_decision_log.md
├── src/                     # reusable Phase 2 modules
├── tests/                   # Phase 2 verification tests
└── requirements.txt
```

### Reproduce the Phase 2 checks

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_phase2_pipeline.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_phase2_grouping.py -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_phase2_diagnostics.py -v
```

Verified result: **30 tests passed, 0 failed**.

Executed notebook: [`phase2_implementation/notebooks/pipeline_execution.ipynb`](phase2_implementation/notebooks/pipeline_execution.ipynb)

## Current status

**Phase 2 preprocessing and diagnostic validation is complete and reproducible.**  
This repository currently exposes the verified preprocessing, leakage-control, diagnostic, and reproducibility evidence. Later model-training, explainability, and deployment artifacts are intentionally kept outside this Phase 2 repository scope until separately reviewed.

## Limitations

- Patient identities are not fully verified.
- Filename relations and pHash edges are conservative leakage constraints, not confirmed patient relationships.
- Metadata associations can reflect acquisition/storage processes rather than disease mechanisms.
- Phase 2 validates preprocessing and leakage controls; it does **not** establish classifier performance or clinical validity.

## Reference

D. S. Kermany *et al.*, “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning,” *Cell*, vol. 172, no. 5, pp. 1122–1131.e9, 2018. https://doi.org/10.1016/j.cell.2018.02.010
