# Baseline missingness and treatment assessment

The two existing CSVs are audited separately. The original inventory contains
17,568 file paths and 25 columns; the provisional canonical table contains
5,824 unique image contents and 33 columns. Counts include all columns,
including target, provenance and optional error text; this is not CNN-input
missingness. No source images were read or changed in this rubric pass.

`phase1_missingness_summary.csv` contains one COLUMN row per field in each
table and a MATRIX_SUMMARY row per table. Empty CSV fields are explicitly read
as missing; literal strings such as `NA` are not automatically treated as
missing. Empty JSON arrays would be recorded values, not nulls. Percentages
for columns use the table's row count; overall percentages use all matrix cells.

| Table | Rows x columns | Matrix cells | Missing cells | Overall missing | Complete rows | Rows with >=1 missing |
|---|---|---:|---:|---:|---:|---:|
| Original audit | 17,568 x 25 | 439,200 | 96,864 | 22.054645% | 0 | 17,568 |
| Canonical manifest | 5,824 x 33 | 192,192 | 13,228 | 6.882701% | 0 | 5,824 |

The zero complete-row counts are largely a consequence of intentionally empty
error columns. Excluding only `read_error` and `hash_error`, complete rows are
8,544 original and 4,244 canonical. These counts mean candidate-ID fields are
filled; they do not mean the patients are identified or independent.

## Columns with nonzero missingness

All columns not listed below have zero missing cells; the CSV includes them too.

| Table | Column(s) | Missing count per column | Percentage |
|---|---|---:|---:|
| Original | patient_id_if_parseable | 9,024 | 51.366120% |
| Original | width, height, aspect_ratio, image_mode, channel_count, mean_intensity, std_intensity, min_intensity, max_intensity, intensity_basis | 5,856 each | 33.333333% each |
| Original | read_error | 11,712 | 66.666667% |
| Original | hash_error | 17,568 | 100% |
| Canonical | patient_id_if_parseable | 1,580 | 27.129121% |
| Canonical | read_error | 5,824 | 100% |
| Canonical | hash_error | 5,824 | 100% |

Canonical missing IDs comprise all 1,579 NORMAL images plus the viral filename
`person447_virus_921_1.jpeg`, rejected by the strict candidate-ID parser.
The remaining 4,244 candidate values are populated but **unverified**. Thus
patient identity is not established for any row by these filenames alone.
Both tables have zero missing target labels. Canonical dimensions, modes and
intensity diagnostics are complete.

## Mechanism classification and treatment

| Field/problem | Observed missingness | Mechanism classification | Evidence | Planned treatment |
|---|---|---|---|---|
| Patient ID availability and confidence | 1,580 canonical blanks; 4,244 unverified candidates | Unavailable identity metadata; MAR-like availability conditional on observed class/filename convention, not proven MAR for true identity | NORMAL naming families lack validated ID semantics; one viral suffix fails parsing | Never impute, invent, or promote candidates to confirmed identities; document grouping limitation |
| AppleDouble image measurements | Ten fields absent on 5,856 non-image sidecars | Structurally undefined, not clinical MCAR/MAR/MNAR | Verified filesystem metadata cannot define image dimensions/pixels | Exclude from modeling discovery while retaining raw inventory; no imputation or image repair |
| Optional error text | Successful operations have blank error fields | Structural/not applicable | read_error is populated only on failed decoding; no hash failures | Retain blank as success/not-applicable; do not impute messages or discard complete images |
| Pixels of valid decoded images | No missing-pixel representation observed; canonical diagnostics complete | No missing-value mechanism demonstrated; tabular MCAR/MAR/MNAR classification is not applicable here | Valid L/RGB arrays decode; black pixels/padding are values, not nulls | No pixel mean/median imputation; cropping, exposure and clinical adequacy need separate review |

There is no evidence supporting a blanket MCAR assumption or a proven MNAR
mechanism. Dependence of parse availability on observed class/filename justifies
the limited phrase MAR-like; it does not establish the latent patient identity
mechanism. Missing anatomy due to cropping is a quality/content issue and is
not equivalent to a missing dataframe cell. Do not replace it with fabricated
pixels. No metadata or image imputation is proposed.
