# Patient identity assessment — local evidence only

Patient-level grouping cannot currently be justified from the available filename metadata.

The current recommendation is
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` after content
deduplication and manual selection review. Five folds meet the assignment,
shuffling removes source-order dependence, and seed 42 makes the future
allocation reproducible. No folds have been created. Do not claim patient
independence: repeated patients and related images can still cross folds.
Only validated global patient identities could justify revisiting
`StratifiedGroupKFold`.

## Evidence searched

Only local evidence was used for this follow-up. The dataset tree contains
17,591 files: 17,568 image-extension paths, 15 other AppleDouble-style metadata
paths and eight `.DS_Store` files. A search for `.md`, `.txt`, `.csv`, `.json`,
`.pdf`, `.html`, and `.xml` documentation found no candidates in the dataset.
The ZIP central directory contains 17,591 entries and no documentation
candidates. All 23 non-image paths are listed in
`data/processed/local_documentation_inventory.json`. The project's README,
audit reports and empty report placeholders were created during this work;
they are not independent source documentation or a patient identity key.
No external sources were searched for this follow-up.

## Observed conventions after byte-content deduplication

| Class | Observed filename convention | Canonical count | ID interpretation |
|---|---|---:|---|
| NORMAL | `IM-<number>-<number>[-<number>].jpeg` | 666 | Unproven; left blank |
| NORMAL | `NORMAL2-IM-<number>-<number>[-<number>].jpeg` | 913 | Unproven; left blank |
| BACTERIAL_PNEUMONIA | `person<number>_bacteria_<number>.jpeg` | 2,760 | Candidate only |
| VIRAL_PNEUMONIA | `person<number>_virus_<number>.jpeg` | 1,484 | Candidate only |
| VIRAL_PNEUMONIA | `person447_virus_921_1.jpeg` | 1 | Extra suffix; strict ID parser rejects |

The final viral filename still has an unambiguous whole `virus` token and so
has a parsed viral label. Its extra `_1` suffix has unknown meaning and is
not silently removed to invent a patient identity.

`patient_id_semantics_audit.csv` contains filename, parsed_candidate_id,
class, convention and path for all 5,824 canonical contents.
`patient_filename_examples.csv` provides exactly 20 representative filenames
from each class (60 total), including both normal naming families and the
viral suffix exception. Examples are chosen deterministically within each
family and sorted by class and filename. Missing candidate IDs remain blank.

The original 8,544 paths matching the strict `personNNN_(bacteria|virus)_NNN`
pattern yielded 1,674 numeric candidate identifiers. Of those, 170 spanned
source splits and 979 spanned pneumonia subtypes. For example,
`person100_bacteria_475.jpeg` and `person100_virus_184.jpeg` share a numeric
token, but the available local files do not establish that they depict one
patient. The overlaps are compatible with different namespaces, repeated
patients or another undocumented convention. Neither shared identity nor
independent identity can be inferred with confidence. Adding subtype or split
to the candidate ID would conceal the ambiguity rather than resolve it.

## Label provenance is separate from patient identity

Labels come from local folder/filename annotations, not independent clinical
verification. `NORMAL` requires a `NORMAL` ancestor folder and no pneumonia
subtype token. Pneumonia subtypes require `PNEUMONIA` plus exactly one whole
`bacteria` or `virus` token, using the parser in `src/transformers.py`.
`label_provenance.csv` supplies examples, rules and canonical class totals.
The observed patterns support subtype parsing, but do not establish
microbiological diagnoses or validate the namespace of patient numbers.

## Remaining manual review

Obtain source documentation or a verified identity key before making patient
group claims. Investigate the extra viral suffix, inspect near-duplicates and
quality flags, and confirm label provenance. Keep the candidate ID column for
diagnostics only under the current plan; do not feed it as a feature or as a
validated split group. Train-derived normalization, augmentation and weighting
remain restricted to each future training fold.
