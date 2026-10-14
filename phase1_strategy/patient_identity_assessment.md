# Identity evidence and leakage-control grouping

The available dataset files do not provide a verified patient identity key, so
filename tokens are not presented as confirmed patient IDs. They nevertheless
contain repeated relation signals that are relevant to leakage prevention.

## Evidence

After SHA-256 content deduplication:

| Class | Filename convention | Canonical count | Leakage-control interpretation |
|---|---|---:|---|
| NORMAL | `IM-<number>-<number>[-<number>].jpeg` | 666 | group within the IM family by leading numeric token |
| NORMAL | `NORMAL2-IM-<number>-<number>[-<number>].jpeg` | 913 | group within the NORMAL2-IM family by leading numeric token |
| BACTERIAL_PNEUMONIA | `person<number>_bacteria_<number>.jpeg` | 2,760 | group by `person<number>` token |
| VIRAL_PNEUMONIA | `person<number>_virus_<number>.jpeg` | 1,484 | group by `person<number>` token |
| VIRAL_PNEUMONIA | `person447_virus_921_1.jpeg` | 1 | label is unambiguous; extra suffix prevents strict ID parsing |

The strict parser yields 1,674 pneumonia candidate tokens; 170 appear in more
than one original source split and 979 appear under both pneumonia subtypes.
These observations do not prove that the tokens are globally valid patient
identifiers. Treating every image as independent, however, could place
plausibly related examinations in different folds.

## Leakage-control grouping decision

The planned primary evaluation uses conservative relation groups with
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
The groups are leakage-control units, not verified patient identities.

The filename-only base structure contains:

- 1,443 Normal relation groups for 1,579 images;
- 1,437 bacterial-class groups for 2,760 images;
- 1,216 viral-class groups for 1,485 images;
- 3,117 groups across all 5,824 images;
- 1,183 multi-image groups;
- largest group size: 31 images.

The per-class group counts are not additive because 979 `person<number>`
tokens appear under both pneumonia subtypes. Therefore
`1,437 + 1,216 - 979 = 1,674` merged pneumonia groups, and
`1,674 + 1,443 = 3,117` groups overall.

Cross-subtype merging is deliberately conservative and is not evidence that a
shared numeric token proves one patient across subtypes. A subtype-separated
sensitivity structure contains 4,096 total groups, 722 multi-image groups, and
largest group size 30. This alternative is retained to expose the effect of the
namespace assumption.

## Near-duplicate relations

The 223 pHash candidate pairs are not automatically merged into identity
groups. For each candidate edge:

- a confirmed relation is merged into the leakage-control graph;
- a rejected edge is omitted;
- an unresolved edge is retained only as a split constraint so its endpoints
  cannot cross training and evaluation roles.

The current 3,117 count is therefore the filename-only base group count, not a
guarantee of the final post-review count. Connected components are formed and
group feasibility is checked before fold assignment.

No patient-independent performance claim is justified until the grouping and
candidate-review policy has been applied consistently.
