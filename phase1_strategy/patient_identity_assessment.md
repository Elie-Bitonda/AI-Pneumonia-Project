# Identity evidence and leakage-control grouping

The local files do not provide a verified patient identity key, so I do not
claim that filename tokens establish clinical identity. Phase 1 nevertheless
shows clear relation evidence that should not be ignored during validation.

## Evidence

After SHA-256 content deduplication:

| Class | Filename convention | Canonical count | Leakage-control interpretation |
|---|---|---:|---|
| NORMAL | `IM-<number>-<number>[-<number>].jpeg` | 666 | group within IM family by leading numeric token |
| NORMAL | `NORMAL2-IM-<number>-<number>[-<number>].jpeg` | 913 | group within NORMAL2-IM family by leading numeric token |
| BACTERIAL_PNEUMONIA | `person<number>_bacteria_<number>.jpeg` | 2,760 | group by person<number> token |
| VIRAL_PNEUMONIA | `person<number>_virus_<number>.jpeg` | 1,484 | group by person<number> token |
| VIRAL_PNEUMONIA | `person447_virus_921_1.jpeg` | 1 | label is unambiguous; extra suffix blocks the strict identity parser |

The strict patient-ID parser originally yielded 1,674 pneumonia candidate
tokens; 170 crossed original source splits and 979 appeared under both
pneumonia subtypes. Manual-review examples also show repeated adjacent
filename tokens such as `person279_bacteria_1315/1316`,
`person1544_bacteria_4033/4035`, and
`person325_virus_659/660/661`.

These observations do not prove that the tokens are globally valid patient IDs.
However, treating every image as independent would risk placing plausibly
related examinations in different folds. For leakage prevention I therefore
use conservative relation groups without converting them into patient claims.

The current canonical data produce:

- 1,443 Normal relation groups for 1,579 images; 74 groups contain >1 image.
- 1,437 Bacterial groups for 2,760 images; 491 groups contain >1 image.
- 1,216 Viral groups for 1,485 images; 157 groups contain >1 image.
- 3,117 groups across all 5,824 images; 1,183 groups contain >1 image.
- largest conservative group size: 31 images.

## Phase 2 split decision

Primary outer evaluation:

```python
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

The relation-group key is a leakage-control device, not a verified patient
identifier. Over-grouping can slightly reduce class-balance precision, but
under-grouping can inflate evaluation by allowing related images to cross
training and evaluation roles. Ordinary `StratifiedKFold(5)` is retained only
as a secondary sensitivity analysis; the grouped-versus-ungrouped performance
difference will be reported as a leakage-sensitivity diagnostic.

No fold files have been generated in Phase 1. Near-duplicate candidates remain
review candidates; only a manually confirmed relation would be merged into the
group structure before folds are frozen.


## Subtype-namespace sensitivity and count reconciliation

The per-class group counts are not additive because the same conservative
`person<number>` relation group can contain both pneumonia subtypes. There are
1,437 bacterial-class groups and 1,216 viral-class groups, with 979 shared
cross-subtype tokens. Therefore the merged pneumonia total is
`1,437 + 1,216 - 979 = 1,674` groups; adding the 1,443 Normal groups gives
3,117 groups overall.

Cross-subtype merging is a deliberately conservative leakage-control choice,
not evidence that a shared numeric token proves one patient across subtypes.
To expose this assumption, Phase 2 will also run a subtype-namespace sensitivity
scheme in which bacterial and viral tokens are kept separate. That scheme has
4,096 total groups, 722 multi-image groups, and largest group size 30, compared
with 3,117 groups, 1,183 multi-image groups, and largest group size 31 under
cross-subtype merging. The primary grouped estimate uses the conservative
merged scheme; the subtype-only scheme is reported as a sensitivity analysis.
