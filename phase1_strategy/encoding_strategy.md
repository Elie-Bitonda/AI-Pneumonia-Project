# Target and diagnostic encoding strategy

Task: **Multi-Class Classification**, with one mutually exclusive label per
image. Target Y is `target_label`; its semantic type is **nominal categorical**.

| Target label | Planned integer index |
|---|---:|
| NORMAL | 0 |
| BACTERIAL_PNEUMONIA | 1 |
| VIRAL_PNEUMONIA | 2 |

These indices are stable class identities, not severity, ordering, distance or
ranks. No ordinal predictor variables are used. Save this mapping alongside
future model artifacts and use the same order for logits, loss weights,
confusion matrices and prediction decoding. Unknown/failed labels must trigger
review, not an implicit default class. The manifest's string labels remain
unchanged in Phase 1.

For the planned hard-label PyTorch CrossEntropyLoss path, targets are integer
class indices in a `torch.long` tensor, shape `(batch,)`, and model outputs are
three unnormalized logits per image, shape `(batch, 3)`. No one-hot encoding
is required and softmax need not be applied before this loss. The loss also
supports probability targets, but that optional mode is not this project's
plan. See [official CrossEntropyLoss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

Input X is the decoded image intensity tensor. Image mode, channel count,
dimensions, bytes, intensity summaries, IDs, hashes, paths and other inventory
fields are diagnostics/provenance, not additional CNN predictors. They receive
no ordinal or one-hot predictor encoding. The target code is not an input
feature. Three-channel grayscale replication for a pretrained model repeats
the same intensity information and does not create independent features.
