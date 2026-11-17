# Model Selection and Robustness Summary

## Status — 17 November 2026

This document records the model evidence used to move the project from model evaluation into deployment preparation.

> Research prototype only. The model predicts dataset labels and is not clinically validated.

## Final architecture direction

**DenseNet-121** is the current deployment architecture.

It was preferred because it was the most consistent architecture in the controlled five-fold comparison:

| Fold | DenseNet-121 Macro-F1 | EfficientNet-B0 Macro-F1 | ConvNeXt-Tiny Macro-F1 | Fold winner |
|---|---:|---:|---:|---|
| 1 | 0.826210 | 0.823629 | 0.821882 | DenseNet-121 |
| 2 | 0.855360 | 0.835634 | 0.862563 | ConvNeXt-Tiny |
| 3 | 0.859508 | 0.844623 | 0.842935 | DenseNet-121 |
| 4 | 0.869190 | 0.854031 | 0.859647 | DenseNet-121 |
| 5 | 0.874968 | 0.838963 | 0.872329 | DenseNet-121 |

DenseNet-121 won **4 of 5 folds**. EfficientNet-B0 won none. ConvNeXt-Tiny won Fold 2.

The exact checkpoint used for deployment will be frozen separately. The Fold 4 DenseNet-121 checkpoint was used as the representative model for the detailed robustness and explainability audit because its outer performance was close to the overall five-fold result and DenseNet-121 was the most consistently selected architecture.

## Five-fold outer evaluation

Across the locked five-fold out-of-fold evaluation:

- Accuracy: **81.89%**
- Macro-F1: **81.30%**
- Normal F1: **94.97%**
- Bacterial pneumonia F1: **83.17%**
- Viral pneumonia F1: **65.75%**
- Normal-vs-pneumonia accuracy: **97.34%**
- Bacterial-vs-viral accuracy: **78.23%**

Approximately **85% of errors** were bacterial-versus-viral confusions. Viral pneumonia remained the most difficult class.

## Representative Fold 4 robustness audit

The representative Fold 4 DenseNet-121 outer set contained **1,166 images**.

Baseline Fold 4 metrics:

- Accuracy: **0.8199**
- Macro-F1: **0.8169**
- Balanced accuracy: **0.8163**

### JPEG recompression

Controlled JPEG recompression was applied before the same preprocessing pipeline:

| Condition | Prediction flips | Mean maximum probability shift |
|---|---:|---:|
| JPEG quality 95 | 0 / 1,166 | 0.0023 |
| JPEG quality 75 | 0 / 1,166 | 0.0012 |
| JPEG quality 50 | 14 / 1,166 (1.20%) | 0.0163 |

Overall accuracy at JPEG quality 50 remained **0.8199**.

**Interpretation:** file-size association in the dataset does not by itself show that the Fold 4 model relies strongly on JPEG compression artifacts.

### Resolution perturbation

Images were downsampled and then restored to their original dimensions before the same preprocessing pipeline:

| Retained linear resolution | Prediction flips | Macro-F1 |
|---|---:|---:|
| 75% | 1 / 1,166 | 0.8179 |
| 50% | 3 / 1,166 | 0.8156 |
| 25% | 4 / 1,166 | 0.8166 |

**Interpretation:** the model was highly stable to substantial resolution reduction.

### Padding perturbation

The original G1 preprocessing preserves aspect ratio and adds synthetic black padding. Replacing only this synthetic padding produced much larger changes.

| Condition | Prediction flips | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Original black padding | — | 0.8169 | 0.8163 |
| Edge replacement | 159 / 1,166 (13.64%) | 0.7547 | 0.7743 |
| Reflection replacement | 204 / 1,166 (17.50%) | 0.7256 | 0.7389 |

Viral pneumonia was the most sensitive class:

- Edge replacement flips: **23.57%**
- Reflection replacement flips: **34.68%**

Padding fraction was positively associated with prediction sensitivity:

- Edge perturbation: Spearman **rho = 0.4344**
- Reflection perturbation: Spearman **rho = 0.4569**

This supports a real **padding-associated robustness weakness**. It does not by itself prove that the model uses padding as a causal shortcut in every case.

## Explainability audit

A 30-case HiResCAM audit compared stable cases with padding-sensitive cases.

Mean area-adjusted padding attribution ratio:

| Group | Padding attribution ratio |
|---|---:|
| Stable correct Normal | 0.448 |
| Stable correct Bacterial | 0.703 |
| Stable correct Viral | 0.669 |
| Padding-sensitive Bacterial | **1.063** |
| Padding-sensitive Viral | **1.117** |
| Stable bacterial/viral errors | 0.707 |

A ratio above 1 means padding received more positive attribution than expected from its share of image area.

**Interpretation:** padding-sensitive bacterial and viral cases also tended to show stronger padding attribution than stable cases. This strengthens the evidence that synthetic padding is entangled with the decision function.

## Alternative geometry checks

### G3 center crop

G3 removes padding but center-crops heavily:

- Mean retained fraction: **0.7129**
- Mean crop loss: **0.2871**
- 76.07% of Fold 4 images retained less than 80%
- Worst retained fraction: **0.3178**

G3 was therefore not adopted as a post-hoc replacement because it can remove too much image content.

### Conservative border removal (G4 audit)

A conservative border-removal strategy retained most source image area:

- Mean original area retained: **0.9937**
- Mean G1 padding: **0.2870**
- Mean G4 padding: **0.2825**
- Mean padding reduction: only **0.0045**

It also removed detected borders much more often in Normal images than pneumonia images.

**Interpretation:** G4 was safe but did not reduce the padding problem enough and could introduce another class-associated preprocessing pattern.

## Deployment decision

The project now moves into deployment with **DenseNet-121 as the selected architecture**.

The current five-fold results remain the official scientific performance estimate. The post-hoc robustness audit is documented as a limitation and does not retroactively change the locked outer evaluation.

Before production-like deployment, the exact checkpoint must be frozen and its complete inference contract recorded:

- checkpoint identity and SHA-256
- architecture
- input size
- preprocessing geometry
- contrast policy
- normalization
- class order
- output probability format

The first deployment target is a Dockerized FastAPI service with a `/health` endpoint and a `/predict` endpoint, ready for Azure Container Apps.
