# Class metadata and acquisition-bias assessment

Unit of analysis: **5,824 unique readable canonical SHA-256 contents** (1,579
NORMAL, 2,760 BACTERIAL_PNEUMONIA, 1,485 VIRAL_PNEUMONIA). Alternative source
paths never contribute extra observations. No model or metadata classifier
was fitted. These variables are diagnostics, not manually engineered CNN inputs.

## Methods and assumptions

`class_metadata_summary.csv` reports class N, mean, median, sample SD
(`ddof=1`), linear-interpolated Q1/Q3/IQR, minimum and maximum, unbiased
skewness, Fisher excess kurtosis, and Shapiro-Wilk W/p for six continuous
diagnostics. `metadata_class_distributions.png` uses common pooled
Freedman-Diaconis histogram edges per variable, capped at 60 bins for legibility.
`metadata_normal_qq.png` displays normal Q-Q plots. Class sample sizes are all
below 5,000, but large enough for normality tests to detect minor deviations.

Distribution shape, tails, unequal dispersion and measurement scale motivate
Kruskal-Wallis for all six comparisons, rather than choosing from a normality
p-value alone. File size is strongly right-skewed (class skewness 1.69–2.39,
excess kurtosis 5.35–11.97); aspect ratios have substantial right tails.
Mean intensity is much closer to normal (W about 0.994–0.998), despite small
normality p-values. Intensity spreads still differ by class. Median-centered
Levene/Brown-Forsythe statistics accompany the comparisons as supporting
dispersion diagnostics. Unequal distribution shapes mean Kruskal-Wallis is
interpreted as a rank/distribution association, **not a pure median test**.
ANOVA's common-variance assumptions are not sufficiently defensible here.

Kruskal-Wallis uses tie correction. Effect size is explicitly
`eta_squared_H = max(0, (H-k+1)/(N-k))`, where k=3. This is a rank-test effect
estimate, not literal explained pixel variance. Categorical tests use Pearson
Chi-square, no continuity correction (the tables are 2x3), after checking every
expected count is at least five. Cramer's V is
`sqrt(chi2 / (N * min(rows-1, columns-1)))`. All expected counts pass; the
minimum is 72.159. Mode and channel count encode the same partition here,
so their identical results are redundant evidence, not independent findings.

Holm adjustment covers the eight primary association tests. Exact independence
of patients is unverified and near-duplicates may remain, so p-values and
permutation references are **conditional, exploratory evidence**, not reliable
population-level causal inference. Effect size and visible distribution shifts
carry more practical weight. Extremely small p-values underflow to zero in
floating-point arithmetic; `log10_p_value` preserves magnitude and
`p_value_underflow` identifies them. Zero must not be presented as literal zero
probability. See [SciPy statistical methods](https://docs.scipy.org/doc/scipy/reference/stats.html)
and [Shapiro-Wilk documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html).

Mutual information is a descriptive plug-in estimate in **nats**, using ten
pooled quantile bins for continuous variables (duplicate boundaries dropped),
and original categories otherwise. Seed 42 and 499 label permutations provide
a finite-sample null mean and a coarse reference p-value (minimum 0.002).
Subtracting that mean is only a bias diagnostic, not a validated correction.
MI depends on binning; neither MI nor these tests select CNN features.

## Results

| Diagnostic | Rank effect eta-squared H | Binned MI (nats) |
|---|---:|---:|
| File size | 0.5686 | 0.4517 |
| Height | 0.4174 | 0.2688 |
| Width | 0.3827 | 0.2380 |
| Aspect ratio | 0.2770 | 0.1584 |
| Intensity SD | 0.1704 | 0.1335 |
| Mean intensity | 0.0085 | 0.0248 |

All eight primary comparisons remain below 0.05 after Holm adjustment.
Mean intensity illustrates why statistical significance is not practical
importance: its rank effect is small despite p=7.57e-12. File size is the
strongest measured association. Median bytes are **518,200 NORMAL**, **64,697.5
bacterial**, and **84,946 viral**. Median widths/heights are respectively
**1,654/1,323**, **1,120/752**, and **1,264/824** pixels. Median aspect ratios
are **1.2347**, **1.4830**, and **1.5070**.

Mode and channel count each have Chi-square=148.303, df=2,
p=6.26e-33, Cramer's V=**0.1596**, and MI=0.0181 nats. There are no NORMAL
RGB images, versus 143 bacterial and 140 viral RGB images. This is evidence
of class-related encoding differences, not evidence that color carries disease
information. Pixel-level RGB findings are in `channel_handling_assessment.md`.

Image dimensions and storage size are strongly associated with class, consistent
with acquisition/export differences that merit shortcut-risk investigation.
This audit cannot identify the scanner, institution, protocol or causal origin
of the differences. Brightness and contrast may also reflect anatomy/pathology;
their class association cannot be attributed entirely to acquisition. File size
itself is not given to a CNN, but encoding, resolution and compression can leave
pixel-level traces. Standardizing input shape/channels does not prove removal
of those traces.

## Correlations

Spearman correlation uses ranks to accommodate skew and monotonic relations.
The upper-triangle heatmap flags three pairs at |rho| >= 0.85:

- Width/height: **0.9296**.
- Width/file size: **0.9128**.
- Height/file size: **0.9383**.

These are diagnostics about interrelated resolution/storage properties.
Do not remove image pixels or select CNN input channels because metadata are
correlated. Aspect ratio and future padding geometry may preserve class cues.

## Why quality flags are class-imbalanced

Existing IQR fences are retained exactly to explain the previous audit. They
were defined on the earlier readable-path inventory and are not refitted here;
all rates and flag counts now use unique canonical hashes. Retaining those
historical fences makes this a bias analysis of the existing rule, not a fresh
class-specific outlier detector.

| Class | Flagged / total | Rate | Single flag | Multiple flags |
|---|---:|---:|---:|---:|
| NORMAL | 857 / 1,579 | 54.27% | 740 | 117 |
| BACTERIAL_PNEUMONIA | 148 / 2,760 | 5.36% | 132 | 16 |
| VIRAL_PNEUMONIA | 86 / 1,485 | 5.79% | 79 | 7 |

The pooled file-size upper fence is 505,943.125 bytes, **below the NORMAL median
of 518,200**. It flags 832 normal images (52.69% of that class), versus two
bacterial and one viral. Normal images also account for 70 of 73 width flags
and 116 of 127 height flags. Larger normal-image resolution and storage size
are consistent with these flags, while the correlations support their overlap.
The measurements do not determine which acquisition step produced them.

Aspect-ratio flags instead affect 1 normal, 51 bacterial and 46 viral images.
Mean-intensity flags affect 3/74/30 and intensity-SD flags 0/29/11 respectively.
Those distributions are compatible with the wider pneumonia-class intensity
and aspect-ratio distributions. `quality_flag_class_analysis.csv` includes
per-metric counts and rates with explicit class denominators.

A **STATISTICAL OUTLIER** lies outside a chosen reference distribution's fences.
An **INVALID MEDICAL IMAGE** requires an independent reason, such as unusable
content or a verified data error. One does not establish the other. Blanket
exclusion here would disproportionately remove NORMAL images, distort balance,
and potentially discard useful resolution. No flagged image is excluded.
