# Channel handling assessment

Every one of the 5,824 canonical source images was decoded without changing the
file. There are **5,541 L images and 283 RGB images**. For all **283 RGB images,
R == G == B at every decoded pixel**. Zero RGB images have channel differences.
Both maximum and mean channel difference are zero for every RGB row.

| Class | L | RGB, identical channels | RGB, differing channels |
|---|---:|---:|---:|
| NORMAL | 1,579 | 0 | 0 |
| BACTERIAL_PNEUMONIA | 2,617 | 143 | 0 |
| VIRAL_PNEUMONIA | 1,345 | 140 | 0 |
| Total | 5,541 | 283 | 0 |

`channel_mode_audit.csv` contains one row per canonical content, with mode,
channel count, pHash, exact equality, maximum channel difference and mean
channel difference. For RGB, maximum difference is the maximum over all pixels
of `max(R,G,B)-min(R,G,B)`. Mean difference averages `|R-G|`, `|R-B|`, `|G-B|`
over all pixels and the three pairs. Calculations use signed 16-bit arithmetic
to avoid uint8 subtraction wraparound. The nonzero-spread fraction and fraction
with spread >=20 are also reported; the latter is a color-screening heuristic,
not an annotation detector. Non-RGB difference measurements are not applicable
and remain blank rather than being represented as measured zeros.

There is **no actual decoded chromatic information** in the RGB subset. Colored
annotations cannot be present as differing color channels in these files.
This does **not** establish absence of annotations: grayscale letters, markers,
text and borders are visible in sampled contact sheets and may still act as
shortcuts. Their diagnostic or institutional meaning has not been inferred.
The requested RGB-difference contact sheet records that no images qualify.

For later pretrained CNN use, standardize all inputs to an intensity image
and replicate it into three equal channels. For the verified RGB images,
taking any one equal channel preserves decoded intensity exactly; there is
no need to average or apply luminance weights. L images supply their existing
single channel. Replication accommodates a three-channel pretrained input
contract; it does not create color information. For a one-channel baseline,
the same intensity image can remain single-channel. Exact dimensions and model
normalization are discussed in `preprocessing_decisions.md`.

No conversion was applied to the source or to a saved modeling dataset. The
three-channel equality must be rechecked if data are added or replaced. If
future RGB channels differ, inspect their source before adopting this rule.
