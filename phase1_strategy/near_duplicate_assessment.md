# Conservative near-duplicate screening

The screen uses all 5,824 canonical readable contents, never copied paths.
Each image is converted in memory to L and resized to 32x32 with Lanczos for
hashing only. A two-dimensional orthonormal DCT-II is computed. The upper-left
8x8 coefficients (including DC) are compared strictly with their median to
create a 64-bit perceptual hash. The fixed candidate threshold is Hamming
distance **<=4**. It was selected as a narrow screening radius, not calibrated
against known duplicate/nonduplicate pairs. This is not the proposed CNN
resize or an augmentation operation.

A BK-tree uses Hamming distance's triangle inequality to prune its exact radius
queries. Each hash is queried against previous hashes and then inserted, so
each unordered candidate pair appears once. No full-resolution all-pairs
image comparison occurs. The search is deterministic in SHA-256 order; BK-tree
worst-case hash-distance work can approach quadratic, but it does not compare
all pairs of source pixels. Synthetic exhaustive small-hash checks validate
candidate retrieval.

**223 pairs** involving **262 unique contents** meet the threshold. There are
16 pairs at distance 2 and 207 at distance 4; none at distance 0. Of these,
**48 pairs cross target classes**, **27 cross original source splits**, and
**10 cross both**. These overlapping categories must not be added. Every pair
has different SHA-256 hashes. CSV fields include both source paths, labels,
hashes, distance, original splits and all alternative source paths.

`near_duplicate_candidates.csv` is sorted by cross-class status, cross-source
status, then increasing distance and hashes. Contact sheets prioritize eight
cross-class pairs, eight cross-source pairs and the 24 lowest-distance pairs,
deduplicated and capped at 40. The current selection contains 32 pairs on four
sheets. The manual-review index maps labels such as “pair 49” to one-based CSV
data-row positions. Original source splits remain provenance, not final folds.

Visual inspection of representative sheets shows that similar global chest
layout can produce small hash distances even when positioning, framing or
other visible details differ. These are examples of why pHash similarity
does not prove identical radiographs or shared patients. No candidate has
been promoted automatically to a confirmed duplicate or a conflicting-label
image. No clinical judgment, relabeling or deletion was made.

Manually adjudicate all cross-class and cross-source candidates first, followed
by the remaining pairs. Inspect full-resolution source images when thumbnails
are insufficient and record decisions with evidence. Confirmed related copies
must not later cross evaluation partitions; retain a documented representative
or explicitly revisit the splitting plan before folds are created. If identity
cannot be determined, record unresolved status rather than merging by hash.

False positives arise from similar composition; false negatives can arise from
cropping, rotation, local edits, or the conservative threshold. This screen is
not proof that all related images were found. Patient independence remains
unverified even after exact-content deduplication and this candidate screen.
