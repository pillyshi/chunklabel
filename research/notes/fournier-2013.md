# Notes: fournier-2013

> Fournier, C. (2013). Evaluating Text Segmentation using Boundary Edit Distance.
> ACL 2013, pp. 1702–1712. https://aclanthology.org/P13-1167
> Builds on [fournier-2012] (Segmentation Similarity S).

## Summary

Proposes **boundary similarity (B)**, a segmentation metric based on boundary edit
distance, plus a boundary confusion matrix that yields **B-precision / B-recall**.
Argues that Pk, WindowDiff, and S all have known biases, and that B is the only one of
the three that ranks near misses, false positives, and false negatives sensibly.

## Method

- Boundary edit distance pairs the boundaries of two segmentations with three
  operations: additions/deletions (full misses), n-wise transpositions (near misses
  within `n_t` units), and substitutions (confusing boundary *types*).
- Each boundary pair gets a correctness score: match 1, addition/deletion 0,
  transposition `1 − distance/(n_t − 1)` (linear), substitution scaled by type distance.
- `B = mean correctness over all boundary pairs`. Range [0, 1], symmetric (no
  "reference vs hypothesis" asymmetry, unlike Pk/WD whose window size comes from the
  reference).
- The confusion matrix sums pair correctness, so precision/recall award partial credit
  for near misses.

## Key findings

- Known flaws of Pk/WD (§2.1): they under-penalise errors at the start and end of a
  text, are biased towards segmentations with few or tightly clustered boundaries,
  compute `k` inconsistently (rounding), and are not symmetric.
- S gives cosmetically high values and favours very sparse segmentations.
- Worked example (§4): only B ranks near miss > false positive > false negative and
  clustered false positives. WD penalises a near miss as harshly as a full miss.
- Inter-coder agreement on real data is much lower than S suggests (Moonstone:
  S-based π* ≈ 0.9 vs B-based ≈ 0.2–0.4). Human coders disagree a lot on where
  topical boundaries go and on how many to place.

## Key findings relevant to chunklabel

- **Explains our Wiki-50 observation**: Pk rewarded "no boundaries" (0.402) over most
  methods. Pk/WD's bias towards few boundaries is documented here.
- **B-precision is the granularity-tolerant measure we want**: a segmenter that is
  finer than the reference but places its coarse boundaries correctly gets high
  B-recall for the reference boundaries, and its extra boundaries show up only as
  lower B-precision. Reporting BP and BR separately makes "finer granularity" and
  "wrong boundaries" distinguishable, which a single Pk number cannot.
- Boundary *types* and substitutions could later evaluate hierarchical (section vs
  paragraph) boundaries.
- Reference implementation: the `segeval` Python package (by the author).

## Limitations

- Still needs a reference segmentation, and still measures boundaries only, not
  labels.
- `n_t` (near-miss window) is a free parameter; mackenzie-2025 uses `n = 2` and notes
  the metric becomes noisier with larger `n`.
- Units are discrete positions (sentences/paragraphs); for chunklabel's
  character-level spans, a unit must be chosen.
