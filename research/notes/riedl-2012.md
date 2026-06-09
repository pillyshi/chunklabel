# Notes: riedl-2012

> Riedl, M., & Biemann, C. (2012). TopicTiling: A text segmentation algorithm based
> on LDA. *ACL 2012 Student Research Workshop*, pp. 37–42.

## Summary

TopicTiling is TextTiling with LDA topic IDs substituted for word tokens. Each word
is annotated with its most probable topic ID (the "mode" across multiple Gibbs
inference runs), and adjacent sentence blocks are compared by cosine similarity over
their topic-ID frequency vectors. The result is state-of-the-art performance in
O(n) time.

## Algorithm

1. **Train an LDA model** on a corpus similar to the target domain (T=100 topics,
   α=50/T, β=0.01, 500 sampling iterations in training).
2. **Annotate target documents** with topic IDs via Gibbs inference (100 iterations).
   Run inference multiple times; assign each word the *mode* (most frequent) topic ID
   across all runs. This "mode" trick stabilizes the stochastic inference.
3. **Compute coherence score** between two adjacent windows of w sentences:
   represent each window as a T-dimensional vector (topic-ID frequency) and compute
   cosine similarity. Low similarity → candidate boundary.
4. **Detect boundaries** via depth scores (identical to TextTiling):
   `d_p = 0.5 * (hl(p) - c_p + hr(p) - c_p)`.
   If segment count n is given: take the n highest depth scores.
   Otherwise: apply threshold `depth > µ - σ/2`.

## Parameters

| Parameter | Role | Recommended value |
|-----------|------|-------------------|
| d (mode) | Stabilize topic IDs across inference runs | Always true |
| w (window) | Sentences per block | Corpus-dependent; w=2 for Choi (3–11 sent), w=5 for WSJ (full articles) |
| T | Number of LDA topics | 100 (standard) |

## Results

**Choi dataset** (700 artificial docs, 10 segments of 3–11 sentences):
- TopicTiling (d=true, w=2): Pk=0.95, WD=1.08 — far below prior SOTA (M09: Pk=2.3)

**Galley WSJ dataset** (500 docs, full articles, 4–22 segments):
- TopicTiling (d=true, w=5, filtered): Pk=11.89, WD=17.41 — below LCseg (Pk=12.21)

## Key findings relevant to chunklabel

**Mode stabilization applies to LLMs too.** The "run inference N times, take mode"
trick directly transfers: running chunklabel N times on the same text and taking the
most frequent category per span could reduce LLM stochasticity. Worth exploring as
an ensemble option for high-stakes use cases.

**Window size is corpus-dependent.** Short segments need small w; long articles need
larger w. chunklabel could expose a similar context window parameter controlling
how many surrounding sentences the LLM sees when labeling a chunk.

**O(n) linear time.** The DP approaches are O(n²); TopicTiling is O(n). chunklabel's
LLM calls are O(1) per document (single prompt), so this is less of a concern, but
the depth-score boundary detection step is also O(n) and can be kept that way.

**Topic model must match domain.** Training on in-domain data is essential; a generic
Wikipedia model underperforms for specialized corpora (Table 6 vs Table 3 results gap).
For chunklabel, this implies that a generic embedding model for boundary_score
computation may underperform on domain-specific text — a domain-specific model
should be preferred.

**Agglomerative coherence merging (Eisenstein 2009) is cited as the hierarchical
alternative.** This is the basis for the two-pass merge idea in
`ideas/two-pass-coherence-merge.md`.

## Limitations

- Requires a pre-trained LDA topic model; training is expensive and domain-dependent.
- Topic IDs are global (corpus-level) constructs; they do not adapt to individual
  document structure.
- The Choi dataset is partly contaminated (up to 10% of test sentences appear in
  training due to how it was generated).

## Actionable ideas

- The depth-score formula and µ - σ/2 threshold from TextTiling/TopicTiling can be
  directly implemented as the `boundary_score` computation
  (see `ideas/boundary-confidence-score.md`).
- Mode stabilization: run `ChunkLabeler.split()` N times, majority-vote categories
  per span. Add as an optional `ensemble_runs` parameter.
