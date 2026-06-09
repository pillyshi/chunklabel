# Notes: hearst-1994

> Hearst, M. A. (1994). Multi-paragraph segmentation of expository text. *ACL 1994*, pp. 9–16.

## Summary

TextTiling partitions expository text into multi-paragraph subtopic segments using
lexical cohesion alone — no thesaurus, no knowledge base. The core insight is that
a drop in vocabulary overlap between adjacent text windows signals a topic boundary.

## Algorithm (3 steps)

**1. Tokenization**
Text is divided into pseudo-sentences of w=20 tokens (not true sentences, to avoid
sentence-length normalization problems). Stop words are removed. Token-sequence
position and frequency are recorded.

**2. Similarity determination**
For each gap between token-sequences, a cosine similarity score is computed between
two adjacent blocks of k=6 token-sequences. The window slides across the entire text;
each token-sequence participates in 2k similarity computations. Scores are smoothed
with a window-3 average.

**3. Boundary identification**
Rather than cutting at valley minima directly, a *depth score* is computed for each gap:
```
depth(i) = (peak_left - score_i) + (peak_right - score_i)
```
Peaks are found by walking left/right from i as long as scores increase. Gaps are
ranked by depth score and a boundary is placed when `depth > mean - σ/2`. A
proviso prevents boundaries fewer than 3 token-sequences apart.

## Evaluation

- 13 magazine articles (1800–2500 words), 7 human judges each.
- Boundary = 3+ judges agree (majority). 41% of paragraph gaps are "true" boundaries.
- Block comparison: precision 0.66, recall 0.61 (vs random baseline ≈ 0.43/0.42).
- Allowing ±1 paragraph tolerance raises precision to 0.83, recall 0.78.

## Key findings relevant to chunklabel

**Depth score → boundary_score**: The depth score is a direct analogue to the
`boundary_score` idea. For chunklabel, instead of vocabulary overlap, use cosine
similarity of chunk embeddings; instead of token-sequence windows, use adjacent
chunk pairs.

**Adaptive boundary count**: The `mean - σ/2` threshold for auto-determining segment
count is elegant and parameter-free. chunklabel currently requires the LLM to decide
segment count implicitly; this heuristic could serve as a post-hoc quality check.

**Paragraph-aligned boundaries**: TextTiling snaps valleys to the nearest true paragraph
gap. chunklabel's rapidfuzz alignment does something analogous — snapping LLM quotes
to actual source text positions.

**Thesaural information degrades performance**: The paper notes that adding WordNet
information *hurt* results compared to raw term frequencies. This suggests that
semantic enrichment is not always better; a strong baseline using direct text overlap
may outperform a noisier semantic signal.

## Limitations

- Term-frequency similarity misses synonymy and polysemy.
- Fixed pseudo-sentence length (w) and block size (k) are global parameters; locally
  variable text density (dense argument vs sparse narrative) hurts performance.
- Evaluated on a single genre (magazine science articles). Chunklabel targets more
  heterogeneous text types.

## Actionable ideas

- Implement depth score as `boundary_score` on `Chunk` (see `ideas/boundary-confidence-score.md`).
- Use `mean - σ/2` threshold as a sanity check on the number of chunks the LLM produces.
