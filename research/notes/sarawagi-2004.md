# Notes: sarawagi-2004

> Sarawagi, S., Cohen, W. W. (2004). Semi-Markov Conditional Random Fields for
> Information Extraction. *NIPS 2004* (Advances in Neural Information Processing
> Systems 17), pp. 1185–1192.

## Summary

Introduces semi-Markov CRFs (semi-CRFs): a conditionally trained model that outputs a
*segmentation* of the input sequence, where labels are assigned to segments rather
than to individual tokens. Features can describe a whole segment (its length, the
phrase inside it, its similarity to dictionary entries), which a token-level CRF
cannot express directly.

## Problem formulation

- A segmentation is `s = <s_1, ..., s_p>`, with each segment `s_j = <t_j, u_j, y_j>`
  (start, end, label), and `y_j` from a **fixed label set** `Y`.
- Segments have positive length, **do not overlap, and cover the whole sequence**
  (`t_1 = 1`, `u_p = |x|`, `t_{j+1} = u_j + 1`).
- For NER, non-entity words are put in unit-length segments labeled `O`.
- Model: `Pr(s | x, W) ∝ exp(W · Σ_j g(y_j, y_{j-1}, x, t_j, u_j))`. Segment features
  may depend only on the segment and the previous segment's label (semi-Markov
  assumption).

## Inference and learning

- Decoding is a semi-Markov Viterbi: `V(i, y) = max_{y', d=1..L} V(i-d, y') + W·g(y, y', x, i-d+1, i)`.
  Cost is linear in the maximum segment length `L` on top of a normal CRF; polynomial
  even when `L` is unbounded (`L ≤ |x|`).
- Training maximizes conditional log-likelihood (convex), with a forward-backward
  style dynamic program for the partition function and feature expectations.
- With fixed `L`, semi-CRFs are no more expressive than order-`L` CRFs, but much
  cheaper (they only consider runs where all `L` positions share one label).

## Experiments

- Five NER tasks on three small corpora (Address: city/state; Jobs: company/title;
  Email: person). `L` was set per dataset from observed entity lengths (4–6).
- Baselines: CRF/1 (I/O tags) and CRF/4 (B/E/C/U tags), also order-2/3 CRFs.
- Without dictionaries, semi-CRF is best on all five tasks (e.g. state 25.6 vs 20.8
  / 15.0 F1). Segment-level dictionary similarity features help semi-CRFs, while the
  token-level versions often hurt CRFs.

## Key findings relevant to chunklabel

- **Same problem shape as chunklabel**: partition text into non-overlapping, fully
  covering segments and give each a label. The paper itself calls this output a
  "segmentation" and notes it also applies to NP chunking.
- **What differs is the setting, not the structure**: semi-CRFs are supervised and use
  a fixed label set; chunklabel has no training data and lets the LLM name labels
  freely. chunklabel can be framed as this problem with both assumptions removed.
- **Decoding idea**: chunklabel currently takes the LLM's segmentation as is. The
  semi-Markov Viterbi recursion is a ready-made way to pick the best full
  segmentation if candidate spans get scores (e.g. from LLM log-probabilities or
  several sampled outputs). See `research/ideas/boundary-confidence-score.md`.
- **Segment-level features matter**: the gain comes from scoring whole segments,
  which is closer to how an LLM judges a chunk than token-level tagging is.

## Limitations

- Segments in the experiments are short (`L` = 4–6 tokens); chunklabel segments are
  sentences or paragraphs, so a naive `O(|x| · L · |Y|)` decoder over tokens is large.
  Working over sentence units instead of tokens would keep it small.
- Small datasets, binary I/O labeling per task; no multi-class segment labeling is
  tested.
- Needs labeled training data and a fixed label set, neither of which chunklabel has.
