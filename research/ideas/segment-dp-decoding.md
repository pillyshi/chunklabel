# Segment-level DP decoding with a length prior

**Status**: Draft

## Motivation

chunklabel takes the LLM's segmentation as is. There is no global choice among
alternative segmentations, no confidence, and no control over segment length. Users
want different granularities (a few sentences vs whole sections), and the LLM picks
one implicitly; on PubMed RCT both tested models chunk at sentence level
(`benchmarks/label_agreement/README.md`).

Semi-Markov models solve this by scoring candidate segments and choosing the best full
segmentation with dynamic programming. The same decoder can sit on top of LLM scores.

## Evidence

- `catalog.yaml` [sarawagi-2004]: semi-Markov Viterbi
  `V(i, y) = max_{y', d<=L} V(i-d, y') + score(y, y', i-d+1, i)`; cost is only linear
  in the maximum segment length `L`. A Gaussian segment-length feature
  `(d - mu)^2` lets the model prefer a typical length.
- `catalog.yaml` [kong-2015]: scores every candidate span independently, then a DP
  picks a consistent segmentation; segment-level scoring beats BIO tagging.
- `research/ideas/boundary-confidence-score.md`: DP marginals would give a
  per-boundary confidence.

## Proposed Scope

1. Units are sentences, not tokens, so the number of candidate spans stays small
   (`O(n * L)` for `n` sentences and maximum length `L` sentences).
2. Candidate spans come from several LLM outputs (sampling at temperature > 0, or
   one_pass and two_pass), not from all `O(n^2)` spans. A span's score is how often it
   appears, optionally plus a length prior.
3. Decode with the semi-Markov Viterbi recursion over sentence boundaries, with an
   optional length prior parameter (target sentences per chunk).
4. Labels: use the label most often given to the chosen span (free mode) or the given
   label set.

## Acceptance Criteria

- With one LLM output and no length prior, the decoder returns that output unchanged.
- With a length prior, average chunk length moves toward the target on PubMed RCT and
  one longer-document dataset.
- Fidelity (coverage, verbatim quotes) is not worse than `split`.

## Out Of Scope

- Scoring all `O(n^2)` spans with LLM log-probabilities (possible later, costly).
- Training any scoring model.
- Hierarchical (nested) segmentation.

## Open Questions

- How many samples are needed before agreement counts are a useful score?
- Is the sentence splitter reliable enough for the target genres (transcripts with poor
  punctuation)?
- In free-label mode the label set cannot be enumerated, so labels are chosen after the
  segmentation rather than jointly. Is that good enough?
