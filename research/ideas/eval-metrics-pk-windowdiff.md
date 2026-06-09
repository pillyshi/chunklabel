# Evaluation metrics: Pk and WindowDiff

**Status**: Draft

## Motivation

chunklabel has no quantitative baseline for segmentation quality. Without a
standard metric, it is impossible to compare prompt variants, measure
regressions, or benchmark against non-LLM baselines.

## Evidence

- `catalog.yaml` [beeferman-1999]: Introduces the Pk metric — a sliding-window
  measure of segmentation error. Standard in the field.
- `catalog.yaml` [pevzner-2002]: Identifies an asymmetry in Pk (false negatives
  penalised less than false positives) and proposes WindowDiff as a correction.
  Both metrics are now expected in segmentation evaluations.

## Proposed Scope

Add a `chunklabel.eval` module with two public functions:

```python
from chunklabel.eval import pk, window_diff

score = pk(predicted_chunks, reference_chunks, source_text)
score = window_diff(predicted_chunks, reference_chunks, source_text)
```

Input: lists of `Chunk` objects and the original source string.
Output: a float in [0, 1] (lower is better).

No new dependencies beyond the stdlib are needed (boundary arrays can be
computed from `Chunk.start` / `Chunk.end`).

## Acceptance Criteria

- `pk()` and `window_diff()` return the same values as the reference
  implementations on the standard test cases from the respective papers.
- A benchmark script runs against the WikiSection dataset and prints scores.
- Existing tests continue to pass.

## Out Of Scope

- Collecting or distributing annotated corpora.
- Integrating evaluation into the main `ChunkLabeler.split()` call path.
- Metrics beyond Pk and WindowDiff (e.g. F1 on boundaries).

## Open Questions

- Which publicly available corpus is best for an initial benchmark?
  WikiSection and the Choi dataset are candidates.
- Should the eval module be an optional extra (`chunklabel[eval]`) to avoid
  adding heavy dataset dependencies to the core package?
