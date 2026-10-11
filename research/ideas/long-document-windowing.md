# Long documents: overlapping windows stitched by DP

**Status**: Draft

## Motivation

chunklabel targets relatively long segments, so its natural inputs (meeting
transcripts, reports, contracts) often exceed what a local model can handle well in
one call. Splitting the text into windows and processing each one independently leaves
artificial boundaries at the window edges and may give the same segment two labels.

## Evidence

- `catalog.yaml` [sarawagi-2004]: the semi-Markov DP picks one consistent,
  fully covering segmentation from scored candidate segments; candidates from
  overlapping windows can be combined by the same recursion.
- `catalog.yaml` [kong-2015]: an upper bound on segment length keeps span computation
  linear in sequence length, which is what makes long inputs feasible.

## Proposed Scope

1. Split the text into overlapping windows on sentence boundaries (window size and
   overlap as parameters).
2. Run `split` on each window and map spans back to document offsets.
3. In overlap regions, candidate segments from both windows compete; choose the final
   segmentation with the decoder from `segment-dp-decoding.md`. Without that decoder,
   a simple rule (prefer the window where the segment is farther from the edge) is the
   fallback.
4. Free-label mode: run `Normalizer` over all windows' labels so that one document
   uses one vocabulary.

## Acceptance Criteria

- For a document that fits in one window, the result equals plain `split`.
- No boundary is placed at a window edge unless one of the windows put it there away
  from its edge.
- Fidelity (coverage, verbatim quotes) is reported for long inputs.

## Out Of Scope

- Hierarchical summaries or multi-level segmentation.
- Choosing window sizes from the model's context length automatically.

## Open Questions

- Which dataset of long documents with labeled sections should be used for testing?
- How much overlap is enough for segments of the target length?
