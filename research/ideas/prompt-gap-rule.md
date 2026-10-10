# Revisit the "gaps are acceptable" rule in the split prompts

**Status**: Draft

## Motivation

`SPLIT_SYSTEM` and `BOUNDARY_SYSTEM` both say:

> Gaps between chunks are acceptable; do not force the entire text into chunks.

This explicitly invites the model to skip text. Skipped text is filled as
`uncategorized` by `postprocess`, so it carries no category, and from the user's side
it looks like a fidelity loss.

## Evidence

- `benchmarks/fidelity/README.md`: with the same prompt, Qwen2.5-7B one_pass left
  5.4% of 40 PubMed abstracts unquoted (6.7% after the example labels were removed),
  while Gemma 4 E4B left 0.1–0.2%. two_pass gaps were 0% for both models.
- `benchmarks/label_agreement/README.md`: both models already chunk at roughly
  sentence granularity, so gaps are not needed to avoid over-long chunks.
- No literature evidence. This idea comes from our own benchmarks only.

## Proposed Scope

1. Decide the intended behaviour: should every part of the input belong to a chunk
   (gaps only for whitespace and boilerplate), or are gaps a feature (e.g. skipping
   noise such as headers, citations, or tables)?
2. If full coverage is intended, replace the rule with one that asks for complete
   coverage, e.g. "Cover the whole text: every part of the input should belong to
   exactly one chunk, in order."
3. Measure with `benchmarks/fidelity` (gap coverage, verbatim rate) for both models,
   both modes, before and after.

## Acceptance Criteria

- The intended gap behaviour is documented in the README ("Lenient mode").
- Gap coverage for Qwen2.5-7B one_pass drops, and the verbatim rate does not drop,
  on the 40-abstract benchmark.

## Out Of Scope

- Model-specific prompt tuning: results are model-dependent by nature, and users check
  fidelity on their own model with `split_with_report`.
- Changing `postprocess`'s gap filling.

## Open Questions

- Was the rule added on purpose, for example to stop models from forcing
  noise into chunks or to save output tokens? It has been in the prompt since the
  first implementation (eaff719, 2026-04-18, then named "seam"), and the commit
  records no rationale. Only the author can say.
- Does demanding full coverage push models to paraphrase or merge text to "fill"
  chunks, which would lower the verbatim rate?
