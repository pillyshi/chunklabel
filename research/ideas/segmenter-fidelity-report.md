# Fidelity reports for the segmenter estimators

**Status**: Promoted to issue #20 (2026-10-14)

## Motivation

`ZeroShotPromptSegmenter` and `FewShotPromptSegmenter` join any text that no aligned
quote covers to the preceding segment. Every segmentation therefore looks complete,
even when the LLM skipped most of the text, and the labels of the skipped parts are
guesses. `ChunkLabeler` has `split_with_report` for this. The estimators have
nothing, so users cannot see the problem.

Fidelity is the evaluation target that was decided for the schema-free mode
(`sklearn-estimator-api.md`, 2026-10-12). Returning it to users is the natural
counterpart.

## Evidence

- `benchmarks/label_induction/README.md` (#18): with the induced 144-label set, Qwen2.5-7B
  left 8.3% of the text unquoted (4.7% with free labels). In 8 of 40 abstracts more
  than 20% was missing, up to 59%.
- `benchmarks/few_shot/README.md` (#19): with three examples, Qwen left 12–14%
  unquoted. In 5–6 abstracts more than 20% was missing, up to 90%. Its label accuracy
  fell, mostly because of this.
- Gemma 4 E4B stayed below 1% in both cases, so the problem depends on the model. Users
  need to check it on their own model and data, as `README.md` already advises for
  `ChunkLabeler`.
- No literature evidence. This comes from our own benchmarks.

## Proposed Scope

1. `predict_with_report(X)` on both estimators returns
   `(segmentations, reports)`. It gives one `chunklabel.eval.FidelityReport` per text,
   the same type that `split_with_report` returns. Like `split_with_report`, it never
   raises on alignment errors: unaligned quotes are counted and dropped.
2. `ZeroShotPromptSegmenter.fit` also stores `fit_reports_`, the reports of its
   own LLM pass over `X`. `fit` already makes that pass, so this costs nothing. It
   covers both the free-label pass and the given-label pass. `FewShotPromptSegmenter.fit`
   makes no LLM call and stores none.
3. The report is computed on the quotes after label mapping. With
   `on_unknown_label="drop"`, dropped quotes are not counted. This is documented.
4. README: show `predict_with_report` next to the existing fidelity check.

## Acceptance Criteria

- `predict_with_report(X)[0]` equals `predict(X)` when every quote aligns.
- A skipped sentence shows up as `gap_coverage > 0`, while the segmentation still
  covers the text.
- `fit_reports_` has one report per text in `X`.
- Existing tests pass unchanged.

## Out Of Scope

- Retrying when the gap is large (issue #6). The report is the prerequisite for it.
- Changing prompts to reduce skipping.

## Open Questions

- Should `predict` warn when a gap exceeds a threshold? Not by default: it would be
  noisy for models like Qwen. It can be decided later with #6.
