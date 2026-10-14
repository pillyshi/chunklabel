# Fill skipped text with a second LLM call

**Status**: Promoted to issue #6 (rewritten 2026-10-14). Implemented; results in `benchmarks/gap_fill/README.md`

## Motivation

The LLM sometimes skips part of the text: it quotes nothing for it. Quotes that
cannot be aligned leave the same kind of hole. Today the skipped text is handled
like this:

- `ChunkLabeler` marks it as `uncategorized`.
- The segmenter estimators join it to the preceding segment.

The second way is invisible, and usually wrong for long skips. #20 made the problem
measurable (`gap_coverage`). This idea fixes it: send only the skipped text back to
the LLM and label it.

Issue #6 (2026-05) proposed to enforce exact matches and, on a mismatch, either
repair the offset locally or retry the request. Since then:

- Alignment searches exact, then exact anywhere, then fuzzy. The probes found
  99–100% of quotes verbatim (`benchmarks/fidelity`).
- `on_align_error="skip"` exists.

So mismatched quotes are rare. The failure that remains is skipped text.

Retrying the whole request does not help with local models. `LlamaCppClient` uses
temperature 0 by default (#15), so the same prompt is expected to give the same answer. This is
expected, not measured. A smaller prompt with only the skipped text is a different
request.

## Evidence

Probe on 2026-10-14. The inputs were the Qwen2.5-7B Q4_K_M outputs of
`benchmarks/few_shot` (#19): 40 PubMed abstracts, 3 examples, random or similar
selection. In each output, I took every gap between aligned quotes with at least 20
non-whitespace characters. I sent each gap's text alone to the same model, with the
five role labels and no examples (`extract_labeled_chunks`), and aligned the quotes
within the gap.

| selection | abstracts with gaps | gaps | gap text (share of all text) | still uncovered after the fill | label accuracy on gap text: join preceding / fill | label accuracy per abstract: join preceding / fill | extra time per abstract |
|---|---|---|---|---|---|---|---|
| random | 13 / 39 | 15 | 12.8% | 10.1% | 0.115 / 0.627 | 0.745 / 0.814 | 5.2 s |
| similar | 15 / 40 | 19 | 10.8% | 0.5% | 0.180 / 0.539 | 0.783 / 0.824 | 4.6 s |

- **Joining the preceding segment almost always mislabels a long gap.** Abstracts change
  role at each part: background, then methods, then results. Text that the model
  skipped near the end gets the label of an earlier part.
- **Where the gaps were.** Out of 34:
  - 12 were at the end of the text.
  - 9 were at the start. A gap at the start joins the first segment.
  - 13 were in the middle.

  The median gap was 218 characters and the largest 1,436.
- **Every fill call returned chunks.** One call per gap. Each gap is short, so the
  call is cheap.
- **Gemma 4 E4B left less than 1% unquoted** in the same benchmarks, so it would make
  almost no fill calls.
- **Limits of the probe.** It used one model and one dataset, and labels came from a
  zero-shot prompt with no examples. As with the other benchmarks, these numbers are
  diagnostics, not targets.

No literature evidence. This comes from our own benchmarks.

## Proposed Scope

1. **Fill step.** After alignment, find each gap that has at least
   `min_gap_chars` non-whitespace characters. Default 20, which skips punctuation and
   short connectives. For each gap:
   - Call the LLM once, with the gap text alone and the same kind of request as the
     first call.
   - Align the returned quotes within the gap and shift their spans by the gap's start.
   - Merge them with the first call's spans before gaps are resolved.

   One round only: text the fill call also skips is handled as today.
2. **Which request the fill call uses.** The same kind of request as the first call,
   without examples:

   | first call | fill call |
   |---|---|
   | `ChunkLabeler.split`, one pass | `extract_chunks` |
   | `ChunkLabeler.split`, two passes | `extract_boundaries` (the labeling pass then covers the new chunks too) |
   | `ZeroShotPromptSegmenter.fit` without `labels` | free labels, before `Normalizer` |
   | given or fitted labels | `extract_labeled_chunks` with the same label set |
   | `FewShotPromptSegmenter` | `extract_labeled_chunks` with no examples |

   The fill call uses no examples, because the long few-shot prompts are what made
   Qwen skip.
3. **Parameters.**
   - `fill_gaps: bool` on `ChunkLabeler` and both estimators.
   - `min_gap_chars: int = 20`.
   - Default `True` for the estimators. They hide gaps, and nothing depends on their
     old behaviour (experimental since 0.5.0).
   - Default `False` for `ChunkLabeler`. It shows gaps as `uncategorized`, and turning
     the fill on changes its output for current users.
4. **Report.** `FidelityReport` gains `n_filled_gaps: int = 0`. Its other fields
   describe all quotes, from the first call and the fill calls together, so
   `gap_coverage` is what is left after the fill.
5. **Benchmark.** `benchmarks/gap_fill/` turns the probe into a script, with a README
   and aggregate results. Outputs stay outside the repository, as in the other
   benchmarks.

## Acceptance Criteria

- With a fake backend that skips a sentence, `fill_gaps=True` labels it from a second
  call. The segmentation and `gap_coverage` reflect the fill.
- Gaps below `min_gap_chars` make no extra call.
- With `fill_gaps=False`, output and the number of LLM calls are unchanged.
- The fill calls send no examples, and use the label set when there is one.
- Existing tests pass. Tests that count LLM calls on gapped outputs get an explicit
  `fill_gaps=False`.
- The benchmark reproduces the probe's direction: less unquoted text, and higher label
  accuracy on the gap text.

## Out Of Scope

- Several fill rounds.
- Context around the gap (the neighbouring sentences) in the fill prompt. The probe
  worked without it. Context could help with labels and could be an option later.
- Retrying a call that raised, for example on invalid JSON. That is client-level
  retrying.
- Local repair of offsets for quotes that do not match. Fuzzy alignment covers this
  now, and a quote that fails to align leaves a gap that the fill handles.

## Open Questions

- Should `ChunkLabeler` also default to `True` in a later release?
- Should a gap that is the whole text (the first call returned nothing usable) be
  filled? It would repeat the first request. Proposed: no. In that case the output
  is `[]`, as today.
