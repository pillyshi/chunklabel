# Prompt-free boundary detection via surprisal / embedding change point detection

**Status**: Draft — primary approach rejected after Wiki-50 probe; see the end of this file

## Motivation

Both current modes ask a chat LLM to *generate* quotes, which it may paraphrase.
The rapidfuzz alignment step (`alignment.py`) and lenient post-processing
exist only to repair this, and `on_align_error` exposes the failure to users.

`two_pass` mode already separates the problem into boundaries
(`LLMBackend.extract_boundaries`) and labels (`LLMBackend.label_chunks`). If
boundaries came from a signal computed *over the source text* instead of from
generated quotes, the chunk spans would be character offsets by construction.
Then alignment would be unnecessary for this path, and the LLM would only
generate labels.

## Evidence

- `notes/zhao-2024.md` [zhao-2024]: a prompt-free perplexity series from a
  0.5B causal LM (sentence-mean token PPL conditioned on all preceding text)
  is a usable boundary signal. On RAG QA it is competitive with
  LumberChunker on a 14B model at ~1/30 of the time, and works with base
  models. Boundaries are placed at PPL *minima* with an ad hoc threshold rule.
- `notes/jia-2026.md` [jia-2026], [diazrodriguez-2025]: penalised kernel
  change point detection with PELT over sentence embeddings (Embed-KCPD) beats
  TextTiling / GraphSeg / Coherence on standard benchmarks without training.
  Its penalty `C·sqrt(T log T)` with elbow-selected `C` chooses the number of
  segments without labels, and it has consistency guarantees under
  short-range dependence.
- `notes/tsipidi-2024.md` [tsipidi-2024]: LM surprisal contours are
  significantly predicted by discourse structure, but the effect is small
  relative to token-level noise. They also define a context-PMI quantity
  (`surprisal_local − surprisal_global`) that factors out sentence-intrinsic
  difficulty.
- Algorithmic background: [killick-2012] (PELT), [truong-2019] (review /
  `ruptures`), [adams-2007] (online BOCPD).

## Proposed Scope

An experiment first, with a library change only if the experiment supports it.

1. **Signal extraction** (sentence-level series, `T` = number of sentences):
   - (a) `s_i` = mean token surprisal of sentence `i` given all preceding
     text, from a small GGUF causal LM via llama-cpp (`logits_all=True`,
     one forward pass).
   - (b) `pmi_i` = surprisal of sentence `i` with no or local-only context
     minus `s_i` (as in tsipidi-2024).
   - (c) sentence embeddings (Embed-KCPD baseline).
2. **Change point detection** with `ruptures`:
   - PELT with an L2 / normal cost on (a) and (b);
   - PELT with a cosine kernel on (c);
   - the zhao-2024 local-minima rule on (a) as a baseline;
   - penalty `C·sqrt(T log T)`, `C` chosen by elbow.
3. **Evaluation** on Choi and Wiki-50 with Pk / WindowDiff (depends on
   `ideas/eval-metrics-pk-windowdiff.md`). Compare against the current
   `two_pass` LLM boundaries on a small held-out set, including alignment
   failure counts.
4. If a variant is competitive, add it as a boundary source that
   `ChunkLabeler.split(mode="two_pass")` can use in place of
   `extract_boundaries` + `align`. Sentence index spans map directly to
   `Chunk.start` / `Chunk.end`. Labels come from the existing
   `label_chunks`.

## Acceptance Criteria

- A reproducible script (under `research/` or `examples/`, not shipped in the
  package) reports Pk / WindowDiff for the variants (a), (b), (c) and the
  minima rule on at least Choi and Wiki-50.
- Report of runtime per 1k sentences for each variant on CPU with a ≤1B GGUF
  model.
- Decision recorded in this file: which signal (if any) proceeds to a
  library change.
- If implemented, chunks from the new boundary source always satisfy
  `text[c.start:c.end] == c.quote` without calling `align`.

## Out Of Scope

- **Labeling strategy.** Two options are discussed below, but each is its own
  idea:
  - label each chunk independently, then normalise and merge adjacent
    same-label chunks (`Normalizer`, `ideas/two-pass-coherence-merge.md`);
  - pass the full list of chunk quotes to the LLM in one call and have it
    choose a small fixed label set first, then assign each chunk to one label.
- Hierarchical / multi-scale segmentation.
- Online / streaming segmentation (BOCPD); noted only as a later option.
- Fine-tuning any model.

## Open Questions

- **Sign of the signal** (resolved by the probe, see below). zhao-2024 cuts after PPL *minima*, while the
  intuition from surprisal theory is a *spike* at the first sentence of a new
  segment. A mean-shift CPD cost is agnostic to this, but a peak-picking rule
  is not. Check which pattern holds on the benchmarks.
- **Granularity mismatch.** Benchmarks are topic segmentation with
  multi-sentence segments. chunklabel's examples split at sub-sentence or
  single-sentence "functional" units (initiation / obstacle / outcome). Does
  surprisal carry signal at that scale? Do we need clause-level units?
- **Minimum segment length.** jia-2026's guarantees assume segments grow with
  `T`; short documents with 1-sentence chunks may be under-segmented by the
  penalty.
- **Model coupling.** Signal (a)/(b) needs token logprobs, which `OpenAIClient`
  cannot provide for arbitrary input text. This likely requires a new
  protocol separate from `BaseLLMClient` (e.g. a `LogprobScorer`) backed by
  llama-cpp or transformers.
- **Combining signals.** Is a multivariate CPD over `[s_i, pmi_i, embedding]`
  better than any single series?
- **Sentence splitting.** Which splitter to use (rule-based vs. neural sentence segmenters such as
  Segment Any Text) and whether its errors dominate boundary error.

## Probe Results (2026-10-09)

Details: `research/experiments/surprisal_cpd_probe/README.md` (Choi set 1, 100
documents, Qwen2.5-0.5B, all-MiniLM-L6-v2).

- **Sign resolved:** surprisal spikes (+0.97σ) and context-PMI dips (−1.11σ) on the
  *first sentence of the new segment* only. It is an impulse, not a level shift.
  Meta-Chunking's "cut after the minimum" rule is near random (Pk 0.43 vs 0.48).
- Mean-shift CPD on surprisal/PMI is the wrong model (Pk ≈ 0.40). Peak picking on
  −PMI reaches Pk 0.145 with the true K, or 0.188 with a z-threshold.
- Embedding + cosine KCPD gives Pk 0.036 with the true K, or 0.054 penalised, at
  ~1/100 of the compute. Naively appending PMI to the embeddings did not help.
- Caveat: every Choi boundary is also an unrelated document's opening sentence,
  which flatters both the spike and the embeddings.

**Provisional decision (superseded by the Wiki-50 results below):** use embedding + KCPD (PELT, `C·sqrt(T log T)`) as the
primary prompt-free boundary source. Keep surprisal/PMI as a secondary signal
(e.g. a boundary-confidence score, cf. `ideas/boundary-confidence-score.md`), and
revisit it on data with related, gradual segments before dropping it. Next check:
a non-Choi dataset (Wiki-50) and chunklabel-scale short texts.

## Wiki-50 Results (2026-10-09)

Details: `research/experiments/surprisal_cpd_probe/README.md`, Wiki-50 section.

- The Choi spike does **not** transfer. At section openings within a Wikipedia
  article, surprisal and PMI move only slightly (≈ −0.2σ).
- Nothing beats the no-boundary baseline (Pk 0.402) except embedding + KCPD with
  the true K (0.369, in line with jia-2026). Every label-free way of choosing K
  (penalty, z-threshold) degenerates to ≈ the trivial baseline or worse.

**Revised decision:** prompt-free boundary detection is not good enough to replace
LLM boundaries on realistic text. Surprisal/PMI as a primary boundary signal is
dropped. Embedding KCPD remains useful only as a candidate generator or a
cheap baseline.

The underlying goal, extractive boundaries with no paraphrase and no alignment,
can be met differently: number the sentences and have the LLM return **boundary
sentence indices** instead of quotes. This is still prompting, but the output
cannot be paraphrased. It is worth a separate idea candidate, evaluated with
`chunklabel.eval` against the current `two_pass` mode on Wiki-50.
