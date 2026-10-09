# Evaluation suite for chunklabel: fidelity, label-aware agreement, tolerant boundaries

**Status**: Draft

## Motivation

`chunklabel.eval` now has Pk and WindowDiff, but the Wiki-50 experiments showed they do
not fit chunklabel:

- **Single granularity.** chunklabel leaves granularity to the LLM. Qwen2.5-7B placed
  13 boundaries per document where Wiki-50 has 3.7, and Pk scored that worse than
  placing none (0.489 vs 0.396). This measures a granularity mismatch, not quality.
  (Quick run: sentence-index prompting, 37 Wiki-50 documents of ≤ 8000 characters,
  2026-10-09; not committed.)
- **No labels.** Half of chunklabel's output (free-form categories) is ignored.
- **No fidelity.** The cost of the quote-based design (paraphrased quotes, rapidfuzz
  alignment, `uncategorized` gaps) is invisible.

Quote-based extraction is kept deliberately: it allows sub-sentence cuts and needs no
sentence splitter. The evaluation should therefore measure its costs directly
rather than replace it with sentence-index methods.

## Evidence

- `notes/fournier-2013.md` [fournier-2013]: Pk/WD are biased towards few or clustered
  boundaries, asymmetric, and use an ill-defined `k`. This matches our Wiki-50 result
  where "no boundaries" beat most methods. Boundary similarity B and B-precision /
  B-recall give partial credit for near misses, and report over- and
  under-segmentation separately.
- `notes/mackenzie-2025.md` [mackenzie-2025]: LLM topic segmentation evaluated with B
  (n = 2) plus BP/BR. Embedding methods win on concatenated texts and lose on natural
  ones, the same pattern as our Choi vs Wiki-50 probe.
- `notes/vinh-2010.md` [vinh-2010], [rosenberg-2007]: label-name-free comparison of
  two partitions (V-measure: homogeneity / completeness; ARI / AMI). Unadjusted
  measures inflate with the number of clusters, so finer chunking looks better;
  adjusted measures or the separate homogeneity/completeness components avoid this.
  At character level, N/K ≥ 100 usually holds.
- `notes/dernoncourt-2017.md` [dernoncourt-2017]: PubMed 20k RCT, short abstracts
  with contiguous role segments (background, objective, method, result, conclusion).
  These are functional segments like chunklabel's README example.
- `notes/arnold-2019.md` [arnold-2019]: WikiSection, sections with normalised topic
  labels. Same output shape as chunklabel (segment + category), longer documents.

## Proposed Scope

Three layers, in order of cost.

### 1. Fidelity (no gold data; runs on any text)

Computed from `RawChunk` + `align` output. Expose as
`chunklabel.eval.fidelity(raw_chunks, text, threshold)` or as diagnostics returned
alongside `split()`:

- `exact_rate`: fraction of quotes found verbatim (`text.find` hit).
- `fuzzy_rate`: fraction aligned only by rapidfuzz, with the distribution of scores.
- `unaligned_rate`: fraction below the threshold (dropped with `on_error="skip"`).
- `gap_coverage`: fraction of characters in `uncategorized` gap chunks (excluding
  whitespace-only gaps).
- `overlap_trimmed`: characters removed by overlap resolution.

Use case: **choose the default `fuzzy_threshold`** (currently 80). Run `split()` over a
corpus with alignment scores logged, inspect fuzzy matches by score band, and pick the
threshold where false alignments start to appear.

### 2. Label-aware agreement (needs gold segments + labels)

Give each non-whitespace character its gold label and its predicted category, then
report:

- homogeneity, completeness, V-measure [rosenberg-2007];
- ARI and AMI [vinh-2010].

No mapping between label names is needed. Datasets: PubMed 20k RCT test set (short,
functional) and WikiSection en_disease / en_city test sets (longer, topical).
Optionally also run after `Normalizer` to measure its effect.

### 3. Granularity-tolerant boundaries (needs gold segments)

Boundary similarity B, B-precision, and B-recall [fournier-2013] at sentence or word
level, via the `segeval` package as an optional dependency. Keep Pk/WD for
comparability with the literature, but do not use them for decisions.

## Acceptance Criteria

- Layer 1 runs on chunklabel output without gold data and has unit tests (exact,
  fuzzy, unaligned, gap cases).
- A report of layer-1 metrics for at least one local model (llama.cpp, Qwen2.5-7B) on
  a fixed text set, with a recommended `fuzzy_threshold` and the evidence for it.
- Layer 2 metrics are implemented with tests on toy inputs where label names differ
  but partitions are identical (score 1.0), and on partitions that differ only in
  granularity (completeness drops, homogeneity stays).
- `one_pass` and `two_pass` scored on PubMed 20k RCT (test subset) with layers 1–3.

## Out Of Scope

- Changing the extraction approach (sentence-index boundaries are noted as prior art
  in mackenzie-2025 but are not pursued, because they require sentence splitting).
- LLM-as-judge evaluation of label quality.
- Building a new annotated corpus.

## Open Questions

- Unit for layers 2–3: characters (natural for chunklabel spans; weights long chunks)
  or tokens/words (closer to the literature)?
- How should `uncategorized` gaps count in layer 2: as their own cluster, or
  excluded?
- Is `segeval` still maintained and compatible with Python ≥ 3.10, or should B be
  re-implemented?
- PubMed RCT license and WikiSection CC BY-SA: fine for local evaluation, but check
  before committing any derived files.
- Granularity control: should `split()` take a granularity hint (e.g. target number
  of chunks) so that evaluation against a fixed-granularity gold standard is fair?

## Layer 1 Results (2026-10-09)

`chunklabel.eval.fidelity` and `chunklabel.alignment.align_detailed` were added. See
`research/experiments/fidelity/README.md`.

- On 40 PubMed abstracts, Gemma 4 E4B quoted 100% verbatim and Qwen2.5-7B 99.5%.
  The larger loss is coverage: Qwen one_pass left 5.5% of the text unquoted.
- The current fuzzy scorer has a length-dependent ceiling (`2L / (2L + 20)`). At
  threshold 80, no non-verbatim quote under ~40 characters can align.
- Proposed (needs a decision): switch to `fuzz.partial_ratio_alignment` with
  default threshold 85, require exact matches for very short quotes, strip quotes
  before the exact search, and fall back to a global exact search for out-of-order
  verbatim quotes.
