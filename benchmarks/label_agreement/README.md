# Label-aware agreement with PubMed RCT roles

Layer 2 of `research/ideas/evaluation-suite.md`: do chunklabel's chunks and free-form
categories group the text the way a reference with labeled segments does?

## Setup

- Same 40 PubMed 20k RCT test abstracts and the same raw LLM outputs as
  `benchmarks/fidelity` (Qwen2.5-7B Q4_K_M and Gemma 4 E4B QAT q4_0, llama.cpp, library
  prompts, temperature 0). No new LLM calls.
- Reference: contiguous sentences with the same role (background, objective, methods,
  results, conclusions) form one labeled chunk. On average 12.2 sentences, 4.1
  reference segments, and 4.0 distinct roles per abstract.
- Predictions: `align` (threshold 85) + `postprocess`, as in `ChunkLabeler.split`.
- Metrics: `chunklabel.eval.chunk_agreement` on non-whitespace characters, averaged
  over abstracts. `by=category` groups characters by category name, so separate
  chunks with the same name form one group. `by=chunk` treats every chunk as its own
  group (segmentation only).
  - homogeneity: each predicted group stays within one reference role.
  - completeness: each reference role stays within one predicted group.
  - ARI: 0 = chance, 1 = identical.
- two_pass raw outputs contain only boundaries (labels come from a later LLM call that
  was not run), so they are scored `by=chunk` only.

## Results

| method | by | chunks / doc | categories / doc | homogeneity | completeness | V | ARI |
|---|---|---|---|---|---|---|---|
| whole abstract as one chunk | – | 1.0 | 1.0 | 0.000 | 1.000 | 0.000 | 0.000 |
| every sentence its own chunk | – | 12.2 | 12.2 | 1.000 | 0.537 | 0.695 | 0.418 |
| gold role per sentence (ceiling) | category | 12.2 | 4.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| Gemma 4 E4B one_pass | category | 12.2 | 10.0 | 0.982 | 0.604 | 0.738 | 0.518 |
| Gemma 4 E4B one_pass | chunk | | | 0.998 | 0.541 | 0.697 | 0.418 |
| Gemma 4 E4B two_pass | chunk | 12.1 | – | 0.998 | 0.542 | 0.699 | 0.421 |
| Qwen2.5-7B one_pass | category | 10.6 | 9.1 | 0.960 | 0.616 | 0.741 | 0.536 |
| Qwen2.5-7B one_pass | chunk | | | 0.988 | 0.574 | 0.721 | 0.467 |
| Qwen2.5-7B two_pass | chunk | 12.2 | – | 1.000 | 0.540 | 0.698 | 0.417 |

Category names produced (one_pass, 40 abstracts):

- Qwen2.5-7B: 180 distinct names. Most common: results 31, conclusion 30, method 23,
  procedure 21, **initiation 20**, **outcome 20**, result 12, study_design 10.
- Gemma 4 E4B: 255 distinct names. Most common: conclusion 36, study_design 27,
  study_objective 19, results 17, measurement 12, method 12.
- Example (same abstract): Qwen `study_type, sample_size, treatment, observation,
  statistical_test, results_maropitant, results_placebo, comparison, ...`; Gemma
  `study_objective, study_design, subject_inclusion, intervention_protocol, ...`.

## Findings

1. **Both models chunk at sentence granularity.** Gemma produces as many chunks as there
   are sentences (12.2), and Qwen nearly so (10.6). by=chunk, both modes score the same
   as the trivial "every sentence" baseline (ARI ≈ 0.42). On these abstracts, the LLM's
   boundaries add nothing beyond sentence splitting.
2. **Boundaries rarely cross roles** (homogeneity 0.96–1.00). The errors are in grouping,
   not in cutting.
3. **Labels carry all the gain, and only a little of it.** Grouping by category lifts
   ARI from ≈ 0.42 to 0.52–0.54, against a ceiling of 1.0. The cause is label
   granularity: about 9–10 distinct categories per abstract against 4 roles. Labels are
   content-specific (`results_maropitant` vs `results_placebo`) and inconsistent in form
   (`result` / `results`, `method` / `methodology`).
4. **Prompt-example leakage:** Qwen used `initiation` and `outcome`, the examples in
   the system prompt (`"initiation", "obstacle", "outcome"`), 20 times each, including
   for medical abstracts where they do not fit naturally.

## Implications

- The levers are on the labeling side: a document-level label set (choose a small set
  of categories first, then assign), normalisation (`Normalizer`) within or across
  documents, or a granularity hint. Each can now be measured with this script by the
  completeness / ARI gain at fixed homogeneity.
- The prompt examples influence the label vocabulary. Consider removing them or making
  them domain-neutral, and measure the effect here.

## Caveats

- PubMed abstracts have short, conventional structure, and their roles are one valid
  grouping among several. A content-oriented grouping (`results_maropitant`) is not
  wrong for every use case, so low completeness means "finer than the reference", not
  necessarily "bad".
- 40 abstracts, two small models, English only, no confidence intervals.
