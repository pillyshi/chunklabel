# Label-aware agreement with PubMed RCT roles

**Status: diagnostic only** for the schema-free mode (`ChunkLabeler`). Chunking quality
is task-dependent, so those scores are not optimisation targets. See the decision in
`research/ideas/evaluation-suite.md`. For the given-label-set mode
(`ZeroShotPromptSegmenter`), the roles are the target; see the last section.

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

## Removing the example labels from the prompts (2026-10-10)

The example labels (`"initiation", "obstacle", "outcome"`) were removed from
`SPLIT_SYSTEM` and `LABEL_SYSTEM`. The rule now reads "Categories are free-form, short,
lowercase labels that describe what each chunk is about." one_pass was regenerated
for the same 40 abstracts (`results_no_prompt_examples.json`).

| model | prompt | distinct labels (40 docs) | categories / doc | leaked example labels | ARI by category | verbatim | gap |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | with examples | 180 | 9.1 | 41 | 0.536 | 99.8% | 5.4% |
| Qwen2.5-7B | without | 253 | 9.9 | 0 | 0.454 | 99.2% | 6.7% |
| Gemma 4 E4B | with examples | 255 | 10.0 | 0 | 0.518 | 100% | 0.2% |
| Gemma 4 E4B | without | 303 | 10.9 | 0 | 0.463 | 100% | 0.1% |

Paired per-abstract ARI change (bootstrap 95% CI, 5000 resamples):

- Qwen2.5-7B: −0.082 [−0.148, −0.021], worse/better/same = 19/13/8
- Gemma 4 E4B: −0.056 [−0.097, −0.021], worse/better/same = 15/5/20

Findings:

- Leakage is gone, but **agreement with the role grouping dropped for both models**,
  including Gemma, which never used the example words. Labels became more specific
  and less often shared within a document (by-category ARI is now close to the
  by-chunk ARI).
- Interpretation: the abstract, function-like examples nudged models towards coarser
  "role" labels. Two changes are confounded here, though: removing the examples, and
  the new wording "describe what each chunk is about", which itself invites
  content-specific labels.
- Neither version controls label granularity explicitly. Coarser, shared labels should
  be requested directly (a rule to reuse categories for chunks with the same role, or a
  document-level label set) rather than induced by example words.

## Given label set (2026-10-12, issue #17)

`ZeroShotPromptSegmenter(labels=[background, objective, methods, results,
conclusions])`: the five role names, without descriptions or examples, with the label
enum enforced by the llama.cpp JSON-schema grammar. Predictions follow
`ZeroShotPromptSegmenter.predict`: closest-label mapping, `align` (threshold 85), and
gaps joined to the preceding segment. one_pass was regenerated in the same run with the
current prompt (full-coverage rule) as the free-label comparison. Same 40 abstracts and
models (`results_given_labels.json`). `label accuracy` is the share of characters whose
label equals the gold role.

| model | mode | by | chunks / doc | categories / doc | homogeneity | completeness | V | ARI | label accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | one_pass | category | 10.8 | 9.8 | 0.960 | 0.572 | 0.707 | 0.476 | 0.070 |
| Qwen2.5-7B | given_labels | category | 6.5 | 4.2 | 0.898 | 0.883 | 0.887 | **0.840** | **0.862** |
| Qwen2.5-7B | given_labels | chunk | | | 0.934 | 0.767 | 0.827 | 0.697 | |
| Gemma 4 E4B | one_pass | category | 12.2 | 11.1 | 0.997 | 0.559 | 0.712 | 0.457 | 0.016 |
| Gemma 4 E4B | given_labels | category | 11.7 | 4.4 | 0.877 | 0.844 | 0.857 | **0.798** | **0.839** |
| Gemma 4 E4B | given_labels | chunk | | | 0.998 | 0.558 | 0.710 | 0.445 | |

Paired per-abstract ARI change (by category, given_labels − one_pass, bootstrap 95% CI,
5000 resamples):

- Qwen2.5-7B: +0.364 [+0.304, +0.421], better/worse = 38/2
- Gemma 4 E4B: +0.341 [+0.284, +0.398], better/worse = 40/0

Findings:

- **Giving the label set closes most of the gap to the ceiling.** ARI by category goes
  from ≈ 0.47 to 0.80–0.84, and 84–86% of characters get the gold role, with no
  examples and no descriptions.
- No label outside the set was returned (the enum was enforced), so the closest-label
  mapping was never used here.
- The models differ in segmentation. Gemma still cuts every sentence (11.7 chunks,
  by-chunk ARI 0.45, the same as the "every sentence" baseline); the shared labels do
  the grouping. Qwen merges consecutive sentences with the same role (6.5 chunks,
  by-chunk ARI 0.70), at a small cost in homogeneity.
- Most label errors are between adjacent, similar roles: background ↔ objective
  (46–47% of wrong characters), methods ↔ results (26–28%), and conclusions labeled
  results (16–17%). Label descriptions may help with these and are untested.
- Caveat: PubMed roles are a familiar scheme, and the role names may be in the
  models' training data. Results on a less conventional label set may be lower.
