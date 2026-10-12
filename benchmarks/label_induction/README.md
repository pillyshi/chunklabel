# Label set induction with `ZeroShotPromptSegmenter.fit` (issue #18)

`fit(X)` splits every text with free labels, then merges the label names over all of
`X` with `Normalizer`. `predict` then segments with the merged label set (`labels_`) as
a closed set. As decided in `research/ideas/sklearn-estimator-api.md`, the target is
fidelity only. Agreement with the PubMed roles is a diagnostic: how close the induced
set comes to one familiar grouping, not a measure of label quality.

## Setup

- Same 40 PubMed 20k RCT test abstracts (seed 0) and models as `benchmarks/fidelity`
  (Qwen2.5-7B Q4_K_M, Gemma 4 E4B QAT q4_0, llama.cpp, M2 16 GB, temperature 0,
  `n_ctx` 8192).
- fit's split step replays the one_pass outputs of `benchmarks/fidelity` (current
  prompt, regenerated 2026-10-12), so the free labels are exactly those scored there.
  The LLM is called for `Normalizer` (once per model) and `predict` (once per abstract).
- `on_align_error="skip"`. Gaps join the preceding segment.
- Agreement: `chunk_agreement(by="category")` on non-whitespace characters, averaged
  over abstracts. Label accuracy does not apply, because induced names differ from
  the roles.
- `results.json` has the aggregates and both induced label sets (`label_sets`).

## Results

| model | step | label set size | segments / doc | labels / doc | homogeneity | completeness | ARI | verbatim | gap |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | one_pass (free labels, per text) | 246 free | 10.8 | 9.8 | 0.960 | 0.572 | 0.476 | 99.3% | 4.7% |
| Qwen2.5-7B | fit | 144 | 10.1 | 8.8 | 0.944 | 0.619 | 0.527 | (same as one_pass) | |
| Qwen2.5-7B | predict with `labels_` | 144 | 10.3 | 7.8 | 0.935 | 0.659 | 0.600 | 99.3% | **8.3%** |
| Gemma 4 E4B | one_pass (free labels, per text) | 314 free | 12.2 | 11.1 | 0.997 | 0.559 | 0.457 | 100% | 0.0% |
| Gemma 4 E4B | fit | 246 | 12.2 | 10.3 | 0.997 | 0.584 | 0.501 | (same as one_pass) | |
| Gemma 4 E4B | predict with `labels_` | 246 | 12.2 | 9.9 | 0.977 | 0.595 | 0.522 | 100% | 0.2% |
| (reference) | given 5 roles | 5 | 6.5–11.7 | 4.2–4.4 | | | 0.80–0.84 | | |

The one_pass rows come from `benchmarks/label_agreement` (`results_given_labels.json`).

Paired per-abstract ARI change (bootstrap 95% CI, 5000 resamples):

| model | fit − one_pass | predict − fit |
|---|---|---|
| Qwen2.5-7B | +0.051 [+0.027, +0.078] (better/worse 20/4) | +0.073 [+0.026, +0.119] (27/10) |
| Gemma 4 E4B | +0.044 [+0.023, +0.068] (16/0) | +0.021 [−0.018, +0.061] (17/17) |

`Normalizer` took 207 s (Qwen) and 241 s (Gemma) for one call. predict took 43–48 s per
abstract, a little slower than one_pass (about 35 s), because the prompt lists every
label.

## Findings

1. **Normalizer merges little.** 246 → 144 labels (Qwen) and 314 → 246 (Gemma). It
   merges spelling variants and modifiers (`adverse-events` / `adverse effects` →
   `adverse events`, `MACS results` / `additional_results` → `results`), but keeps
   content-specific labels (`MMSE score`, `antibody`, `TAC levels`). The set is
   still about 30–50 times larger than the 5 roles. This is the decided behaviour
   (no target number of labels), but it means the label set is a cleaned-up
   inventory, not a schema.
2. **Merging across the corpus helps a little and never hurts much.** ARI rises by
   about 0.05 for both models; Gemma got worse on no abstract.
3. **Re-labeling with the closed set helps Qwen, not clearly Gemma.** With `labels_`,
   Qwen reuses labels more within an abstract (7.8 labels/doc against 8.8), and ARI
   rises by another 0.07.
4. **Fidelity cost for Qwen: gaps double** (4.7% → 8.3%). In 8 abstracts more than 20%
   of the text is not quoted (up to 59%). The long label list in the prompt seems to
   make Qwen skip sentences. Gemma is unaffected (0.2%). In the segmenter, the gaps
   join the preceding segment, so their labels are guesses. `split_with_report`-style
   reporting for the segmenter would make this visible to users.
5. All induced-set results stay far below the given 5-role set (ARI 0.52–0.60 against
   0.80–0.84). Coarser grouping would need a label set much smaller than what
   `Normalizer` produces.

## Implications

- With the current `Normalizer`, `fit` gives a usable label inventory for mining:
  `mapping_` shows which free labels were merged, and `labels_` can be edited by hand
  and passed back as `labels`.
- If a schema-sized label set is wanted, the levers are a coarser normalisation step
  (target size, or several merge rounds) or a label set chosen from the inventory by
  the user. Both are outside #18.
- A per-text fidelity report for `ZeroShotPromptSegmenter` (gap coverage) would let
  users spot the Qwen-style skipped sentences.

## Caveats

- One run per model, 40 short abstracts, English only. Stability of `labels_` across
  samples and runs is not measured.
- 314 label names were the largest Normalizer input here. Larger inventories may not
  fit a small local model's context in one call.
