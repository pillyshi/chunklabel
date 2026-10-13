# FewShotPromptSegmenter on PubMed RCT (issue #19)

Does showing labeled examples change how the LLM segments and labels? Here `y` is a
correct answer, so agreement with the gold roles is the metric. With the current small
local models the numbers are a smoke test of the method, not a performance claim.

## Setup

- Test: the same 40 PubMed 20k RCT test abstracts (seed 0) as `benchmarks/fidelity`
  and `benchmarks/label_agreement`.
- Example pool: 200 abstracts sampled from the train split (seed 0). Their gold
  segmentations are contiguous sentences with the same role, labeled with the
  lowercased role (background, objective, methods, results, conclusions).
- Models: Gemma 4 E4B QAT q4_0 and Qwen2.5-7B Q4_K_M, llama.cpp on an M2 16 GB,
  temperature 0, `n_ctx` 8192. Library prompts, `on_align_error="skip"`.
- Selection: `random` (same examples for every input, `random_state=0`) and `similar`
  (default character 3-gram TF-IDF).
- Reduced grid (about 3 hours): both models with `random:3` and `similar:3`. Gemma
  also ran `similar:1` and `similar:5`. Qwen `n_examples` 1 and 5 were not run.
- Baseline: `ZeroShotPromptSegmenter` given the five role names, from
  `benchmarks/label_agreement` (same models and abstracts).
- Metrics: `chunk_agreement` on non-whitespace characters. `by=category` groups by
  label, `by=chunk` treats every segment as its own group (boundaries only). Label
  accuracy is the share of characters labeled with the gold role.

## Results

| model | method | segments / doc | ARI by category | ARI by chunk (boundaries) | label accuracy | verbatim | gap | s / doc |
|---|---|---|---|---|---|---|---|---|
| (reference) | gold | 4.1 | 1.000 | 1.000 | 1.000 | | | |
| Gemma 4 E4B | zero-shot, given labels | 11.7 | 0.798 | 0.445 | 0.839 | 100% | 0.3% | ~35 |
| Gemma 4 E4B | few-shot random:3 | 4.2 | 0.770 | **0.775** | 0.829 | 99.4% | 0.2% | 41 |
| Gemma 4 E4B | few-shot similar:1 | 4.2 | 0.762 | 0.768 | 0.813 | 98.8% | 0.9% | 36 |
| Gemma 4 E4B | few-shot similar:3 | 4.2 | 0.763 | 0.765 | 0.822 | 99.4% | 0.1% | 51 |
| Gemma 4 E4B | few-shot similar:5 | 4.1 | 0.783 | **0.785** | 0.821 | 100% | 0.8% | 56 |
| Qwen2.5-7B | zero-shot, given labels | 6.5 | 0.840 | 0.697 | 0.862 | 97.7% | 3.5% | ~35 |
| Qwen2.5-7B | few-shot random:3 | 4.1 | 0.719 | 0.735 | 0.745 | 98.1% | **13.7%** | 37 |
| Qwen2.5-7B | few-shot similar:3 | 4.0 | 0.772 | 0.778 | 0.783 | 98.1% | **12.0%** | 53 |

One Qwen `random:3` call failed: after 401 s the output was invalid JSON (a control
character). It is excluded (39 docs).

Paired per-abstract change against zero-shot with given labels (bootstrap 95% CI,
5000 resamples):

| model | method | ARI by chunk | ARI by category | label accuracy |
|---|---|---|---|---|
| Gemma 4 E4B | random:3 | +0.330 [+0.265, +0.392] | −0.028 [−0.058, −0.001] | −0.010 [−0.034, +0.012] |
| Gemma 4 E4B | similar:3 | +0.320 [+0.255, +0.384] | −0.035 [−0.068, −0.004] | −0.017 [−0.043, +0.008] |
| Gemma 4 E4B | similar:5 | +0.340 [+0.274, +0.402] | −0.015 [−0.044, +0.013] | −0.017 [−0.052, +0.015] |
| Qwen2.5-7B | random:3 | +0.038 [−0.090, +0.155] | −0.124 [−0.233, −0.029] | −0.120 [−0.214, −0.040] |
| Qwen2.5-7B | similar:3 | +0.081 [−0.054, +0.201] | −0.068 [−0.180, +0.027] | −0.079 [−0.172, −0.001] |

## Findings

1. **Examples teach the granularity.** With examples, both models produce about 4.1
   segments per abstract, the same as the reference. Zero-shot, Gemma cut every
   sentence (11.7). Gemma's boundary agreement (ARI by chunk) rose from 0.45 to
   0.77–0.79, in 37–38 of 40 abstracts. A label set alone cannot convey this.
2. **Labels did not improve.** Label accuracy and ARI by category stayed at or slightly
   below the zero-shot level. The remaining errors are the same adjacent-role
   confusions as before.
3. **Qwen skips text with examples.** Gaps rose from 3.5% to 12–14%. In 5–6 abstracts
   more than 20% of the text was not quoted, up to 90%. The gaps join the preceding
   segment, so their labels are guesses, and this explains Qwen's lower accuracy.
   With long label lists in #18, Qwen also skipped more. Longer prompts seem to make it
   stop early.
4. **random vs similar, and 1 vs 3 vs 5 examples: no clear difference** at this size.
   PubMed abstracts share one structure, so similarity by topic (character n-grams)
   should not matter much here. Datasets with more varied structure are the place to
   test it.
5. Cost: each example adds prompt tokens. With 5 examples, a call took 56 s instead of
   about 35 s.

## Caveats

- Reduced grid, one run per configuration, 40 short abstracts, English only, two small
  models.
- The pool and the test set come from the same corpus and have the same structure, the
  easiest case for few-shot prompting.
