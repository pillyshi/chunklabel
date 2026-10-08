# Notes: tsipidi-2024

> Tsipidi, E., Nowak, F., Cotterell, R., Wilcox, E., Giulianelli, M.,
> Warstadt, A. (2024). Surprise! Uniform Information Density Isn't the Whole
> Story: Predicting Surprisal Contours in Long-form Discourse. EMNLP 2024.
> arXiv:2410.16062. Code: https://github.com/rycolab/surprisal-discourse

## Summary

Proposes the **Structured Context Hypothesis**: the per-token surprisal
contour of a document is partly determined by its hierarchical discourse
structure (paragraphs/sentences, or RST trees), not just noise around a
uniform rate (UID). Tested by regressing LM surprisal on structural
predictors.

## Method

- Surprisal `s(u_t) = −log p(u_t | u_<t)` from long-context 7B LMs
  (Yarn-Llama-2-7B-64k for English, a Mistral 7B for Spanish), conditioned on
  the full preceding document.
- Dependent variables:
  1. document surprisal (full-context);
  2. its rolling average (window 3/5/7 tokens);
  3. **PMI between a unit and its preceding context given the local context**
     = surprisal under local (current sentence / EDU) context − surprisal under
     full document context;
  4. the same PMI with no local context (document − unigram surprisal).
- Predictors: relative position within a unit, distance to nearest boundary,
  hierarchical position, parser PUSH/POP transitions; both for RST trees and
  prose structure.
- Bayesian linear regression, 5-fold CV, ΔMSE vs. a baseline (unit length +
  previous surprisal), permutation tests.
- Data: English RST-DT (347 WSJ docs), Spanish RST-DT.

## Results

- All structural predictor groups significantly improve fit over baseline
  (p < 0.001), but explain only a **small portion** of the variance.
- Relative position within a discourse unit is the best predictor; its effect
  is better modelled as monotonic than as a "near-boundary" U-shape.
- RST (deeper hierarchy) beats paragraph/sentence prose structure.

## Key findings relevant to chunklabel

- Supports the core assumption of a surprisal-CPD approach: the surprisal
  series carries discourse-structure signal, and position within a unit
  shifts it systematically. A monotonic within-unit trend is consistent with
  zhao-2024's observation of declining PPL inside a logical unit.
- But the signal is weak relative to token-level noise. Raw token surprisal
  is a poor CPD input; aggregation (sentence-level mean, rolling windows)
  is needed.
- **The PMI quantity is a promising alternative series**:
  `PMI_i = s_local(x_i) − s_global(x_i)` measures how much the preceding
  document helps predict sentence `i`. At a topic/discourse boundary the
  preceding context should help *less*, so PMI should dip. This removes much
  of the sentence-intrinsic difficulty (rare words, length) that pollutes raw
  surprisal. Cost: one extra forward pass per sentence without context (or a
  short local context).

## Limitations

- Analysis, not a segmentation method: shows predictability of surprisal
  *from* known structure, not recoverability of structure *from* surprisal.
  The inverse problem is untested here.
- Only WSJ news and Spanish specialist texts; small corpora.
- Effects are statistically significant but small; boundary detection from
  surprisal alone may have low recall for subtle boundaries.

## Actionable ideas

- In `ideas/surprisal-cpd-boundaries.md`, evaluate three candidate series:
  raw sentence-mean surprisal, rolling-average surprisal, and
  context-PMI (`s_local − s_global`).
