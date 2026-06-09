# Notes: pevzner-2002

> Pevzner, L., & Hearst, M. A. (2002). A critique and improvement of an evaluation
> metric for text segmentation. *Computational Linguistics*, 28(1), 19–36.

## Summary

The paper identifies five structural flaws in the Pk metric (Beeferman et al. 1997/1999)
and proposes WindowDiff as a drop-in replacement that addresses all of them.

## Pk recap

Pk slides a probe of width k across the text. At each position i, it checks whether
both ends of the probe (positions i and i+k) are in the same segment according to both
the reference and the hypothesis. A penalty is incurred when they disagree. k is set
to half the average true segment length. Score ∈ [0, 1]; lower is better.

## The five problems with Pk

**Problem 1 — False negatives penalized more than false positives.**
A missed boundary (FN) is always penalized k times. A spurious boundary (FP) in the
middle of a segment is penalized only k/2 times on average, because the probe only
overlaps the false boundary half the time. Thus Pk is biased toward over-segmentation.

**Problem 2 — Number of boundaries within window is ignored.**
If the reference has 1 boundary within a k-wide window and the hypothesis has 2,
Pk records no error because both ends of the probe are still in "different" segments.
Multiple errors within one window go undetected.

**Problem 3 — Sensitive to segment size variation.**
When segments are smaller than k, the probability that a false negative is detected
falls, because the probe may span multiple missed boundaries at once. The metric
becomes lenient in exactly the cases where many small errors accumulate.

**Problem 4 — Near-miss errors penalized too much.**
An algorithm that places a boundary 1 sentence away from the true boundary (a
near miss) is penalized as severely as one that places a false positive far from any
true boundary. Near misses should be better than pure false positives.

**Problem 5 — Non-intuitive interpretation.**
Pk measures the probability that two sentences k apart are mislabeled. This is not a
natural measure of segmentation quality and is difficult to communicate.

## WindowDiff formula

```
WindowDiff(ref, hyp) = 1/(N-k) * Σ_{i=1}^{N-k} (|b(ref_i, ref_{i+k}) - b(hyp_i, hyp_{i+k})| > 0)
```

where `b(i, j)` counts the number of segment boundaries between positions i and j,
and N is the total number of sentences. The probe now counts *how many* boundaries
fall within its window and penalizes whenever that count differs between reference
and hypothesis — regardless of direction.

**Properties:**
- Treats FP and FN symmetrically (fixes Problem 1).
- Catches multiple errors within one window (fixes Problem 2).
- Less sensitive to segment size variation (partially fixes Problem 3).
- Near-miss errors receive a smaller penalty because they shift the boundary count
  by a smaller amount (fixes Problem 4).
- Still not perfectly intuitive, but easier to reason about than Pk.

## Simulation results

With 50% FN probability: Pk ≈ 0.245, WD ≈ 0.245 (nearly identical — both detect FNs well).
With 50% FP probability: Pk ≈ 0.128, WD ≈ 0.240 (Pk is misleadingly lenient on FPs).
Mixed FP+FN: Pk ≈ 0.317, WD ≈ 0.376 (WD correctly assigns a higher penalty).

## Key findings relevant to chunklabel

**Implement both.** Both Pk and WD are used in the literature (see lo-2021, riedl-2012).
WD is the stricter and more informative metric; Pk is reported for comparability with
older baselines.

**k = average_segment_length / 2.** This is the standard setting, confirmed by
lo-2021 ("k is set to be half of the average ground-truth segment length").

**Neither metric measures label quality.** Pk and WD measure boundary placement
only, not whether the assigned category is correct. chunklabel's category evaluation
requires a separate metric (e.g. category agreement across annotators).

## Actionable ideas

- Implement `pk()` and `window_diff()` in `chunklabel.eval`
  (see `ideas/eval-metrics-pk-windowdiff.md`).
- Set default k automatically from the ground-truth corpus statistics.
- Report both Pk and WD side-by-side in benchmarks so results are comparable with
  both old (Pk-only) and new (WD-preferred) papers.
