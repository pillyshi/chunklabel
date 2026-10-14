# Gap filling on PubMed RCT (issue #6)

When the LLM skips text, `fill_gaps` sends each skipped stretch back to it, alone and
once. Does that cover the skipped text, and label it better than joining it to the
preceding segment? With the current small local models the numbers are a smoke test
of the method, not a performance claim.

## Setup

- Inputs: the Qwen2.5-7B Q4_K_M records of `benchmarks/few_shot`. That is 40
  PubMed 20k RCT test abstracts (seed 0) with 3 examples, selected `random` or
  `similar`. One `random` abstract failed there, so `random` has 39.
- The first call of each abstract is replayed from its record. Only the fill calls
  reach the model (llama.cpp on an M2 16 GB, temperature 0). This measures what
  `fill_gaps` adds, at the cost of the fill calls alone.
- Both settings run through `FewShotPromptSegmenter.predict_with_report`:
  - `fill_gaps=False`: the gap joins the preceding segment.
  - `fill_gaps=True`: default `min_gap_chars=20`. The fill calls get no examples.
- Gemma 4 E4B was not run. It left less than 1% of the text unquoted in
  `benchmarks/few_shot`, so it would make almost no fill calls.
- Metrics:
  - **Gap**: `gap_coverage`, the mean per abstract.
  - **Label accuracy**: the share of non-whitespace characters labeled with the gold
    role. It is measured over the whole abstract, and over the gap text alone (the
    text the first call left, which is what the fill calls received).

## Results

| selection | abstracts with a fill call | fill calls | gap: off → on | label accuracy on the gap text: off → on | label accuracy, whole abstract: off → on | extra time / abstract |
|---|---|---|---|---|---|---|
| random | 13 / 39 | 15 | 0.137 → 0.013 | 0.115 → 0.631 | 0.745 → 0.817 | 4.9 s |
| similar | 15 / 40 | 19 | 0.120 → 0.000 | 0.180 → 0.600 | 0.783 → 0.833 | 4.6 s |

The aggregates are in `results.json`.

- **The fill calls cover almost all of the skipped text.** The remaining 1.3% under
  `random` is text that a fill call skipped again. There is only one round.
- **Joining the gap to the preceding segment mislabels most of it.** Abstracts move
  from background to methods to results, so text skipped later in the abstract gets an
  earlier role. Labels from the fill call, without examples or context, are right
  about 60% of the time.
- **Fill calls are cheap.** They happen only on abstracts with a gap. Each one sends
  only the gap's text (median about 200 characters), with no examples.
- These are small models on one dataset. The direction is clear, but the exact numbers
  are diagnostics, not targets.

## Reproduce

See the docstring of `run.py`. Outputs contain abstract text, so they stay outside the
repository.
