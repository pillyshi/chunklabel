# Notes: mackenzie-2025

> Mackenzie, P., Shah, M., Frenett, P. (2025). Topic Segmentation Using Generative
> Language Models. arXiv:2601.03276 (v1, Dec 2025).

## Summary

Prompt-based topic segmentation with **sentence enumeration**: sentences are
separated by `[1]`, `[2]`, … markers and the LLM returns a list of boundary indices.
Long documents use overlapping windows; segments that are too long are split
recursively and segments that are too short are merged. Evaluated with
**boundary similarity** (Fournier 2013, `n = 2`) and B-precision/recall.

## Method

- Rejects the "copy the text with delimiters" prompting style (as in Xing 2024)
  because the output is not guaranteed to be unedited and wastes tokens. The index
  output is guaranteed to refer to the original text and is far cheaper.
- Few-shot prompt plus a system prompt. Overlap of 1500 tokens between windows;
  segment length kept in 50–500 words by recursion and merging.
- Data: 10 human-segmented articles ("Human"), Wikipedia segmented by headings
  ("Wiki"), concatenated Wikipedia segments ("Conc-Wiki"), and a GPT-3.5 synthetic set.

## Results (Table 1, B with n = 2)

| | Human | Wiki | Conc-Wiki |
|---|---|---|---|
| GPT-3.5 (index prompting) | 0.38 | 0.25 | 0.29 |
| Fine-tuned Flan-T5 | 0.25 | 0.24 | 0.41 |
| SBERT similarity | 0.18 | 0.09 | 0.46 |
| Every 5 sentences | 0.13 | 0.19 | 0.19 |

- LLMs win on natural text (Human, Wiki) with higher recall. SBERT wins on the
  artificial concatenations (Conc-Wiki).
- Failure mode: on messy input (tables, artefacts) GPT-3.5 sometimes emits a regular
  pattern of indices that extends beyond the number of sentences.

## Key findings relevant to chunklabel

- **Prior art for the index-boundary idea**, with the same motivation as ours
  (unedited output). Our quick Qwen run showed the same over-segmentation tendency
  when no granularity is specified.
- **Same pattern as our probe**: embedding methods do well on concatenated
  unrelated documents (Choi / Conc-Wiki) and poorly on natural documents. This
  supports treating Choi results as unrepresentative.
- Endorses boundary similarity over Pk/WD for this setting.
- Their method needs a sentence splitter and fixes boundaries to sentence edges,
  which is the main limitation relative to chunklabel's quote-based spans (sub-sentence
  cuts, texts without clean sentence boundaries).

## Limitations

- Small evaluation (≤ 150 documents per set, 10 human-annotated), code and data
  proprietary, single prompting method, no comparison to copy-with-delimiters methods.
- Granularity is controlled by hand-set min/max segment lengths, not learned.
