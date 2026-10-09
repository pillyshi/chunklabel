# Notes: arnold-2019

> Arnold, S., Schneider, R., Cudré-Mauroux, P., Gers, F. A., Löser, A. (2019).
> SECTOR: A Neural Model for Coherent Topic Segmentation and Classification.
> TACL 7. arXiv:1902.04793. Data: https://github.com/sebastianarnold/WikiSection
> (CC BY-SA 3.0).

## Summary

Introduces the joint task of **segmenting a document into sections and assigning a
topic label to each section**, the WikiSection dataset for it, and the SECTOR model.

## Dataset

- Wikipedia articles from two domains, diseases and cities, in English and German.
- Every section has its original heading and a **normalised topic label** (e.g.
  `disease.symptom`, `disease.cause`), obtained by clustering headings via BabelNet
  synsets. Rare or unmatched headings are labeled `other`.
- Headings are removed from the text. The task input is plain sentences.
- 8.5k headings over the disease set and 23.0k over the city set (Table 1).

## Key findings relevant to chunklabel

- The closest public match to chunklabel's output format: contiguous segments, each
  with a category. The categories are a closed set, but they are derived from free-text
  headings, much like chunklabel's free-form labels plus `Normalizer`.
- Gives a **second, longer-document benchmark** for label-aware clustering evaluation,
  complementary to PubMed RCT (short, functional) and Wiki-50 (no labels).
- The heading → normalised label mapping is itself a reference point for evaluating
  chunklabel's `Normalizer`.

## Limitations

- Wikipedia only. Section boundaries follow editorial conventions, so it shares
  Wiki-50's granularity issue: an LLM may split sections into finer chunks.
- Long documents: the cost of quote-based LLM runs grows with document length.
