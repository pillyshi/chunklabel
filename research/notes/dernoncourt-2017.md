# Notes: dernoncourt-2017

> Dernoncourt, F., Lee, J. Y. (2017). PubMed 200k RCT: a Dataset for Sequential
> Sentence Classification in Medical Abstracts. IJCNLP 2017. arXiv:1710.06071.
> Data: https://github.com/Franck-Dernoncourt/pubmed-rct

## Summary

About 200k structured abstracts of randomized controlled trials from PubMed (2.3M
sentences). Each sentence is labeled with its rhetorical role, derived from the
abstract's own structured headings: background, objective, method, result,
conclusion. A smaller **PubMed 20k RCT** split has 15k / 2.5k / 2.5k abstracts
(train / validation / test).

## Key findings relevant to chunklabel

- Consecutive sentences with the same role form contiguous segments, so each abstract
  is a **segmentation with labels**: functional segments of the same kind as the
  README example (initiation / obstacle / outcome), rather than topic sections.
- Short documents (around 12 sentences), so LLM runs are cheap, even on a local
  7B model.
- Gold labels are a small closed set and chunklabel's labels are free-form, which fits
  clustering-based evaluation (ARI/AMI, homogeneity/completeness; see `vinh-2010.md`).
- The structure is derived from author-written headings, so it is "natural" rather
  than concatenated, unlike Choi.

## Limitations

- Medical RCT abstracts only, and very conventional structure (often already
  signposted by phrases like "We conducted…").
- Labels come from structured abstracts with the headings removed. A model may
  over-split within long "method" or "result" runs, which is reasonable but penalised.
- Dataset license not checked yet (to do before redistributing anything derived).
