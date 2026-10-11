# Label schema induction: free labels over a corpus, then a fixed label set

**Status**: Draft

## Motivation

The schema-free mode is chunklabel's distinctive feature, but on its own it gives
labels that differ per document. A natural way to keep it useful is to treat it as the
first step of a loop:

1. Run schema-free chunklabel over a sample of a corpus.
2. Merge the free labels into a small label set (schema).
3. Re-label the whole corpus with that fixed set (`given-label-set.md`).

This matches how qualitative researchers code interview transcripts (open coding →
codebook → focused coding), and fits other long-segment corpora whose schema is not
known in advance: meeting minutes, support conversations, reviews, contracts. This
demand is a hypothesis, not yet backed by literature in the catalog.

## Evidence

- `benchmarks/label_agreement/README.md`: free labels are fine-grained and
  inconsistent in form; "a document-level label set" and "normalisation" are listed as
  the next levers.
- `catalog.yaml` [sarawagi-2004], [kong-2015]: once the label set is fixed, the problem
  is the standard segmentation-and-labeling setting, so standard evaluation applies.
- `catalog.yaml` [vinh-2010]: the induced schema can be compared with a gold role set
  by clustering measures (ARI/AMI), since induced and gold label names differ.
- `chunklabel/normalizer.py`: `Normalizer.build_mapping` already maps free labels to
  canonical ones; it is the starting point for step 2.

## Proposed Scope

1. A corpus-level step that runs `split` on a sample of documents, collects the free
   labels with counts, and asks the LLM (via `Normalizer`) for a small label set plus a
   mapping from free labels to it. Optional target size (e.g. at most `k` labels).
2. The result is a label set with short descriptions, which can be saved, edited by
   hand, and passed to `ChunkLabeler(labels=...)`.
3. Experiment on PubMed RCT: induce a schema from N abstracts, re-label the test set
   with it, and report category-level ARI/AMI and number of labels vs the free-label
   run and the gold-label run.

## Acceptance Criteria

- The induced label set is saved and loaded in the same format that the given-label-set
  feature accepts.
- On PubMed RCT, re-labeling with the induced schema gives fewer labels per abstract
  than the free-label run, with ARI reported for all three settings.

## Out Of Scope

- The scikit-learn style API (`sklearn-estimator-api.md`), although `fit(X)` there
  would wrap this step.
- Hierarchical schemas (labels with sub-labels).

## Open Questions

- How many documents are needed for a stable schema, and how much does it vary across
  samples and runs?
- Should the schema come from label names only (current `Normalizer`) or also from
  chunk text (e.g. a few example quotes per label)?
- Is there literature on LLM-assisted qualitative coding to back the demand
  hypothesis? Worth a literature search.
