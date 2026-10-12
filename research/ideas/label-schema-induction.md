# Label schema induction: free labels over a corpus, then a fixed label set

**Status**: Promoted to issue #18 (2026-10-13)

## Motivation

The schema-free mode is chunklabel's distinctive feature, but on its own it gives
labels that differ per document. A natural way to keep it useful is to treat it as the
first step of a loop:

1. Run schema-free chunklabel over the texts the user passes (the user chooses how
   many).
2. Merge the free label names with `Normalizer` into a label set (schema).
3. Re-label the whole corpus with that fixed set (`given-label-set.md`).

Related: issue #7 (recursive structure discovery) builds on the same
split → `Normalizer` step.

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

1. `ZeroShotPromptSegmenter.fit(X)` (see `sklearn-estimator-api.md`): split every
   text, merge label names with `Normalizer` as it is (no target number of labels),
   and store the label set as `labels_`.
2. The label set can be saved, edited by hand, and passed back as
   `ZeroShotPromptSegmenter(labels=...)`.
3. Evaluation is fidelity only (decision 2026-10-12 in `sklearn-estimator-api.md`).
   Whether the label set is good is up to the user. On PubMed RCT, number of labels and
   ARI against gold roles may be reported as diagnostics, not as targets.

## Acceptance Criteria

- The induced label set is saved and loaded in the same format that the given-label-set
  feature accepts.
- Fidelity of `fit` and of `predict` with the induced labels is reported, at least on
  PubMed RCT.

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
