# Zero-shot chunklabel vs supervised segmental models

**Status**: Draft

## Motivation

A clear research question for chunklabel: with a fixed label set and no training
data, how close does an LLM get to a supervised segmentation-and-labeling model on the
same data? The answer tells users when chunklabel is enough and when they should
collect labels and train a model.

The reverse direction is also useful: chunklabel's outputs can be training data for a
small segmental model (distillation), which would make large corpora cheap to process.

## Evidence

- `catalog.yaml` [sarawagi-2004], [kong-2015]: supervised semi-Markov models for the
  same problem, with fixed label sets; [kong-2015] also trains with latent boundaries
  (labels only), which matters when gold boundaries are noisy.
- `catalog.yaml` [dernoncourt-2017]: PubMed RCT has gold rhetorical roles and a
  standard train/test split, so a supervised baseline can be trained on it.
- `benchmarks/label_agreement/README.md`: the evaluation code already scores chunks
  against PubMed roles.

## Proposed Scope

1. Depends on `given-label-set.md`.
2. Supervised baseline on PubMed 20k RCT: a sentence-level semi-CRF or a BiLSTM-CRF
   over sentence embeddings (the standard model for this dataset), trained on
   increasing amounts of data (learning curve, as in [sarawagi-2004] Figure 1).
3. Compare with chunklabel given the five roles, on the same test abstracts, using the
   existing label agreement metrics plus sentence-level role accuracy.
4. Report the training size at which the supervised model overtakes zero-shot
   chunklabel.

## Acceptance Criteria

- One table and one learning-curve plot comparing the two on the same test set.
- Costs (LLM calls, training time) are reported with the scores.

## Out Of Scope

- Distillation (training on chunklabel outputs). Kept as an open question until the
  comparison shows whether it is worth it.
- Datasets other than PubMed RCT in the first round.

## Open Questions

- PubMed abstracts are short and very conventional. Does the comparison carry over to
  long, loosely structured documents, which are chunklabel's target?
- Which supervised baseline is fair: a classic semi-CRF, or a strong pretrained
  sentence classifier?
