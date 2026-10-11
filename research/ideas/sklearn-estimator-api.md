# scikit-learn style estimator API: fit(X), fit(X, y), predict(X)

**Status**: Draft

## Motivation

With a given label set, schema induction, and supervised comparisons, chunklabel gains
corpus-level state (a label set, examples, a mapping). `split(text)` handles one text
and keeps no state. A scikit-learn style API expresses the "amount of supervision"
cleanly, split across two estimators in the way scikit-learn separates clustering from
classification:

| estimator | call | supervision | what `fit` stores |
|---|---|---|---|
| unsupervised | `fit(X)` / `fit_predict(X)` | none | an induced label set (like a clustering model's clusters) |
| supervised | `fit(X, y)` | segmentations | whatever its method needs (prompt examples, fine-tuned weights, ...) |
| supervised | `fit(X, y)` with some `y[i] is None` | partial | same, using unlabeled texts too |
| supervised | `labels=[...]` hyperparameter, no `y` | label set only | nothing learned; zero-shot with a fixed label set |

Here `X` is a list of texts and `y` is a list of segmentations, one per text. A
segmentation is a list of `(start, end, label)` triples. `predict(X)` and
`fit_predict(X)` return segmentations in the same format.

Design precedent: sklearn-crfsuite and pystruct use `fit(X, y)` with one label
sequence per sample, without full scikit-learn compliance. Not yet in the catalog.

## Evidence

- `catalog.yaml` [sarawagi-2004]: the training set is pairs `(x_l, s_l)`, an input and
  its segmentation. That is the natural shape of `fit(X, y)` here.
- `catalog.yaml` [kong-2015]: partially supervised training where only labels are
  observed and boundaries are latent. It supports allowing incomplete `y`.
- `research/ideas/label-schema-induction.md`: the unsupervised `fit(X)` would wrap
  this step.
- `chunklabel/normalizer.py`: `build_mapping` / `apply` / `save` / `load` already work
  like `fit` / `transform` / persistence.

## Decision (2026-10-12): two estimators

First decided as one class with `fit(X, y=None)`; revised the same day to two classes.

- The supervised `fit(X, y)` may be implemented completely differently from the
  unsupervised one: a different prompt with examples, or fine-tuning a pretrained
  model. One class would only switch its whole internals on whether `y` is given.
  scikit-learn keeps clustering and classification apart for the same reason.
- The argument for one class was that partial supervision would need a third class.
  It does not: as in scikit-learn's `LabelSpreading` (unlabeled samples marked `-1`),
  partial supervision belongs to the supervised estimator, with `y[i] is None` for
  unlabeled texts.
- Shared parts stay shared: quote alignment, post-processing, and the
  `(start, end, label)` output format are common components used by both.

## Output format: `(start, end, label)`

`Chunk.quote` is always `text[start:end]`, so it duplicates information. Estimators
return `(start, end, label)` triples only. A small helper recovers the quote text from
the source when needed.

## Meaning of the unsupervised `fit_predict(X)`

As in clustering, labels are consistent across all of `X`: `fit_predict(X)` runs the
schema-free split on each text, then unifies the label names over the whole corpus
(as `Normalizer` does today). The current `split(text)` is the special case of a
single document.

## Proposed Scope

1. Follow scikit-learn conventions without aiming for `check_estimator` compliance:
   hyperparameters only in `__init__`, learned state in attributes ending with `_`,
   `fit` returns `self`, `get_params`/`set_params` (via `BaseEstimator`) so `clone` and
   parameter search work.
2. Unsupervised estimator:
   - `fit(X)`: schema induction → `labels_` (and the free-label → label `mapping_`).
   - `fit_predict(X)`: segmentations of `X` with corpus-wide consistent labels.
   - `predict(X_new)`: re-label new texts with the fitted `labels_` (closed
     vocabulary, as in `given-label-set.md`), like `KMeans.predict`.
3. Supervised estimator:
   - `fit(X, y)`: first version takes `labels_` from `y` and selects a few
     `(text, segmentation)` pairs as in-context examples → `examples_`. Other methods
     (fine-tuning) can replace this later behind the same interface.
   - `y[i] is None` marks an unlabeled text.
   - `predict(X)`: raises `NotFittedError` if not fitted and no `labels`
     hyperparameter is given.
4. Shared components: alignment, post-processing, `(start, end, label)` output, and a
   helper to get the quote text.
5. Keep `split(text)` working, unchanged, during the transition.
6. Optional: `transform(X)` → per-document share of text per label (a
   `(n_documents, n_labels)` matrix), so segment labels can feed ordinary scikit-learn
   pipelines (document classification, clustering).
7. Optional: `score(X, y)` on the supervised estimator, using the label agreement
   metrics.

## Acceptance Criteria

- `clone(...)` works on both estimators and keeps hyperparameters.
- Unsupervised `fit_predict(X)` returns the same label names across all texts.
- Supervised `predict` only returns labels from `labels_`.
- Both return `(start, end, label)` triples; `text[start:end]` gives the quote.
- Existing `split` tests pass unchanged.

## Out Of Scope

- Full `check_estimator` compliance (it assumes array-shaped `X` and `y`).
- Fine-tuning inside the supervised `fit` in the first version (see
  `zero-shot-vs-supervised.md`). The interface should allow it later.

## Open Questions

- Class names. Does `ChunkLabeler` become the unsupervised estimator?
- Should `split` and `Chunk` be deprecated once the estimators exist?
- Should `y` also accept `list[Chunk]` or BIO over sentences, or only triples?
- `fit` calls the LLM over a corpus and is expensive. Should it take a sample size,
  and should it cache raw outputs?
- Are LLM calls with temperature 0 deterministic enough that `fit` is reproducible?
