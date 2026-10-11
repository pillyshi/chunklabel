# scikit-learn style estimator API: fit(X), fit(X, y), predict(X)

**Status**: Draft

## Motivation

With a given label set, schema induction, and supervised comparisons, chunklabel gains
corpus-level state (a label set, examples, a mapping). `split(text)` handles one text
and keeps no state. A scikit-learn style API expresses the "amount of supervision"
cleanly, with one class:

| call | supervision | what `fit` stores |
|---|---|---|
| `fit(X)` | none | an induced label set (like a clustering model's clusters) |
| `fit(X, y)` | segmentations | the label set from `y`, plus examples for the prompt |
| `fit(X, y)` with some `y[i] is None` | partial | both of the above |
| `ChunkLabeler(labels=[...])` | label set only | a hyperparameter, not learned |

Here `X` is a list of texts and `y` is a list of segmentations, one per text (a list
of `(start, end, label)` or of `Chunk`). `predict(X)` returns a list of segmentations.
`fit_predict(X)` mirrors clustering estimators.

Design precedent: sklearn-crfsuite and pystruct use `fit(X, y)` with one label
sequence per sample, without full scikit-learn compliance. Not yet in the catalog.

## Evidence

- `catalog.yaml` [sarawagi-2004]: the training set is pairs `(x_l, s_l)`, an input and
  its segmentation. That is the natural shape of `fit(X, y)` here.
- `catalog.yaml` [kong-2015]: partially supervised training where only labels are
  observed and boundaries are latent. It supports allowing incomplete `y`.
- `research/ideas/label-schema-induction.md`: `fit(X)` would wrap this step.
- `chunklabel/normalizer.py`: `build_mapping` / `apply` / `save` / `load` already work
  like `fit` / `transform` / persistence.

## Decision (2026-10-12): one class

Use a single class with `fit(X, y=None)` for all supervision levels, instead of
separate classes as scikit-learn does for clustering vs classification. One class
covers the partially supervised case (some `y[i] is None`) without a third class, and
keeps the "amount of supervision" a property of the data rather than of the class.

## Proposed Scope

1. Follow scikit-learn conventions without aiming for `check_estimator` compliance:
   hyperparameters only in `__init__`, learned state in attributes ending with `_`
   (`labels_`, `examples_`, `mapping_`), `fit` returns `self`, `get_params`/`set_params`
   (via `BaseEstimator`) so `clone` and parameter search work.
2. `fit(X, y=None)`:
   - `y is None`: schema induction on `X` → `labels_`.
   - `y` given: `labels_` from the labels in `y`; select a few `(text, segmentation)`
     pairs as in-context examples → `examples_`.
3. `predict(X)`: closed-vocabulary split using `labels_` and `examples_`. Raises
   `NotFittedError` if not fitted and no `labels` hyperparameter is given.
4. Keep `split(text)` as the schema-free, stateless entry point, unchanged.
5. Optional: `transform(X)` → per-document share of text per label (a
   `(n_documents, n_labels)` matrix), so segment labels can feed ordinary scikit-learn
   pipelines (document classification, clustering).
6. Optional: a `score(X, y)` using the label agreement metrics.

## Acceptance Criteria

- `clone(ChunkLabeler(...))` works and keeps hyperparameters.
- `fit(X)` and `fit(X, y)` both set `labels_`; `predict` uses only labels from it.
- Existing `split` tests pass unchanged.

## Out Of Scope

- Full `check_estimator` compliance (it assumes array-shaped `X` and `y`).
- Training model weights inside `fit` (see `zero-shot-vs-supervised.md` for
  distillation).

## Open Questions

- Format of `y`: `(start, end, label)` triples are simplest; also accept `list[Chunk]`
  and BIO over sentences?
- `fit` calls the LLM over a corpus and is expensive. Should it take a sample size,
  and should it cache raw outputs?
- Are LLM calls with temperature 0 deterministic enough that `fit` is reproducible?
