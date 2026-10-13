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
| `ZeroShotPromptSegmenter()` | `fit(X)` / `fit_predict(X)` | none | an induced label set (like a clustering model's clusters) |
| `ZeroShotPromptSegmenter(labels=[...])` | `predict(X)` | label set only | nothing; the given labels replace the induced ones |
| `FewShotPromptSegmenter` (and later methods) | `fit(X, y)` | segmentations | whatever its method needs (prompt examples, fine-tuned weights, ...) |
| same | `fit(X, y)` with some `y[i] is None` | partial | same, using unlabeled texts too |

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

## Decision (2026-10-12): class names

- Classes are named after their method, as in scikit-learn (`KMeans`,
  `LogisticRegression`). Each supervised method gets its own class.
- `ChunkLabeler` stays unchanged. Its usefulness has not been shown experimentally
  yet, so new classes are added next to it rather than replacing it.
- Unsupervised: **`ZeroShotPromptSegmenter`**. "Prompt" alone would not separate it
  from the supervised in-context method, which also prompts an LLM; "zero-shot" does.
- First supervised method: **`FewShotPromptSegmenter`**, which puts examples from `y`
  in the prompt. It pairs naturally with the zero-shot class.
- The "label set only" case (fixed labels, no examples) belongs to
  `ZeroShotPromptSegmenter(labels=[...])`, like `KMeans(init=centers)` taking given
  cluster centers. Supervised classes are then exactly those that use `y`.
- "Segmenter" matches the problem name (segmentation and labeling) and avoids
  confusion with `ChunkLabeler`, even though the classes also label.

## Output format: `(start, end, label)`

`Chunk.quote` is always `text[start:end]`, so it duplicates information. Estimators
return `(start, end, label)` triples only. A small helper recovers the quote text from
the source when needed.

## Decision (2026-10-12): `ZeroShotPromptSegmenter` follows `KMeans`

- `fit(X)` runs the schema-free split on each text, unifies the label names over the
  whole corpus with `Normalizer`, and stores:
  - `segmentations_`: the segmentations of `X` with unified labels (like
    `KMeans.labels_`),
  - `labels_`: the unified label set,
  - `mapping_`: free label → unified label.
- `fit_predict(X)` returns `fit(X).segmentations_`. One LLM pass over `X`, no
  re-labeling.
- `predict(X_new)` is for new texts: it splits them with `labels_` as a closed label
  set. Called on the training texts, it may give different boundaries than
  `segmentations_`, because it calls the LLM again with a different prompt. This is
  documented, not prevented.
- With `labels=[...]` given, `predict` uses that set and works without `fit`.
- The current `split(text)` corresponds to `fit_predict([text])` without the corpus-wide
  unification.

## Decision (2026-10-12): scope of the zero-shot estimator

- **Evaluation**: for the schema-free mode, only fidelity is evaluated (verbatim quotes,
  alignment, coverage), as decided on 2026-10-10. Whether labels are good can only be
  judged by the user; at this stage the mode is closer to text mining than to a task
  with a correct answer. Agreement with gold labels stays a diagnostic.
- **No examples in prompts**: zero-shot means the prompts contain no example
  segmentations or labels (example labels were already removed in e141317).
- **`Normalizer` as is**: it only merges label names; no target number of labels.
- **Cost**: `fit` processes every text it receives. Users limit the cost by passing
  fewer texts; there is no built-in sampling.

## Proposal (2026-10-13): `FewShotPromptSegmenter`

`ZeroShotPromptSegmenter` is done (#17 given labels, #18 `fit`). This section proposes
the first supervised estimator before it becomes an issue.

**Interface**

```python
seg = FewShotPromptSegmenter(n_examples=3, random_state=0, client=...)
seg.fit(X_train, y_train)      # y: list of segmentations, or None for unlabeled texts
seg.predict(X_test)            # list[list[tuple[int, int, str]]], labels from labels_
```

- Hyperparameters: `n_examples` (default 3), `random_state`, and the ones shared with
  `ZeroShotPromptSegmenter` (`client`, `backend`, `fuzzy_threshold`, `on_align_error`,
  `on_unknown_label`, `timeout`).
- `fit(X, y)` stores:
  - `labels_`: sorted labels that occur in `y`.
  - `examples_`: the chosen `(text, segmentation)` pairs.
- `fit` makes no LLM call. It only checks `y` and chooses examples.
- `predict(X)`: the same prompt as the given-label-set mode (label set listed, schema
  enum), plus the examples. Then the shared steps: closest-label mapping, alignment,
  and gaps joined to the preceding segment.
- `predict` before `fit` raises `NotFittedError`. This is a chunklabel class that
  subclasses `ValueError` and `AttributeError`, as scikit-learn's does, so it needs no
  scikit-learn import. `ZeroShotPromptSegmenter` should raise it too.

**Checking `y`**

- `len(y) == len(X)`.
- Each `y[i]` is `None` or a list of `(start, end, label)` triples. The triples are
  within the text, sorted, and do not overlap.
- Coverage is not required. Uncovered text simply does not appear in the example.
- At least one labeled text, with at least one label.

**Prompt format for examples**

Each example is a user / assistant turn pair before the real text. The user turn is the
example text. The assistant turn is the exact JSON the model must produce,
`{"chunks": [{"category": label, "quote": text[start:end]}, ...]}`. The model sees
the output format filled in, and no new prompt template is needed beyond the system
prompt. `LLMBackend.extract_labeled_chunks` gets an optional `examples` argument.

**Decisions to make (with a recommendation)**

1. *How to choose examples.* Recommended: random, seeded by `random_state`. Every label
   is listed in the system prompt and enforced by the schema, so examples do not need
   to cover every label. Similarity-based selection (choose the training texts closest
   to each input) needs an embedding model or TF-IDF, and changes the prompt per
   input. Keep it for a later option (`example_selection="similar"`).
2. *Unlabeled texts (`y[i] is None`).* Recommended: accepted and ignored in the first
   version, so the interface is fixed now. Possible later uses: self-training, or
   choosing examples close to the unlabeled distribution.
3. *Context length.* An example costs roughly twice its text length in tokens (text
   plus quoted JSON). Recommended: no automatic truncation. Document the cost, and let
   users lower `n_examples` or pass shorter texts. Raise a clear error if the backend
   reports a context overflow.
4. *Shared code.* Recommended: move the shared parameters and `_segment` /
   `_make_backend` into a private base class used by both estimators. Keep behaviour
   unchanged.

**Evaluation**

Here `y` is a correct answer, so agreement is a real metric, not only a diagnostic.
Examples come from the PubMed 20k RCT train split, and scores are measured on the same
40 test abstracts. Compare with `ZeroShotPromptSegmenter` given the five roles
(ARI 0.80–0.84, label accuracy 0.84–0.86), and vary `n_examples` (1, 3, 5). With the
current small local models, the numbers are a smoke test of the method, not a
performance claim. Fidelity is reported as for the other modes.

**Out of scope for the first version**: similarity-based selection, use of unlabeled
texts, `score(X, y)`, `y` given as `list[Chunk]` or BIO tags, fine-tuning.

## Proposed Scope

1. Follow scikit-learn conventions without aiming for `check_estimator` compliance:
   hyperparameters only in `__init__`, learned state in attributes ending with `_`,
   `fit` returns `self`, `get_params`/`set_params` (via `BaseEstimator`) so `clone` and
   parameter search work.
2. `ZeroShotPromptSegmenter`:
   - `fit(X)`: split every text, unify labels with `Normalizer` → `segmentations_`,
     `labels_`, `mapping_`.
   - `fit_predict(X)`: returns `segmentations_`.
   - `predict(X_new)`: re-label new texts with the fitted `labels_` (closed
     vocabulary, as in `given-label-set.md`), like `KMeans.predict`.
   - `labels=[...]` hyperparameter: use the given label set; `predict` works without
     `fit`.
3. `FewShotPromptSegmenter` (first supervised method):
   - `fit(X, y)`: takes `labels_` from `y` and selects a few `(text, segmentation)`
     pairs as in-context examples → `examples_`. Other methods (e.g. fine-tuning) get
     their own classes with the same interface.
   - `y[i] is None` marks an unlabeled text.
   - `predict(X)`: raises `NotFittedError` if not fitted.
4. Shared components: alignment, post-processing, `(start, end, label)` output, and a
   helper to get the quote text.
5. `ChunkLabeler` and `split(text)` stay unchanged.
6. Optional: `transform(X)` → per-document share of text per label (a
   `(n_documents, n_labels)` matrix), so segment labels can feed ordinary scikit-learn
   pipelines (document classification, clustering).
7. Optional: `score(X, y)` on the supervised estimator, using the label agreement
   metrics.

## Acceptance Criteria

- `clone(...)` works on both estimators and keeps hyperparameters.
- `ZeroShotPromptSegmenter.fit_predict(X)` returns the same label names across all
  texts and equals `fit(X).segmentations_`.
- `ZeroShotPromptSegmenter` reports fidelity; no label-quality target.
- Supervised `predict` only returns labels from `labels_`.
- Both return `(start, end, label)` triples; `text[start:end]` gives the quote.
- Existing `ChunkLabeler` tests pass unchanged.

## Out Of Scope

- Full `check_estimator` compliance (it assumes array-shaped `X` and `y`).
- Fine-tuning inside the supervised `fit` in the first version (see
  `zero-shot-vs-supervised.md`). The interface should allow it later.

## Open Questions

- Should `ChunkLabeler` and `Chunk` be deprecated if the estimators prove useful?
- Should `y` also accept `list[Chunk]` or BIO over sentences, or only triples?
- Should `fit` cache raw LLM outputs so that re-fitting is cheap?
- Are LLM calls with temperature 0 deterministic enough that `fit` is reproducible?
