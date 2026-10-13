"""scikit-learn style estimators for segmentation and labeling.

A segmentation of a text is a list of ``(start, end, label)`` triples that cover the
text without overlapping; ``text[start:end]`` is the segment's text. See
``research/ideas/sklearn-estimator-api.md`` for the design.
"""

from __future__ import annotations

import inspect
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Literal, Union

from rapidfuzz import fuzz, process
from typing_extensions import Self

from chunklabel.alignment import AlignmentDetail, AlignmentError, align_detailed
from chunklabel.eval import FidelityReport, fidelity_from_details
from chunklabel.llm.base import LLMBackend
from chunklabel.llm.client import BaseLLMClient
from chunklabel.normalizer import Normalizer
from chunklabel.postprocess import absorb_gaps, resolve_spans
from chunklabel.types import Chunk, RawChunk

Segment = tuple[int, int, str]
Vector = dict[Hashable, float]


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit``. Like scikit-learn's, it subclasses
    ``ValueError`` and ``AttributeError``."""


class _ParamsMixin:
    """``get_params`` / ``set_params`` as scikit-learn expects, so ``sklearn.clone`` and
    parameter searches work without depending on scikit-learn.

    Subclasses store every ``__init__`` argument unchanged under the same name.
    """

    @classmethod
    def _param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> Self:
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for {type(self).__name__}")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class _PromptSegmenter(_ParamsMixin):
    """Shared steps: prompt with a closed label set, map labels, align, fill gaps."""

    client: Union[BaseLLMClient, str]
    backend: LLMBackend | None
    fuzzy_threshold: int
    on_align_error: Literal["raise", "skip"]
    on_unknown_label: Literal["closest", "drop", "raise"]
    timeout: float | None

    def _segment(
        self,
        text: str,
        backend: LLMBackend,
        labels: Mapping[str, str],
        examples: Sequence[tuple[str, list[RawChunk]]] = (),
        strict: bool = True,
    ) -> tuple[list[Segment], FidelityReport]:
        """Segment ``text`` and report fidelity. ``strict`` applies ``on_align_error``;
        otherwise unaligned quotes are only counted."""
        if examples:
            raw = backend.extract_labeled_chunks(text, labels, examples)
        else:
            raw = backend.extract_labeled_chunks(text, labels)
        raw_chunks = map_labels(raw, list(labels), self.on_unknown_label)
        return self._resolve(raw_chunks, text, strict)

    def _resolve(
        self, raw_chunks: list[RawChunk], text: str, strict: bool
    ) -> tuple[list[Segment], FidelityReport]:
        details = align_detailed(raw_chunks, text, self.fuzzy_threshold)
        if strict and self.on_align_error == "raise":
            _raise_unaligned(raw_chunks, details, self.fuzzy_threshold)
        spans = [d.span for d in details]
        segments = absorb_gaps(resolve_spans(raw_chunks, spans), len(text))
        return segments, fidelity_from_details(details, text)

    def _make_client(self) -> BaseLLMClient:
        if isinstance(self.client, str):
            from chunklabel.llm.client import OpenAIClient

            return OpenAIClient(model=self.client, timeout=self.timeout)
        return self.client

    def _make_backend(self) -> LLMBackend:
        if self.backend is not None:
            return self.backend
        from chunklabel.llm.backend import ClientBackend

        return ClientBackend(self._make_client())


class ZeroShotPromptSegmenter(_PromptSegmenter):
    """Split texts into labeled segments by prompting an LLM, without examples.

    Like ``KMeans``: ``fit(X)`` finds a label set in ``X`` and segments ``X`` with it;
    ``predict(X_new)`` segments new texts using that label set.

    - ``fit(X)`` without ``labels``: the LLM splits each text with free labels, and
      ``Normalizer`` merges the label names over all of ``X``. Every text in ``X`` is
      processed, so limit ``X`` yourself.
    - ``fit(X)`` with ``labels``: segments ``X`` with the given labels.
    - ``predict(X)``: segments with ``labels`` if given, otherwise with the fitted
      ``labels_``. The LLM may only use those labels. Called on the training texts, it
      can give different boundaries than ``segmentations_``, because it prompts again.

    Args:
        labels: Allowed labels, or a mapping from label to a short description.
        client: Model name for the OpenAI API, or a ``BaseLLMClient``. ``fit`` also
            uses it for ``Normalizer``.
        backend: An ``LLMBackend`` to use for splitting instead of ``client``.
        fuzzy_threshold: Minimum score for aligning a quote that is not verbatim.
        on_align_error: ``"raise"`` or ``"skip"`` a quote that cannot be aligned.
        on_unknown_label: What to do with a label outside the label set (only possible
            when the backend cannot enforce the schema): ``"closest"`` maps it to the
            most similar allowed label, ``"drop"`` discards the segment, ``"raise"``
            raises ``ValueError``.
        timeout: Request timeout when ``client`` is a model name.

    Attributes (after ``fit``):
        segmentations_: Segmentations of the training texts.
        labels_: The label set, sorted.
        mapping_: Free label → label in ``labels_`` (empty when ``labels`` is given).
        fit_reports_: Fidelity of the LLM's quotes for each training text.

    Gaps between aligned quotes are joined to the preceding segment, so every
    segmentation covers its text exactly, even when the LLM skipped text. Use
    ``predict_with_report`` (or ``fit_reports_``) to see how much it skipped. A text
    with no aligned quote gets ``[]``.
    """

    def __init__(
        self,
        labels: Sequence[str] | Mapping[str, str] | None = None,
        client: Union[BaseLLMClient, str] = "gpt-4o",
        backend: LLMBackend | None = None,
        fuzzy_threshold: int = 85,
        on_align_error: Literal["raise", "skip"] = "raise",
        on_unknown_label: Literal["closest", "drop", "raise"] = "closest",
        timeout: float | None = 120.0,
    ) -> None:
        self.labels = labels
        self.client = client
        self.backend = backend
        self.fuzzy_threshold = fuzzy_threshold
        self.on_align_error = on_align_error
        self.on_unknown_label = on_unknown_label
        self.timeout = timeout

    def fit(self, X: Sequence[str], y: None = None) -> Self:
        """Find a label set in ``X`` (unless ``labels`` is given) and segment ``X``."""
        _check_texts(X)
        if self.labels is not None:
            labels = self._label_descriptions()
            backend = self._make_backend()
            results = [self._segment(text, backend, labels) for text in X]
            self.segmentations_ = [segments for segments, _ in results]
            self.fit_reports_ = [report for _, report in results]
            self.labels_ = sorted(labels)
            self.mapping_: dict[str, str] = {}
            return self

        backend = self._make_backend()
        resolved = []
        self.fit_reports_ = []
        for text in X:
            raw_chunks = [c for c in backend.extract_chunks(text) if c.category.strip()]
            details = align_detailed(raw_chunks, text, self.fuzzy_threshold)
            if self.on_align_error == "raise":
                _raise_unaligned(raw_chunks, details, self.fuzzy_threshold)
            resolved.append(resolve_spans(raw_chunks, [d.span for d in details]))
            self.fit_reports_.append(fidelity_from_details(details, text))

        free = sorted({label for segments in resolved for _, _, label in segments})
        mapping: dict[str, str] = {}
        if free:
            normalizer = Normalizer(client=self._make_client())
            mapping = normalizer.build_mapping(
                [Chunk(category=label, quote="", start=0, end=0) for label in free]
            )
        self.mapping_ = {label: _unified(mapping, label) for label in free}
        self.labels_ = sorted(set(self.mapping_.values()))
        self.segmentations_ = [
            absorb_gaps([(s, e, self.mapping_[label]) for s, e, label in segments], len(text))
            for segments, text in zip(resolved, X)
        ]
        return self

    def fit_predict(self, X: Sequence[str], y: None = None) -> list[list[Segment]]:
        """``fit(X)`` and return ``segmentations_``. One pass over ``X``, no re-labeling."""
        return self.fit(X).segmentations_

    def predict(self, X: Sequence[str]) -> list[list[Segment]]:
        """Return one segmentation per text in ``X``, using only the label set."""
        return [segments for segments, _ in self._predict(X, strict=True)]

    def predict_with_report(
        self, X: Sequence[str]
    ) -> tuple[list[list[Segment]], list[FidelityReport]]:
        """Like ``predict``, but also report how faithfully the LLM quoted each text.

        Never raises on alignment errors: unaligned quotes are counted and dropped. The
        report covers the quotes after label mapping.
        """
        results = self._predict(X, strict=False)
        return [s for s, _ in results], [r for _, r in results]

    def _predict(
        self, X: Sequence[str], strict: bool
    ) -> list[tuple[list[Segment], FidelityReport]]:
        _check_texts(X)
        labels = self._label_descriptions()
        backend = self._make_backend()
        return [self._segment(text, backend, labels, strict=strict) for text in X]

    def _label_descriptions(self) -> dict[str, str]:
        labels = self.labels
        if labels is None:
            labels = getattr(self, "labels_", None)
            if labels is None:
                raise NotFittedError(
                    "This ZeroShotPromptSegmenter has no label set: call fit() first "
                    "or pass labels"
                )
            if not labels:
                raise ValueError("fit() found no labels: no quote could be aligned")
        if isinstance(labels, str):
            raise TypeError("labels must be a sequence or mapping of labels, not a string")
        if isinstance(labels, Mapping):
            descriptions = {str(k): str(v) for k, v in labels.items()}
        else:
            descriptions = {str(label): "" for label in labels}
        if not descriptions or not all(label.strip() for label in descriptions):
            raise ValueError("labels must contain at least one non-blank label")
        return descriptions


class FewShotPromptSegmenter(_PromptSegmenter):
    """Split texts into labeled segments by prompting an LLM with labeled examples.

    ``fit(X, y)`` keeps the labeled texts as an example pool; it makes no LLM call.
    ``predict`` puts ``n_examples`` of them in the prompt as user / assistant turns,
    and restricts labels to those in ``y``.

    Args:
        n_examples: Number of examples per prompt.
        example_selection: ``"random"`` uses the same examples, drawn once with
            ``random_state``, for every text. ``"similar"`` uses, for each text, the
            pool texts most similar to it; the most similar comes last.
        embedding: For ``"similar"``: a function from a list of texts to a list of
            vectors (for example a sentence-transformers model's ``encode``). By
            default, TF-IDF over character 3-grams is used.
        random_state: Seed for ``"random"``.
        client, backend, fuzzy_threshold, on_align_error, on_unknown_label, timeout:
            As in ``ZeroShotPromptSegmenter``.

    A pool text identical to the input is never used as its example. Each example
    costs about twice its length in prompt tokens (text plus expected JSON), so lower
    ``n_examples`` for long texts.

    Attributes (after ``fit``):
        labels_: Labels that occur in ``y``, sorted.
        example_pool_: The labeled ``(text, segmentation)`` pairs. Texts with
            ``y[i] is None`` are accepted and not used.
    """

    def __init__(
        self,
        n_examples: int = 3,
        example_selection: Literal["random", "similar"] = "random",
        embedding: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
        random_state: int | None = None,
        client: Union[BaseLLMClient, str] = "gpt-4o",
        backend: LLMBackend | None = None,
        fuzzy_threshold: int = 85,
        on_align_error: Literal["raise", "skip"] = "raise",
        on_unknown_label: Literal["closest", "drop", "raise"] = "closest",
        timeout: float | None = 120.0,
    ) -> None:
        self.n_examples = n_examples
        self.example_selection = example_selection
        self.embedding = embedding
        self.random_state = random_state
        self.client = client
        self.backend = backend
        self.fuzzy_threshold = fuzzy_threshold
        self.on_align_error = on_align_error
        self.on_unknown_label = on_unknown_label
        self.timeout = timeout

    def fit(self, X: Sequence[str], y: Sequence[Sequence[Segment] | None]) -> Self:
        """Check ``y`` and keep the labeled texts as the example pool."""
        _check_texts(X)
        if self.n_examples < 0:
            raise ValueError("n_examples must be >= 0")
        if self.example_selection not in ("random", "similar"):
            raise ValueError(f"Unknown example_selection: {self.example_selection!r}")
        if len(y) != len(X):
            raise ValueError(f"X and y have different lengths: {len(X)} and {len(y)}")
        pool: list[tuple[str, list[Segment]]] = []
        for i, (text, segmentation) in enumerate(zip(X, y)):
            if segmentation is not None:
                pool.append((text, _check_segmentation(segmentation, text, i)))
        labels = sorted({label for _, segs in pool for _, _, label in segs})
        if not labels:
            raise ValueError("y has no labeled segment")

        self.labels_ = labels
        self.example_pool_ = pool
        texts = [text for text, _ in pool]
        if self.example_selection == "random":
            self.example_order_ = random.Random(self.random_state).sample(
                range(len(pool)), len(pool)
            )
        elif self.embedding is None:
            self.vectorizer_: _CharNgramTfidf | None = _CharNgramTfidf().fit(texts)
            self.pool_vectors_ = self.vectorizer_.transform(texts)
        else:
            self.vectorizer_ = None
            self.pool_vectors_ = _dense_to_vectors(self.embedding(texts))
        return self

    def predict(self, X: Sequence[str]) -> list[list[Segment]]:
        """Return one segmentation per text in ``X``, using only ``labels_``."""
        return [segments for segments, _ in self._predict(X, strict=True)]

    def predict_with_report(
        self, X: Sequence[str]
    ) -> tuple[list[list[Segment]], list[FidelityReport]]:
        """Like ``predict``, but also report how faithfully the LLM quoted each text.

        Never raises on alignment errors: unaligned quotes are counted and dropped. The
        report covers the quotes after label mapping.
        """
        results = self._predict(X, strict=False)
        return [s for s, _ in results], [r for _, r in results]

    def _predict(
        self, X: Sequence[str], strict: bool
    ) -> list[tuple[list[Segment], FidelityReport]]:
        _check_texts(X)
        if not hasattr(self, "example_pool_"):
            raise NotFittedError("This FewShotPromptSegmenter is not fitted: call fit() first")
        labels = {label: "" for label in self.labels_}
        backend = self._make_backend()
        chosen = self.select_examples(X)
        return [
            self._segment(text, backend, labels, [self._as_raw(k) for k in indices], strict)
            for text, indices in zip(X, chosen)
        ]

    def select_examples(self, X: Sequence[str]) -> list[list[int]]:
        """Indices into ``example_pool_`` of the examples used for each text, in prompt
        order."""
        _check_texts(X)
        if not hasattr(self, "example_pool_"):
            raise NotFittedError("This FewShotPromptSegmenter is not fitted: call fit() first")
        pool_texts = [text for text, _ in self.example_pool_]
        if self.example_selection == "random":
            return [
                [k for k in self.example_order_ if pool_texts[k] != text][: self.n_examples]
                for text in X
            ]
        if self.vectorizer_ is not None:
            vectors = self.vectorizer_.transform(list(X))
        else:
            assert self.embedding is not None
            vectors = _dense_to_vectors(self.embedding(list(X)))
        chosen = []
        for text, vector in zip(X, vectors):
            scored = [
                (_dot(vector, pool_vector), -k)
                for k, pool_vector in enumerate(self.pool_vectors_)
                if pool_texts[k] != text
            ]
            top = sorted(scored, reverse=True)[: self.n_examples]
            chosen.append([-neg_k for _, neg_k in reversed(top)])
        return chosen

    def _as_raw(self, k: int) -> tuple[str, list[RawChunk]]:
        text, segmentation = self.example_pool_[k]
        return text, [RawChunk(category=label, quote=text[s:e]) for s, e, label in segmentation]


def _raise_unaligned(
    raw_chunks: Sequence[RawChunk], details: Sequence[AlignmentDetail], threshold: int
) -> None:
    """Raise like ``align(on_error="raise")`` for the first quote that did not align."""
    for chunk, detail in zip(raw_chunks, details):
        if detail.span is None:
            raise AlignmentError(
                f"Could not align quote (score={detail.score:.1f} < threshold={threshold}): "
                f"{chunk.quote!r}"
            )


def _check_texts(X: Sequence[str]) -> None:
    if isinstance(X, str):
        raise TypeError("X must be a sequence of texts, not a single string")


def _unified(mapping: Mapping[str, str], label: str) -> str:
    """The label ``Normalizer`` maps ``label`` to; itself if missing or blank."""
    target = mapping.get(label, label).strip()
    return target or label


def map_labels(
    raw_chunks: list[RawChunk],
    labels: list[str],
    on_unknown: Literal["closest", "drop", "raise"],
) -> list[RawChunk]:
    """Keep chunks whose category is in ``labels``; handle the others per ``on_unknown``.

    A category that differs from a label only in case or surrounding whitespace counts
    as that label.
    """
    by_key = {label.strip().lower(): label for label in labels}
    out: list[RawChunk] = []
    for chunk in raw_chunks:
        label = by_key.get(chunk.category.strip().lower())
        if label is None:
            if on_unknown == "raise":
                raise ValueError(f"LLM returned a label outside the given set: {chunk.category!r}")
            if on_unknown == "drop":
                continue
            match = process.extractOne(
                chunk.category.strip().lower(), list(by_key), scorer=fuzz.ratio
            )
            label = by_key[match[0]]
        out.append(RawChunk(category=label, quote=chunk.quote))
    return out


def _check_segmentation(segmentation: Sequence[Segment], text: str, i: int) -> list[Segment]:
    out: list[Segment] = []
    for triple in segmentation:
        start, end, label = triple
        if not (isinstance(label, str) and label.strip()):
            raise ValueError(f"y[{i}]: labels must be non-blank strings, got {label!r}")
        if not 0 <= start < end <= len(text):
            raise ValueError(f"y[{i}]: segment {triple!r} is outside the text")
        if out and start < out[-1][1]:
            raise ValueError(f"y[{i}]: segments must be sorted and must not overlap")
        out.append((start, end, label))
    return out


class _CharNgramTfidf:
    """TF-IDF over character n-grams (default 3), L2-normalized; pure Python.

    Character n-grams need no tokenizer, so this also works for languages written
    without spaces.
    """

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self.idf: dict[str, float] = {}

    def _grams(self, text: str) -> Counter[str]:
        t = re.sub(r"\s+", " ", text.lower()).strip()
        if len(t) <= self.n:
            return Counter([t]) if t else Counter()
        return Counter(t[i : i + self.n] for i in range(len(t) - self.n + 1))

    def fit(self, texts: Sequence[str]) -> _CharNgramTfidf:
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(self._grams(text)))
        n = len(texts)
        # Smoothed IDF, as scikit-learn's TfidfVectorizer(smooth_idf=True).
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
        return self

    def transform(self, texts: Sequence[str]) -> list[Vector]:
        out = []
        for text in texts:
            v: Vector = {g: c * self.idf[g] for g, c in self._grams(text).items() if g in self.idf}
            out.append(_normalize(v))
        return out


def _dense_to_vectors(rows: Sequence[Sequence[float]]) -> list[Vector]:
    return [_normalize({i: float(x) for i, x in enumerate(row)}) for row in rows]


def _normalize(v: Vector) -> Vector:
    norm = math.sqrt(sum(x * x for x in v.values()))
    return {k: x / norm for k, x in v.items()} if norm else v


def _dot(a: Vector, b: Vector) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(k, 0.0) for k, x in a.items())
