"""scikit-learn style estimators for segmentation and labeling.

A segmentation of a text is a list of ``(start, end, label)`` triples that cover the
text without overlapping; ``text[start:end]`` is the segment's text. See
``research/ideas/sklearn-estimator-api.md`` for the design.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Union

from rapidfuzz import fuzz, process
from typing_extensions import Self

from chunklabel.alignment import align
from chunklabel.llm.base import LLMBackend
from chunklabel.llm.client import BaseLLMClient
from chunklabel.normalizer import Normalizer
from chunklabel.postprocess import absorb_gaps, resolve_spans
from chunklabel.types import Chunk, RawChunk

Segment = tuple[int, int, str]


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


class ZeroShotPromptSegmenter(_ParamsMixin):
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

    Gaps between aligned quotes are joined to the preceding segment, so every
    segmentation covers its text exactly. A text with no aligned quote gets ``[]``.
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
            self.segmentations_ = [self._segment(text, backend, labels) for text in X]
            self.labels_ = sorted(labels)
            self.mapping_: dict[str, str] = {}
            return self

        backend = self._make_backend()
        resolved = []
        for text in X:
            raw_chunks = [c for c in backend.extract_chunks(text) if c.category.strip()]
            spans = align(raw_chunks, text, self.fuzzy_threshold, on_error=self.on_align_error)
            resolved.append(resolve_spans(raw_chunks, spans))

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
        _check_texts(X)
        labels = self._label_descriptions()
        backend = self._make_backend()
        return [self._segment(text, backend, labels) for text in X]

    def _segment(
        self, text: str, backend: LLMBackend, labels: Mapping[str, str]
    ) -> list[Segment]:
        raw_chunks = map_labels(
            backend.extract_labeled_chunks(text, labels), list(labels), self.on_unknown_label
        )
        spans = align(raw_chunks, text, self.fuzzy_threshold, on_error=self.on_align_error)
        return absorb_gaps(resolve_spans(raw_chunks, spans), len(text))

    def _label_descriptions(self) -> dict[str, str]:
        labels = self.labels
        if labels is None:
            labels = getattr(self, "labels_", None)
            if labels is None:
                raise ValueError(
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
