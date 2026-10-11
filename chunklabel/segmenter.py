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
from chunklabel.postprocess import absorb_gaps, resolve_spans
from chunklabel.types import RawChunk

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

    This version needs a given label set: the LLM may only use ``labels``. Inducing a
    label set with ``fit`` is not implemented yet.

    Args:
        labels: Allowed labels, or a mapping from label to a short description.
        client: Model name for the OpenAI API, or a ``BaseLLMClient``.
        backend: An ``LLMBackend`` to use instead of ``client``.
        fuzzy_threshold: Minimum score for aligning a quote that is not verbatim.
        on_align_error: ``"raise"`` or ``"skip"`` a quote that cannot be aligned.
        on_unknown_label: What to do with a label outside ``labels`` (only possible
            when the backend cannot enforce the schema): ``"closest"`` maps it to the
            most similar allowed label, ``"drop"`` discards the segment, ``"raise"``
            raises ``ValueError``.
        timeout: Request timeout when ``client`` is a model name.

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

    def predict(self, X: Sequence[str]) -> list[list[Segment]]:
        """Return one segmentation per text in ``X``."""
        if isinstance(X, str):
            raise TypeError("X must be a sequence of texts, not a single string")
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
        if self.labels is None:
            raise ValueError(
                "labels must be given: inducing labels with fit() is not implemented yet"
            )
        if isinstance(self.labels, str):
            raise TypeError("labels must be a sequence or mapping of labels, not a string")
        if isinstance(self.labels, Mapping):
            descriptions = {str(k): str(v) for k, v in self.labels.items()}
        else:
            descriptions = {str(label): "" for label in self.labels}
        if not descriptions or not all(label.strip() for label in descriptions):
            raise ValueError("labels must contain at least one non-blank label")
        return descriptions

    def _make_backend(self) -> LLMBackend:
        if self.backend is not None:
            return self.backend
        from chunklabel.llm.backend import ClientBackend
        from chunklabel.llm.client import OpenAIClient

        if isinstance(self.client, str):
            return ClientBackend(OpenAIClient(model=self.client, timeout=self.timeout))
        return ClientBackend(self.client)


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
