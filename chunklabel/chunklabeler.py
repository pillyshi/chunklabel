from __future__ import annotations

import dataclasses
from typing import Literal

from chunklabel.alignment import align
from chunklabel.llm.base import LLMBackend
from chunklabel.postprocess import postprocess
from chunklabel.types import Chunk


class ChunkLabeler:
    def __init__(
        self,
        llm: object = None,
        fuzzy_threshold: int = 80,
        on_align_error: Literal["raise", "skip"] = "raise",
        timeout: float | None = 120.0,
        backend: LLMBackend | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        else:
            from chunklabel.llm.langchain_backend import LangChainBackend

            self._backend = LangChainBackend(llm=llm, timeout=timeout)  # type: ignore[arg-type]
        self.fuzzy_threshold = fuzzy_threshold
        self.on_align_error = on_align_error

    def split(self, text: str) -> list[Chunk]:
        raw_chunks = self._backend.extract_chunks(text)
        spans = align(raw_chunks, text, self.fuzzy_threshold, on_error=self.on_align_error)
        return postprocess(raw_chunks, spans, text)

    def split_two_pass(self, text: str) -> list[Chunk]:
        # Pass 1 — boundary detection only
        raw_chunks = self._backend.extract_boundaries(text)
        spans = align(raw_chunks, text, self.fuzzy_threshold, on_error=self.on_align_error)
        chunks = postprocess(raw_chunks, spans, text)

        # Pass 2 — single batched labeling call (covers gap-fills too)
        labels = self._backend.label_chunks(chunks)
        return [dataclasses.replace(chunk, category=label) for chunk, label in zip(chunks, labels)]
