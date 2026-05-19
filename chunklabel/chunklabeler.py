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

    def split(self, text: str, mode: Literal["one_pass", "two_pass"] = "one_pass") -> list[Chunk]:
        if mode == "two_pass":
            raw_chunks = self._backend.extract_boundaries(text)
            spans = align(raw_chunks, text, self.fuzzy_threshold, on_error=self.on_align_error)
            chunks = postprocess(raw_chunks, spans, text)

            non_ws_indices = [i for i, c in enumerate(chunks) if c.quote.strip()]
            labels_from_llm = self._backend.label_chunks([chunks[i] for i in non_ws_indices])
            label_map = dict(zip(non_ws_indices, labels_from_llm))

            return [
                dataclasses.replace(c, category=label_map.get(i, "whitespace"))
                for i, c in enumerate(chunks)
            ]

        raw_chunks = self._backend.extract_chunks(text)
        spans = align(raw_chunks, text, self.fuzzy_threshold, on_error=self.on_align_error)
        return postprocess(raw_chunks, spans, text)
