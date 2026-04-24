from __future__ import annotations

from typing import Literal

from seam.alignment import align
from seam.llm.base import LLMBackend
from seam.postprocess import postprocess
from seam.types import Chunk


class Seam:
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
            from seam.llm.langchain_backend import LangChainBackend

            self._backend = LangChainBackend(llm=llm, timeout=timeout)  # type: ignore[arg-type]
        self.fuzzy_threshold = fuzzy_threshold
        self.on_align_error = on_align_error

    def split(self, text: str) -> list[Chunk]:
        raw_chunks = self._backend.extract_chunks(text)
        spans = align(raw_chunks, text, self.fuzzy_threshold, on_error=self.on_align_error)
        return postprocess(raw_chunks, spans, text)
