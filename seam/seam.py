from __future__ import annotations

from seam.alignment import align
from seam.llm.base import LLMBackend
from seam.postprocess import postprocess
from seam.types import Chunk


class Seam:
    def __init__(
        self,
        llm: object = None,
        fuzzy_threshold: int = 80,
        backend: LLMBackend | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        else:
            from seam.llm.langchain_backend import LangChainBackend

            self._backend = LangChainBackend(llm=llm)  # type: ignore[arg-type]
        self.fuzzy_threshold = fuzzy_threshold

    def split(self, text: str) -> list[Chunk]:
        raw_chunks = self._backend.extract_chunks(text)
        spans = align(raw_chunks, text, self.fuzzy_threshold)
        return postprocess(raw_chunks, spans, text)
