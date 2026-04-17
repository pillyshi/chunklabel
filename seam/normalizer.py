from __future__ import annotations

import dataclasses

from seam.llm.base import LLMBackend
from seam.types import Chunk


class Normalizer:
    def __init__(self, backend: LLMBackend | None = None) -> None:
        if backend is not None:
            self._backend = backend
        else:
            from seam.llm.langchain_backend import LangChainBackend

            self._backend = LangChainBackend()

    def build_mapping(self, chunks: list[Chunk]) -> dict[str, str]:
        categories = sorted({c.category for c in chunks})
        return self._backend.build_category_mapping(categories)

    def apply(self, chunks: list[Chunk], mapping: dict[str, str]) -> list[Chunk]:
        return [
            dataclasses.replace(chunk, category=mapping.get(chunk.category, chunk.category))
            for chunk in chunks
        ]
