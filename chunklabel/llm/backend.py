from __future__ import annotations


from pydantic import BaseModel

from chunklabel.llm.base import LLMBackend
from chunklabel.llm.client import BaseLLMClient
from chunklabel.llm.prompts import BOUNDARY_SYSTEM, LABEL_SYSTEM, SPLIT_SYSTEM
from chunklabel.types import Chunk, RawChunk


class _RawChunkSchema(BaseModel):
    category: str
    quote: str


class _ChunkListSchema(BaseModel):
    chunks: list[_RawChunkSchema]


class _BoundarySchema(BaseModel):
    quote: str


class _BoundaryListSchema(BaseModel):
    segments: list[_BoundarySchema]


class _SingleLabelSchema(BaseModel):
    label: str


class ClientBackend(LLMBackend):
    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client

    def extract_chunks(self, text: str) -> list[RawChunk]:
        result = self._client.complete_structured(
            [{"role": "system", "content": SPLIT_SYSTEM}, {"role": "user", "content": text}],
            _ChunkListSchema,
        )
        return [RawChunk(category=c.category, quote=c.quote) for c in result.chunks]

    def extract_boundaries(self, text: str) -> list[RawChunk]:
        result = self._client.complete_structured(
            [{"role": "system", "content": BOUNDARY_SYSTEM}, {"role": "user", "content": text}],
            _BoundaryListSchema,
        )
        return [RawChunk(category="", quote=s.quote) for s in result.segments]

    def label_chunks(self, chunks: list[Chunk]) -> list[str]:
        labels = []
        for chunk in chunks:
            result = self._client.complete_structured(
                [{"role": "system", "content": LABEL_SYSTEM}, {"role": "user", "content": chunk.quote}],
                _SingleLabelSchema,
            )
            labels.append(result.label)
        return labels

