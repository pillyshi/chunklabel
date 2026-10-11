from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field, JsonValue, create_model

from chunklabel.llm.base import LLMBackend
from chunklabel.llm.client import BaseLLMClient
from chunklabel.llm.prompts import (
    BOUNDARY_SYSTEM,
    GIVEN_LABELS_SYSTEM,
    LABEL_SYSTEM,
    SPLIT_SYSTEM,
)
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


def _labeled_chunk_list_schema(labels: list[str]) -> type[BaseModel]:
    # The category is a plain str with an enum in the JSON schema: backends that honour
    # the schema (OpenAI structured outputs, llama.cpp grammars) only produce allowed
    # labels, while validation stays lenient so that the caller decides what to do
    # with a label outside the set.
    enum: list[JsonValue] = list(labels)
    chunk = create_model(
        "_LabeledChunkSchema",
        category=(str, Field(json_schema_extra={"enum": enum})),
        quote=(str, ...),
    )
    return create_model("_LabeledChunkListSchema", chunks=(list[chunk], ...))  # type: ignore[valid-type]


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

    def extract_labeled_chunks(self, text: str, labels: Mapping[str, str]) -> list[RawChunk]:
        categories = "\n".join(
            f"- {name}: {desc}" if desc else f"- {name}" for name, desc in labels.items()
        )
        result = self._client.complete_structured(
            [
                {"role": "system", "content": GIVEN_LABELS_SYSTEM.format(categories=categories)},
                {"role": "user", "content": text},
            ],
            _labeled_chunk_list_schema(list(labels)),
        )
        return [RawChunk(category=c.category, quote=c.quote) for c in result.chunks]  # type: ignore[attr-defined]

    def label_chunks(self, chunks: list[Chunk]) -> list[str]:
        labels = []
        for chunk in chunks:
            result = self._client.complete_structured(
                [{"role": "system", "content": LABEL_SYSTEM}, {"role": "user", "content": chunk.quote}],
                _SingleLabelSchema,
            )
            labels.append(result.label)
        return labels

