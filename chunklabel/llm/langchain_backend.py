from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from chunklabel.llm.base import LLMBackend
from chunklabel.llm.prompts import BOUNDARY_SYSTEM, LABEL_SYSTEM, NORMALIZE_SYSTEM, SPLIT_SYSTEM
from chunklabel.types import Chunk, RawChunk


class _RawChunkSchema(BaseModel):
    category: str
    quote: str


class _ChunkListSchema(BaseModel):
    chunks: list[_RawChunkSchema]


class _MappingSchema(BaseModel):
    mapping: dict[str, str]


class _BoundarySchema(BaseModel):
    quote: str


class _BoundaryListSchema(BaseModel):
    segments: list[_BoundarySchema]


class _LabeledSegmentSchema(BaseModel):
    index: int
    label: str


class _LabelListSchema(BaseModel):
    segments: list[_LabeledSegmentSchema]


class LangChainBackend(LLMBackend):
    def __init__(self, llm: BaseChatModel | None = None, timeout: float | None = 120.0) -> None:
        if llm is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o")
        self._llm = llm
        self._timeout = timeout

    def extract_chunks(self, text: str) -> list[RawChunk]:
        structured: Any = self._llm.with_structured_output(_ChunkListSchema)
        result: _ChunkListSchema = structured.invoke(
            [SystemMessage(content=SPLIT_SYSTEM), HumanMessage(content=text)],
            config={"timeout": self._timeout},
        )
        return [RawChunk(category=c.category, quote=c.quote) for c in result.chunks]

    def extract_boundaries(self, text: str) -> list[RawChunk]:
        structured: Any = self._llm.with_structured_output(_BoundaryListSchema)
        result: _BoundaryListSchema = structured.invoke(
            [SystemMessage(content=BOUNDARY_SYSTEM), HumanMessage(content=text)],
            config={"timeout": self._timeout},
        )
        return [RawChunk(category="", quote=s.quote) for s in result.segments]

    def label_chunks(self, chunks: list[Chunk]) -> list[str]:
        lines = [f"{i + 1}. {chunk.quote}" for i, chunk in enumerate(chunks)]
        structured: Any = self._llm.with_structured_output(_LabelListSchema)
        result: _LabelListSchema = structured.invoke(
            [SystemMessage(content=LABEL_SYSTEM), HumanMessage(content="\n".join(lines))],
            config={"timeout": self._timeout},
        )
        sorted_segs = sorted(result.segments, key=lambda s: s.index)
        if len(sorted_segs) != len(chunks):
            raise ValueError(
                f"label_chunks: expected {len(chunks)} labels, got {len(sorted_segs)}"
            )
        return [s.label for s in sorted_segs]

    def build_category_mapping(self, categories: list[str]) -> dict[str, str]:
        prompt = f"{json.dumps(sorted(categories), indent=2)}"
        structured: Any = self._llm.with_structured_output(_MappingSchema)
        result: _MappingSchema = structured.invoke(
            [SystemMessage(content=NORMALIZE_SYSTEM), HumanMessage(content=prompt)],
            config={"timeout": self._timeout},
        )
        return result.mapping
