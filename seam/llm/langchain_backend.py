from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from seam.llm.base import LLMBackend
from seam.llm.prompts import NORMALIZE_SYSTEM, SPLIT_SYSTEM
from seam.types import RawChunk


class _RawChunkSchema(BaseModel):
    category: str
    quote: str


class _ChunkListSchema(BaseModel):
    chunks: list[_RawChunkSchema]


class _MappingSchema(BaseModel):
    mapping: dict[str, str]


class LangChainBackend(LLMBackend):
    def __init__(self, llm: BaseChatModel | None = None) -> None:
        if llm is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o")
        self._llm = llm

    def extract_chunks(self, text: str) -> list[RawChunk]:
        structured: Any = self._llm.with_structured_output(_ChunkListSchema)
        result: _ChunkListSchema = structured.invoke(
            [SystemMessage(content=SPLIT_SYSTEM), HumanMessage(content=text)]
        )
        return [RawChunk(category=c.category, quote=c.quote) for c in result.chunks]

    def build_category_mapping(self, categories: list[str]) -> dict[str, str]:
        prompt = f"{json.dumps(sorted(categories), indent=2)}"
        structured: Any = self._llm.with_structured_output(_MappingSchema)
        result: _MappingSchema = structured.invoke(
            [SystemMessage(content=NORMALIZE_SYSTEM), HumanMessage(content=prompt)]
        )
        return result.mapping
