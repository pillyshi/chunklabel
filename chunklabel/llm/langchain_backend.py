from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from chunklabel.llm.base import LLMBackend
from chunklabel.llm.prompts import NORMALIZE_SYSTEM, SPLIT_SYSTEM
from chunklabel.types import RawChunk


class _RawChunkSchema(BaseModel):
    category: str
    quote: str


class _ChunkListSchema(BaseModel):
    chunks: list[_RawChunkSchema]


class _MappingSchema(BaseModel):
    mapping: dict[str, str]


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

    def build_category_mapping(self, categories: list[str]) -> dict[str, str]:
        prompt = f"{json.dumps(sorted(categories), indent=2)}"
        structured: Any = self._llm.with_structured_output(_MappingSchema)
        result: _MappingSchema = structured.invoke(
            [SystemMessage(content=NORMALIZE_SYSTEM), HumanMessage(content=prompt)],
            config={"timeout": self._timeout},
        )
        return result.mapping
