from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from pydantic import BaseModel

from chunklabel.llm.client import BaseLLMClient, OpenAIClient
from chunklabel.llm.prompts import NORMALIZE_SYSTEM
from chunklabel.types import Chunk


class _MappingSchema(BaseModel):
    mapping: dict[str, str]


class Normalizer:
    def __init__(self, client: BaseLLMClient | str = "gpt-4o") -> None:
        self._mapping: dict[str, str] | None = None
        if isinstance(client, str):
            self._client: BaseLLMClient = OpenAIClient(model=client)
        else:
            self._client = client

    def build_mapping(self, chunks: list[Chunk]) -> dict[str, str]:
        categories = sorted({c.category for c in chunks})
        result = self._client.complete_structured(
            [{"role": "system", "content": NORMALIZE_SYSTEM}, {"role": "user", "content": json.dumps(categories, indent=2)}],
            _MappingSchema,
        )
        self._mapping = result.mapping
        return self._mapping

    def apply(self, chunks: list[Chunk], mapping: dict[str, str] | None = None) -> list[Chunk]:
        m = mapping if mapping is not None else self._mapping
        if m is None:
            raise ValueError("No mapping available. Call build_mapping first or pass a mapping.")
        return [
            dataclasses.replace(chunk, category=m.get(chunk.category, chunk.category))
            for chunk in chunks
        ]

    def save(self, path: str | Path) -> None:
        if self._mapping is None:
            raise ValueError("No mapping to save. Call build_mapping first.")
        Path(path).write_text(json.dumps(self._mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        obj = cls.__new__(cls)
        obj._mapping = json.loads(Path(path).read_text(encoding="utf-8"))
        return obj
