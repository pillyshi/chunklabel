from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing_extensions import Self

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
            self._client: BaseLLMClient | None = OpenAIClient(model=client)
        else:
            self._client: BaseLLMClient | None = client

    @classmethod
    def _from_mapping(cls, mapping: dict[str, str]) -> Self:
        obj = object.__new__(cls)
        obj._client = None
        obj._mapping = mapping
        return obj

    def build_mapping(self, chunks: list[Chunk]) -> dict[str, str]:
        if self._client is None:
            raise ValueError("No LLM client available. Instantiate Normalizer with a client to call build_mapping.")
        categories = sorted({c.category for c in chunks if c.category})
        if not categories:
            self._mapping = {}
            return {}
        result = self._client.complete_structured(
            [{"role": "system", "content": NORMALIZE_SYSTEM}, {"role": "user", "content": json.dumps(categories, indent=2)}],
            _MappingSchema,
        )
        self._mapping = result.mapping
        return dict(self._mapping)

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
    def load(cls, path: str | Path) -> Self:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(k, str) and k and isinstance(v, str) for k, v in data.items()
        ):
            raise ValueError(
                f"Expected dict[str, str], got {type(data).__name__} (values must all be strings)"
            )
        return cls._from_mapping(data)
