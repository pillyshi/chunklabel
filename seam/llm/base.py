from abc import ABC, abstractmethod

from seam.types import RawChunk


class LLMBackend(ABC):
    @abstractmethod
    def extract_chunks(self, text: str) -> list[RawChunk]: ...

    def build_category_mapping(self, categories: list[str]) -> dict[str, str]: ...
