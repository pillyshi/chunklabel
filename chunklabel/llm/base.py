from abc import ABC, abstractmethod

from chunklabel.types import Chunk, RawChunk


class LLMBackend(ABC):
    @abstractmethod
    def extract_chunks(self, text: str) -> list[RawChunk]: ...

    def extract_boundaries(self, text: str) -> list[RawChunk]:
        """Pass 1: return verbatim quote spans with empty category strings."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extract_boundaries(). "
            "Override this method to use two-pass mode."
        )

    def label_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Pass 2: return one label per chunk (same length as input), in order."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement label_chunks(). "
            "Override this method to use two-pass mode."
        )
