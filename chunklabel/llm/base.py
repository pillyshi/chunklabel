from abc import ABC, abstractmethod
from collections.abc import Mapping

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

    def extract_labeled_chunks(self, text: str, labels: Mapping[str, str]) -> list[RawChunk]:
        """Like ``extract_chunks``, but categories should come from ``labels``.

        ``labels`` maps each allowed category to a description ("" for none). Backends
        should restrict categories to these keys where they can; callers still check.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extract_labeled_chunks(). "
            "Override this method to use a given label set."
        )

    def label_chunks(self, chunks: list[Chunk]) -> list[str]:
        """Pass 2: return one label per chunk (same length as input), in order."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement label_chunks(). "
            "Override this method to use two-pass mode."
        )
