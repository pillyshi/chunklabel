from chunklabel.llm.base import LLMBackend
from chunklabel.chunklabeler import ChunkLabeler
from chunklabel.types import Chunk, RawChunk

TEXT = (
    "The project kicked off in January with a small team. "
    "Budget constraints forced a scope reduction in March. "
    "Despite the setbacks, the product launched successfully in June."
)


class MockBackend(LLMBackend):
    def extract_chunks(self, text: str) -> list[RawChunk]:
        return [
            RawChunk(category="initiation", quote="The project kicked off in January with a small team"),
            RawChunk(category="obstacle", quote="Budget constraints forced a scope reduction in March"),
            RawChunk(category="outcome", quote="the product launched successfully in June"),
        ]

    def build_category_mapping(self, categories: list[str]) -> dict[str, str]:
        return {c: c for c in categories}


def test_full_pipeline_returns_chunks() -> None:
    labeler = ChunkLabeler(backend=MockBackend())
    chunks = labeler.split(TEXT)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_positions_are_valid() -> None:
    labeler = ChunkLabeler(backend=MockBackend())
    chunks = labeler.split(TEXT)
    for chunk in chunks:
        assert 0 <= chunk.start < chunk.end <= len(TEXT)
        assert TEXT[chunk.start:chunk.end] == chunk.quote


def test_expected_categories_present() -> None:
    labeler = ChunkLabeler(backend=MockBackend())
    chunks = labeler.split(TEXT)
    cats = {c.category for c in chunks}
    assert "initiation" in cats
    assert "obstacle" in cats
    assert "outcome" in cats


def test_chunks_cover_full_text() -> None:
    labeler = ChunkLabeler(backend=MockBackend())
    chunks = labeler.split(TEXT)
    reconstructed = "".join(c.quote for c in chunks)
    assert reconstructed == TEXT


class MockBackendWithBadChunk(LLMBackend):
    def extract_chunks(self, text: str) -> list[RawChunk]:
        return [
            RawChunk(category="initiation", quote="The project kicked off in January with a small team"),
            RawChunk(category="bad", quote="completely unrelated text that will never match"),
            RawChunk(category="outcome", quote="the product launched successfully in June"),
        ]

    def build_category_mapping(self, categories: list[str]) -> dict[str, str]:
        return {c: c for c in categories}


def test_skip_unaligned_covers_full_text() -> None:
    labeler = ChunkLabeler(backend=MockBackendWithBadChunk(), on_align_error="skip")
    chunks = labeler.split(TEXT)
    reconstructed = "".join(c.quote for c in chunks)
    assert reconstructed == TEXT


def test_skip_unaligned_bad_chunk_becomes_uncategorized() -> None:
    labeler = ChunkLabeler(backend=MockBackendWithBadChunk(), on_align_error="skip")
    chunks = labeler.split(TEXT)
    cats = {c.category for c in chunks}
    assert "bad" not in cats
    assert "uncategorized" in cats


class MockTwoPassBackend(LLMBackend):
    def extract_chunks(self, text: str) -> list[RawChunk]:
        raise NotImplementedError

    def build_category_mapping(self, categories: list[str]) -> dict[str, str]:
        return {c: c for c in categories}

    def extract_boundaries(self, text: str) -> list[RawChunk]:
        return [
            RawChunk(category="", quote="The project kicked off in January with a small team"),
            RawChunk(category="", quote="Budget constraints forced a scope reduction in March"),
            RawChunk(category="", quote="the product launched successfully in June"),
        ]

    def label_chunks(self, chunks: list[Chunk]) -> list[str]:
        mapping = {"kicked off": "initiation", "Budget": "obstacle", "launched": "outcome"}
        return [
            next((v for k, v in mapping.items() if k in chunk.quote), "transition")
            for chunk in chunks
        ]


def test_two_pass_returns_chunks() -> None:
    labeler = ChunkLabeler(backend=MockTwoPassBackend())
    chunks = labeler.split(TEXT, mode="two_pass")
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_two_pass_positions_are_valid() -> None:
    labeler = ChunkLabeler(backend=MockTwoPassBackend())
    chunks = labeler.split(TEXT, mode="two_pass")
    for chunk in chunks:
        assert 0 <= chunk.start < chunk.end <= len(TEXT)
        assert TEXT[chunk.start:chunk.end] == chunk.quote


def test_two_pass_covers_full_text() -> None:
    labeler = ChunkLabeler(backend=MockTwoPassBackend())
    chunks = labeler.split(TEXT, mode="two_pass")
    assert "".join(c.quote for c in chunks) == TEXT


def test_two_pass_no_empty_or_uncategorized() -> None:
    labeler = ChunkLabeler(backend=MockTwoPassBackend())
    chunks = labeler.split(TEXT, mode="two_pass")
    for chunk in chunks:
        assert chunk.category != ""
        assert chunk.category != "uncategorized"


def test_two_pass_expected_labels() -> None:
    labeler = ChunkLabeler(backend=MockTwoPassBackend())
    chunks = labeler.split(TEXT, mode="two_pass")
    cats = {c.category for c in chunks}
    assert "initiation" in cats
    assert "obstacle" in cats
    assert "outcome" in cats


def test_two_pass_raises_not_implemented_on_plain_backend() -> None:
    import pytest
    labeler = ChunkLabeler(backend=MockBackend())
    with pytest.raises(NotImplementedError):
        labeler.split(TEXT, mode="two_pass")
