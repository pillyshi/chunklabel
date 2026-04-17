from seam.llm.base import LLMBackend
from seam.seam import Seam
from seam.types import Chunk, RawChunk

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
    seam = Seam(backend=MockBackend())
    chunks = seam.split(TEXT)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_positions_are_valid() -> None:
    seam = Seam(backend=MockBackend())
    chunks = seam.split(TEXT)
    for chunk in chunks:
        assert 0 <= chunk.start < chunk.end <= len(TEXT)
        assert TEXT[chunk.start:chunk.end] == chunk.quote


def test_expected_categories_present() -> None:
    seam = Seam(backend=MockBackend())
    chunks = seam.split(TEXT)
    cats = {c.category for c in chunks}
    assert "initiation" in cats
    assert "obstacle" in cats
    assert "outcome" in cats


def test_chunks_cover_full_text() -> None:
    seam = Seam(backend=MockBackend())
    chunks = seam.split(TEXT)
    reconstructed = "".join(c.quote for c in chunks)
    assert reconstructed == TEXT
