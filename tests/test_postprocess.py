from seam.postprocess import postprocess
from seam.types import RawChunk

TEXT = "abcdefghij"  # len=10


def _raw(category: str) -> RawChunk:
    return RawChunk(category=category, quote="")


def test_no_gap_no_overlap() -> None:
    raw = [_raw("a"), _raw("b")]
    spans = [(0, 5), (5, 10)]
    chunks = postprocess(raw, spans, TEXT)
    assert len(chunks) == 2
    assert chunks[0].start == 0 and chunks[0].end == 5
    assert chunks[1].start == 5 and chunks[1].end == 10


def test_gap_between_chunks_filled() -> None:
    raw = [_raw("a"), _raw("b")]
    spans = [(0, 3), (7, 10)]
    chunks = postprocess(raw, spans, TEXT)
    assert len(chunks) == 3
    gap = chunks[1]
    assert gap.category == "uncategorized"
    assert gap.start == 3 and gap.end == 7
    assert gap.quote == TEXT[3:7]


def test_leading_gap_filled() -> None:
    raw = [_raw("a")]
    spans = [(3, 8)]
    chunks = postprocess(raw, spans, TEXT)
    assert chunks[0].category == "uncategorized"
    assert chunks[0].start == 0 and chunks[0].end == 3


def test_trailing_gap_filled() -> None:
    raw = [_raw("a")]
    spans = [(0, 5)]
    chunks = postprocess(raw, spans, TEXT)
    assert chunks[-1].category == "uncategorized"
    assert chunks[-1].start == 5 and chunks[-1].end == 10


def test_overlap_earlier_wins() -> None:
    raw = [_raw("a"), _raw("b")]
    spans = [(0, 7), (5, 10)]  # overlap at 5-7
    chunks = postprocess(raw, spans, TEXT)
    cats = [c.category for c in chunks if c.category != "uncategorized"]
    assert "a" in cats
    assert "b" in cats
    a = next(c for c in chunks if c.category == "a")
    b = next(c for c in chunks if c.category == "b")
    assert a.end <= b.start


def test_fully_consumed_chunk_dropped() -> None:
    raw = [_raw("a"), _raw("b")]
    spans = [(0, 10), (3, 6)]  # b is entirely inside a
    chunks = postprocess(raw, spans, TEXT)
    cats = [c.category for c in chunks]
    assert "a" in cats
    assert "b" not in cats


def test_quote_comes_from_source_text() -> None:
    raw = [RawChunk(category="a", quote="WRONG")]
    spans = [(2, 5)]
    chunks = postprocess(raw, spans, TEXT)
    chunk_a = next(c for c in chunks if c.category == "a")
    assert chunk_a.quote == TEXT[2:5]
