import pytest

from seam.alignment import AlignmentError, align
from seam.types import RawChunk

TEXT = "The project kicked off in January with a small team. Budget constraints forced a scope reduction in March. Despite the setbacks, the product launched successfully in June."


def test_exact_match() -> None:
    raw = [RawChunk(category="x", quote="The project kicked off in January with a small team")]
    spans = align(raw, TEXT, threshold=80)
    assert spans[0] == (0, 51)


def test_noisy_quote_aligns() -> None:
    # One word changed — should still align above threshold=60
    raw = [RawChunk(category="x", quote="The project kicked of in January with a small team")]
    spans = align(raw, TEXT, threshold=60)
    start, end = spans[0]
    assert start == 0
    assert end > 0


def test_below_threshold_raises() -> None:
    raw = [RawChunk(category="x", quote="completely unrelated text that will never match")]
    with pytest.raises(AlignmentError):
        align(raw, TEXT, threshold=80)


def test_ordering_preserved() -> None:
    raw = [
        RawChunk(category="a", quote="Budget constraints forced a scope reduction in March"),
        RawChunk(category="b", quote="the product launched successfully in June"),
    ]
    spans = align(raw, TEXT, threshold=80)
    assert spans[0][0] < spans[1][0]


def test_multiple_chunks() -> None:
    raw = [
        RawChunk(category="a", quote="The project kicked off in January with a small team"),
        RawChunk(category="b", quote="Budget constraints forced a scope reduction in March"),
        RawChunk(category="c", quote="the product launched successfully in June"),
    ]
    spans = align(raw, TEXT, threshold=80)
    assert len(spans) == 3
    starts = [s[0] for s in spans]
    assert starts == sorted(starts)
