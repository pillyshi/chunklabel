import pytest

from chunklabel.alignment import AlignmentDetail, AlignmentError, align, align_detailed
from chunklabel.types import RawChunk

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


def test_skip_returns_none_for_unmatched() -> None:
    raw = [RawChunk(category="x", quote="completely unrelated text that will never match")]
    spans = align(raw, TEXT, threshold=80, on_error="skip")
    assert spans == [None]


def test_skip_preserves_other_chunks() -> None:
    raw = [
        RawChunk(category="a", quote="The project kicked off in January with a small team"),
        RawChunk(category="x", quote="completely unrelated text that will never match"),
        RawChunk(category="b", quote="the product launched successfully in June"),
    ]
    spans = align(raw, TEXT, threshold=80, on_error="skip")
    assert spans[0] is not None
    assert spans[1] is None
    assert spans[2] is not None


def test_ordering_preserved() -> None:
    raw = [
        RawChunk(category="a", quote="Budget constraints forced a scope reduction in March"),
        RawChunk(category="b", quote="the product launched successfully in June"),
    ]
    spans = align(raw, TEXT, threshold=80)
    assert spans[0][0] < spans[1][0]


def test_short_quote_exact_match() -> None:
    raw = [RawChunk(category="x", quote="March")]
    spans = align(raw, TEXT, threshold=80)
    idx = TEXT.index("March")
    assert spans[0] == (idx, idx + len("March"))


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


def test_align_detailed_reports_scores() -> None:
    raw = [
        RawChunk(category="a", quote="The project kicked off in January with a small team"),
        RawChunk(category="b", quote="Budget constraint forced a scope reduction in March"),
        RawChunk(category="c", quote="completely unrelated text that will never match"),
    ]
    details = align_detailed(raw, TEXT, threshold=80)
    assert details[0].exact and details[0].score == 100.0 and details[0].span == (0, 51)
    assert not details[1].exact and 80 <= details[1].score < 100 and details[1].span is not None
    assert details[2].span is None and details[2].score < 80
    assert [d.span for d in details] == align(raw, TEXT, threshold=80, on_error="skip")


def test_trailing_whitespace_is_exact() -> None:
    raw = [RawChunk(category="x", quote="Budget constraints forced a scope reduction in March. ")]
    details = align_detailed_for(raw)
    start = TEXT.index("Budget")
    assert details[0].exact
    assert details[0].span == (start, start + len("Budget constraints forced a scope reduction in March."))


def test_out_of_order_verbatim_quote_is_found() -> None:
    raw = [
        RawChunk(category="b", quote="Budget constraints forced a scope reduction in March"),
        RawChunk(category="a", quote="The project kicked off in January with a small team"),
        RawChunk(category="c", quote="the product launched successfully in June"),
    ]
    spans = align(raw, TEXT, threshold=85)
    assert spans[1] == (0, 51)
    # The moved quote must not disturb the forward search for later quotes.
    assert spans[2] is not None and spans[2][0] > spans[0][0]


def test_fuzzy_score_does_not_depend_on_quote_length() -> None:
    # 20-character quote with one character missing: rejected by the old
    # window-based ratio (max ~65), accepted by substring alignment.
    quote = "Budget constrints fo"
    details = align_detailed_for([RawChunk(category="x", quote=quote)])
    assert details[0].span is not None and details[0].score >= 85
    assert details[0].span[0] == TEXT.index("Budget")


def test_fuzzy_span_uses_matched_substring_end() -> None:
    # Inserted words make the quote longer than the source span it matches.
    quote = "Budget constraints forced a big scope reduction in March"
    details = align_detailed_for([RawChunk(category="x", quote=quote)], threshold=80)
    start = TEXT.index("Budget")
    end = TEXT.index("March") + len("March")
    assert details[0].span is not None
    assert abs(details[0].span[0] - start) <= 1
    assert abs(details[0].span[1] - end) <= 1


def test_short_quote_requires_exact_match() -> None:
    # Fewer than MIN_FUZZY_CHARS characters: a typo is not tolerated.
    details = align_detailed_for([RawChunk(category="x", quote="in Marhc")])
    assert details[0].span is None
    assert align_detailed_for([RawChunk(category="x", quote="in March")])[0].exact


def align_detailed_for(raw: list[RawChunk], threshold: int = 85) -> list[AlignmentDetail]:
    return align_detailed(raw, TEXT, threshold=threshold)
