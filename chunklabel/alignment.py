from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz

from chunklabel.types import RawChunk


# Quotes shorter than this must match verbatim: short phrases find spurious fuzzy
# matches too easily (see benchmarks/fidelity/README.md).
MIN_FUZZY_CHARS = 16


class AlignmentError(Exception):
    pass


@dataclass
class AlignmentDetail:
    """How one quote was aligned to the source text."""

    span: tuple[int, int] | None  # None if the best fuzzy score is below the threshold
    score: float  # 100.0 for a verbatim match, else the best partial_ratio score
    exact: bool  # True if the quote was found verbatim


def align_detailed(
    raw_chunks: list[RawChunk], text: str, threshold: int
) -> list[AlignmentDetail]:
    """Align quotes like ``align``, but report scores and never raise."""
    return list(_iter_align(raw_chunks, text, threshold))


def _iter_align(
    raw_chunks: list[RawChunk], text: str, threshold: int
) -> Iterator[AlignmentDetail]:
    search_start = 0

    for chunk in raw_chunks:
        q = chunk.quote.strip()
        if not q:
            yield AlignmentDetail(None, 0.0, False)
            continue

        idx = text.find(q, search_start)
        if idx != -1:
            yield AlignmentDetail((idx, idx + len(q)), 100.0, True)
            search_start = idx
            continue

        # A verbatim quote the LLM moved out of order lies before search_start.
        # Accept it without moving search_start, so later quotes are unaffected.
        idx = text.find(q)
        if idx != -1:
            yield AlignmentDetail((idx, idx + len(q)), 100.0, True)
            continue

        # Best-matching substring; unlike fuzz.ratio against a fixed-size window, the
        # score does not depend on the quote's length.
        result = fuzz.partial_ratio_alignment(q, text[search_start:])
        score = result.score if result is not None else 0.0
        if result is None or score < threshold or len(q) < MIN_FUZZY_CHARS:
            yield AlignmentDetail(None, score, False)
            continue

        start, end = _refine_span(
            q, text, search_start + result.dest_start, search_start + result.dest_end
        )
        yield AlignmentDetail((start, end), score, False)
        search_start = start


def _refine_span(q: str, text: str, start: int, end: int) -> tuple[int, int]:
    """Nudge span edges to best fit the quote.

    partial_ratio_alignment returns a window as long as the quote, so when the LLM
    inserted or dropped words the window is shifted. Moving each edge by up to a few
    characters to maximise fuzz.ratio recovers the matched region's real edges.
    """
    slack = min(20, max(3, len(q) // 10))
    best_start = max(
        range(max(0, start - slack), min(end - 1, start + slack) + 1),
        key=lambda s: (fuzz.ratio(q, text[s:end]), -abs(s - start)),
    )
    best_end = max(
        range(max(best_start + 1, end - slack), min(len(text), end + slack) + 1),
        key=lambda e: (fuzz.ratio(q, text[best_start:e]), -abs(e - end)),
    )
    return best_start, best_end


def align(
    raw_chunks: list[RawChunk],
    text: str,
    threshold: int,
    on_error: Literal["raise", "skip"] = "raise",
) -> list[tuple[int, int] | None]:
    spans: list[tuple[int, int] | None] = []
    for chunk, detail in zip(raw_chunks, _iter_align(raw_chunks, text, threshold)):
        if detail.span is None and on_error == "raise":
            raise AlignmentError(
                f"Could not align quote (score={detail.score:.1f} < threshold={threshold}): "
                f"{chunk.quote!r}"
            )
        spans.append(detail.span)
    return spans
