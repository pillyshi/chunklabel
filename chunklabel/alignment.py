from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz

from chunklabel.types import RawChunk


class AlignmentError(Exception):
    pass


@dataclass
class AlignmentDetail:
    """How one quote was aligned to the source text."""

    span: tuple[int, int] | None  # None if the best fuzzy score is below the threshold
    score: float  # 100.0 for a verbatim match, else the best fuzz.ratio score
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
        q = chunk.quote
        window_size = len(q) + 20
        best_score = -1.0
        best_start = -1

        idx = text.find(q, search_start)
        if idx != -1:
            yield AlignmentDetail((idx, idx + len(q)), 100.0, True)
            search_start = idx
            continue

        for i in range(search_start, max(search_start + 1, len(text) - len(q) + 1)):
            window = text[i : i + window_size]
            score = fuzz.ratio(q, window)
            if score > best_score:
                best_score = score
                best_start = i

        if best_score < threshold:
            yield AlignmentDetail(None, best_score, False)
            continue

        yield AlignmentDetail((best_start, best_start + len(q)), best_score, False)
        search_start = best_start


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
