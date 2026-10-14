"""Fill text the LLM skipped with a second call on that text alone (issue #6).

See ``research/ideas/gap-fill-retry.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from chunklabel.alignment import AlignmentDetail, align_detailed
from chunklabel.types import RawChunk


def find_gaps(
    spans: Sequence[tuple[int, int] | None], text: str, min_chars: int
) -> list[tuple[int, int]]:
    """Maximal stretches of ``text`` covered by no span, trimmed of surrounding
    whitespace, with at least ``min_chars`` non-whitespace characters.

    Returns nothing when no span aligned: filling the whole text would repeat the
    first request.
    """
    aligned = [span for span in spans if span is not None]
    if not aligned:
        return []
    covered = [False] * len(text)
    for start, end in aligned:
        for i in range(start, min(end, len(text))):
            covered[i] = True

    gaps = []
    i = 0
    while i < len(text):
        if covered[i] or text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not covered[j]:
            j += 1
        end = j
        while text[end - 1].isspace():
            end -= 1
        if sum(not ch.isspace() for ch in text[i:end]) >= min_chars:
            gaps.append((i, end))
        i = j
    return gaps


def fill_gaps(
    raw_chunks: list[RawChunk],
    details: list[AlignmentDetail],
    text: str,
    extract: Callable[[str], list[RawChunk]],
    threshold: int,
    min_chars: int,
) -> tuple[list[RawChunk], list[AlignmentDetail], int]:
    """Call ``extract`` once on each gap's text and align its quotes within the gap.

    Returns the first call's chunks and details followed by the fill calls' (their spans
    shifted to positions in ``text``), and the number of gaps sent to ``extract``.
    """
    gaps = find_gaps([d.span for d in details], text, min_chars)
    raw_out = list(raw_chunks)
    details_out = list(details)
    for gap_start, gap_end in gaps:
        more = extract(text[gap_start:gap_end])
        for chunk, detail in zip(more, align_detailed(more, text[gap_start:gap_end], threshold)):
            span = detail.span
            if span is not None:
                span = (gap_start + span[0], gap_start + span[1])
            raw_out.append(chunk)
            details_out.append(AlignmentDetail(span, detail.score, detail.exact))
    return raw_out, details_out, len(gaps)
