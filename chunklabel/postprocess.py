from chunklabel.types import Chunk, RawChunk


def resolve_spans(
    raw_chunks: list[RawChunk], spans: list[tuple[int, int] | None]
) -> list[tuple[int, int, str]]:
    """Sorted, non-overlapping ``(start, end, category)``; unaligned chunks are dropped."""
    paired = sorted(
        [(span, raw) for span, raw in zip(spans, raw_chunks) if span is not None],
        key=lambda x: x[0][0],
    )

    # Resolve overlaps (earlier chunk wins)
    resolved: list[tuple[int, int, str]] = []
    for (start, end), raw in paired:
        if resolved and start < resolved[-1][1]:
            start = resolved[-1][1]
        if start >= end:
            continue
        resolved.append((start, end, raw.category))
    return resolved


def absorb_gaps(resolved: list[tuple[int, int, str]], n: int) -> list[tuple[int, int, str]]:
    """Extend segments over the gaps between them so they cover ``[0, n)`` exactly.

    Each gap joins the segment before it; a leading gap joins the first segment.
    """
    if not resolved:
        return []
    starts = [0] + [start for start, _, _ in resolved[1:]]
    ends = starts[1:] + [n]
    return [(s, e, label) for s, e, (_, _, label) in zip(starts, ends, resolved)]


def postprocess(
    raw_chunks: list[RawChunk], spans: list[tuple[int, int] | None], text: str
) -> list[Chunk]:
    resolved = resolve_spans(raw_chunks, spans)

    # Build final chunks with gap-filling
    chunks: list[Chunk] = []
    cursor = 0

    for start, end, category in resolved:
        if cursor < start:
            chunks.append(Chunk(
                category="uncategorized",
                quote=text[cursor:start],
                start=cursor,
                end=start,
            ))
        chunks.append(Chunk(
            category=category,
            quote=text[start:end],
            start=start,
            end=end,
        ))
        cursor = end

    if cursor < len(text):
        chunks.append(Chunk(
            category="uncategorized",
            quote=text[cursor:],
            start=cursor,
            end=len(text),
        ))

    return chunks
