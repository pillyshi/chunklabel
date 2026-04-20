from seam.types import Chunk, RawChunk


def postprocess(
    raw_chunks: list[RawChunk], spans: list[tuple[int, int] | None], text: str
) -> list[Chunk]:
    # Pair and sort by start position, dropping unaligned chunks
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
