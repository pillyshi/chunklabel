from rapidfuzz import fuzz

from seam.types import RawChunk


class AlignmentError(Exception):
    pass


def align(raw_chunks: list[RawChunk], text: str, threshold: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    search_start = 0

    for chunk in raw_chunks:
        q = chunk.quote
        window_size = len(q) + 20
        best_score = -1.0
        best_start = -1

        idx = text.find(q, search_start)
        if idx != -1:
            spans.append((idx, idx + len(q)))
            search_start = idx
            continue

        for i in range(search_start, max(search_start + 1, len(text) - len(q) + 1)):
            window = text[i : i + window_size]
            score = fuzz.ratio(q, window)
            if score > best_score:
                best_score = score
                best_start = i

        if best_score < threshold:
            raise AlignmentError(
                f"Could not align quote (score={best_score:.1f} < threshold={threshold}): "
                f"{q!r}"
            )

        best_end = best_start + len(q)
        spans.append((best_start, best_end))
        search_start = best_start

    return spans
