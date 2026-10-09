"""Segmentation metrics: Pk (Beeferman et al., 1999) and WindowDiff (Pevzner & Hearst, 2002).

Both metrics slide a window of width ``k`` over a sequence of ``n`` units and count
disagreements between the reference and predicted segmentations. Lower is better.

A segmentation is given as a collection of boundary positions: ``b`` means a new
segment starts at unit ``b`` (so ``0 < b < n``). Chunk-based wrappers use characters
as units.
"""

from collections.abc import Collection, Sequence

from chunklabel.types import Chunk


def pk_from_boundaries(
    reference: Collection[int], predicted: Collection[int], n: int, k: int | None = None
) -> float:
    """Pk over ``n`` units. ``k`` defaults to half the mean reference segment length."""
    ref, hyp, k = _prepare(reference, predicted, n, k)
    errors = sum(
        ((ref[i + k] - ref[i]) > 0) != ((hyp[i + k] - hyp[i]) > 0) for i in range(n - k)
    )
    return errors / (n - k)


def window_diff_from_boundaries(
    reference: Collection[int], predicted: Collection[int], n: int, k: int | None = None
) -> float:
    """WindowDiff over ``n`` units. ``k`` defaults to half the mean reference segment length."""
    ref, hyp, k = _prepare(reference, predicted, n, k)
    errors = sum((ref[i + k] - ref[i]) != (hyp[i + k] - hyp[i]) for i in range(n - k))
    return errors / (n - k)


def pk(
    predicted: Sequence[Chunk], reference: Sequence[Chunk], text: str, k: int | None = None
) -> float:
    """Character-level Pk between two chunkings of ``text``."""
    n = len(text)
    return pk_from_boundaries(chunk_boundaries(reference, n), chunk_boundaries(predicted, n), n, k)


def window_diff(
    predicted: Sequence[Chunk], reference: Sequence[Chunk], text: str, k: int | None = None
) -> float:
    """Character-level WindowDiff between two chunkings of ``text``."""
    n = len(text)
    return window_diff_from_boundaries(
        chunk_boundaries(reference, n), chunk_boundaries(predicted, n), n, k
    )


def chunk_boundaries(chunks: Sequence[Chunk], n: int) -> set[int]:
    """Start offsets of chunks, as boundary positions in a text of length ``n``.

    Whitespace-only chunks (e.g. the gaps that two-pass mode labels "whitespace") are
    absorbed into the preceding chunk so that they do not count as extra boundaries.
    """
    return {c.start for c in chunks if c.quote.strip() and 0 < c.start < n}


def _prepare(
    reference: Collection[int], predicted: Collection[int], n: int, k: int | None
) -> tuple[list[int], list[int], int]:
    for name, bounds in (("reference", reference), ("predicted", predicted)):
        if any(not 0 < b < n for b in bounds):
            raise ValueError(f"{name} boundaries must satisfy 0 < b < n (n={n})")
    if k is None:
        num_segments = len(set(reference)) + 1
        k = max(1, int(n / (2 * num_segments) + 0.5))
    if not 0 < k < n:
        raise ValueError(f"window size k must satisfy 0 < k < n (k={k}, n={n})")
    return _cumulative(reference, n), _cumulative(predicted, n), k


def _cumulative(boundaries: Collection[int], n: int) -> list[int]:
    # counts[j] = number of boundaries b with b <= j, so a window (i, i+k] holds
    # counts[i + k] - counts[i] boundaries.
    is_boundary = [0] * n
    for b in boundaries:
        is_boundary[b] = 1
    counts = []
    total = 0
    for flag in is_boundary:
        total += flag
        counts.append(total)
    return counts
