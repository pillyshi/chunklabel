"""Evaluation helpers.

Segmentation metrics: Pk (Beeferman et al., 1999) and WindowDiff (Pevzner & Hearst, 2002).

Both metrics slide a window of width ``k`` over a sequence of ``n`` units and count
disagreements between the reference and predicted segmentations. Lower is better.

A segmentation is given as a collection of boundary positions: ``b`` means a new
segment starts at unit ``b`` (so ``0 < b < n``). Chunk-based wrappers use characters
as units.

Fidelity: how faithfully LLM quotes map back onto the source text (no gold data needed).
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from chunklabel.alignment import align_detailed
from chunklabel.types import Chunk, RawChunk


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


@dataclass
class FidelityReport:
    """Quote fidelity for one text. Rates are fractions of quotes; coverage is a fraction
    of the text's non-whitespace characters."""

    n_quotes: int
    n_exact: int
    n_fuzzy: int  # aligned by fuzzy matching at or above the threshold
    n_unaligned: int  # best fuzzy score below the threshold
    gap_coverage: float  # non-whitespace characters covered by no aligned quote
    overlap_chars: int  # characters claimed by more than one aligned quote
    fuzzy_scores: list[float] = field(default_factory=list)  # every non-exact quote's score

    @property
    def exact_rate(self) -> float:
        return self.n_exact / self.n_quotes if self.n_quotes else 0.0

    @property
    def fuzzy_rate(self) -> float:
        return self.n_fuzzy / self.n_quotes if self.n_quotes else 0.0

    @property
    def unaligned_rate(self) -> float:
        return self.n_unaligned / self.n_quotes if self.n_quotes else 0.0


def fidelity(raw_chunks: Sequence[RawChunk], text: str, threshold: int = 85) -> FidelityReport:
    """Measure how LLM quotes align to ``text`` under the given fuzzy ``threshold``."""
    details = align_detailed(list(raw_chunks), text, threshold)
    spans = [d.span for d in details if d.span is not None]

    covered = [False] * len(text)
    for start, end in spans:
        for i in range(start, min(end, len(text))):
            covered[i] = True
    non_ws = [i for i, ch in enumerate(text) if not ch.isspace()]
    gap = sum(not covered[i] for i in non_ws)

    return FidelityReport(
        n_quotes=len(details),
        n_exact=sum(d.exact for d in details),
        n_fuzzy=sum(d.span is not None and not d.exact for d in details),
        n_unaligned=sum(d.span is None for d in details),
        gap_coverage=gap / len(non_ws) if non_ws else 0.0,
        overlap_chars=sum(min(e, len(text)) - s for s, e in spans) - sum(covered),
        fuzzy_scores=[d.score for d in details if not d.exact],
    )
