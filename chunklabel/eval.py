"""Evaluation helpers.

Segmentation metrics: Pk (Beeferman et al., 1999) and WindowDiff (Pevzner & Hearst, 2002).

Both metrics slide a window of width ``k`` over a sequence of ``n`` units and count
disagreements between the reference and predicted segmentations. Lower is better.

A segmentation is given as a collection of boundary positions: ``b`` means a new
segment starts at unit ``b`` (so ``0 < b < n``). Chunk-based wrappers use characters
as units.

Fidelity: how faithfully LLM quotes map back onto the source text (no gold data needed).

Agreement: how well a chunking with free-form categories matches a reference chunking,
compared as two partitions of the text's characters (Rosenberg & Hirschberg, 2007;
Hubert & Arabie, 1985). Category names never need to match.
"""

import math
from collections import Counter
from collections.abc import Collection, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from chunklabel.alignment import AlignmentDetail, align_detailed
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
    return fidelity_from_details(align_detailed(list(raw_chunks), text, threshold), text)


def fidelity_from_details(details: Sequence[AlignmentDetail], text: str) -> FidelityReport:
    """Build a FidelityReport from alignment results that were already computed."""
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


@dataclass
class Agreement:
    """Agreement between a predicted and a reference partition (1.0 = identical)."""

    homogeneity: float  # each predicted group holds a single reference group
    completeness: float  # each reference group falls in a single predicted group
    v_measure: float  # harmonic mean of homogeneity and completeness
    ari: float  # adjusted Rand index: 0 for chance agreement, 1 for identical


def agreement(predicted: Sequence[Hashable], reference: Sequence[Hashable]) -> Agreement:
    """Compare two labelings of the same items as partitions; label names are ignored."""
    if len(predicted) != len(reference):
        raise ValueError("predicted and reference must have the same length")
    n = len(reference)
    if n == 0:
        raise ValueError("cannot compare empty labelings")
    joint = Counter(zip(predicted, reference))
    pred_counts = Counter(predicted)
    ref_counts = Counter(reference)

    def entropy(counts: Collection[int]) -> float:
        return -sum(c / n * math.log(c / n) for c in counts)

    h_ref, h_pred = entropy(ref_counts.values()), entropy(pred_counts.values())
    # Conditional entropies H(ref | pred) and H(pred | ref).
    h_ref_given_pred = -sum(c / n * math.log(c / pred_counts[p]) for (p, _), c in joint.items())
    h_pred_given_ref = -sum(c / n * math.log(c / ref_counts[r]) for (_, r), c in joint.items())
    homogeneity = 1.0 if h_ref == 0 else 1 - h_ref_given_pred / h_ref
    completeness = 1.0 if h_pred == 0 else 1 - h_pred_given_ref / h_pred
    v = (
        0.0 if homogeneity + completeness == 0
        else 2 * homogeneity * completeness / (homogeneity + completeness)
    )

    def pairs(k: int) -> int:
        return k * (k - 1) // 2

    index = sum(pairs(c) for c in joint.values())
    sum_pred = sum(pairs(c) for c in pred_counts.values())
    sum_ref = sum(pairs(c) for c in ref_counts.values())
    expected = sum_pred * sum_ref / pairs(n) if n > 1 else 0.0
    max_index = (sum_pred + sum_ref) / 2
    ari = 1.0 if max_index == expected else (index - expected) / (max_index - expected)
    return Agreement(homogeneity, completeness, v, ari)


def char_labels(
    chunks: Sequence[Chunk], n: int, by: Literal["category", "chunk"] = "category"
) -> list[Hashable | None]:
    """Label of each character position (None where no chunk covers it).

    ``by="category"`` groups characters by chunk category, so separate chunks with the
    same category form one group. ``by="chunk"`` makes every chunk its own group, which
    compares segmentations only.
    """
    labels: list[Hashable | None] = [None] * n
    for i, c in enumerate(chunks):
        key: Hashable = c.category if by == "category" else i
        for j in range(max(0, c.start), min(c.end, n)):
            labels[j] = key
    return labels


def chunk_agreement(
    predicted: Sequence[Chunk],
    reference: Sequence[Chunk],
    text: str,
    by: Literal["category", "chunk"] = "category",
) -> Agreement:
    """Character-level agreement between two chunkings of ``text``.

    Only non-whitespace characters covered by the reference are compared. Characters no
    predicted chunk covers form one extra group of their own.
    """
    pred = char_labels(predicted, len(text), by)
    ref = char_labels(reference, len(text), by)
    keep = [i for i, ch in enumerate(text) if not ch.isspace() and ref[i] is not None]
    return agreement([pred[i] for i in keep], [ref[i] for i in keep])
