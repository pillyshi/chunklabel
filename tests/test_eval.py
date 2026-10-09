import pytest

from chunklabel.eval import (
    chunk_boundaries,
    pk,
    pk_from_boundaries,
    window_diff,
    window_diff_from_boundaries,
)
from chunklabel.types import Chunk


def test_identical_segmentation_scores_zero() -> None:
    assert pk_from_boundaries({3, 7}, {3, 7}, 10) == 0.0
    assert window_diff_from_boundaries({3, 7}, {3, 7}, 10) == 0.0


def test_default_k_is_half_mean_segment_length() -> None:
    # 12 units, 3 reference segments -> mean length 4 -> k = 2.
    # Shifting one boundary by one unit flips exactly two windows of width 2.
    assert pk_from_boundaries({4, 8}, {5, 8}, 12) == pytest.approx(2 / 10)
    assert pk_from_boundaries({4, 8}, {5, 8}, 12, k=2) == pytest.approx(2 / 10)


def test_missed_boundary() -> None:
    # Reference {4}, predicted none, k=2: windows (i, i+2] containing 4 are i=2,3.
    assert pk_from_boundaries({4}, set(), 8, k=2) == pytest.approx(2 / 6)
    assert window_diff_from_boundaries({4}, set(), 8, k=2) == pytest.approx(2 / 6)


def test_window_diff_penalises_extra_boundary_that_pk_misses() -> None:
    # Two predicted boundaries inside one window where the reference has one:
    # Pk sees "different segment" on both sides and scores no error, WindowDiff does.
    ref, hyp, n, k = {4}, {4, 5}, 10, 3
    assert pk_from_boundaries(ref, hyp, n, k) < window_diff_from_boundaries(ref, hyp, n, k)


def test_matches_nltk_reference_values() -> None:
    # Values from nltk.metrics.segmentation (nltk 3.9) on the same segmentations,
    # encoded there as gap strings of length n - 1.
    ref, hyp, n = {3, 9, 14}, {2, 10, 14, 17}, 20
    assert pk_from_boundaries(ref, hyp, n, k=3) == pytest.approx(6 / 17)
    assert window_diff_from_boundaries(ref, hyp, n, k=3) == pytest.approx(6 / 17)


def test_invalid_boundaries_raise() -> None:
    with pytest.raises(ValueError):
        pk_from_boundaries({0}, set(), 10)
    with pytest.raises(ValueError):
        window_diff_from_boundaries(set(), {10}, 10)
    with pytest.raises(ValueError):
        pk_from_boundaries(set(), set(), 10, k=10)


def _chunk(text: str, start: int, end: int, category: str = "x") -> Chunk:
    return Chunk(category=category, quote=text[start:end], start=start, end=end)


def test_chunk_boundaries_skip_whitespace_chunks() -> None:
    text = "Alpha one. Beta two."
    chunks = [_chunk(text, 0, 10), _chunk(text, 10, 11, "whitespace"), _chunk(text, 11, 20)]
    assert chunk_boundaries(chunks, len(text)) == {11}


def test_chunk_wrappers_use_character_units() -> None:
    text = "Alpha one. Beta two. Gamma three."
    reference = [_chunk(text, 0, 11), _chunk(text, 11, 21), _chunk(text, 21, 33)]
    predicted = [_chunk(text, 0, 21), _chunk(text, 21, 33)]
    n = len(text)
    assert pk(predicted, reference, text) == pk_from_boundaries({11, 21}, {21}, n)
    assert window_diff(predicted, reference, text) == window_diff_from_boundaries(
        {11, 21}, {21}, n
    )
    assert pk(reference, reference, text) == 0.0
