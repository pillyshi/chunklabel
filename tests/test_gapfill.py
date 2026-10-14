from collections.abc import Mapping, Sequence
from typing import Any

from chunklabel import ChunkLabeler, FewShotPromptSegmenter, ZeroShotPromptSegmenter
from chunklabel.gapfill import find_gaps
from chunklabel.llm.base import LLMBackend
from chunklabel.types import RawChunk

FIRST = "The project kicked off in January with a small team."
SKIPPED = "Budget constraints forced a scope reduction in March."
LAST = "Despite the setbacks, the product launched successfully in June."
TEXT = f"{FIRST} {SKIPPED} {LAST}"
LABELS = ["background", "problem", "result"]


class SkippingBackend(LLMBackend):
    """Skips the middle sentence of TEXT; labels any other text as "problem"."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []

    def _answer(self, text: str, label: str) -> list[RawChunk]:
        if text == TEXT:
            return [RawChunk("background", FIRST), RawChunk("result", LAST)]
        return [RawChunk(label, text)]

    def extract_chunks(self, text: str) -> list[RawChunk]:
        self.calls.append(("free", text, None))
        return self._answer(text, "blocker")

    def extract_boundaries(self, text: str) -> list[RawChunk]:
        self.calls.append(("boundaries", text, None))
        return [RawChunk("", c.quote) for c in self._answer(text, "")]

    def extract_labeled_chunks(
        self,
        text: str,
        labels: Mapping[str, str],
        examples: Sequence[tuple[str, list[RawChunk]]] = (),
    ) -> list[RawChunk]:
        self.calls.append(("labeled", text, list(examples)))
        return self._answer(text, "problem")

    def label_chunks(self, chunks: Any) -> list[str]:
        return ["label"] * len(chunks)


def test_find_gaps_trims_whitespace_and_skips_short_gaps() -> None:
    first = (0, len(FIRST))
    last = (len(TEXT) - len(LAST), len(TEXT))
    assert find_gaps([first, last], TEXT, 20) == [(len(FIRST) + 1, len(FIRST) + 1 + len(SKIPPED))]
    assert find_gaps([first, last], TEXT, 100) == []
    assert find_gaps([first, None], TEXT, 20)[-1][1] == len(TEXT)


def test_find_gaps_does_not_fill_the_whole_text() -> None:
    assert find_gaps([None], TEXT, 20) == []
    assert find_gaps([], TEXT, 20) == []


def test_segmenter_labels_skipped_text_from_a_second_call() -> None:
    backend = SkippingBackend()
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=backend)
    [segments], [report] = seg.predict_with_report([TEXT])
    assert [label for _, _, label in segments] == ["background", "problem", "result"]
    assert [text for _, text, _ in backend.calls] == [TEXT, SKIPPED]
    assert report.n_filled_gaps == 1
    assert report.n_quotes == 3
    assert report.gap_coverage == 0.0


def test_fill_gaps_false_keeps_one_call_and_joins_the_gap() -> None:
    backend = SkippingBackend()
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=backend, fill_gaps=False)
    [segments], [report] = seg.predict_with_report([TEXT])
    assert [label for _, _, label in segments] == ["background", "result"]
    assert len(backend.calls) == 1
    assert report.n_filled_gaps == 0 and report.gap_coverage > 0


def test_short_gaps_make_no_extra_call() -> None:
    backend = SkippingBackend()
    ZeroShotPromptSegmenter(labels=LABELS, backend=backend, min_gap_chars=100).predict([TEXT])
    assert len(backend.calls) == 1


def test_fit_without_labels_fills_with_free_labels() -> None:
    class NormalizerClient:
        def complete_structured(self, messages: Any, response_model: Any) -> Any:
            return response_model.model_validate({"mapping": {}})

    backend = SkippingBackend()
    seg = ZeroShotPromptSegmenter(client=NormalizerClient(), backend=backend).fit([TEXT])
    assert [kind for kind, _, _ in backend.calls] == ["free", "free"]
    assert [label for _, _, label in seg.segmentations_[0]] == ["background", "blocker", "result"]
    assert seg.fit_reports_[0].n_filled_gaps == 1


def test_few_shot_fill_calls_send_no_examples() -> None:
    backend = SkippingBackend()
    other = "A team formed. Money ran out. It shipped anyway."
    seg = FewShotPromptSegmenter(n_examples=1, backend=backend)
    seg.fit([other], [[(0, 14, "background"), (14, 29, "problem"), (29, len(other), "result")]])
    [segments] = seg.predict([TEXT])
    assert [label for _, _, label in segments] == ["background", "problem", "result"]
    (_, _, first_examples), (_, _, fill_examples) = backend.calls
    assert len(first_examples) == 1 and fill_examples == []


def test_chunk_labeler_fills_only_when_asked() -> None:
    off = ChunkLabeler(backend=SkippingBackend()).split(TEXT)
    assert "uncategorized" in {c.category for c in off}

    backend = SkippingBackend()
    chunks, report = ChunkLabeler(backend=backend, fill_gaps=True).split_with_report(TEXT)
    assert [c.category for c in chunks if c.quote.strip()] == ["background", "blocker", "result"]
    assert report.n_filled_gaps == 1


def test_chunk_labeler_two_pass_fills_with_boundaries() -> None:
    backend = SkippingBackend()
    chunks = ChunkLabeler(backend=backend, fill_gaps=True).split(TEXT, mode="two_pass")
    assert [kind for kind, _, _ in backend.calls] == ["boundaries", "boundaries"]
    assert "uncategorized" not in {c.category for c in chunks}
