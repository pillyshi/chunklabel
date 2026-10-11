from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from chunklabel import ZeroShotPromptSegmenter
from chunklabel.llm.backend import ClientBackend
from chunklabel.llm.base import LLMBackend
from chunklabel.segmenter import map_labels
from chunklabel.types import RawChunk

TEXT = (
    "The project kicked off in January with a small team. "
    "Budget constraints forced a scope reduction in March. "
    "Despite the setbacks, the product launched successfully in June."
)
LABELS = ["background", "problem", "result"]


class MockBackend(LLMBackend):
    def __init__(self, chunks: list[RawChunk]) -> None:
        self.chunks = chunks
        self.seen_labels: Mapping[str, str] | None = None

    def extract_chunks(self, text: str) -> list[RawChunk]:
        raise NotImplementedError

    def extract_labeled_chunks(self, text: str, labels: Mapping[str, str]) -> list[RawChunk]:
        self.seen_labels = labels
        return self.chunks


GOOD = [
    RawChunk("background", "The project kicked off in January with a small team."),
    RawChunk("problem", "Budget constraints forced a scope reduction in March."),
    RawChunk("result", "the product launched successfully in June."),
]


def test_predict_returns_triples_covering_text() -> None:
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(GOOD))
    [segments] = seg.predict([TEXT])
    assert [label for _, _, label in segments] == ["background", "problem", "result"]
    assert segments[0][0] == 0 and segments[-1][1] == len(TEXT)
    assert all(e == s for (_, e, _), (s, _, _) in zip(segments, segments[1:]))
    assert "".join(TEXT[s:e] for s, e, _ in segments) == TEXT


def test_gap_joins_preceding_segment() -> None:
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(GOOD))
    [segments] = seg.predict([TEXT])
    # "Despite the setbacks, " is not quoted, so it joins the "problem" segment.
    assert TEXT[segments[1][0]:segments[1][1]].endswith("Despite the setbacks, ")


def test_leading_gap_joins_first_segment() -> None:
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(GOOD[1:]))
    [segments] = seg.predict([TEXT])
    assert segments[0][0] == 0 and segments[0][2] == "problem"


def test_labels_are_passed_with_descriptions() -> None:
    backend = MockBackend(GOOD)
    labels = {"background": "context", "problem": "", "result": "outcome"}
    ZeroShotPromptSegmenter(labels=labels, backend=backend).predict([TEXT])
    assert backend.seen_labels == labels


def test_unknown_label_maps_to_closest_by_default() -> None:
    chunks = [RawChunk("Background ", GOOD[0].quote), RawChunk("results", GOOD[2].quote)]
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(chunks))
    [segments] = seg.predict([TEXT])
    assert [label for _, _, label in segments] == ["background", "result"]


def test_unknown_label_drop_and_raise() -> None:
    chunks = [GOOD[0], RawChunk("other", GOOD[1].quote), GOOD[2]]
    dropped = map_labels(chunks, LABELS, "drop")
    assert [c.category for c in dropped] == ["background", "result"]
    with pytest.raises(ValueError, match="outside the given set"):
        map_labels(chunks, LABELS, "raise")


def test_every_label_is_in_the_set() -> None:
    chunks = [RawChunk("intro", GOOD[0].quote), RawChunk("issue", GOOD[1].quote)]
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(chunks))
    [segments] = seg.predict([TEXT])
    assert {label for _, _, label in segments} <= set(LABELS)


def test_no_aligned_quote_gives_empty_segmentation() -> None:
    chunks = [RawChunk("result", "completely unrelated text that will never match")]
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(chunks), on_align_error="skip")
    assert seg.predict([TEXT]) == [[]]


def test_predict_requires_labels_and_a_list_of_texts() -> None:
    with pytest.raises(ValueError, match="labels must be given"):
        ZeroShotPromptSegmenter(backend=MockBackend(GOOD)).predict([TEXT])
    with pytest.raises(TypeError):
        ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(GOOD)).predict(TEXT)


def test_params_and_clone() -> None:
    sklearn_base = pytest.importorskip("sklearn.base")
    seg = ZeroShotPromptSegmenter(labels=LABELS, client="gpt-4o-mini", fuzzy_threshold=90)
    cloned = sklearn_base.clone(seg)
    assert cloned is not seg
    assert cloned.get_params() == seg.get_params()
    assert cloned.set_params(fuzzy_threshold=70).fuzzy_threshold == 70
    with pytest.raises(ValueError):
        seg.set_params(nope=1)


class FakeClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.schema: dict[str, Any] = {}

    def complete_structured(self, messages: list[dict[str, str]], response_model: type[Any]) -> Any:
        self.messages = messages
        self.schema = response_model.model_json_schema()
        payload = {"chunks": [{"category": c.category, "quote": c.quote} for c in GOOD]}
        result: BaseModel = response_model.model_validate(payload)
        return result


def test_client_backend_prompt_and_schema_enum() -> None:
    client = FakeClient()
    raw = ClientBackend(client).extract_labeled_chunks(TEXT, {"background": "context", "result": ""})
    assert [c.category for c in raw] == ["background", "problem", "result"]
    system = client.messages[0]["content"]
    assert "- background: context" in system and "- result\n" in system
    item = client.schema["$defs"]["_LabeledChunkSchema"]["properties"]["category"]
    assert item["enum"] == ["background", "result"]
