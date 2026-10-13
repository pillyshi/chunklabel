import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from chunklabel import FewShotPromptSegmenter, NotFittedError, ZeroShotPromptSegmenter
from chunklabel.alignment import AlignmentError
from chunklabel.llm.backend import ClientBackend
from chunklabel.llm.base import LLMBackend
from chunklabel.segmenter import _CharNgramTfidf, _dot, map_labels
from chunklabel.types import RawChunk

TEXT = (
    "The project kicked off in January with a small team. "
    "Budget constraints forced a scope reduction in March. "
    "Despite the setbacks, the product launched successfully in June."
)
LABELS = ["background", "problem", "result"]


class MockBackend(LLMBackend):
    def __init__(self, chunks: list[RawChunk], free: list[RawChunk] | None = None) -> None:
        self.chunks = chunks
        self.free = free
        self.seen_labels: Mapping[str, str] | None = None

    def extract_chunks(self, text: str) -> list[RawChunk]:
        if self.free is None:
            raise AssertionError("extract_chunks should not be called")
        return self.free

    def extract_labeled_chunks(
        self,
        text: str,
        labels: Mapping[str, str],
        examples: Sequence[tuple[str, list[RawChunk]]] = (),
    ) -> list[RawChunk]:
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
    with pytest.raises(ValueError, match="call fit"):
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


FREE = [
    RawChunk("kick-off", "The project kicked off in January with a small team."),
    RawChunk("blocker", "Budget constraints forced a scope reduction in March."),
    RawChunk("launch", "the product launched successfully in June."),
]


class NormalizerClient:
    """Fake client answering only the Normalizer's request."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls = 0

    def complete_structured(self, messages: list[dict[str, str]], response_model: type[Any]) -> Any:
        self.calls += 1
        return response_model.model_validate({"mapping": self.mapping})


def test_fit_induces_label_set() -> None:
    client = NormalizerClient({"kick-off": "start", "blocker": "problem", "launch": "outcome"})
    seg = ZeroShotPromptSegmenter(client=client, backend=MockBackend(GOOD, free=FREE))
    assert seg.fit([TEXT, TEXT]) is seg
    assert seg.labels_ == ["outcome", "problem", "start"]
    assert seg.mapping_ == {"blocker": "problem", "kick-off": "start", "launch": "outcome"}
    assert client.calls == 1
    for segments in seg.segmentations_:
        assert [label for _, _, label in segments] == ["start", "problem", "outcome"]
        assert "".join(TEXT[s:e] for s, e, _ in segments) == TEXT


def test_fit_merges_synonyms_and_keeps_unmapped_labels() -> None:
    client = NormalizerClient({"kick-off": "launch"})
    seg = ZeroShotPromptSegmenter(client=client, backend=MockBackend(GOOD, free=FREE))
    seg.fit([TEXT])
    assert seg.mapping_ == {"blocker": "blocker", "kick-off": "launch", "launch": "launch"}
    assert seg.labels_ == ["blocker", "launch"]


def test_fit_predict_returns_segmentations() -> None:
    client = NormalizerClient({})
    seg = ZeroShotPromptSegmenter(client=client, backend=MockBackend(GOOD, free=FREE))
    result = seg.fit_predict([TEXT])
    assert result is seg.segmentations_
    assert [label for _, _, label in result[0]] == ["kick-off", "blocker", "launch"]


def test_predict_after_fit_uses_fitted_labels() -> None:
    backend = MockBackend(GOOD, free=FREE)
    client = NormalizerClient({"kick-off": "background", "blocker": "problem", "launch": "result"})
    seg = ZeroShotPromptSegmenter(client=client, backend=backend).fit([TEXT])
    [segments] = seg.predict([TEXT])
    assert backend.seen_labels == {"background": "", "problem": "", "result": ""}
    assert [label for _, _, label in segments] == ["background", "problem", "result"]


def test_fit_with_given_labels_skips_induction() -> None:
    client = NormalizerClient({})
    seg = ZeroShotPromptSegmenter(labels=LABELS, client=client, backend=MockBackend(GOOD))
    seg.fit([TEXT])
    assert client.calls == 0
    assert seg.labels_ == sorted(LABELS)
    assert seg.mapping_ == {}
    assert seg.segmentations_ == seg.predict([TEXT])


def test_fit_with_nothing_aligned() -> None:
    client = NormalizerClient({})
    free = [RawChunk("x", "completely unrelated text that will never match")]
    seg = ZeroShotPromptSegmenter(
        client=client, backend=MockBackend(GOOD, free=free), on_align_error="skip"
    ).fit([TEXT])
    assert (seg.labels_, seg.segmentations_, client.calls) == ([], [[]], 0)
    with pytest.raises(ValueError, match="found no labels"):
        seg.predict([TEXT])


def test_clone_drops_fitted_attributes() -> None:
    sklearn_base = pytest.importorskip("sklearn.base")
    client = NormalizerClient({})
    seg = ZeroShotPromptSegmenter(client=client, backend=MockBackend(GOOD, free=FREE)).fit([TEXT])
    assert not hasattr(sklearn_base.clone(seg), "labels_")


# --- FewShotPromptSegmenter -------------------------------------------------


class ExampleBackend(LLMBackend):
    """Records the examples it is given and answers with GOOD."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, str], list[tuple[str, list[RawChunk]]]]] = []

    def extract_chunks(self, text: str) -> list[RawChunk]:
        raise AssertionError("extract_chunks should not be called")

    def extract_labeled_chunks(
        self,
        text: str,
        labels: Mapping[str, str],
        examples: Sequence[tuple[str, list[RawChunk]]] = (),
    ) -> list[RawChunk]:
        self.calls.append((text, labels, list(examples)))
        return GOOD


POOL_X = [
    "Cats sleep a lot. Cats purr when happy.",
    "Dogs bark at night. Dogs like walks.",
    "The stock market fell. Investors were worried.",
    "Unlabeled text here.",
]
POOL_Y: list[list[tuple[int, int, str]] | None] = [
    [(0, 17, "background"), (18, 39, "result")],
    [(0, 19, "background"), (20, 35, "result")],
    [(0, 22, "problem")],
    None,
]


def test_fewshot_fit_stores_labels_and_pool_without_llm() -> None:
    backend = ExampleBackend()
    seg = FewShotPromptSegmenter(backend=backend).fit(POOL_X, POOL_Y)
    assert seg.labels_ == ["background", "problem", "result"]
    assert [text for text, _ in seg.example_pool_] == POOL_X[:3]
    assert backend.calls == []


def test_fewshot_predict_passes_examples_and_labels() -> None:
    backend = ExampleBackend()
    seg = FewShotPromptSegmenter(n_examples=2, random_state=0, backend=backend)
    [segments] = seg.fit(POOL_X, POOL_Y).predict([TEXT])
    _, labels, examples = backend.calls[0]
    assert labels == {"background": "", "problem": "", "result": ""}
    assert len(examples) == 2
    for text, chunks in examples:
        k = POOL_X.index(text)
        expected = POOL_Y[k]
        assert expected is not None
        assert [(c.category, c.quote) for c in chunks] == [(lab, text[s:e]) for s, e, lab in expected]
    assert "".join(TEXT[s:e] for s, e, _ in segments) == TEXT


def test_fewshot_random_is_reproducible_and_shared_across_inputs() -> None:
    a = FewShotPromptSegmenter(n_examples=2, random_state=1).fit(POOL_X, POOL_Y)
    b = FewShotPromptSegmenter(n_examples=2, random_state=1).fit(POOL_X, POOL_Y)
    chosen = a.select_examples([TEXT, "another text"])
    assert chosen == b.select_examples([TEXT, "another text"])
    assert chosen[0] == chosen[1] and len(chosen[0]) == 2


def test_fewshot_skips_identical_text() -> None:
    selections: tuple[Literal["random", "similar"], ...] = ("random", "similar")
    for selection in selections:
        seg = FewShotPromptSegmenter(n_examples=3, example_selection=selection)
        seg.fit(POOL_X, POOL_Y)
        [chosen] = seg.select_examples([POOL_X[0]])
        assert 0 not in chosen and len(chosen) == 2


def test_fewshot_similar_puts_most_similar_last() -> None:
    seg = FewShotPromptSegmenter(n_examples=2, example_selection="similar").fit(POOL_X, POOL_Y)
    [chosen] = seg.select_examples(["Dogs bark loudly. Dogs like long walks."])
    assert chosen[-1] == 1


def test_fewshot_similar_with_custom_embedding() -> None:
    def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "market" in t or "stock" in t else [0.0, 1.0] for t in texts]

    seg = FewShotPromptSegmenter(n_examples=1, example_selection="similar", embedding=embed)
    seg.fit(POOL_X, POOL_Y)
    assert seg.select_examples(["stock prices"]) == [[2]]


def test_fewshot_validates_y() -> None:
    seg = FewShotPromptSegmenter()
    with pytest.raises(ValueError, match="different lengths"):
        seg.fit(POOL_X, POOL_Y[:2])
    with pytest.raises(ValueError, match="outside the text"):
        seg.fit(["short"], [[(0, 99, "a")]])
    with pytest.raises(ValueError, match="must not overlap"):
        seg.fit(["some text here"], [[(0, 6, "a"), (4, 9, "b")]])
    with pytest.raises(ValueError, match="no labeled segment"):
        seg.fit(["a text"], [None])


def test_not_fitted_error() -> None:
    with pytest.raises(NotFittedError):
        FewShotPromptSegmenter().predict([TEXT])
    with pytest.raises(NotFittedError):
        ZeroShotPromptSegmenter().predict([TEXT])
    assert issubclass(NotFittedError, ValueError) and issubclass(NotFittedError, AttributeError)


def test_fewshot_clone() -> None:
    sklearn_base = pytest.importorskip("sklearn.base")
    seg = FewShotPromptSegmenter(n_examples=5, example_selection="similar", random_state=3)
    seg.fit(POOL_X, POOL_Y)
    cloned = sklearn_base.clone(seg)
    assert cloned.get_params() == seg.get_params()
    assert not hasattr(cloned, "example_pool_")


def test_char_ngram_tfidf_works_without_spaces() -> None:
    tfidf = _CharNgramTfidf().fit(["猫はよく眠る。", "株価が下がった。"])
    query, cat, stock = tfidf.transform(["猫がよく眠っている。", "猫はよく眠る。", "株価が下がった。"])
    assert _dot(query, cat) > _dot(query, stock)


def test_client_backend_puts_examples_as_turns() -> None:
    client = FakeClient()
    examples = [("Example text.", [RawChunk("background", "Example text.")])]
    ClientBackend(client).extract_labeled_chunks(TEXT, {"background": ""}, examples)
    roles = [m["role"] for m in client.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert client.messages[1]["content"] == "Example text."
    assert json.loads(client.messages[2]["content"]) == {
        "chunks": [{"category": "background", "quote": "Example text."}]
    }
    assert client.messages[3]["content"] == TEXT


# --- fidelity reports ---------------------------------------------------------


def test_predict_with_report_matches_predict_and_shows_gap() -> None:
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(GOOD))
    segmentations, [report] = seg.predict_with_report([TEXT])
    assert segmentations == seg.predict([TEXT])
    # "Despite the setbacks, " is not quoted: reported, though the segments cover it.
    assert report.gap_coverage > 0
    assert report.n_quotes == 3 and report.n_exact == 3
    assert "".join(TEXT[s:e] for s, e, _ in segmentations[0]) == TEXT


def test_predict_with_report_never_raises_on_alignment_errors() -> None:
    chunks = [GOOD[0], RawChunk("result", "completely unrelated text that will never match")]
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(chunks))
    with pytest.raises(AlignmentError):
        seg.predict([TEXT])
    [segments], [report] = seg.predict_with_report([TEXT])
    assert report.n_unaligned == 1
    assert [label for _, _, label in segments] == ["background"]


def test_fit_reports_for_both_fit_paths() -> None:
    client = NormalizerClient({})
    seg = ZeroShotPromptSegmenter(client=client, backend=MockBackend(GOOD, free=FREE))
    assert len(seg.fit([TEXT, TEXT]).fit_reports_) == 2
    seg = ZeroShotPromptSegmenter(labels=LABELS, backend=MockBackend(GOOD))
    [report] = seg.fit([TEXT]).fit_reports_
    assert report.gap_coverage > 0


def test_fewshot_predict_with_report() -> None:
    seg = FewShotPromptSegmenter(n_examples=1, backend=ExampleBackend()).fit(POOL_X, POOL_Y)
    segmentations, [report] = seg.predict_with_report([TEXT])
    assert segmentations == seg.predict([TEXT])
    assert report.n_quotes == 3
