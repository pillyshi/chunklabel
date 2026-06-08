import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from chunklabel.normalizer import Normalizer
from chunklabel.types import Chunk


def _chunk(category: str) -> Chunk:
    return Chunk(category=category, quote="x", start=0, end=1)


def _mock_client(mapping: dict[str, str] | None = None) -> MagicMock:
    client = MagicMock()
    client.complete_structured.return_value = MagicMock(mapping=mapping or {})
    return client


def test_apply_renames_categories() -> None:
    normalizer = Normalizer(client=MagicMock())
    mapping = {"kick-off": "initiation", "blocker": "obstacle"}
    chunks = [_chunk("kick-off"), _chunk("blocker"), _chunk("outcome")]
    result = normalizer.apply(chunks, mapping)
    assert [c.category for c in result] == ["initiation", "obstacle", "outcome"]


def test_apply_identity_mapping() -> None:
    normalizer = Normalizer(client=MagicMock())
    chunks = [_chunk("a"), _chunk("b")]
    result = normalizer.apply(chunks, {"a": "a", "b": "b"})
    assert [c.category for c in result] == ["a", "b"]


def test_apply_does_not_mutate_original() -> None:
    normalizer = Normalizer(client=MagicMock())
    original = _chunk("old")
    result = normalizer.apply([original], {"old": "new"})
    assert original.category == "old"
    assert result[0].category == "new"


def test_build_mapping_passes_unique_categories() -> None:
    mock_client = _mock_client({"a": "a", "b": "a"})
    normalizer = Normalizer(client=mock_client)
    chunks = [_chunk("a"), _chunk("b"), _chunk("a")]
    normalizer.build_mapping(chunks)
    user_content = mock_client.complete_structured.call_args[0][0][1]["content"]
    assert sorted(json.loads(user_content)) == ["a", "b"]


def test_build_mapping_stores_internally() -> None:
    mock_client = _mock_client({"a": "b"})
    normalizer = Normalizer(client=mock_client)
    normalizer.build_mapping([_chunk("a")])
    result = normalizer.apply([_chunk("a")])
    assert result[0].category == "b"


def test_apply_without_mapping_raises_when_none() -> None:
    normalizer = Normalizer(client=MagicMock())
    with pytest.raises(ValueError):
        normalizer.apply([_chunk("a")])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    mock_client = _mock_client({"kick-off": "initiation"})
    normalizer = Normalizer(client=mock_client)
    normalizer.build_mapping([_chunk("kick-off")])

    path = tmp_path / "mapping.json"
    normalizer.save(path)

    loaded = Normalizer.load(path, client=MagicMock())
    result = loaded.apply([_chunk("kick-off")])
    assert result[0].category == "initiation"


def test_save_raises_when_no_mapping() -> None:
    normalizer = Normalizer(client=MagicMock())
    with pytest.raises(ValueError):
        normalizer.save("mapping.json")


def test_build_mapping_returns_copy() -> None:
    mock_client = _mock_client({"a": "b"})
    normalizer = Normalizer(client=mock_client)
    returned = normalizer.build_mapping([_chunk("a")])
    returned["a"] = "mutated"
    assert normalizer._mapping == {"a": "b"}


def test_build_mapping_empty_chunks_skips_llm() -> None:
    mock_client = MagicMock()
    normalizer = Normalizer(client=mock_client)
    result = normalizer.build_mapping([])
    assert result == {}
    mock_client.complete_structured.assert_not_called()


def test_load_raises_on_invalid_json_structure(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        Normalizer.load(path, client=MagicMock())


def test_load_build_mapping_works_with_client(tmp_path: Path) -> None:
    mock_client = _mock_client({"kick-off": "initiation"})
    normalizer = Normalizer(client=mock_client)
    normalizer.build_mapping([_chunk("kick-off")])
    path = tmp_path / "mapping.json"
    normalizer.save(path)

    new_mock = _mock_client({"x": "y"})
    loaded = Normalizer.load(path, client=new_mock)
    loaded.build_mapping([_chunk("x")])
    assert loaded._mapping == {"x": "y"}
