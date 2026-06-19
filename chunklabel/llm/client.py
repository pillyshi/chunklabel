from __future__ import annotations

import json
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class BaseLLMClient(Protocol):
    def complete_structured(self, messages: list[dict[str, str]], response_model: type[T]) -> T: ...


class OpenAIClient:
    def __init__(self, model: str, api_key: str | None = None, timeout: float | None = 120.0) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    def complete_structured(self, messages: list[dict[str, str]], response_model: type[T]) -> T:
        response = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            response_format=response_model,
        )
        if not response.choices:
            raise ValueError("OpenAI returned an empty choices list")
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned a refusal or null parsed response")
        return parsed


class LlamaCppClient:
    """In-process inference backend backed by a llama-cpp-python Llama instance.

    Example::

        from llama_cpp import Llama
        client = LlamaCppClient(Llama(model_path="path/to/model.gguf", n_ctx=4096))
    """

    def __init__(self, model: Any, temperature: float = 0.0) -> None:
        self._model = model
        self._temperature = temperature

    def complete_structured(self, messages: list[dict[str, str]], response_model: type[T]) -> T:
        schema = _inline_refs(response_model.model_json_schema())
        result = self._model.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": schema},
            temperature=self._temperature,
        )
        choices = result.get("choices") or []
        if not choices:
            raise ValueError("llama-cpp returned no choices")
        content = choices[0]["message"]["content"]
        if not content:
            raise ValueError("llama-cpp returned empty content")
        return response_model.model_validate(json.loads(content))


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve $ref pointers against $defs, returning a flat schema.

    llama-cpp-python's grammar converter may not resolve $ref in all versions,
    so we inline before passing the schema.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, _visiting: frozenset[str] = frozenset()) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref: str = node["$ref"]
                if ref.startswith("#/$defs/") and ref not in _visiting:
                    key = ref[len("#/$defs/"):]
                    if key in defs:
                        return resolve(defs[key], _visiting | {ref})
                return node
            return {k: resolve(v, _visiting) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(item, _visiting) for item in node]
        return node

    resolved = resolve(schema)
    resolved.pop("$defs", None)
    return cast(dict[str, Any], resolved)
