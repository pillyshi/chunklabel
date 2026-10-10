from __future__ import annotations

import dataclasses
import warnings
from typing import Literal, Union

from chunklabel.alignment import align, align_detailed
from chunklabel.eval import FidelityReport, fidelity_from_details
from chunklabel.llm.base import LLMBackend
from chunklabel.llm.client import BaseLLMClient
from chunklabel.postprocess import postprocess
from chunklabel.types import Chunk


class ChunkLabeler:
    def __init__(
        self,
        client: Union[BaseLLMClient, str] = "gpt-4o",
        fuzzy_threshold: int = 85,
        on_align_error: Literal["raise", "skip"] = "raise",
        timeout: float | None = 120.0,
        backend: LLMBackend | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        else:
            from chunklabel.llm.backend import ClientBackend
            from chunklabel.llm.client import OpenAIClient

            if isinstance(client, str):
                _client: BaseLLMClient = OpenAIClient(model=client, timeout=timeout)
            else:
                if timeout != 120.0:
                    warnings.warn(
                        "timeout is ignored when a pre-built BaseLLMClient is passed; "
                        "configure it on the client directly.",
                        UserWarning,
                        stacklevel=2,
                    )
                _client = client
            self._backend = ClientBackend(_client)
        self.fuzzy_threshold = fuzzy_threshold
        self.on_align_error = on_align_error

    def split(self, text: str, mode: Literal["one_pass", "two_pass"] = "one_pass") -> list[Chunk]:
        if mode == "two_pass":
            raw_chunks = self._backend.extract_boundaries(text)
            spans = align(raw_chunks, text, self.fuzzy_threshold, on_error=self.on_align_error)
            return self._label(postprocess(raw_chunks, spans, text))

        raw_chunks = self._backend.extract_chunks(text)
        spans = align(raw_chunks, text, self.fuzzy_threshold, on_error=self.on_align_error)
        return postprocess(raw_chunks, spans, text)

    def split_with_report(
        self, text: str, mode: Literal["one_pass", "two_pass"] = "one_pass"
    ) -> tuple[list[Chunk], FidelityReport]:
        """Like ``split``, but also report how faithfully the LLM's quotes matched ``text``.

        Use it to check fidelity for your model and data. Unlike ``split``, it never raises
        on alignment errors: quotes that cannot be aligned are counted in the report and
        dropped, as with ``on_align_error="skip"``.
        """
        if mode == "two_pass":
            raw_chunks = self._backend.extract_boundaries(text)
        else:
            raw_chunks = self._backend.extract_chunks(text)
        details = align_detailed(raw_chunks, text, self.fuzzy_threshold)
        chunks = postprocess(raw_chunks, [d.span for d in details], text)
        if mode == "two_pass":
            chunks = self._label(chunks)
        return chunks, fidelity_from_details(details, text)

    def _label(self, chunks: list[Chunk]) -> list[Chunk]:
        """Two-pass labeling step: ask the LLM for a category per non-whitespace chunk."""
        non_ws_indices = [i for i, c in enumerate(chunks) if c.quote.strip()]
        labels_from_llm = self._backend.label_chunks([chunks[i] for i in non_ws_indices])
        label_map = dict(zip(non_ws_indices, labels_from_llm))

        return [
            dataclasses.replace(c, category=label_map.get(i, "whitespace"))
            for i, c in enumerate(chunks)
        ]
