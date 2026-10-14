from __future__ import annotations

import dataclasses
import warnings
from typing import Literal, Union

from chunklabel.alignment import AlignmentDetail, align_detailed, raise_unaligned
from chunklabel.eval import FidelityReport, fidelity_from_details
from chunklabel.gapfill import fill_gaps
from chunklabel.llm.base import LLMBackend
from chunklabel.llm.client import BaseLLMClient
from chunklabel.postprocess import postprocess
from chunklabel.types import Chunk, RawChunk


class ChunkLabeler:
    def __init__(
        self,
        client: Union[BaseLLMClient, str] = "gpt-4o",
        fuzzy_threshold: int = 85,
        on_align_error: Literal["raise", "skip"] = "raise",
        timeout: float | None = 120.0,
        backend: LLMBackend | None = None,
        fill_gaps: bool = False,
        min_gap_chars: int = 20,
    ) -> None:
        """
        Args:
            fill_gaps: Send each stretch of text that no aligned quote covers (at least
                ``min_gap_chars`` non-whitespace characters) back to the LLM, alone and
                once, instead of marking it ``uncategorized``.
        """
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
        self.fill_gaps = fill_gaps
        self.min_gap_chars = min_gap_chars

    def split(self, text: str, mode: Literal["one_pass", "two_pass"] = "one_pass") -> list[Chunk]:
        chunks, _ = self._split(text, mode, strict=True)
        return chunks

    def split_with_report(
        self, text: str, mode: Literal["one_pass", "two_pass"] = "one_pass"
    ) -> tuple[list[Chunk], FidelityReport]:
        """Like ``split``, but also report how faithfully the LLM's quotes matched ``text``.

        Use it to check fidelity for your model and data. Unlike ``split``, it never raises
        on alignment errors: quotes that cannot be aligned are counted in the report and
        dropped, as with ``on_align_error="skip"``.
        """
        return self._split(text, mode, strict=False)

    def _split(
        self, text: str, mode: Literal["one_pass", "two_pass"], strict: bool
    ) -> tuple[list[Chunk], FidelityReport]:
        def extract(t: str) -> list[RawChunk]:
            if mode == "two_pass":
                return self._backend.extract_boundaries(t)
            return self._backend.extract_chunks(t)

        raise_on_unaligned = strict and self.on_align_error == "raise"
        raw_chunks = extract(text)
        details: list[AlignmentDetail] = align_detailed(raw_chunks, text, self.fuzzy_threshold)
        if raise_on_unaligned:
            raise_unaligned(raw_chunks, details, self.fuzzy_threshold)
        n_filled = 0
        if self.fill_gaps:
            raw_chunks, details, n_filled = fill_gaps(
                raw_chunks, details, text, extract, self.fuzzy_threshold, self.min_gap_chars
            )
            if raise_on_unaligned:
                raise_unaligned(raw_chunks, details, self.fuzzy_threshold)
        chunks = postprocess(raw_chunks, [d.span for d in details], text)
        if mode == "two_pass":
            chunks = self._label(chunks)
        return chunks, fidelity_from_details(details, text, n_filled)

    def _label(self, chunks: list[Chunk]) -> list[Chunk]:
        """Two-pass labeling step: ask the LLM for a category per non-whitespace chunk."""
        non_ws_indices = [i for i, c in enumerate(chunks) if c.quote.strip()]
        labels_from_llm = self._backend.label_chunks([chunks[i] for i in non_ws_indices])
        label_map = dict(zip(non_ws_indices, labels_from_llm))

        return [
            dataclasses.replace(c, category=label_map.get(i, "whitespace"))
            for i, c in enumerate(chunks)
        ]
