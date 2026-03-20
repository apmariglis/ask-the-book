"""
Abstract interface for LLM generation providers.

A provider receives a system prompt, a list of context chunks, and the
user question, and returns an answer string. Conversation history is
passed as an optional list of message dicts so memory can be added later
without changing the signature of existing providers.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TypedDict


class Message(TypedDict):
    role: str  # "user" or "assistant"
    content: str


@dataclass
class Excerpt:
    text: str
    supports: str = ""          # which part of the answer this passage backs up
    book_page: int | None = None  # filled in by the engine, not the LLM


@dataclass
class GenerationResult:
    found_in_context: bool  # whether the answer could be grounded in retrieved chunks
    summary: str            # one-sentence answer
    detail: str             # full explanation
    caveats: str            # limitations or gaps; empty string if none
    excerpts: list[Excerpt]
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None if the model is not in the pricing table


class LLMProvider(ABC):
    """Generates an answer given retrieved context and a question."""

    @abstractmethod
    def generate(
        self,
        question: str,
        context_chunks: list[str],
        history: list[Message] | None = None,
    ) -> GenerationResult:
        """
        Produce an answer.

        Parameters
        ----------
        question:
            The user's question.
        context_chunks:
            Retrieved text passages that should ground the answer.
        history:
            Prior conversation turns (unused for now; reserved for
            memory support). Pass ``None`` or ``[]`` to ignore.
        """
        ...
