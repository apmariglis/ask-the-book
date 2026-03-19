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
    book_page: int | None = None  # filled in by the engine, not the LLM


@dataclass
class GenerationResult:
    answer: str
    excerpts: list[Excerpt]


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
