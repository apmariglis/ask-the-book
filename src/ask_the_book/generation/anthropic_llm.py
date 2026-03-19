"""
Anthropic (Claude) generation provider.

The system prompt instructs the model to answer *only* from the supplied
context and to say so clearly when the context is insufficient, which
keeps hallucinations in check for a book-scoped RAG system.
"""

from __future__ import annotations

import json

import anthropic
from ask_the_book.generation.base import Excerpt
from ask_the_book.generation.base import GenerationResult
from ask_the_book.generation.base import LLMProvider
from ask_the_book.generation.base import Message

_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions strictly based on the \
provided excerpts from a book.

Rules:
- Base your answer ONLY on the context passages below.
- If the answer cannot be found in the context, say so in the "answer" field \
and leave "excerpts" as an empty list.
- Do not speculate or bring in outside knowledge.
- In "excerpts", copy verbatim only the specific sentences or passages from \
the context that your answer is directly based on. Do not include passages \
that are merely related but not actually used.

Respond with ONLY a valid JSON object in this exact format:
{
  "answer": "your answer here",
  "excerpts": [
    "verbatim passage 1 you based your answer on",
    "verbatim passage 2 you based your answer on"
  ]
}
"""


class AnthropicLLM(LLMProvider):
    """
    Wraps the Anthropic Messages API.

    Parameters
    ----------
    api_key:
        Anthropic API key.
    model:
        Claude model identifier.
    max_tokens:
        Upper bound on the generated response length.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(
        self,
        question: str,
        context_chunks: list[str],
        history: list[Message] | None = None,
    ) -> GenerationResult:
        context_block = _format_context(context_chunks)
        user_message = f"{context_block}\n\nQuestion: {question}"

        messages: list[dict] = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": "{"})

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=messages,
        )

        raw = "{" + response.content[0].text  # re-attach the prefill
        print("DEBUG RAW:", repr(raw))  # add this temporarily

        try:
            data = json.loads(raw)
            return GenerationResult(
                answer=data.get("answer", ""),
                excerpts=[Excerpt(text=t) for t in data.get("excerpts", [])],
            )
        except json.JSONDecodeError:
            # Graceful fallback: treat the whole response as the answer
            return GenerationResult(answer=raw, excerpts=[])


def _format_context(chunks: list[str]) -> str:
    """Wrap retrieved passages in a clearly labelled block."""
    numbered = "\n\n---\n\n".join(
        f"[Passage {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return f"Context passages from the book:\n\n{numbered}"
