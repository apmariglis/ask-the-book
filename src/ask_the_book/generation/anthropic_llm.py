"""
Anthropic (Claude) generation provider.

The system prompt instructs the model to answer *only* from the supplied
context and to say so clearly when the context is insufficient, which
keeps hallucinations in check for a book-scoped RAG system.
"""

from __future__ import annotations

import json
import re
import time

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
- If the answer cannot be found in the context, set "found_in_context" to false, \
write a short explanation in "summary", leave "detail" and "caveats" empty, \
and leave "excerpts" as an empty list.
- Do not speculate or bring in outside knowledge.
- In "excerpts", copy verbatim only the specific sentences or passages from \
the context that your answer is directly based on. For each excerpt, add a \
"supports" field explaining which part of your answer it backs up.

Respond with ONLY a valid JSON object in this exact format:
{
  "found_in_context": true,
  "summary": "one-sentence answer",
  "detail": "full explanation in Markdown — use - for bullet points, numbered lists, or ## headers where they help clarity; never use • characters",
  "caveats": "any limitations or gaps in the context, or empty string if none",
  "excerpts": [
    {
      "text": "verbatim passage from the context",
      "supports": "which part of the answer this passage backs up"
    }
  ]
}
"""

_BULLET_RE = re.compile(r"^[ \t]*[•▪▸◦●‣⁃∙◆◇→]\s*", re.MULTILINE)


def _normalize_markdown(text: str) -> str:
    """Replace non-standard bullet characters at line starts with Markdown `- `."""
    return _BULLET_RE.sub("- ", text)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return estimated cost in USD using LiteLLM's pricing database, or None on failure."""
    try:
        import litellm
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return prompt_cost + completion_cost
    except Exception:
        return None


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
        model: str = "claude-sonnet-4-5",
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

        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=messages,
        )
        elapsed = time.monotonic() - t0

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        raw = "{" + response.content[0].text  # re-attach the prefill

        try:
            data = json.loads(raw)
            return GenerationResult(
                found_in_context=bool(data.get("found_in_context", True)),
                summary=data.get("summary", ""),
                detail=_normalize_markdown(data.get("detail", "")),
                caveats=data.get("caveats", ""),
                excerpts=[
                    Excerpt(
                        text=e["text"],
                        supports=e.get("supports", ""),
                    )
                    for e in data.get("excerpts", [])
                    if isinstance(e, dict)
                ],
                elapsed_seconds=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=_estimate_cost(self._model, input_tokens, output_tokens),
            )
        except json.JSONDecodeError:
            # Graceful fallback: treat the whole response as the summary
            return GenerationResult(
                found_in_context=False,
                summary=raw,
                detail="",
                caveats="",
                excerpts=[],
                elapsed_seconds=elapsed,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=_estimate_cost(self._model, input_tokens, output_tokens),
            )


def _format_context(chunks: list[str]) -> str:
    """Wrap retrieved passages in a clearly labelled block."""
    numbered = "\n\n---\n\n".join(
        f"[Passage {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return f"Context passages from the book:\n\n{numbered}"
