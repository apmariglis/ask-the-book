"""Tests for generation data structures and LLM parsing."""

from __future__ import annotations

from ask_the_book.generation.base import Excerpt
from ask_the_book.generation.base import GenerationResult


def test_generation_result_has_summary() -> None:
    result = GenerationResult(
        found_in_context=True,
        summary="A brief answer.",
        detail="A longer explanation.",
        caveats="",
        excerpts=[],
        elapsed_seconds=1.0,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )
    assert result.summary == "A brief answer."


def test_excerpt_has_supports_field() -> None:
    excerpt = Excerpt(text="verbatim passage", supports="backs up the main claim")
    assert excerpt.supports == "backs up the main claim"


def test_parse_llm_response_structured(monkeypatch) -> None:
    """AnthropicLLM correctly parses a well-formed structured JSON response."""
    import json
    from unittest.mock import MagicMock
    from ask_the_book.generation.anthropic_llm import AnthropicLLM

    payload = {
        "found_in_context": True,
        "summary": "A brief answer.",
        "detail": "A detailed explanation.",
        "caveats": "Only one source covers this.",
        "excerpts": [
            {"text": "verbatim passage", "supports": "backs up the summary"}
        ],
    }
    # Strip the leading "{" since the LLM prefill adds it back
    raw_text = json.dumps(payload)[1:]

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text=raw_text)]

    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm._model = "test-model"
    llm._max_tokens = 1024
    llm._client = MagicMock()
    llm._client.messages.create.return_value = fake_response

    result = llm.generate(question="What?", context_chunks=["some context"])

    assert result.found_in_context is True
    assert result.summary == "A brief answer."
    assert result.detail == "A detailed explanation."
    assert result.caveats == "Only one source covers this."
    assert len(result.excerpts) == 1
    assert result.excerpts[0].text == "verbatim passage"
    assert result.excerpts[0].supports == "backs up the summary"
