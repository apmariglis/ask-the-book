"""Tests for generation data structures and LLM parsing."""

from __future__ import annotations

from unittest.mock import MagicMock

from ask_the_book.generation.anthropic_llm import AnthropicLLM
from ask_the_book.generation.anthropic_llm import _normalize_markdown
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
    fake_response.usage = MagicMock(input_tokens=100, output_tokens=50)

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


def test_parse_llm_response_falls_back_gracefully_when_json_is_invalid() -> None:
    """When the LLM returns non-JSON, the raw text becomes the summary."""
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="Sorry, I cannot answer that.")]
    fake_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm._model = "test-model"
    llm._max_tokens = 1024
    llm._client = MagicMock()
    llm._client.messages.create.return_value = fake_response

    result = llm.generate(question="What?", context_chunks=["context"])

    assert result.found_in_context is False
    assert "Sorry" in result.summary
    assert result.excerpts == []


def test_normalize_markdown_replaces_bullet_characters_with_markdown_list_syntax() -> None:
    text = "• First point\n• Second point"
    assert _normalize_markdown(text) == "- First point\n- Second point"


def test_normalize_markdown_handles_multiple_bullet_variants() -> None:
    text = "▪ Point A\n▸ Point B\n◦ Point C\n● Point D"
    result = _normalize_markdown(text)
    assert result == "- Point A\n- Point B\n- Point C\n- Point D"


def test_normalize_markdown_leaves_standard_markdown_lists_unchanged() -> None:
    text = "- Already correct\n- Second item"
    assert _normalize_markdown(text) == text
