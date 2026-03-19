"""Tests for the RAG engine."""

from __future__ import annotations

from unittest.mock import MagicMock

from ask_the_book.rag.engine import RAGEngine
from ask_the_book.rag.engine import RAGResponse
from ask_the_book.vectorstore.base import SearchResult


def _make_search_result(page: int = 1, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=f"page-{page}",
        text=f"Text from page {page}.",
        page=page,
        book_page=page,
        title=f"Chapter {page}",
        score=score,
    )


def test_engine_returns_rag_response() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_search_result()]

    llm = MagicMock()
    llm.generate.return_value = "The answer."

    engine = RAGEngine(retriever=retriever, llm=llm)
    response = engine.query("What is the answer?")

    assert isinstance(response, RAGResponse)
    assert response.answer == "The answer."
    assert len(response.sources) == 1


def test_engine_passes_context_chunks_to_llm() -> None:
    results = [_make_search_result(page=i) for i in range(1, 4)]

    retriever = MagicMock()
    retriever.retrieve.return_value = results

    llm = MagicMock()
    llm.generate.return_value = "Answer."

    engine = RAGEngine(retriever=retriever, llm=llm)
    engine.query("Question?")

    _, call_kwargs = llm.generate.call_args
    assert len(call_kwargs["context_chunks"]) == 3


def test_engine_passes_history_to_llm() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = []

    llm = MagicMock()
    llm.generate.return_value = "Answer."

    history = [{"role": "user", "content": "Prior question"}]
    engine = RAGEngine(retriever=retriever, llm=llm)
    engine.query("New question?", history=history)

    _, call_kwargs = llm.generate.call_args
    assert call_kwargs["history"] == history


def test_engine_sources_match_results() -> None:
    results = [_make_search_result(page=3, score=0.75)]

    retriever = MagicMock()
    retriever.retrieve.return_value = results

    llm = MagicMock()
    llm.generate.return_value = "Answer."

    engine = RAGEngine(retriever=retriever, llm=llm)
    response = engine.query("Q?")

    assert response.sources[0].page == 3
    assert response.sources[0].score == 0.75
