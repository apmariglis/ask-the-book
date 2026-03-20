"""
RAG engine: the core, UI-agnostic question-answering entry point.

The engine knows nothing about CLI, HTTP, or any other interface.
It takes a question and returns a structured response that any
frontend (CLI, GUI, API) can render however it likes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ask_the_book.generation.base import Excerpt
from ask_the_book.generation.base import LLMProvider
from ask_the_book.generation.base import Message
from ask_the_book.retrieval.retriever import Retriever
from ask_the_book.vectorstore.base import SearchResult


@dataclass
class Source:
    """Metadata about a single retrieved chunk shown as a citation."""

    page: int
    book_page: int | None
    title: str | None
    score: float


@dataclass
class RAGResponse:
    """
    The full result of a single RAG query.

    Separating the answer from the sources lets any frontend decide
    independently how and whether to display citations.
    """

    found_in_context: bool
    summary: str
    detail: str
    caveats: str
    sources: list[Source]
    excerpts: list[Excerpt]
    elapsed_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None if the model is not in the pricing table


class RAGEngine:
    """
    Orchestrates retrieval → generation.

    Parameters
    ----------
    retriever:
        Finds relevant chunks for a query.
    llm:
        Generates an answer from the retrieved chunks.
    """

    def __init__(self, retriever: Retriever, llm: LLMProvider) -> None:
        self._retriever = retriever
        self._llm = llm

    def query(
        self,
        question: str,
        history: list[Message] | None = None,
    ) -> RAGResponse:
        """
        Answer *question* using only content retrieved from the book.

        Parameters
        ----------
        question:
            Natural-language question from the user.
        history:
            Conversation history for future memory support.
            Pass ``None`` (default) for a stateless query.

        Returns
        -------
        RAGResponse
            The generated answer and the source chunks it was based on.
        """
        results: list[SearchResult] = self._retriever.retrieve(question)

        context_chunks = [r.text for r in results]

        sources = [
            Source(
                page=r.page,
                book_page=r.book_page,
                title=r.title,
                score=r.score,
            )
            for r in results
        ]

        result = self._llm.generate(
            question=question,
            context_chunks=context_chunks,
            history=history,
        )

        # Match each excerpt to the chunk it most likely came from
        annotated_excerpts = _annotate_excerpts(result.excerpts, results)

        return RAGResponse(
            found_in_context=result.found_in_context,
            summary=result.summary,
            detail=result.detail,
            caveats=result.caveats,
            sources=sources,
            excerpts=annotated_excerpts,
            elapsed_seconds=result.elapsed_seconds,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )


def _annotate_excerpts(
    excerpts: list[Excerpt],
    results: list[SearchResult],
) -> list[Excerpt]:
    """
    For each excerpt, find which retrieved chunk contains it and attach
    that chunk's book_page. Matching is done by substring search.
    """
    annotated = []
    for excerpt in excerpts:
        book_page = None
        for result in results:
            if excerpt.text.strip() in result.text:
                book_page = result.book_page
                break
        annotated.append(Excerpt(text=excerpt.text, supports=excerpt.supports, book_page=book_page))
    return annotated
