"""
Abstract interface for vector stores.

Swapping ChromaDB for Pinecone, Weaviate, pgvector, etc. means
implementing these two methods and changing one line in the wiring code.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

from ask_the_book.ingestion.chunker import Chunk


@dataclass
class SearchResult:
    """A single result returned by a similarity search."""

    chunk_id: str
    text: str
    page: int
    book_page: int | None
    title: str | None
    score: float  # Higher is more similar (normalised to [0, 1] where possible)


class VectorStore(ABC):
    """Persist chunks + vectors and retrieve the closest ones to a query."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """
        Insert or update *chunks* in the store using the provided *vectors*.

        ``chunks[i]`` corresponds to ``vectors[i]``.
        """
        ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int) -> list[SearchResult]:
        """
        Return the *top_k* chunks most similar to *vector*.

        Results should be ordered by descending similarity.
        """
        ...
