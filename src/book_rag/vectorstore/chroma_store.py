"""
ChromaDB vector store implementation.

ChromaDB runs entirely locally (no server needed) and persists data to
disk at the configured path. It uses cosine similarity by default.
"""

from __future__ import annotations

import chromadb
from book_rag.ingestion.chunker import Chunk
from book_rag.vectorstore.base import SearchResult
from book_rag.vectorstore.base import VectorStore
from chromadb.config import Settings


class ChromaStore(VectorStore):
    """
    Wraps a local ChromaDB collection.

    Parameters
    ----------
    path:
        Directory where ChromaDB will persist its data.
    collection_name:
        Name of the collection inside ChromaDB.
    """

    def __init__(self, path: str, collection_name: str = "book") -> None:
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[_chunk_metadata(c) for c in chunks],
        )

    def query(self, vector: list[float], top_k: int) -> list[SearchResult]:
        results = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
            # Convert to a similarity score in [0, 1].
            score = 1.0 - (distance / 2.0)

            search_results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    text=doc,
                    page=meta["page"],
                    book_page=meta.get("book_page"),
                    title=meta.get("title"),
                    score=score,
                )
            )

        return search_results


def _chunk_metadata(chunk: Chunk) -> dict:
    """Flatten a Chunk's fields into the flat dict ChromaDB metadata expects."""
    return {
        "chunk_id": chunk.chunk_id,
        "page": chunk.page,
        # ChromaDB metadata values must be str | int | float | bool.
        # Store None as an empty string so we can round-trip cleanly.
        "book_page": chunk.book_page if chunk.book_page is not None else "",
        "title": chunk.title if chunk.title is not None else "",
    }
