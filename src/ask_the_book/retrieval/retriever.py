"""
Retriever: embed a query and fetch the most relevant chunks.

This is intentionally a thin layer — it delegates embedding to the
provider and searching to the vector store. Its only job is to
connect them and expose a clean interface to the RAG engine.
"""

from __future__ import annotations

from ask_the_book.embedding.base import EmbeddingProvider
from ask_the_book.vectorstore.base import SearchResult
from ask_the_book.vectorstore.base import VectorStore


class Retriever:
    """
    Finds the *top_k* chunks most relevant to a natural-language query.

    Parameters
    ----------
    embedder:
        Provider used to embed the query text.
    store:
        Vector store to search.
    top_k:
        Number of results to return.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    def retrieve(self, query: str) -> list[SearchResult]:
        """Embed *query* and return the closest chunks from the store."""
        [query_vector] = self._embedder.embed([query])
        return self._store.query(query_vector, top_k=self._top_k)
