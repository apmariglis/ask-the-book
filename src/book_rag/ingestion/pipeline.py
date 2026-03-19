"""
Ingestion pipeline: load → chunk → embed → store.

This module wires the individual pieces together into a single
``ingest`` function. Each dependency is injected, so the pipeline
is fully testable and provider-agnostic.
"""

from __future__ import annotations

from pathlib import Path

from book_rag.embedding.base import EmbeddingProvider
from book_rag.ingestion.chunker import Chunk
from book_rag.ingestion.chunker import Chunker
from book_rag.ingestion.chunker import WholePageChunker
from book_rag.ingestion.loader import load_pages
from book_rag.vectorstore.base import VectorStore


def ingest(
    jsonl_path: str | Path,
    embedder: EmbeddingProvider,
    store: VectorStore,
    chunker: Chunker | None = None,
    batch_size: int = 100,
) -> int:
    """
    Full ingestion run.

    Parameters
    ----------
    jsonl_path:
        Path to the source JSONL file.
    embedder:
        Provider used to turn chunk text into vectors.
    store:
        Vector store where chunks and their embeddings are persisted.
    chunker:
        Splitting strategy. Defaults to ``WholePageChunker``.
    batch_size:
        Number of chunks to embed and upsert per API call.

    Returns
    -------
    int
        Total number of chunks indexed.
    """
    if chunker is None:
        chunker = WholePageChunker()

    pages = load_pages(jsonl_path)
    chunks = chunker.chunk(pages)

    for batch in _batched(chunks, batch_size):
        texts = [c.text for c in batch]
        vectors = embedder.embed(texts)
        store.upsert(chunks=batch, vectors=vectors)

    return len(chunks)


def _batched(items: list[Chunk], size: int):
    """Yield successive slices of *items* of length *size*."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
