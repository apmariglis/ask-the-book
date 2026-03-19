"""
OpenAI embedding provider.

Uses the ``text-embedding-3-small`` model by default (fast and cheap).
Switch to ``text-embedding-3-large`` via the EMBEDDING_MODEL env var
if you need higher accuracy.
"""

from __future__ import annotations

from ask_the_book.embedding.base import EmbeddingProvider
from openai import OpenAI


class OpenAIEmbedder(EmbeddingProvider):
    """
    Wraps the OpenAI Embeddings API.

    Parameters
    ----------
    api_key:
        OpenAI API key. Reads from the environment if not provided.
    model:
        Embedding model name.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(input=texts, model=self._model)

        # The API returns embeddings in the same order as the input.
        return [item.embedding for item in response.data]
