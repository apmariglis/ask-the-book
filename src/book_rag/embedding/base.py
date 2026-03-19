"""
Abstract interface for embedding providers.

Any new provider (Cohere, local sentence-transformers, etc.) only needs
to implement this one method to plug into the rest of the system.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class EmbeddingProvider(ABC):
    """Turns a list of strings into a list of float vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed *texts* and return one vector per input string.

        The order of the returned vectors must match the order of *texts*.
        """
        ...
