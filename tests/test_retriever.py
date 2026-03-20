"""Tests for the Retriever."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from ask_the_book.retrieval.retriever import Retriever
from fixtures import make_search_result


def test_retriever_embeds_the_query_and_searches_the_store() -> None:
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    store = MagicMock()
    store.query.return_value = [make_search_result()]

    results = Retriever(embedder, store, top_k=3).retrieve("What is X?")

    embedder.embed.assert_called_once_with(["What is X?"])
    store.query.assert_called_once_with([0.1, 0.2, 0.3], top_k=3)
    assert len(results) == 1


def test_retriever_raises_if_embedder_returns_wrong_number_of_vectors() -> None:
    embedder = MagicMock()
    embedder.embed.return_value = []  # should always return exactly 1

    store = MagicMock()

    with pytest.raises(ValueError, match="Expected 1 embedding vector"):
        Retriever(embedder, store).retrieve("query")
