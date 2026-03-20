"""Tests for ChromaDB metadata round-trips."""

from __future__ import annotations

from ask_the_book.vectorstore.chroma_store import _chunk_metadata
from ask_the_book.ingestion.chunker import Chunk


def _make_chunk(**kwargs) -> Chunk:
    defaults = dict(
        chunk_id="page-1",
        text="Some text.",
        page=1,
        book_page=None,
        title=None,
        table_title=None,
    )
    return Chunk(**{**defaults, **kwargs})


def test_book_page_stored_as_empty_string_when_none() -> None:
    meta = _chunk_metadata(_make_chunk(book_page=None))
    assert meta["book_page"] == ""


def test_book_page_stored_as_integer_when_present() -> None:
    meta = _chunk_metadata(_make_chunk(book_page=42))
    assert meta["book_page"] == 42


def test_title_stored_as_empty_string_when_none() -> None:
    meta = _chunk_metadata(_make_chunk(title=None))
    assert meta["title"] == ""


def test_table_title_stored_as_empty_string_when_none() -> None:
    meta = _chunk_metadata(_make_chunk(table_title=None))
    assert meta["table_title"] == ""


def test_table_title_stored_as_string_when_present() -> None:
    meta = _chunk_metadata(_make_chunk(table_title="Table 8: Racial Adjustments"))
    assert meta["table_title"] == "Table 8: Racial Adjustments"
