"""Tests for the chunking strategies."""

from __future__ import annotations

from ask_the_book.ingestion.chunker import WholePageChunker
from ask_the_book.ingestion.loader import Table
from fixtures import make_page


def test_whole_page_chunker_one_chunk_per_page() -> None:
    pages = [make_page(page=i) for i in range(1, 4)]
    chunks = WholePageChunker().chunk(pages)
    assert len(chunks) == 3


def test_whole_page_chunker_chunk_id_format() -> None:
    pages = [make_page(page=42)]
    chunks = WholePageChunker().chunk(pages)
    assert chunks[0].chunk_id == "page-42"


def test_whole_page_chunker_table_included_in_text() -> None:
    table = Table(title="Table 1", caption="A caption", content="| col1 | col2 |")
    page = make_page(text="Intro text.", tables=[table])
    chunks = WholePageChunker().chunk([page])
    assert "Table 1" in chunks[0].text
    assert "col1" in chunks[0].text
    assert "Intro text." in chunks[0].text


def test_whole_page_chunker_skips_empty_pages() -> None:
    page = make_page(text="", tables=[])
    chunks = WholePageChunker().chunk([page])
    assert chunks == []


def test_whole_page_chunker_preserves_metadata() -> None:
    page = make_page(page=7, book_page=5, title="Chapter 3")
    chunks = WholePageChunker().chunk([page])
    assert chunks[0].page == 7
    assert chunks[0].book_page == 5
    assert chunks[0].title == "Chapter 3"
