"""Tests for the chunking strategies."""

from __future__ import annotations

from book_rag.ingestion.chunker import WholePageChunker
from book_rag.ingestion.loader import Page
from book_rag.ingestion.loader import Table


def _make_page(**kwargs) -> Page:
    defaults = dict(page=1, book_page=1, title="Title", text="Body text.", tables=[])
    return Page(**{**defaults, **kwargs})


def test_whole_page_chunker_one_chunk_per_page() -> None:
    pages = [_make_page(page=i) for i in range(1, 4)]
    chunks = WholePageChunker().chunk(pages)
    assert len(chunks) == 3


def test_whole_page_chunker_chunk_id_format() -> None:
    pages = [_make_page(page=42)]
    chunks = WholePageChunker().chunk(pages)
    assert chunks[0].chunk_id == "page-42"


def test_whole_page_chunker_table_included_in_text() -> None:
    table = Table(title="Table 1", caption="A caption", content="| col1 | col2 |")
    page = _make_page(text="Intro text.", tables=[table])
    chunks = WholePageChunker().chunk([page])
    assert "Table 1" in chunks[0].text
    assert "col1" in chunks[0].text
    assert "Intro text." in chunks[0].text


def test_whole_page_chunker_skips_empty_pages() -> None:
    page = _make_page(text="", tables=[])
    chunks = WholePageChunker().chunk([page])
    assert chunks == []


def test_whole_page_chunker_preserves_metadata() -> None:
    page = _make_page(page=7, book_page=5, title="Wisdom")
    chunks = WholePageChunker().chunk([page])
    assert chunks[0].page == 7
    assert chunks[0].book_page == 5
    assert chunks[0].title == "Wisdom"
