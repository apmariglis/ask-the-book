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


def test_page_with_text_and_two_tables_produces_three_chunks() -> None:
    table1 = Table(title="Table 1", caption="", content="| a |")
    table2 = Table(title="Table 2", caption="", content="| b |")
    page = make_page(text="Some text.", tables=[table1, table2])
    chunks = WholePageChunker().chunk([page])
    assert len(chunks) == 3


def test_table_chunk_contains_table_content() -> None:
    table = Table(title="Table 1", caption="A caption", content="| col1 | col2 |")
    page = make_page(text="Intro text.", tables=[table])
    chunks = WholePageChunker().chunk([page])
    table_chunk = next(c for c in chunks if c.table_title is not None)
    assert "col1" in table_chunk.text
    assert "Table 1" in table_chunk.text


def test_text_chunk_does_not_contain_table_content() -> None:
    table = Table(title="Table 1", caption="", content="| col1 |")
    page = make_page(text="Intro text.", tables=[table])
    chunks = WholePageChunker().chunk([page])
    text_chunk = next(c for c in chunks if c.table_title is None)
    assert "col1" not in text_chunk.text
    assert "Intro text." in text_chunk.text


def test_table_chunk_id_format() -> None:
    table = Table(title="Table 1", caption="", content="| a |")
    page = make_page(page=5, tables=[table])
    chunks = WholePageChunker().chunk([page])
    table_chunk = next(c for c in chunks if c.table_title is not None)
    assert table_chunk.chunk_id == "page-5-table-0"


def test_table_chunk_carries_table_title() -> None:
    table = Table(title="Table 8: Racial Adjustments", caption="", content="| a |")
    page = make_page(tables=[table])
    chunks = WholePageChunker().chunk([page])
    table_chunk = next(c for c in chunks if c.table_title is not None)
    assert table_chunk.table_title == "Table 8: Racial Adjustments"


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


def test_text_chunk_has_no_table_title() -> None:
    page = make_page(text="Some text.")
    chunks = WholePageChunker().chunk([page])
    assert chunks[0].table_title is None
