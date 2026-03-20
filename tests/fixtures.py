"""Shared factory functions for test data."""

from __future__ import annotations

from ask_the_book.ingestion.loader import Page
from ask_the_book.ingestion.loader import Table
from ask_the_book.vectorstore.base import SearchResult


def make_page(**kwargs) -> Page:
    defaults = dict(page=1, book_page=1, title="Title", text="Body text.", tables=[])
    return Page(**{**defaults, **kwargs})


def make_table(**kwargs) -> Table:
    defaults = dict(title="Table 1", caption="A caption", content="| col |")
    return Table(**{**defaults, **kwargs})


def make_search_result(**kwargs) -> SearchResult:
    page = kwargs.get("page", 1)
    defaults = dict(
        chunk_id=f"page-{page}",
        text=f"Text from page {page}.",
        page=page,
        book_page=page,
        title=f"Chapter {page}",
        score=0.9,
    )
    return SearchResult(**{**defaults, **kwargs})
