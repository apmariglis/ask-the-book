"""
Converts loaded Pages into Chunks ready for embedding.

Strategy (v1 — simplest possible):
    One chunk per page. Tables are serialised as Markdown and appended
    to the page text so they remain in context together.

The ``Chunker`` base class makes it straightforward to swap in a more
sophisticated strategy later (e.g. paragraph-level splitting, sliding
window, separate chunk per table) without touching any other module.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

from ask_the_book.ingestion.loader import Page


@dataclass
class Chunk:
    """The unit of text that gets embedded and stored."""

    chunk_id: str  # Unique, stable identifier (used as ChromaDB doc id)
    text: str  # The text to embed
    page: int
    book_page: int | None
    title: str | None


class Chunker(ABC):
    """Abstract base — implement ``chunk`` to define a splitting strategy."""

    @abstractmethod
    def chunk(self, pages: list[Page]) -> list[Chunk]:
        """Convert a list of Pages into a list of Chunks."""
        ...


class WholePageChunker(Chunker):
    """
    One chunk per page.

    Tables are converted to Markdown and appended after the main text,
    separated by a blank line and a heading so an LLM can tell them apart.
    """

    def chunk(self, pages: list[Page]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page in pages:
            parts: list[str] = []

            if page.text.strip():
                parts.append(page.text.strip())

            for table in page.tables:
                table_block = _format_table(table.title, table.caption, table.content)
                parts.append(table_block)

            if not parts:
                continue  # Nothing to index on this page

            combined_text = "\n\n".join(parts)
            chunks.append(
                Chunk(
                    chunk_id=f"page-{page.page}",
                    text=combined_text,
                    page=page.page,
                    book_page=page.book_page,
                    title=page.title,
                )
            )

        return chunks


def _format_table(title: str, caption: str, content: str) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"### {title}")
    if caption:
        lines.append(f"_{caption}_")
    if content:
        lines.append(content)
    return "\n".join(lines)
