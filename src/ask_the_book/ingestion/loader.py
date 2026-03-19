"""
Loads raw pages from a JSONL file produced by the OCR pipeline.

Each line in the file is a JSON object representing one book page.
Pages with ``skip: true`` are silently filtered out here so the rest
of the pipeline never has to think about them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass
class Table:
    title: str
    caption: str
    content: str  # Markdown table string


@dataclass
class Page:
    """A single, usable book page after loading and filtering."""

    page: int  # PDF/scan page number
    book_page: int | None  # Printed page number inside the book (may differ)
    title: str | None
    text: str
    tables: list[Table] = field(default_factory=list)


def load_pages(path: str | Path) -> list[Page]:
    """
    Read *path* (a JSONL file) and return all pages that should be indexed.

    Skipped pages (``skip: true``) are excluded. Malformed lines are skipped
    with a warning rather than crashing the whole ingest run.
    """
    pages: list[Page] = []

    with open(path, encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(f"[loader] Warning: skipping malformed line {line_number}: {exc}")
                continue

            if data.get("skip", False):
                continue

            tables = [
                Table(
                    title=t.get("title", ""),
                    caption=t.get("caption", ""),
                    content=t.get("content", ""),
                )
                for t in data.get("tables", [])
            ]

            pages.append(
                Page(
                    page=data["page"],
                    book_page=data.get("book_page"),
                    title=data.get("title"),
                    text=data.get("text", ""),
                    tables=tables,
                )
            )

    return pages
