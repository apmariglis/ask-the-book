"""Tests for the JSONL loader."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from ask_the_book.ingestion.loader import Page
from ask_the_book.ingestion.loader import load_pages


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def _base_record(**overrides) -> dict:
    return {
        "page": 1,
        "book_page": 1,
        "title": "Chapter 1",
        "text": "Some text.",
        "tables": [],
        "skip": False,
        "skip_reason": None,
        **overrides,
    }


def test_load_pages_returns_non_skipped(tmp_path: Path) -> None:
    records = [
        _base_record(page=1, skip=False),
        _base_record(page=2, skip=True),
        _base_record(page=3, skip=False),
    ]
    pages = load_pages(_write_jsonl(tmp_path, records))

    assert len(pages) == 2
    assert {p.page for p in pages} == {1, 3}


def test_load_pages_filters_illustration_only(tmp_path: Path) -> None:
    records = [
        _base_record(page=1, skip=True, skip_reason="illustration_only", text="")
    ]
    pages = load_pages(_write_jsonl(tmp_path, records))
    assert pages == []


def test_load_pages_parses_tables(tmp_path: Path) -> None:
    records = [
        _base_record(
            tables=[
                {"title": "Table 1", "caption": "A caption", "content": "| a | b |"}
            ]
        )
    ]
    pages = load_pages(_write_jsonl(tmp_path, records))
    assert len(pages[0].tables) == 1
    assert pages[0].tables[0].title == "Table 1"


def test_load_pages_handles_missing_optional_fields(tmp_path: Path) -> None:
    # book_page and title are optional
    record = {"page": 5, "text": "Hello", "skip": False, "tables": []}
    pages = load_pages(_write_jsonl(tmp_path, [record]))
    assert pages[0].book_page is None
    assert pages[0].title is None


def test_load_pages_skips_malformed_lines(tmp_path: Path, capsys) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"page": 1, "text": "ok", "skip": false, "tables": []}\nNOT JSON\n',
        encoding="utf-8",
    )
    pages = load_pages(p)
    # The good line is loaded; the bad line is skipped with a warning
    assert len(pages) == 1
