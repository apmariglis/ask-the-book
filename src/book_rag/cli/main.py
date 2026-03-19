"""
CLI entry point.

Two subcommands:
    rag ingest  — index a JSONL file into the vector store
    rag query   — ask a single question (no memory)

The wiring of concrete providers happens here so all other modules
remain decoupled from each other.
"""

from __future__ import annotations

import click
from book_rag.cli.ingest import ingest_command
from book_rag.cli.query import query_command


@click.group()
def cli() -> None:
    """Book RAG — query your OCR'd book with natural language."""


cli.add_command(ingest_command, name="ingest")
cli.add_command(query_command, name="query")

if __name__ == "__main__":
    cli()
