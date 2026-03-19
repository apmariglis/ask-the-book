"""
CLI entry point.

Two subcommands:
    askthebook ingest  — index a JSONL file into the vector store
    askthebook query   — ask a single question (no memory)

The wiring of concrete providers happens here so all other modules
remain decoupled from each other.
"""

from __future__ import annotations

import click
from ask_the_book.cli.ingest import ingest_command
from ask_the_book.cli.query import query_command


@click.group()
def cli() -> None:
    """AskTheBook — query your reference books with natural language."""


cli.add_command(ingest_command, name="ingest")
cli.add_command(query_command, name="query")

if __name__ == "__main__":
    cli()
