"""
``askthebook ingest`` command.

Reads a JSONL file, chunks its pages, embeds them, and stores the
result in ChromaDB. Run this once before querying.
"""

from __future__ import annotations

from pathlib import Path

import click
from ask_the_book.cli.wiring import build_embedder
from ask_the_book.cli.wiring import build_store
from ask_the_book.config import config
from ask_the_book.ingestion.pipeline import ingest
from rich.console import Console
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn

console = Console()


@click.command()
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
def ingest_command(jsonl_path: Path) -> None:
    """
    Index JSONL_PATH into the vector store.

    JSONL_PATH is the path to your OCR output file.
    """
    embedder = build_embedder()
    store = build_store()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Ingesting…", total=None)
        total = ingest(jsonl_path=jsonl_path, embedder=embedder, store=store)

    console.print(
        f"[green]✓[/green] Indexed [bold]{total}[/bold] chunks from [cyan]{jsonl_path}[/cyan]"
    )
    console.print(f"  Vector store: [cyan]{config.chroma_path}[/cyan]")
