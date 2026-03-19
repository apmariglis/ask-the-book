"""
``rag ingest`` command.

Reads a JSONL file, chunks its pages, embeds them, and stores the
result in ChromaDB. Run this once before querying.
"""

from __future__ import annotations

from pathlib import Path

import click
from book_rag.config import config
from book_rag.embedding.openai_embedder import OpenAIEmbedder
from book_rag.ingestion.pipeline import ingest
from book_rag.vectorstore.chroma_store import ChromaStore
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
    embedder = OpenAIEmbedder(
        api_key=config.openai_api_key, model=config.embedding_model
    )
    store = ChromaStore(
        path=config.chroma_path, collection_name=config.chroma_collection
    )

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
