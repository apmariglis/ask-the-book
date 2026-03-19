"""
``askthebook query`` command.

Asks a single question against the indexed book. No conversation memory.
Displays the answer followed by source citations.
"""

from __future__ import annotations

import click
from ask_the_book.config import config
from ask_the_book.embedding.openai_embedder import OpenAIEmbedder
from ask_the_book.generation.anthropic_llm import AnthropicLLM
from ask_the_book.rag.engine import RAGEngine
from ask_the_book.rag.engine import RAGResponse
from ask_the_book.retrieval.retriever import Retriever
from ask_the_book.vectorstore.chroma_store import ChromaStore
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.rule import Rule
from rich.text import Text

console = Console()


@click.command()
@click.argument("question")
def query_command(question: str) -> None:
    """
    Ask QUESTION against the indexed book and print the answer.

    QUESTION should be a natural-language question in quotes,
    e.g.:  askthebook query "What does chapter 3 say about topic X?"
    """
    engine = _build_engine()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Thinking…", total=None)
        response: RAGResponse = engine.query(question)

    _render_response(question, response)


def _build_engine() -> RAGEngine:
    embedder = OpenAIEmbedder(
        api_key=config.openai_api_key, model=config.embedding_model
    )
    store = ChromaStore(
        path=config.chroma_path, collection_name=config.chroma_collection
    )
    retriever = Retriever(embedder=embedder, store=store, top_k=config.retrieval_top_k)
    llm = AnthropicLLM(
        api_key=config.anthropic_api_key,
        model=config.generation_model,
        max_tokens=config.max_tokens,
    )
    return RAGEngine(retriever=retriever, llm=llm)


def _render_response(question: str, response: RAGResponse) -> None:
    console.print()
    console.print(
        Panel(Text(question, style="bold cyan"), title="Question", expand=False)
    )
    console.print()

    console.print(Markdown(response.answer))
    console.print()

    if response.excerpts:
        console.print(Rule("Passages used", style="dim"))
        for i, excerpt in enumerate(response.excerpts, start=1):
            page_label = (
                f"  [dim]Book p. {excerpt.book_page}[/dim]" if excerpt.book_page else ""
            )
            console.print(
                Panel(
                    Text(excerpt.text, style="italic"),
                    title=f"[dim]Excerpt {i}[/dim]{page_label}",
                    border_style="dim",
                )
            )
        console.print()

    console.print(Rule("Sources", style="dim"))
    for i, source in enumerate(response.sources, start=1):
        page_label = (
            f"book p. {source.book_page}"
            if source.book_page
            else f"scan p. {source.page}"
        )
        title_label = f" — {source.title}" if source.title else ""
        score_label = f"  [dim](score: {source.score:.2f})[/dim]"
        console.print(f"  [bold]{i}.[/bold] {page_label}{title_label}{score_label}")
    console.print()
