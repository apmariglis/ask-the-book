"""
``askthebook query`` command.

Asks a single question against the indexed book. No conversation memory.
Displays a structured answer followed by source citations.
"""

from __future__ import annotations

import click
from ask_the_book.cli.wiring import build_engine
from ask_the_book.rag.engine import RAGEngine
from ask_the_book.rag.engine import RAGResponse
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
    engine = build_engine()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Thinking…", total=None)
        response: RAGResponse = engine.query(question)

    _render_response(question, response)


def _render_response(question: str, response: RAGResponse) -> None:
    console.print()
    console.print(
        Panel(Text(question, style="bold cyan"), title="Question", expand=False)
    )
    console.print()

    if not response.found_in_context:
        console.print(
            Panel(
                Text(response.summary, style="yellow"),
                title="[yellow]Not found in book[/yellow]",
                border_style="yellow",
            )
        )
        console.print()
        return

    # Summary — quick one-liner
    console.print(Panel(Text(response.summary, style="bold"), title="Summary", expand=False))
    console.print()

    # Detail — full answer as Markdown
    if response.detail:
        console.print(Rule("Detail", style="dim"))
        console.print(Markdown(response.detail))
        console.print()

    # Caveats
    if response.caveats:
        console.print(
            Panel(
                Text(response.caveats, style="dim italic"),
                title="[dim]Caveats[/dim]",
                border_style="dim",
            )
        )
        console.print()

    # Excerpts with attribution
    if response.excerpts:
        console.print(Rule("Passages used", style="dim"))
        for i, excerpt in enumerate(response.excerpts, start=1):
            page_label = f"  [dim]p. {excerpt.book_page}[/dim]" if excerpt.book_page else ""
            supports_label = f"\n[dim]↳ {excerpt.supports}[/dim]" if excerpt.supports else ""
            console.print(
                Panel(
                    Text(excerpt.text, style="italic") if not excerpt.supports
                    else Text.assemble(
                        (excerpt.text, "italic"),
                        (f"\n↳ {excerpt.supports}", "dim"),
                    ),
                    title=f"[dim]Excerpt {i}[/dim]{page_label}",
                    border_style="dim",
                )
            )
        console.print()

    # Sources
    console.print(Rule("Sources", style="dim"))
    for i, source in enumerate(response.sources, start=1):
        page_label = (
            f"p. {source.book_page}" if source.book_page else f"scan p. {source.page}"
        )
        title_label = f" — {source.title}" if source.title else ""
        score_label = f"  [dim](score: {source.score:.2f})[/dim]"
        console.print(f"  [bold]{i}.[/bold] {page_label}{title_label}{score_label}")
    console.print()
