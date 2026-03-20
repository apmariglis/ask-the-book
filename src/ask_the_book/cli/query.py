"""
``askthebook query`` command.

Asks a single question against the indexed book. No conversation memory.
Displays a structured answer followed by source citations.
"""

from __future__ import annotations

import click
from ask_the_book.cli.wiring import build_engine
from ask_the_book.generation.base import Excerpt
from ask_the_book.rag.engine import RAGResponse
from ask_the_book.rag.engine import Source
from rich.console import Console
from rich.console import Group
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

    answer_parts: list = [Text(response.summary, style="bold")]
    if response.detail:
        answer_parts.append(Rule(style="dim"))
        answer_parts.append(Markdown(response.detail))
    if response.caveats:
        answer_parts.append(Rule("Caveats", style="dim"))
        answer_parts.append(Text(response.caveats, style="dim italic"))

    console.print(Panel(Group(*answer_parts), title="Answer"))
    console.print()

    excerpts_by_chunk: dict[str | None, list[Excerpt]] = {}
    for excerpt in response.excerpts:
        excerpts_by_chunk.setdefault(excerpt.source_chunk_id, []).append(excerpt)

    used: list[Source] = []
    retrieved: list[Source] = []
    for source in response.sources:
        if excerpts_by_chunk.get(source.chunk_id):
            used.append(source)
        else:
            retrieved.append(source)

    if used:
        console.print(Rule("Used in answer", style="dim"))
        console.print()
        for i, source in enumerate(used, start=1):
            page_label = _page_label(source.page, source.book_page)
            title_label = f" — {source.table_title}" if source.table_title else (f" — {source.title}" if source.title else "")
            score_label = f"  (score: {source.score:.2f})"

            excerpt_lines = Text()
            for j, excerpt in enumerate(excerpts_by_chunk.get(source.chunk_id, [])):
                if j > 0:
                    excerpt_lines.append("\n\n")
                excerpt_lines.append(excerpt.text, style="italic")
                if excerpt.supports:
                    excerpt_lines.append(f"\n↳ {excerpt.supports}", style="dim")

            console.print(
                Panel(
                    excerpt_lines,
                    title=f"[bold]{i}.[/bold] {page_label}{title_label}[dim]{score_label}[/dim]",
                    border_style="yellow",
                )
            )

    if retrieved:
        console.print()
        console.print(Rule("Also retrieved", style="dim"))
        for source in retrieved:
            page_label = _page_label(source.page, source.book_page)
            title_label = f" — {source.table_title}" if source.table_title else (f" — {source.title}" if source.title else "")
            score_label = f"  [dim](score: {source.score:.2f})[/dim]"
            console.print(f"  {page_label}{title_label}{score_label}")

    console.print()
    _render_stats(response)


def _page_label(page: int, book_page: int | None) -> str:
    return f"scan p. {page} · book p. {book_page if book_page else 'n/a'}"


def _render_stats(response: RAGResponse) -> None:
    cost = f"${response.cost_usd:.4f}" if response.cost_usd is not None else "n/a"
    console.print(
        f"[dim]  {response.elapsed_seconds:.1f}s · "
        f"{response.input_tokens} in / {response.output_tokens} out · "
        f"cost {cost}[/dim]"
    )
    console.print()
