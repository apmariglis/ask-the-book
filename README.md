# AskTheBook

A clean, modular **Retrieval-Augmented Generation (RAG)** system for querying OCR'd books.
Built with plain Python — no LangChain, no LlamaIndex — as a hands-on demonstration of
the core ideas behind RAG.

## Architecture

```
ask_the_book/
├── config.py               # Central config from environment variables
├── ingestion/
│   ├── loader.py           # Reads & filters JSONL pages
│   ├── chunker.py          # Splits pages into embeddable chunks
│   └── pipeline.py         # Orchestrates load → chunk → embed → store
├── embedding/
│   ├── base.py             # Abstract EmbeddingProvider
│   └── openai_embedder.py  # OpenAI implementation
├── vectorstore/
│   ├── base.py             # Abstract VectorStore
│   └── chroma_store.py     # ChromaDB implementation (local, persistent)
├── retrieval/
│   └── retriever.py        # Embeds a query and fetches top-k chunks
├── generation/
│   ├── base.py             # Abstract LLMProvider
│   └── anthropic_llm.py    # Anthropic (Claude) implementation
├── rag/
│   └── engine.py           # UI-agnostic engine: retrieve → generate
└── cli/
    ├── main.py             # CLI entry point (Click)
    ├── ingest.py           # `askthebook ingest` command
    └── query.py            # `askthebook query` command
```

Each layer depends only on abstractions, making it straightforward to swap providers:

| Layer | Current | Swap with |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` | Cohere, sentence-transformers, … |
| Vector store | ChromaDB (local) | Pinecone, Weaviate, pgvector, … |
| Generation | Anthropic Claude | OpenAI GPT, Mistral, local LLM, … |
| Chunking | One chunk per page text + one per table | Paragraph-level, sliding window, … |

## Quickstart

### 1. Install

```bash
git clone https://github.com/apmariglis/ask-the-book
cd ask-the-book
poetry install
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and fill in all required values
```

### 3. Ingest your book

> **Don't have a JSONL file yet?** Any source that emits the schema described below works. [docpeel](https://github.com/apmariglis/docpeel) is one such tool — it extracts structured JSONL from PDF books.

```bash
askthebook ingest data/book.jsonl
```

This reads the JSONL file, skips pages marked `"skip": true`, and creates one
chunk per page (for prose) and one chunk per table — each embedded and stored
separately in a local ChromaDB database (`.chroma/` by default).
Run this once; it's safe to re-run (upserts).

### 4. Query

```bash
askthebook query "What does chapter 3 say about topic X?"
```

Output:

```
╭─ Question ──────────────────────────────────────────╮
│ What does chapter 3 say about topic X?              │
╰─────────────────────────────────────────────────────╯

╭─ Answer ────────────────────────────────────────────╮
│ Topic X is covered in chapter 3 as follows...       │
│ ──────────────────────────────────────────────────  │
│ ## Key points                                       │
│ - First point from the book                         │
│ - Second point from the book                        │
│ ────────────────────── Caveats ──────────────────── │
│ Only one passage covers this directly.              │
╰─────────────────────────────────────────────────────╯

────────────────── Used in answer ────────────────────
╭─ 1. scan p. 42 · book p. 41 — Chapter 3: Topic X ──╮
│ Verbatim passage from the book...                   │
│ ↳ Backs up the summary claim                        │
╰─────────────────────────────────────────────────────╯

──────────────────── Also retrieved ──────────────────
  scan p. 41 · book p. 40 — Chapter 3: Introduction  (score: 0.71)

  3.2s · 1800 in / 310 out · cost $0.0042
```

## Configuration

All settings can be overridden with environment variables (or in `.env`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** — used for embeddings |
| `ANTHROPIC_API_KEY` | — | **Required** — used for answer generation |
| `CHROMA_PATH` | — | **Required** — directory where ChromaDB stores its data |
| `EMBEDDING_MODEL` | — | **Required** — OpenAI embedding model |
| `GENERATION_MODEL` | — | **Required** — Anthropic model |
| `RETRIEVAL_TOP_K` | — | **Required** — number of chunks retrieved per query |
| `MAX_TOKENS` | — | **Required** — max tokens for the generated answer (4096 recommended) |

## Input data format

The JSONL file must have one JSON object per line. Any extra fields are ignored.

| Field | Type | Required | Description |
|---|---|---|---|
| `page` | int | ✅ | PDF/scan page number |
| `text` | string | | Page body text |
| `skip` | bool | | If `true`, page is excluded from indexing |
| `book_page` | int | | Printed page number inside the book |
| `title` | string | | Page or section title |
| `tables` | array | | List of tables on the page (see below) |

Each object in `tables`:

| Field | Type | Description |
|---|---|---|
| `title` | string | Table title |
| `caption` | string | Table caption |
| `content` | string | Table contents as a Markdown string |

## Running tests

```bash
poetry run pytest
```

## Design decisions

**No frameworks.** The system uses the provider SDKs directly so every step
is visible and easy to reason about.

**Injected dependencies.** The `RAGEngine`, `Retriever`, and `ingest` function
all accept their dependencies as arguments. This makes unit testing with mocks
trivial and provider-swapping a one-line change.

**UI-agnostic engine.** `rag/engine.py` returns a structured `RAGResponse` and
knows nothing about the CLI. Wiring in a web API or GUI is just a matter of
calling `engine.query()` from the new interface.

**Memory-ready.** Every layer that touches generation accepts an optional
`history: list[Message] | None` parameter. Adding conversation memory later
means populating that list — no API changes required.
