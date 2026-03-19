# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest

# Run a single test file
poetry run pytest tests/test_loader.py

# Run a single test by name
poetry run pytest tests/test_loader.py::test_load_pages_returns_non_skipped

# Lint
poetry run ruff check src/ tests/

# Format
poetry run ruff format src/ tests/

# Type check
poetry run mypy src/
```

The CLI is exposed as a `rag` command after `poetry install`:
```bash
askthebook ingest data/book.jsonl
askthebook query "Your question here"
```

## Architecture

This is a framework-free RAG system. All dependencies are injected — no global singletons are used except `config` from `config.py`.

**Data flow:**
1. **Ingest**: `load_pages()` reads JSONL → `Chunker.chunk()` splits → `EmbeddingProvider.embed()` vectorizes → `VectorStore.upsert()` persists
2. **Query**: `Retriever.retrieve()` embeds query + searches store → `LLMProvider.generate()` answers with retrieved context → `RAGEngine` wraps into `RAGResponse`

**Abstractions** (in `base.py` files): `EmbeddingProvider`, `VectorStore`, `Chunker`, `LLMProvider`. Concrete implementations are in the same package. The CLI (`cli/main.py`) is the only place where concrete providers are wired together.

**Key types:**
- `Page` (loader) → `Chunk` (chunker) → stored as vectors in ChromaDB
- `SearchResult` (vectorstore) → `RAGResponse` (engine) with `Source` and `Excerpt` lists
- `Message` TypedDict and `history` parameters exist throughout for future conversation memory support

**Config** (`config.py`): a frozen dataclass loaded from env vars via `python-dotenv`. Import the module-level `config` singleton. Required vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Optional with defaults: `CHROMA_PATH`, `EMBEDDING_MODEL`, `GENERATION_MODEL`, `RETRIEVAL_TOP_K`.

## Input data format

The JSONL source file has one JSON object per line:
```json
{"page": 1, "book_page": 1, "title": "Chapter 1", "text": "...", "tables": [], "skip": false}
```
Pages with `"skip": true` are filtered out by `load_pages()`. The `book_page` and `title` fields are optional. Tables have `title`, `caption`, and `content` (Markdown string) fields.
