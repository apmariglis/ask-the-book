"""
Central configuration loaded from environment variables.

All tuneable knobs live here so nothing is hard-coded elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # --- API keys ---
    openai_api_key: str = field(default_factory=lambda: _require("OPENAI_API_KEY"))
    anthropic_api_key: str = field(
        default_factory=lambda: _require("ANTHROPIC_API_KEY")
    )

    # --- Vector store ---
    chroma_path: str = field(default_factory=lambda: _require("CHROMA_PATH"))
    chroma_collection: str = "book"

    # --- Embedding ---
    embedding_model: str = field(default_factory=lambda: _require("EMBEDDING_MODEL"))

    # --- Generation ---
    generation_model: str = field(default_factory=lambda: _require("GENERATION_MODEL"))
    max_tokens: int = field(
        default_factory=lambda: int(_require("MAX_TOKENS"))
    )

    # --- Retrieval ---
    retrieval_top_k: int = field(
        default_factory=lambda: int(_require("RETRIEVAL_TOP_K"))
    )


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in your keys."
        )
    return value


# Module-level singleton — import this everywhere.
config = Config()
