"""
Provider wiring: constructs concrete dependencies from config.

All CLI commands import from here so provider instantiation stays
in one place and changes propagate automatically.
"""

from __future__ import annotations

from ask_the_book.config import config
from ask_the_book.embedding.openai_embedder import OpenAIEmbedder
from ask_the_book.generation.anthropic_llm import AnthropicLLM
from ask_the_book.rag.engine import RAGEngine
from ask_the_book.retrieval.retriever import Retriever
from ask_the_book.vectorstore.chroma_store import ChromaStore


def build_store() -> ChromaStore:
    return ChromaStore(path=config.chroma_path, collection_name=config.chroma_collection)


def build_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder(api_key=config.openai_api_key, model=config.embedding_model)


def build_engine() -> RAGEngine:
    retriever = Retriever(
        embedder=build_embedder(),
        store=build_store(),
        top_k=config.retrieval_top_k,
    )
    llm = AnthropicLLM(
        api_key=config.anthropic_api_key,
        model=config.generation_model,
        max_tokens=config.max_tokens,
    )
    return RAGEngine(retriever=retriever, llm=llm)
