"""Vector store wrapper around Chroma with Ollama embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

from .config import Config


def get_embeddings(cfg: Config) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=cfg.ollama.embed_model,
        base_url=cfg.ollama.base_url,
    )


def get_chroma_client(cfg: Config) -> chromadb.PersistentClient:
    cfg.storage_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(cfg.storage_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=False),
    )


def get_or_create_collection(cfg: Config):
    client = get_chroma_client(cfg)
    # cosine is the standard for text embeddings
    return client.get_or_create_collection(
        name=cfg.collection,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(cfg: Config, texts: List[str]) -> List[List[float]]:
    embedder = get_embeddings(cfg)
    return embedder.embed_documents(texts)


def embed_query(cfg: Config, text: str) -> List[float]:
    embedder = get_embeddings(cfg)
    return embedder.embed_query(text)
