"""Retrieval over the Chroma collection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import Config, load_config
from .vector_store import embed_query, get_or_create_collection


@dataclass
class RetrievedChunk:
    text: str
    path: str
    score: float
    chunk_index: int

    def format_for_context(self) -> str:
        return f"### {self.path} (chunk {self.chunk_index})\n{self.text}"


def search(query: str, k: int | None = None, cfg: Config | None = None) -> List[RetrievedChunk]:
    cfg = cfg or load_config()
    k = k or cfg.retrieval.top_k
    collection = get_or_create_collection(cfg)
    if collection.count() == 0:
        return []

    qvec = embed_query(cfg, query)
    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    return [
        RetrievedChunk(
            text=d,
            path=m.get("path", "?"),
            score=1.0 - float(dist),  # cosine similarity
            chunk_index=int(m.get("chunk_index", 0)),
        )
        for d, m, dist in zip(docs, metas, dists)
    ]


def build_context(chunks: List[RetrievedChunk], max_chars: int) -> str:
    out, total = [], 0
    for c in chunks:
        piece = c.format_for_context()
        if total + len(piece) > max_chars:
            break
        out.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(out)
