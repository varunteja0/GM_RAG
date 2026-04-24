"""Config loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class OllamaCfg:
    base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"


@dataclass
class ChunkingCfg:
    chunk_size: int = 1200
    chunk_overlap: int = 150


@dataclass
class RetrievalCfg:
    top_k: int = 6
    max_context_chars: int = 12000


@dataclass
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    sources: List[str] = field(default_factory=list)
    storage_dir: str = "./storage/chroma"
    collection: str = "company_repo"
    ollama: OllamaCfg = field(default_factory=OllamaCfg)
    chunking: ChunkingCfg = field(default_factory=ChunkingCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    ignore_dirs: List[str] = field(default_factory=list)
    allowed_extensions: List[str] = field(default_factory=list)
    max_file_size_mb: float = 2.0
    server: ServerCfg = field(default_factory=ServerCfg)

    @property
    def storage_path(self) -> Path:
        p = Path(self.storage_dir)
        if not p.is_absolute():
            p = ROOT / p
        return p


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        sources=raw.get("sources", []),
        storage_dir=raw.get("storage_dir", "./storage/chroma"),
        collection=raw.get("collection", "company_repo"),
        ollama=OllamaCfg(**(raw.get("ollama") or {})),
        chunking=ChunkingCfg(**(raw.get("chunking") or {})),
        retrieval=RetrievalCfg(**(raw.get("retrieval") or {})),
        ignore_dirs=raw.get("ignore_dirs", []),
        allowed_extensions=raw.get("allowed_extensions", []),
        max_file_size_mb=float(raw.get("max_file_size_mb", 2)),
        server=ServerCfg(**(raw.get("server") or {})),
    )
