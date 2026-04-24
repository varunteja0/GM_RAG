"""Ingestion pipeline: walk source folders, chunk, embed, store in Chroma.

Idempotent: deterministic chunk IDs mean re-running safely upserts.
Run:  python -m src.ingest
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, List

import pathspec
from rich.console import Console
from tqdm import tqdm

from .chunking import split_text
from .config import Config, load_config
from .loaders import load_file
from .vector_store import embed_texts, get_or_create_collection

console = Console()


def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    gi = root / ".gitignore"
    if gi.exists():
        return pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text().splitlines())
    return None


def iter_files(source: Path, cfg: Config) -> Iterable[Path]:
    spec = _load_gitignore(source)
    allowed = {e.lower() for e in cfg.allowed_extensions}
    ignore_dirs = set(cfg.ignore_dirs)
    max_bytes = cfg.max_file_size_mb * 1024 * 1024

    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs and not d.startswith(".")]
        dp = Path(dirpath)
        for name in filenames:
            p = dp / name
            if p.suffix.lower() not in allowed:
                continue
            rel = p.relative_to(source)
            if spec and spec.match_file(str(rel)):
                continue
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            yield p


def _chunk_id(source_root: Path, file_path: Path, idx: int, content: str) -> str:
    rel = file_path.relative_to(source_root).as_posix()
    h = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{rel}::{idx}::{h}"


def ingest(cfg: Config | None = None, batch_size: int = 64) -> None:
    cfg = cfg or load_config()
    collection = get_or_create_collection(cfg)

    total_chunks = 0
    for src in cfg.sources:
        root = Path(src).expanduser().resolve()
        if not root.exists():
            console.print(f"[yellow]Skipping missing source:[/] {root}")
            continue
        console.print(f"[bold cyan]Indexing:[/] {root}")

        files = list(iter_files(root, cfg))
        console.print(f"  {len(files)} files to consider")

        ids_buf: List[str] = []
        docs_buf: List[str] = []
        metas_buf: List[dict] = []

        def flush():
            nonlocal ids_buf, docs_buf, metas_buf, total_chunks
            if not docs_buf:
                return
            vectors = embed_texts(cfg, docs_buf)
            collection.upsert(
                ids=ids_buf,
                embeddings=vectors,
                documents=docs_buf,
                metadatas=metas_buf,
            )
            total_chunks += len(docs_buf)
            ids_buf, docs_buf, metas_buf = [], [], []

        for f in tqdm(files, desc="files"):
            text = load_file(f)
            if not text.strip():
                continue
            chunks = split_text(
                text,
                f.suffix,
                cfg.chunking.chunk_size,
                cfg.chunking.chunk_overlap,
            )
            rel = f.relative_to(root).as_posix()
            for i, ch in enumerate(chunks):
                ids_buf.append(_chunk_id(root, f, i, ch))
                docs_buf.append(ch)
                metas_buf.append({
                    "source_root": str(root),
                    "path": rel,
                    "abs_path": str(f),
                    "ext": f.suffix.lower(),
                    "chunk_index": i,
                })
                if len(docs_buf) >= batch_size:
                    flush()
        flush()

    console.print(f"[bold green]Done.[/] {total_chunks} chunks in collection '{cfg.collection}'.")


if __name__ == "__main__":
    ingest()
