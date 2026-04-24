"""Language-aware chunking for source code and documents."""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


EXT_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".cs": Language.CSHARP,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    ".html": Language.HTML,
    ".sol": Language.SOL,
}


def get_splitter_for(ext: str, chunk_size: int, chunk_overlap: int):
    lang = EXT_TO_LANGUAGE.get(ext.lower())
    if lang is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def split_text(text: str, ext: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text.strip():
        return []
    splitter = get_splitter_for(ext, chunk_size, chunk_overlap)
    return splitter.split_text(text)
