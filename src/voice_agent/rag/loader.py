"""Reads the original documents (the clinic FAQ and any other notes) from data/raw."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

SUPPORTED = (".md", ".txt")


def load_documents(path: Path) -> list[Document]:
    """Every .md and .txt file in a folder (in name order), or the one file given. `metadata["source"]` is the file
    name, so an answer can be traced back to where it came from."""
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    else:
        raise FileNotFoundError(f"No documents found at {path}. Put the clinic's .md or .txt files in data/raw.")
    if not files:
        raise ValueError(f"No .md or .txt documents in {path}")
    return [Document(page_content=f.read_text(encoding="utf-8"), metadata={"source": f.name}) for f in files]
