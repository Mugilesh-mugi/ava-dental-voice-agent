"""Saves and searches the vectors, with LangChain's in-memory vector store.

It keeps every vector in memory and finds the closest by cosine similarity (a score from -1, opposite meaning, to 1,
same meaning). That is plenty for a few hundred chunks. For thousands, swap it for pgvector (LangChain has a PGVector
store with the same add_documents / similarity_search_with_score methods).

An index can be saved to a folder (index.json and meta.json) and loaded again, which is what scripts/ingest.py and the
app do. meta.json records the embedding model and a fingerprint of the chunks, so an index that no longer matches the
documents is noticed instead of silently giving old answers.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

INDEX_FILE, META_FILE = "index.json", "meta.json"


def build_store(chunks: list[Document], embedder: Embeddings) -> InMemoryVectorStore:
    store = InMemoryVectorStore(embedder)
    store.add_documents(chunks)                     # embeds every chunk
    return store


def chunks_fingerprint(chunks: list[Document]) -> str:
    """Changes whenever any chunk's text changes."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.page_content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def save_index(store: InMemoryVectorStore, directory: Path, embed_model: str, chunks: list[Document]) -> dict:
    """Write the index to `directory`. Returns the metadata that was saved."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"embed_model": embed_model, "chunks": len(store.store), "fingerprint": chunks_fingerprint(chunks),
            "created": datetime.now(timezone.utc).isoformat()}
    store.dump(str(directory / INDEX_FILE))
    (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def read_meta(directory: Path) -> dict:
    """What a saved index says about itself. Raises FileNotFoundError if there is none."""
    return json.loads((Path(directory) / META_FILE).read_text(encoding="utf-8"))


def load_index(directory: Path, embedder: Embeddings) -> tuple[InMemoryVectorStore, dict]:
    """Read an index written by save_index(). Raises FileNotFoundError if there is none."""
    meta = read_meta(directory)
    store = InMemoryVectorStore.load(str(Path(directory) / INDEX_FILE), embedder)
    if len(store.store) != meta["chunks"]:
        raise ValueError("the saved index is incomplete")
    return store, meta
