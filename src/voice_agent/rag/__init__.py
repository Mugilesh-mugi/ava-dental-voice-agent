"""RAG (Retrieval-Augmented Generation): find the trusted text that answers a question, then let the LLM write a reply
from ONLY that text. The LLM never answers from its own memory. Built on LangChain's documents, splitter, embeddings
and vector store.

    loader.py     reads the documents in data/raw
    chunker.py    splits them into small pieces (one per "##" heading)
    embedder.py   turns text into vectors (lists of numbers that capture meaning)
    store.py      saves and searches the vectors
    retriever.py  search + similarity limit + (optional) reranking

scripts/ingest.py runs loader, chunker, embedder and store once and saves the result in data/processed, so the app does
not have to embed the documents every time it starts.
"""
from pathlib import Path

from langchain_core.documents import Document

from .chunker import split_into_chunks
from .loader import load_documents


def load_chunks(path: Path) -> list[Document]:
    """The documents in `path` (a file or a folder), split into chunks."""
    return split_into_chunks(load_documents(path))


__all__ = ["load_chunks", "load_documents", "split_into_chunks"]
