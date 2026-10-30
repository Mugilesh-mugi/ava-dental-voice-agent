"""Finds the chunks that answer a question: search, then a similarity limit, then (optionally) reranking.

1. The question is embedded and compared with every chunk (cosine similarity).
2. If even the best chunk scores below the threshold, the documents have no answer: nothing is returned, and the agent
   says "I don't have that information" without asking the LLM at all, so it cannot guess.
3. Otherwise the best `top_k` chunks at or above the threshold are returned. An optional reranker can reorder them
   (for example with a cross-encoder model) before the LLM sees them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from ..config import Settings
from .store import build_store, chunks_fingerprint, load_index

log = logging.getLogger(__name__)

Match = tuple[Document, float]                       # a chunk and its similarity score
Reranker = Callable[[str, list[Match]], list[Match]]


@dataclass(frozen=True)
class SearchResult:
    matches: list[Match]     # best first; only filled when the question can be answered
    best_score: float

    @property
    def answerable(self) -> bool:
        return bool(self.matches)


class Retriever:
    def __init__(self, chunks: list[Document], embedder: Embeddings, threshold: float, top_k: int,
                 store: InMemoryVectorStore | None = None, reranker: Reranker | None = None):
        self.threshold = threshold
        self.top_k = top_k
        self._reranker = reranker
        self._store = store if store is not None else build_store(chunks, embedder)      # no saved index: embed now
        log.info("knowledge_loaded", extra={"chunks": len(self._store.store)})

    def search(self, question: str) -> SearchResult:
        matches = self._store.similarity_search_with_score(question, k=self.top_k)
        best = matches[0][1] if matches else 0.0
        if best < self.threshold:
            return SearchResult([], best)
        kept = [m for m in matches if m[1] >= self.threshold]
        if self._reranker is not None:
            kept = self._reranker(question, kept)
        return SearchResult(kept, best)


def build_retriever(settings: Settings, embedder: Embeddings, chunks: list[Document]) -> Retriever:
    """The retriever for the app. Uses the index made by scripts/ingest.py when it matches the documents and the
    embedding model; otherwise embeds the chunks now (slower to start, same answers)."""
    store = None
    if settings.use_prebuilt_index:
        try:
            saved, meta = load_index(settings.processed_dir, embedder)
            model = getattr(embedder, "model_name", type(embedder).__name__)
            if meta.get("embed_model") == model and meta.get("fingerprint") == chunks_fingerprint(chunks):
                store = saved
                log.info("rag_index_loaded", extra={"chunks": meta["chunks"], "index_created": meta["created"]})
            else:
                log.warning("rag_index_out_of_date: the documents or the embedding model changed. "
                            "Run: python scripts/ingest.py")
        except FileNotFoundError:
            log.info("rag_index_missing: embedding the documents now. Run python scripts/ingest.py to save the index.")
        except Exception:
            log.exception("rag_index_unreadable")
    return Retriever(chunks, embedder, settings.match_threshold, settings.top_k, store=store)
