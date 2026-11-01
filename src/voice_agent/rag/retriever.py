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

KEYWORD_BONUS = 0.25      # added to the meaning score when every key word of the question appears in the chunk
# Words that say nothing about the topic. "clinic" and "dentist" are here because every chunk is about them.
_FILLER = frozenset("""a an the is are was were be do does did can could would should will you your yours i me my we our us
    it its this that these those what which who whom whose when why how and or but if to of in on at for with about
    from by as there here any some please tell know like want need get have has had also other all things thing provided
    clinic dentist dental sunrise ava""".split())


def _key_words(text: str) -> set[str]:
    """The topic words of a question, lower case, with a trailing plural 's' removed ("services" matches "service")."""
    words = (w.strip(".,!?;:'\"()") for w in text.lower().split())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w and w not in _FILLER}


def _share_of(words: set[str], doc: Document) -> float:
    """How much of the question's topic words appear in this chunk (0 to 1)."""
    if not words:
        return 0.0
    return len(words & _key_words(doc.page_content)) / len(words)


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
        # Meaning (embeddings) plus a small bonus for shared key words: short questions such as "location" or "what is
        # your email" score low on meaning alone, although the section plainly answers them.
        everything = self._store.similarity_search_with_score(question, k=len(self._store.store) or 1)
        words = _key_words(question)
        boosted = [(doc, min(1.0, score + KEYWORD_BONUS * _share_of(words, doc))) for doc, score in everything]
        matches = sorted(boosted, key=lambda m: m[1], reverse=True)[:self.top_k]
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
