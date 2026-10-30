"""Turns text into an "embedding": a list of numbers that captures its meaning.

Texts with similar meaning get similar numbers, so "what time do you open?" lands close to the Opening hours section
even though they share few words. This is LangChain's HuggingFaceEmbeddings around a sentence-transformers model that
runs on this computer (free, no key). Vectors are scaled to length 1, so cosine similarity is a plain dot product.
"""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


def build_embedder(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"},
                                 encode_kwargs={"normalize_embeddings": True})
