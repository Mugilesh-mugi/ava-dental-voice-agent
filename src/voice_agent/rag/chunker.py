"""Splits documents into small pieces ("chunks"), so a question can be matched to the one paragraph that answers it.

Each "## Title" heading in a document starts a new chunk. Keep each section about one topic and write it as plain
sentences: the whole chunk is what the LLM is given as its source. A chunk's text is "Title: the sentences", and its
metadata holds the title and the source file.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

_SPLITTER = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "title")], strip_headers=True)


def split_into_chunks(documents: list[Document]) -> list[Document]:
    chunks = []
    for document in documents:
        for piece in _SPLITTER.split_text(document.page_content):
            title, body = piece.metadata.get("title"), " ".join(piece.page_content.split())
            if title and body:                      # (text before the first "##" heading has no title: it is skipped)
                chunks.append(Document(page_content=f"{title}: {body}",
                                       metadata={"title": title, "source": document.metadata["source"]}))
    if not chunks:
        names = ", ".join(d.metadata["source"] for d in documents)
        raise ValueError(f"No '## ' sections found in {names}")
    return chunks
