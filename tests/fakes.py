"""Stand-ins for the real models, so tests are fast and need no token or downloads."""
from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

from voice_agent.llm.client import LLMError

_STOPWORDS = set("a an and are at be by can do does for from how i in is it me my of on or the to we what "
                 "when where which who will with you your our there any".split())


class KeywordEmbedder(Embeddings):
    """Bag-of-words embedder: texts sharing words get high similarity. Good enough to test the logic."""

    dim = 4096

    def _vector(self, text: str) -> list[float]:
        out = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in _STOPWORDS:
                out[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return (out / max(float(np.linalg.norm(out)), 1e-9)).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeLLM:
    """Answers with the first sentence of the FAQ text it was given (so key facts show up)."""

    def __init__(self, reply: str | None = None):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        content = messages[0]["content"]
        if "Validate the user's answer" in content:
            return "YES"
        if self.reply is not None:
            return self.reply
        info = content.split("Clinic information:\n", 1)[1]
        return info.split(": ", 1)[1].split(". ")[0] + "."


class FailingLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages: list[dict]) -> str:
        self.calls += 1
        raise LLMError("ConnectionError: network down")


class FakeSTT:
    def __init__(self, text: str = "What time do you open on Saturday?", fail: bool = False):
        self.text = text
        self.fail = fail

    def transcribe(self, sample_rate, audio) -> str:
        if self.fail:
            raise RuntimeError("whisper crashed")
        return self.text


class FakeTTS:
    def speak(self, text: str):
        return 16_000, np.zeros(1600, dtype=np.int16)


class RecordingTool:
    """A consultation tool that remembers what it was sent (or fails, if asked)."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.sent: list[dict] = []

    def submit(self, booking: dict) -> str:
        if self.fail:
            raise ConnectionError("booking system down")
        self.sent.append(booking)
        return f"SD-{booking['booking_id']:06d}"
