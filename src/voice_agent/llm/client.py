"""The only file that talks to an LLM.

Everything else in the project asks for text or for a tool choice through the class here, and never imports a provider
library itself. It is a thin wrapper around LangChain's ChatOpenAI, which speaks the OpenAI chat API: Groq, xAI Grok,
Gemini and a local Ollama (/v1) all do. Swapping the LLM is a change of settings (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL).

    chat(messages)                -> text       (writing an answer from clinic documents)
    chat_tools(messages, tools)   -> ToolReply  (the LLM either answers, or asks us to run a tool)

What the wrapper adds on top of LangChain, because the agent depends on it:
    - every problem (no key, no network, slow, rate limited, bad reply) becomes an LLMError, so the agent handles one
      thing. A rate limit is its own subclass: free tiers allow very little per minute, and "one moment" is a better
      thing to say to a caller than "technical problem".
    - a short rate-limit wait is waited out once (a caller is waiting, so never more than a few seconds)
    - a broken tool call is asked for again, and a model with no "reasoning" setting is retried without it
    - requests and tokens are counted in /metrics
    - text is made plain: the voice reads fancy hyphens and quotes badly
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import openai
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_openai import ChatOpenAI

from .. import monitoring
from ..config import Settings

log = logging.getLogger(__name__)

MIN_OUTPUT_TOKENS = 400        # "thinking" models count their thinking against this, so a small cap gives empty answers
MAX_RATE_LIMIT_WAIT_S = 3.0    # a caller is waiting: longer than this and we give up instead

_PLAIN = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": ", ", "‘": "'",
                        "’": "'", "“": '"', "”": '"', "…": "...", " ": " ", " ": " "})


def plain_text(text: str | None) -> str:
    """One line of plain text. Models like to write non-breaking hyphens and curly quotes, which the
    text-to-speech voice and plain terminals handle badly."""
    joined = " ".join((text or "").translate(_PLAIN).split())
    return joined.replace(" ,", ",")


class LLMError(Exception):
    """The LLM could not give an answer. The message says why."""


class LLMRateLimited(LLMError):
    """The provider says we are over the limit (free tiers allow only a few requests and tokens a minute)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict


@dataclass
class ToolReply:
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


class LLM(Protocol):
    def chat(self, messages: list[dict]) -> str: ...

    def chat_tools(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> ToolReply: ...


def default_reasoning_effort(model: str) -> str | None:
    """Thinking models are slow unless told to think briefly. Other models take no such setting."""
    name = model.lower()
    if "gpt-oss" in name:
        return "low"
    if "qwen3" in name:
        return "none"
    return None


class ChatLLM:
    def __init__(self, model: str, base_url: str, api_key: str | None, max_tokens: int, temperature: float,
                 timeout_s: float, reasoning_effort: str | None = None, http_client=None):
        self.model = model
        self._settings = dict(model=model, base_url=base_url.rstrip("/"), api_key=api_key or "not-needed",
                              temperature=temperature, max_tokens=max(max_tokens, MIN_OUTPUT_TOKENS),
                              timeout=timeout_s, max_retries=0,          # retries are ours: a caller is waiting
                              http_client=http_client)
        self._reasoning = reasoning_effort or default_reasoning_effort(model)
        self._chat = self._build()

    def _build(self) -> ChatOpenAI:
        extra = {"reasoning_effort": self._reasoning} if self._reasoning else None
        return ChatOpenAI(**self._settings, extra_body=extra)

    # ---------- the two public calls ----------
    def chat(self, messages: list[dict]) -> str:
        text = plain_text(self._text(self._invoke(messages)))
        if not text:
            raise LLMError("the model returned an empty reply")
        return text

    def chat_tools(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> ToolReply:
        reply = self._invoke(messages, tools, tool_choice)
        calls = [ToolCallRequest(c.get("id") or f"call_{i + 1}", c["name"], c.get("args") or {})
                 for i, c in enumerate(reply.tool_calls)]
        # A tool call whose arguments were not valid JSON still counts as a call; the tool then says what is missing.
        calls += [ToolCallRequest(c.get("id") or f"call_{len(calls) + i + 1}", c.get("name") or "", {})
                  for i, c in enumerate(reply.invalid_tool_calls)]
        content = plain_text(self._text(reply))
        if not calls and not content:
            raise LLMError("the model returned an empty reply")
        return ToolReply(content, calls)

    # ---------- the request itself ----------
    def _invoke(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto") -> AIMessage:
        prepared = convert_to_messages(messages)
        rate_retries, malformed_retries = 1, 1        # each fix is tried once: a caller is waiting
        while True:
            runnable = self._chat if tools is None else self._chat.bind_tools(tools, tool_choice=tool_choice)
            try:
                reply = runnable.invoke(prepared)
                break
            except openai.APIStatusError as e:
                status = e.status_code
                if status == 429:
                    monitoring.LLM_REQUESTS.labels("rate_limited").inc()
                    wait = _retry_after(e)
                    if rate_retries and wait is not None and wait <= MAX_RATE_LIMIT_WAIT_S:
                        rate_retries -= 1
                        time.sleep(wait)                              # a short wait is worth it; try once more
                        continue
                    raise LLMRateLimited("the provider's rate limit was reached", wait) from e
                if status == 400 and self._reasoning and "reasoning" in str(e):
                    self._reasoning = None                            # this model has no such setting
                    self._chat = self._build()
                    continue
                if status == 400 and _error_kind(e) == "tool_use_failed" and malformed_retries:
                    malformed_retries -= 1                                      # the model wrote a broken tool call: ask again
                    log.warning("llm_tool_call_malformed_retrying")
                    continue
                monitoring.LLM_REQUESTS.labels("error").inc()
                log.warning("llm_http_error", extra={"status": status, "kind": _error_kind(e)})
                raise LLMError(f"HTTP {status}: {str(e)[:200]}") from e
            except Exception as e:                                    # no network, timeout, unreadable reply...
                monitoring.LLM_REQUESTS.labels("error").inc()
                raise LLMError(f"{type(e).__name__}: {str(e)[:200]}") from e
        monitoring.LLM_REQUESTS.labels("ok").inc()
        usage = reply.usage_metadata or {}
        monitoring.LLM_TOKENS.labels("prompt").inc(usage.get("input_tokens") or 0)
        monitoring.LLM_TOKENS.labels("completion").inc(usage.get("output_tokens") or 0)
        return reply

    @staticmethod
    def _text(reply: AIMessage) -> str:
        if isinstance(reply.content, str):
            return reply.content
        return "".join(block.get("text", "") for block in reply.content if isinstance(block, dict))


def _retry_after(error: openai.APIStatusError) -> float | None:
    """How long the provider asks us to wait, in seconds (None if it does not say)."""
    try:
        return float(error.response.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _error_kind(error: openai.APIStatusError) -> str:
    """The provider's short error code (not its message, which can quote the caller's words)."""
    return str(getattr(error, "code", None) or getattr(error, "type", None) or "unknown")


def build_llm(settings: Settings) -> ChatLLM:
    """The LLM chosen in the settings."""
    print(f"  - LLM ({settings.llm_base_url}): {settings.llm_model}", flush=True)
    return ChatLLM(settings.llm_model, settings.llm_base_url, settings.llm_api_key_value, settings.llm_max_tokens,
                   settings.llm_temperature, settings.llm_timeout_s, settings.llm_reasoning_effort)
