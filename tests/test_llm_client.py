"""The LLM client (LangChain's ChatOpenAI behind our wrapper): requests, tool calls, retries, rate limits, token counts.
Every test talks to a fake server (httpx.MockTransport), so no LLM request is made."""
import json

import httpx
import pytest

from voice_agent import monitoring
from voice_agent.llm.client import ChatLLM, LLMError, LLMRateLimited, build_llm, default_reasoning_effort, plain_text
from voice_agent.tools import default_tools

BASE = "https://example.test/v1"
SPECS = [tool.spec() for tool in default_tools()]       # the real tool definitions to send


def make_llm(handler, model="openai/gpt-oss-20b", api_key="secret-key", **kwargs):
    return ChatLLM(model, BASE, api_key, 120, 0.2, 5, http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def ok(content=None, tool_calls=None, usage=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                     "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                                     "usage": usage or {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def tool_call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def failure(status, code="x", headers=None):
    return httpx.Response(status, headers=headers or {}, json={"error": {"message": f"error {status}", "code": code}})


# ---------- Requests ----------
def test_chat_sends_the_right_request():
    seen = {}

    def handler(request):
        seen.update(url=str(request.url), auth=request.headers["authorization"], body=json.loads(request.content))
        return ok("  Hello   there ")

    assert make_llm(handler).chat([{"role": "user", "content": "hi"}]) == "Hello there"
    assert seen["url"] == BASE + "/chat/completions" and seen["auth"] == "Bearer secret-key"
    body = seen["body"]
    assert body["model"] == "openai/gpt-oss-20b" and body["temperature"] == 0.2 and body["messages"][0]["content"] == "hi"
    assert body["max_completion_tokens"] >= 400      # "thinking" models spend part of this on thinking
    assert body["reasoning_effort"] == "low" and "tools" not in body


def test_the_conversation_is_sent_in_the_openai_format():
    seen = {}
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "find it"},
                {"role": "assistant", "content": None, "tool_calls": [tool_call("find_bookings", json.dumps({"phone": "1"}), "c1")]},
                {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'}]
    make_llm(lambda r: seen.update(b=json.loads(r.content)) or ok("fine")).chat(messages)
    sent = seen["b"]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[2]["tool_calls"][0]["id"] == "c1" and sent[2]["tool_calls"][0]["function"]["name"] == "find_bookings"
    assert json.loads(sent[2]["tool_calls"][0]["function"]["arguments"]) == {"phone": "1"} and sent[3]["tool_call_id"] == "c1"


@pytest.mark.parametrize("choice", ["auto", "none", "required"])
def test_chat_tools_sends_the_tools_and_the_choice(choice):
    seen = {}
    make_llm(lambda r: seen.update(b=json.loads(r.content)) or ok("hello")).chat_tools([{"role": "user", "content": "x"}], SPECS, choice)
    assert [t["function"]["name"] for t in seen["b"]["tools"]] == [s["function"]["name"] for s in SPECS]
    assert seen["b"]["tool_choice"] == choice


def test_tool_calls_are_read_back():
    reply = make_llm(lambda r: ok(tool_calls=[tool_call("check_availability", json.dumps({"when": "Saturday morning"}))])).chat_tools([], SPECS)
    assert [(c.id, c.name, c.arguments) for c in reply.tool_calls] == [("call_1", "check_availability", {"when": "Saturday morning"})]


def test_text_answer_and_unreadable_arguments():
    assert make_llm(lambda r: ok("What is your name?")).chat_tools([], SPECS).content == "What is your name?"
    broken = make_llm(lambda r: ok(tool_calls=[tool_call("find_bookings", "{not json")])).chat_tools([], SPECS)
    assert [(c.name, c.arguments) for c in broken.tool_calls] == [("find_bookings", {})]       # the tool then says what is missing


def test_an_empty_answer_is_an_error():
    with pytest.raises(LLMError, match="empty"):
        make_llm(lambda r: ok("   ")).chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMError, match="empty"):
        make_llm(lambda r: ok(None)).chat_tools([{"role": "user", "content": "x"}], SPECS)


# ---------- Rate limits and retries ----------
def test_a_short_rate_limit_is_waited_out_once(monkeypatch):
    waits, attempts = [], []
    monkeypatch.setattr("time.sleep", waits.append)

    def handler(request):
        attempts.append(1)
        return failure(429, headers={"retry-after": "1"}) if len(attempts) == 1 else ok("fine")

    assert make_llm(handler).chat([{"role": "user", "content": "x"}]) == "fine" and waits == [1.0] and len(attempts) == 2


def test_a_long_rate_limit_is_not_waited_out(monkeypatch):
    waits = []
    monkeypatch.setattr("time.sleep", waits.append)
    with pytest.raises(LLMRateLimited) as caught:
        make_llm(lambda r: failure(429, headers={"retry-after": "30"})).chat([{"role": "user", "content": "x"}])
    assert caught.value.retry_after == 30.0 and waits == []         # a caller is waiting: don't sleep for 30 seconds


def test_a_rate_limit_that_persists_is_reported_after_one_retry(monkeypatch):
    waits = []
    monkeypatch.setattr("time.sleep", waits.append)
    with pytest.raises(LLMRateLimited):
        make_llm(lambda r: failure(429, headers={"retry-after": "1"})).chat([{"role": "user", "content": "x"}])
    assert waits == [1.0]                                           # one retry, then it gives up


def test_a_model_without_a_reasoning_setting_is_retried_without_it():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if "reasoning_effort" in bodies[-1]:
            return httpx.Response(400, json={"error": {"message": "reasoning_effort is not supported by this model"}})
        return ok("fine")

    llm = make_llm(handler)
    assert llm.chat([{"role": "user", "content": "x"}]) == "fine"
    assert "reasoning_effort" in bodies[0] and "reasoning_effort" not in bodies[1]
    llm.chat([{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in bodies[2]                       # remembered: no wasted request next time


def test_a_broken_tool_call_is_asked_for_again():
    attempts = []

    def handler(request):
        attempts.append(1)
        return failure(400, "tool_use_failed") if len(attempts) == 1 else ok(tool_calls=[tool_call("check_availability", json.dumps({"when": "today"}))])

    reply = make_llm(handler).chat_tools([{"role": "user", "content": "x"}], SPECS)
    assert reply.tool_calls[0].name == "check_availability" and len(attempts) == 2


def test_a_tool_call_that_keeps_breaking_is_an_error():
    with pytest.raises(LLMError, match="400"):
        make_llm(lambda r: failure(400, "tool_use_failed")).chat_tools([{"role": "user", "content": "x"}], SPECS)


# ---------- Other failures ----------
@pytest.mark.parametrize("status", [401, 500])
def test_http_problems_become_llm_errors(status):
    with pytest.raises(LLMError, match=str(status)):
        make_llm(lambda r: failure(status)).chat([{"role": "user", "content": "x"}])


def test_an_unreadable_reply_becomes_an_llm_error():
    with pytest.raises(LLMError):
        make_llm(lambda r: httpx.Response(200, text="<html>not json</html>")).chat([{"role": "user", "content": "x"}])


def test_network_problems_become_llm_errors():
    def handler(request):
        raise httpx.ConnectError("no network")

    with pytest.raises(LLMError, match="Connect"):
        make_llm(handler).chat([{"role": "user", "content": "x"}])


# ---------- Metrics ----------
def test_tokens_and_requests_are_counted():
    prompt, completion = monitoring.LLM_TOKENS.labels("prompt"), monitoring.LLM_TOKENS.labels("completion")
    requests = monitoring.LLM_REQUESTS.labels("ok")
    before = (prompt._value.get(), completion._value.get(), requests._value.get())
    make_llm(lambda r: ok("hi", usage={"prompt_tokens": 1500, "completion_tokens": 40, "total_tokens": 1540})).chat([{"role": "user", "content": "x"}])
    assert (prompt._value.get() - before[0], completion._value.get() - before[1], requests._value.get() - before[2]) == (1500, 40, 1)


def test_rate_limits_and_errors_are_counted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    limited, errors = monitoring.LLM_REQUESTS.labels("rate_limited"), monitoring.LLM_REQUESTS.labels("error")
    before = (limited._value.get(), errors._value.get())
    with pytest.raises(LLMRateLimited):
        make_llm(lambda r: failure(429, headers={"retry-after": "60"})).chat([{"role": "user", "content": "x"}])
    with pytest.raises(LLMError):
        make_llm(lambda r: failure(500)).chat([{"role": "user", "content": "x"}])
    assert (limited._value.get() - before[0], errors._value.get() - before[1]) == (1, 1)


# ---------- Settings ----------
@pytest.mark.parametrize("model, effort", [("openai/gpt-oss-20b", "low"), ("openai/gpt-oss-120b", "low"),
                                           ("qwen/qwen3.8-27b", "none"), ("llama-3.3-70b-versatile", None)])
def test_thinking_models_are_told_to_think_briefly(model, effort):
    assert default_reasoning_effort(model) == effort


def test_an_explicit_reasoning_setting_wins():
    seen = {}
    make_llm(lambda r: seen.update(b=json.loads(r.content)) or ok("hi"), reasoning_effort="high").chat([{"role": "user", "content": "x"}])
    assert seen["b"]["reasoning_effort"] == "high"


def test_a_model_without_a_default_gets_no_reasoning_setting():
    seen = {}
    make_llm(lambda r: seen.update(b=json.loads(r.content)) or ok("hi"), model="llama-3.3-70b-versatile").chat([{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in seen["b"]


def test_a_local_server_needs_no_key():
    seen = {}
    make_llm(lambda r: seen.update(h=dict(r.headers)) or ok("hi"), api_key=None).chat([{"role": "user", "content": "x"}])
    assert seen["h"]["authorization"] == "Bearer not-needed"


def test_the_llm_is_built_from_the_settings(settings):
    llm = build_llm(settings.model_copy(update={"llm_model": "openai/gpt-oss-20b", "llm_base_url": "https://x.test/v1/"}))
    assert isinstance(llm, ChatLLM) and llm.model == "openai/gpt-oss-20b"


def test_the_text_is_cleaned_for_the_voice():
    hyphen, dash, apostrophe, open_quote, close_quote, nbsp, ellipsis = (
        chr(c) for c in (0x2011, 0x2014, 0x2019, 0x201C, 0x201D, 0xA0, 0x2026))
    messy = f"A check{hyphen}up {dash} it{apostrophe}s {open_quote}free{close_quote}{nbsp}today{ellipsis}"
    assert plain_text(messy) == """A check-up, it's "free" today..."""
    assert plain_text(None) == "" and plain_text("  two lines ") == "two lines"
    assert make_llm(lambda r: ok(f"Your check{hyphen}up is booked.")).chat([{"role": "user", "content": "x"}]) == "Your check-up is booked."
