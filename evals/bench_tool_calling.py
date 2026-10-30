"""How well can a local model choose the right tool, with the right arguments, for what a caller says?

    python evals/bench_tool_calling.py llama3.2:3b qwen2.5:3b qwen3:4b          # models in local Ollama
    python evals/bench_tool_calling.py --hosted openai/gpt-oss-120b             # models behind LLM_BASE_URL in .env

Each case gives the model Ava's instructions, six tools, a short conversation so far, and the caller's
latest sentence. We check which tool it calls (or that it calls none and just talks) and whether the
arguments are right. Temperature 0, so a run can be repeated. Nothing here touches the database: the
tools are only described to the model, never executed.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict

import httpx
from dotenv import dotenv_values

OLLAMA = "http://127.0.0.1:11434"
PAUSE_S = 11.0       # seconds between hosted requests. Groq's free tier allows 8,000 tokens a minute and one case is ~1,500

SYSTEM = """You are Ava, the phone receptionist at Sunrise Dental. Today is Friday 30 October 2026.
Callers speak, so keep replies short and plain.

Use the tools for facts. Never make up an appointment time, a price or an opening hour.
- check_availability: when the caller asks what appointment times are free. Always check before offering a time.
- get_clinic_info: for questions about opening hours, prices, insurance, payment, parking, location, children,
  cancellation fees, emergency slots or new patients.
- find_bookings: to look up a booking. You need the caller's phone number first: ask for it if you don't have it.
- create_booking: only when you have the caller's name, the reason, a time that check_availability offered, and a
  phone number, and the caller has said yes to your read-back.
- reschedule_booking and cancel_booking: only after find_bookings, and only after the caller says yes.
If something is missing, ask for it in one short question instead of guessing. Never give medical advice.
Greetings and thanks need no tool."""

TOOLS = [
    {"type": "function", "function": {
        "name": "check_availability", "description": "Find free appointment times.",
        "parameters": {"type": "object", "properties": {"when": {
            "type": "string", "description": "What the caller asked for, in their words, e.g. 'Saturday morning'. "
                                             "Empty for the soonest times."}}, "required": ["when"]}}},
    {"type": "function", "function": {
        "name": "get_clinic_info", "description": "Answer a question about the clinic from its FAQ.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "find_bookings", "description": "Look up the caller's upcoming appointments by phone number.",
        "parameters": {"type": "object", "properties": {"phone": {
            "type": "string", "description": "Digits only, e.g. 07700900123"}}, "required": ["phone"]}}},
    {"type": "function", "function": {
        "name": "create_booking", "description": "Book an appointment at a time that check_availability offered.",
        "parameters": {"type": "object", "properties": {
            "caller_name": {"type": "string"}, "reason": {"type": "string"},
            "start": {"type": "string", "description": "ISO start time exactly as offered, e.g. 2026-11-07T09:00"},
            "phone": {"type": "string", "description": "Digits only"}},
            "required": ["caller_name", "reason", "start", "phone"]}}},
    {"type": "function", "function": {
        "name": "reschedule_booking", "description": "Move a booking to a time that check_availability offered.",
        "parameters": {"type": "object", "properties": {"booking_id": {"type": "integer"}, "new_start": {"type": "string"}},
                       "required": ["booking_id", "new_start"]}}},
    {"type": "function", "function": {
        "name": "cancel_booking", "description": "Cancel a booking.",
        "parameters": {"type": "object", "properties": {"booking_id": {"type": "integer"}}, "required": ["booking_id"]}}},
]


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def result(name, text):
    return {"role": "tool", "tool_name": name, "content": text}


def say(role, text):
    return {"role": role, "content": text}


OFFER = result("check_availability", json.dumps({"free_times": [
    {"start": "2026-11-07T09:00", "spoken": "Saturday at nine"}, {"start": "2026-11-07T09:30", "spoken": "Saturday at half past nine"},
    {"start": "2026-11-07T10:00", "spoken": "Saturday at ten"}]}))
BOOKING_FLOW = [say("user", "I need an appointment"), say("assistant", "What is your full name?"), say("user", "Priya"),
                say("assistant", "And the reason for your visit?"), say("user", "a check-up"),
                say("assistant", "What day and time would suit you?"), say("user", "Saturday morning"),
                call("check_availability", when="Saturday morning"), OFFER,
                say("assistant", "I can offer Saturday at nine, half past nine or ten. Which would you like?")]
ASKED_PHONE = [say("user", "when is my appointment"), say("assistant", "What phone number did you book with?")]
FOUND = [say("user", "I'd like to change my appointment"), say("assistant", "What phone number did you book with?"),
         say("user", "07700 900123"), call("find_bookings", phone="07700900123"),
         result("find_bookings", json.dumps({"bookings": [{"booking_id": 3, "when": "Saturday 7 November at 9:00", "reason": "check-up"}]}))]

# (category, history, what the caller says, tools that are acceptable (None = just talk), argument checks)
# An argument check is {argument: regex that its lower-cased text must match}.
CASES = [
    ("availability", [], "Do you have anything on Saturday morning?", ["check_availability"], {"when": r"saturday"}),
    ("availability", [], "When is your next free appointment?", ["check_availability"], {}),
    ("availability", [], "Can I come in tomorrow afternoon?", ["check_availability"], {"when": r"tomorrow.*afternoon"}),
    ("availability", [], "Are you free next Tuesday at 3pm?", ["check_availability"], {"when": r"tuesday.*3"}),
    ("availability", [], "Is anything available after 4 on Friday?", ["check_availability"], {"when": r"friday"}),
    ("availability", [], "I'd like to see a hygienist next week, what have you got?", ["check_availability"], {"when": r"next week"}),
    ("availability", [], "What slots do you have on the 12th?", ["check_availability"], {"when": r"12"}),
    ("availability", [], "I need something as soon as possible", ["check_availability"], {}),
    ("clinic info", [], "What time do you open on Saturday?", ["get_clinic_info"], {}),
    ("clinic info", [], "How much is a check-up?", ["get_clinic_info"], {}),
    ("clinic info", [], "Do you take Bupa insurance?", ["get_clinic_info"], {}),
    ("clinic info", [], "Where are you based?", ["get_clinic_info"], {}),
    ("clinic info", [], "Is there parking?", ["get_clinic_info"], {}),
    ("clinic info", [], "Do you see children?", ["get_clinic_info"], {}),
    ("clinic info", [], "What's your cancellation policy?", ["get_clinic_info"], {}),
    ("clinic info", [], "Can I pay by card?", ["get_clinic_info"], {}),
    ("clinic info", [], "Do you sell train tickets?", ["get_clinic_info", None], {}),
    ("find booking", ASKED_PHONE, "07700 900123", ["find_bookings"], {"phone": r"^07700900123$"}),
    ("find booking", ASKED_PHONE, "My number is 0 7 7 0 0 9 0 0 5 5 5", ["find_bookings"], {"phone": r"^07700900555$"}),
    ("find booking", [], "When is my appointment? My number is 07700 900123.", ["find_bookings"], {"phone": r"^07700900123$"}),
    ("find booking", [], "What time is my booking? I booked under 07911 123456", ["find_bookings"], {"phone": r"^07911123456$"}),
    ("find booking", [], "When is my appointment?", [None], {}),                      # no phone number yet: ask for it
    ("find booking", [], "Did you get my booking?", [None], {}),
    ("book", BOOKING_FLOW, "The first one", [None], {}),                               # still needs a phone number
    ("book", BOOKING_FLOW + [say("user", "The first one"), say("assistant", "What phone number can we reach you on?"),
                             say("user", "07700 900321"),
                             say("assistant", "Let me check that. Priya, check-up, Saturday at nine, 07700900321. Shall I book that?")],
     "Yes", ["create_booking"], {"start": r"2026-11-07t?\s?09:00", "phone": r"^07700900321$", "caller_name": r"priya"}),
    ("book", BOOKING_FLOW[:-1] + [say("assistant", "I can offer Saturday at nine, half past nine or ten. Which would you like?"),
                                  say("user", "Half past nine"), say("assistant", "What phone number can we reach you on?"),
                                  say("user", "07700 900321"),
                                  say("assistant", "Let me check that. Priya, check-up, Saturday at half past nine. Shall I book that?")],
     "Yes please", ["create_booking"], {"start": r"2026-11-07t?\s?09:30"}),
    ("book", BOOKING_FLOW + [say("user", "The first one"), say("assistant", "What phone number can we reach you on?"),
                             say("user", "07700 900321"),
                             say("assistant", "Let me check that. Priya, check-up, Saturday at nine. Shall I book that?")],
     "No, make it later", [None, "check_availability"], {}),
    ("book", BOOKING_FLOW, "Actually, what about Tuesday?", ["check_availability"], {"when": r"tuesday"}),
    ("book", [], "I want to book an appointment", [None], {}),                          # start by asking the name
    ("book", [say("user", "I want to book an appointment"), say("assistant", "What is your full name?")], "Priya",
     [None], {}),
    ("change", FOUND, "Move it to Tuesday morning", ["check_availability"], {"when": r"tuesday"}),
    ("change", FOUND + [call("check_availability", when="Tuesday morning"), result("check_availability", json.dumps({"free_times": [
        {"start": "2026-11-03T08:00", "spoken": "Tuesday at eight"}, {"start": "2026-11-03T08:30", "spoken": "Tuesday at half past eight"}]})),
        say("assistant", "I can offer Tuesday at eight or half past eight. Which one?"), say("user", "Half past eight"),
        say("assistant", "Shall I move your appointment to Tuesday at half past eight?")],
     "Yes", ["reschedule_booking"], {"booking_id": r"^3$", "new_start": r"2026-11-03t?\s?08:30"}),
    ("change", [], "Can you move my appointment?", [None], {}),
    ("cancel", FOUND[:-2] + [call("find_bookings", phone="07700900123"), FOUND[-1],
                             say("assistant", "I found your appointment on Saturday at nine. Would you like me to cancel it?")],
     "Yes please", ["cancel_booking"], {"booking_id": r"^3$"}),
    ("cancel", FOUND[:-2] + [call("find_bookings", phone="07700900123"), FOUND[-1],
                             say("assistant", "I found your appointment on Saturday at nine. Would you like me to cancel it?")],
     "No, keep it", [None], {}),
    ("cancel", [], "I need to cancel my appointment", [None], {}),
    ("cancel", [], "Cancel my booking please, it's 07700 900123", ["find_bookings"], {"phone": r"^07700900123$"}),
    ("small talk", [], "Hello", [None], {}),
    ("small talk", [], "Thank you, that's all", [None], {}),
    ("small talk", [], "What should I do about my toothache?", [None], {}),             # no medical advice
]


def digits_ok(value) -> str:
    return str(value).lower().strip()


def grade(case, tool, args, said: str, ms: float, n_calls: int) -> dict:
    category, _history, text, ok_tools, checks = case
    right_tool = tool in ok_tools

    def value(key):
        v = digits_ok(args.get(key, ""))
        return v.replace(" ", "") if key in ("phone", "new_start", "start") else v

    right_args = right_tool and all(re.search(p, value(k)) for k, p in checks.items())
    note = "" if right_args else (f"called {tool}({json.dumps(args)[:90]})" if tool else f"said: {said[:70]!r}")
    return {"category": category, "text": text, "tool": tool, "right_tool": right_tool, "right_args": right_args,
            "ms": ms, "note": note, "multiple": n_calls > 1}


def error_result(case, e: Exception) -> dict:
    return {"category": case[0], "text": case[2], "tool": "ERROR", "right_tool": False, "right_args": False,
            "ms": 0, "note": f"{type(e).__name__}: {e}"[:90]}


def parse_args(args) -> dict:
    if isinstance(args, str):
        try:
            return json.loads(args)
        except ValueError:
            return {}
    return args or {}


def run_case(client: httpx.Client, model: str, case) -> dict:
    """One case against a model in local Ollama."""
    _category, history, text, _ok, _checks = case
    messages = [{"role": "system", "content": SYSTEM}, *history, {"role": "user", "content": text}]
    body = {"model": model, "messages": messages, "tools": TOOLS, "stream": False, "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 200}, "think": False}
    start = time.perf_counter()
    try:
        r = client.post("/api/chat", json=body)
        if r.status_code == 400 and "think" in r.text:                  # a model that has no thinking mode
            body.pop("think")
            r = client.post("/api/chat", json=body)
        r.raise_for_status()
        message = r.json()["message"]
    except Exception as e:
        return error_result(case, e)
    ms = (time.perf_counter() - start) * 1000
    calls = message.get("tool_calls") or []
    tool = calls[0]["function"]["name"] if calls else None
    args = parse_args(calls[0]["function"].get("arguments")) if calls else {}
    return grade(case, tool, args, message.get("content") or "", ms, len(calls))


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """The same conversation in the OpenAI style used by Groq, xAI and Gemini: tool calls have ids, and
    their arguments are JSON text."""
    out, pending, n = [], [], 0
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                n += 1
                calls.append({"id": f"call_{n}", "type": "function",
                              "function": {"name": tc["function"]["name"], "arguments": json.dumps(tc["function"]["arguments"])}})
            pending = [c["id"] for c in calls]
            out.append({"role": "assistant", "content": None, "tool_calls": calls})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else "call_0", "content": m["content"]})
        else:
            out.append(m)
    return out


def run_case_hosted(client: httpx.Client, model: str, case) -> dict:
    """One case against a hosted OpenAI-compatible model (the free tiers allow only a few requests a minute)."""
    _category, history, text, _ok, _checks = case
    messages = to_openai_messages([{"role": "system", "content": SYSTEM}, *history, {"role": "user", "content": text}])
    body = {"model": model, "messages": messages, "tools": TOOLS, "temperature": 0, "max_tokens": 400}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"                                # think briefly: a caller is waiting
    elif "qwen" in model:
        body["reasoning_effort"] = "none"                               # no thinking: a caller is waiting
    start = time.perf_counter()
    try:
        for attempt in range(6):
            r = client.post("/chat/completions", json=body)
            if r.status_code == 429:                                     # over the free rate limit: wait and retry
                time.sleep(min(float(r.headers.get("retry-after", 4)) + 0.5, 30))
                continue
            if r.status_code == 400 and "reasoning" in r.text and "reasoning_effort" in body:
                body.pop("reasoning_effort")
                continue
            break
        r.raise_for_status()
        message = r.json()["choices"][0]["message"]
    except Exception as e:
        return error_result(case, e)
    ms = (time.perf_counter() - start) * 1000
    calls = message.get("tool_calls") or []
    tool = calls[0]["function"]["name"] if calls else None
    args = parse_args(calls[0]["function"].get("arguments")) if calls else {}
    time.sleep(PAUSE_S)                                                  # stay under the free tier's tokens-a-minute limit
    return grade(case, tool, args, message.get("content") or "", ms, len(calls))


def main(argv: list[str]) -> None:
    hosted = "--hosted" in argv
    models = [a for a in argv if not a.startswith("--")] or ["llama3.2:3b", "qwen2.5:3b"]
    if hosted:
        env = dotenv_values(".env")
        client = httpx.Client(base_url=env["LLM_BASE_URL"].rstrip("/"), timeout=90,
                              headers={"Authorization": f"Bearer {env['LLM_API_KEY']}"})
    else:
        client = httpx.Client(base_url=OLLAMA, timeout=120)
    summary = {}
    for model in models:
        print(f"\n=== {model} ===", flush=True)
        if hosted:
            results = [run_case_hosted(client, model, c) for c in CASES]
        else:
            client.post("/api/chat", json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                                           "stream": False, "keep_alive": "10m", "options": {"num_predict": 1}}, timeout=300)
            results = [run_case(client, model, c) for c in CASES]
        by_cat = defaultdict(lambda: [0, 0])
        for r in results:
            by_cat[r["category"]][0] += r["right_args"]
            by_cat[r["category"]][1] += 1
        n = len(results)
        times = sorted(r["ms"] for r in results if r["ms"])
        summary[model] = (sum(r["right_tool"] for r in results), sum(r["right_args"] for r in results), n,
                          sum(times) / max(1, len(times)), times[max(0, int(len(times) * 0.9) - 1)] if times else 0)
        for cat, (ok, total) in by_cat.items():
            print(f"  {cat:14} {ok}/{total}")
        for r in results:
            if not r["right_args"]:
                print(f"  X {r['text'][:52]!r:56} {r['note']}")
    print("\n" + "=" * 78)
    print(f"{'model':26} {'right tool':>11} {'right tool + args':>19} {'avg ms':>8} {'slowest 10%':>12}")
    for model, (tool_ok, args_ok, n, avg, p90) in summary.items():
        print(f"{model:26} {tool_ok:>7}/{n:<3} {args_ok:>13}/{n:<3} {avg:>8.0f} {p90:>12.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])
