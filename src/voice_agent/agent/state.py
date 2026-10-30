"""The state that lives for one call.

LangGraph passes this dictionary from node to node. Each node returns only the keys it changes, and LangGraph
merges them in.

Some keys last for the whole call (what the agent remembers, the unclear counter, history, outcome). Others describe
only the current turn and are reset at its start.
"""
from __future__ import annotations

from typing import TypedDict


class CallState(TypedDict, total=False):
    # ---------- Lasts for the whole call ----------
    call_id: str
    agent: dict | None          # what the tool agent remembers (see new_agent_state)
    unclear_count: int          # unclear messages in a row
    history: list[dict]         # recent {"role", "content"} messages, given to the LLM for context
    outcome: str | None         # most important outcome so far (see rules.OUTCOME_PRIORITY)

    # ---------- Describes the current turn only ----------
    caller_text: str
    intent: str                 # which branch the rules chose (see rules.decide_intent)
    route: str                  # what actually happened, e.g. "faq_answered", "tool_check_availability", "booking_saved"
    reply: str                  # what Ava says
    similarity_score: float | None
    faq_texts: list[str]        # document text given to the LLM
    faq_titles: list[str]
    llm_ms: float               # time spent waiting for the LLM itself (0 if not used)


TURN_DEFAULTS: CallState = {
    "caller_text": "",
    "intent": "",
    "route": "",
    "reply": "",
    "similarity_score": None,
    "faq_texts": [],
    "faq_titles": [],
    "llm_ms": 0.0,
}


def new_agent_state() -> dict:
    """What the tool agent remembers during one call (plain data, because it lives in the call state).

    offered / offered_text   the times just offered, by id (A, B, C): the only times that may be booked
    known                    bookings found for this caller, by booking id: the only ones that may be moved or cancelled
    heard_phones             phone numbers the caller actually said: the only ones a tool may use
    pending                  a change waiting for the caller's spoken yes
    """
    return {"offered": {}, "offered_text": {}, "known": {}, "pending": None, "phone": None, "failures": 0,
            "heard_phones": []}


def new_call_state(call_id: str) -> CallState:
    return {"call_id": call_id, "agent": None, "unclear_count": 0, "history": [], "outcome": None, **TURN_DEFAULTS}
