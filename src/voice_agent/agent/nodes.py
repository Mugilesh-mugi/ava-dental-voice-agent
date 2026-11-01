"""One small function per step of the conversation.

Each node receives the call state and returns only what it changes. The rules (rules.py, guardrails.py) decide WHICH
node runs; the nodes decide WHAT Ava says.

Safety, small talk and unclear input are fixed replies. Everything else is the tool agent:

    caller says something
        -> a change is waiting for a yes?   agent_confirm: the CODE says yes/no, not the LLM
        -> otherwise                        tool_agent:    the LLM answers, or picks a tool;
                                                           the tool runs (tools/) and speaks the facts

The LLM never makes a booking, move or cancellation. It can only stage one (book_appointment, ...), Ava reads it
back, and when the caller says yes the code in _commit makes the change.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
from datetime import datetime

from ..config import Settings
from ..llm.client import LLM, LLMError, LLMRateLimited
from ..rag.retriever import Retriever
from ..services.booking_service import BookingService, SlotTaken
from ..tools import ToolContext, ToolRegistry
from . import guardrails, prompts, rules, speech, timeparse
from .state import CallState, new_agent_state

log = logging.getLogger(__name__)

MAX_LLM_FAILURES = 2          # unavailable twice in a row: stop apologising and pass the caller to a person


def _spoken(text: str) -> str:
    """Plain speech: no markdown marks, one line."""
    return " ".join(re.sub(r"[*#_`]+", " ", text).split())


class AgentNodes:
    def __init__(self, settings: Settings, retriever: Retriever, llm: LLM, bookings: BookingService):
        self._s = settings
        self._llm = llm
        self._bookings = bookings
        self._retriever = retriever
        self._tools = ToolRegistry(ToolContext(settings, bookings, retriever, llm))

    # ==================================================================
    # Router
    # ==================================================================
    def classify(self, state: CallState) -> dict:
        intent = rules.decide_intent(state["caller_text"], agent_pending=bool((state.get("agent") or {}).get("pending")))
        update: dict = {"intent": intent}
        if intent != "unclear":
            update["unclear_count"] = 0      # the counter is for unclear messages IN A ROW
        return update

    # ==================================================================
    # Safety, small talk and unclear input (fixed replies, no LLM)
    # ==================================================================
    def severe_emergency(self, state: CallState) -> dict:
        log.warning("emergency_detected", extra={"tier": "severe"})
        return self._done(state, "emergency_999", prompts.SEVERE_EMERGENCY, outcome="emergency_999", agent=None)

    def urgent_dental(self, state: CallState) -> dict:
        log.warning("emergency_detected", extra={"tier": "urgent"})
        return self._done(state, "urgent_dental", prompts.URGENT_DENTAL, outcome="urgent_dental", agent=None)

    def handoff(self, state: CallState) -> dict:
        return self._done(state, "handed_off", prompts.HANDOFF, outcome="handed_off", agent=None)

    def advice(self, state: CallState) -> dict:
        return self._done(state, "advice_refused", prompts.NO_ADVICE, outcome="answered")

    def greeting(self, state: CallState) -> dict:
        return self._done(state, "greeting", prompts.GREETING_REPLY)

    def thanks(self, state: CallState) -> dict:
        return self._done(state, "thanks", prompts.THANKS_REPLY)

    def unclear(self, state: CallState) -> dict:
        count = state.get("unclear_count", 0) + 1
        if count >= self._s.max_unclear_turns:
            return self._done(state, "unclear_handoff", prompts.UNCLEAR_HANDOFF, outcome="handed_off",
                              unclear_count=0, agent=None)
        return self._done(state, "unclear", prompts.UNCLEAR, unclear_count=count)

    # ==================================================================
    # The tool agent: what the LLM is told each turn
    # ==================================================================
    def _say(self, dt: datetime) -> str:
        return speech.say_slot(dt, self._bookings.today())

    def _say_on(self, dt: datetime) -> str:
        return speech.say_slot_on(dt, self._bookings.today())

    def _agent_context(self, agent: dict, caller_text: str = "") -> str:
        today = self._bookings.today()
        lines = [f"Context: today is {today.strftime('%A')} {today.day} {today.strftime('%B %Y')}."]
        if agent["offered"]:
            lines.append("Times on offer: " + "; ".join(f"{i} = {agent['offered_text'][i]}" for i in agent["offered"]) + ".")
            # The code can tell exactly which time "the second one" or "half past eight" means, so it says so.
            ids = list(agent["offered"])
            picked = timeparse.pick_offered(caller_text, [datetime.fromisoformat(agent["offered"][i]) for i in ids], today)
            if picked is not None:
                lines.append(f"The caller's latest message chooses time {ids[picked]}: use slot_id {ids[picked]}.")
        if agent["known"]:
            lines.append("Bookings found for this caller: " + "; ".join(
                f"booking_id {i} = {self._say(datetime.fromisoformat(b['start']))} ({b['reason']})"
                for i, b in agent["known"].items()) + ".")
        return "\n".join(lines)

    def _agent_messages(self, state: CallState, agent: dict) -> list[dict]:
        history = state.get("history", [])[-2 * self._s.agent_history_turns:]
        system = prompts.AGENT_SYSTEM + "\n" + self._agent_context(agent, state["caller_text"])
        return [{"role": "system", "content": system}, *history, {"role": "user", "content": state["caller_text"]}]

    # ==================================================================
    # The tool agent: the LLM's turn
    # ==================================================================
    def tool_agent(self, state: CallState) -> dict:
        agent = copy.deepcopy(state.get("agent") or new_agent_state())
        agent.setdefault("heard_phones", [])
        said_number = rules.extract_phone(state["caller_text"])      # only numbers the caller says may be used
        if said_number and said_number not in agent["heard_phones"]:
            agent["heard_phones"] = (agent["heard_phones"] + [said_number])[-3:]
        messages = self._agent_messages(state, agent)
        turn: dict = {}
        outcome = None
        choice = "auto"
        chained: tuple[str, str] | None = None                      # (route, words) from a step the LLM may build on
        try:
            for _ in range(self._s.agent_max_tool_rounds):
                start = time.perf_counter()
                reply = self._llm.chat_tools(messages, self._tools.specs(), tool_choice=choice)
                turn["llm_ms"] = turn.get("llm_ms", 0.0) + (time.perf_counter() - start) * 1000
                if not reply.tool_calls and chained:
                    return self._agent_done(state, agent, chained[0], chained[1], turn, outcome)   # nothing more to do
                if not reply.tool_calls:
                    text = _spoken(reply.content)
                    if choice == "auto" and self._answered_without_the_facts(text, state["caller_text"]):
                        # It stated a fact in its own words, or answered a question the clinic documents can answer
                        # without looking them up. Make it use a tool instead.
                        log.warning("llm_answered_without_a_tool")
                        messages += [{"role": "assistant", "content": reply.content},
                                     {"role": "system", "content": "You answered from your own knowledge. Call a tool "
                                      "(get_clinic_info for questions about the clinic) so the facts come from the clinic's system."}]
                        choice = "required"
                        continue
                    if guardrails.looks_like_advice(text):
                        return self._agent_done(state, agent, "advice_blocked", prompts.NO_ADVICE, turn, "answered")
                    return self._agent_done(state, agent, "agent_reply", text, turn, outcome)
                chained = None                                   # it took another step, so what it says next wins
                call = reply.tool_calls[0]                       # one tool at a time, then the caller hears the result
                result = self._tools.run(call.name, call.arguments, agent, state["call_id"])
                agent.update(result.agent)
                turn.update(result.turn)
                outcome = result.outcome or outcome
                if result.staged:
                    agent["pending"] = result.staged
                if result.ok and result.say and not result.chain:
                    route = turn.pop("route", f"tool_{call.name}")
                    return self._agent_done(state, agent, route, result.say, turn, outcome)
                if result.ok and result.say:                        # a lookup the LLM may build on (move / cancel it)
                    chained = (turn.pop("route", f"tool_{call.name}"), result.say)
                # The LLM gets to see the result (a refusal, or a lookup it may build on).
                messages += [
                    {"role": "assistant", "content": None, "tool_calls": [{"id": call.id, "type": "function", "function": {
                        "name": call.name, "arguments": json.dumps(call.arguments)}}]},
                    {"role": "tool", "tool_call_id": call.id, "content": result.for_llm()}]
                choice = "auto" if result.ok else "none"         # after a refusal it must ask the caller, not try something else
            return self._agent_done(state, agent, "agent_reply", prompts.AGENT_CONFUSED, turn, outcome)
        except LLMRateLimited:
            log.warning("llm_rate_limited")
            return self._agent_failed(state, agent, turn, prompts.AGENT_RATE_LIMITED, "llm_rate_limited")
        except LLMError as e:
            log.error("llm_unavailable", extra={"error": str(e)})
            return self._agent_failed(state, agent, turn, prompts.LLM_ERROR, "llm_error")

    def _answered_without_the_facts(self, reply: str, caller_text: str) -> bool:
        """The LLM replied in words, with no tool. That is fine for a question to the caller ("What is your full name?"),
        but not for a statement of fact: a time, price or address, or an answer to something the clinic documents cover."""
        if guardrails.states_unverified_fact(reply):
            return True
        if "?" in reply and not guardrails.refuses(reply):
            return False                                    # a normal question back to the caller
        return self._retriever.search(caller_text).answerable

    def _agent_failed(self, state: CallState, agent: dict, turn: dict, reply: str, route: str) -> dict:
        agent["failures"] = agent.get("failures", 0) + 1
        if agent["failures"] >= MAX_LLM_FAILURES:
            agent["failures"] = 0
            return self._done(state, "unclear_handoff", prompts.UNCLEAR_HANDOFF, outcome="handed_off", agent=None,
                              llm_ms=turn.get("llm_ms", 0.0))
        return self._done(state, route, reply, agent=agent, llm_ms=turn.get("llm_ms", 0.0))

    def _agent_done(self, state: CallState, agent: dict, route: str, reply: str, turn: dict, outcome: str | None) -> dict:
        agent["failures"] = 0
        turn = {k: v for k, v in turn.items() if k != "route"}
        return self._done(state, route, reply, outcome=outcome, agent=agent, **turn)

    # ==================================================================
    # The caller answers "shall I book that?"  (no LLM involved)
    # ==================================================================
    def agent_confirm(self, state: CallState) -> dict:
        agent = copy.deepcopy(state.get("agent") or new_agent_state())
        pending, text = agent.get("pending"), state["caller_text"]
        said_a_time = timeparse.parse_time_request(text, self._bookings.today()) is not None
        if rules.is_yes(text) and not said_a_time:               # "yes, but make it Tuesday" is a new request
            return self._commit(state, agent, pending)
        agent["pending"] = None
        if rules.is_no(text) and not said_a_time:
            return self._done(state, "agent_declined", prompts.AGENT_DECLINED, agent=agent)
        return self.tool_agent({**state, "agent": agent})        # they said something else: carry on from there

    def _commit(self, state: CallState, agent: dict, pending: dict) -> dict:
        agent["pending"] = None
        kind = pending["kind"]
        try:
            if kind == "book":
                info = self._bookings.create(
                    state["call_id"], {k: pending[k] for k in ("caller_name", "reason", "phone", "said_time")},
                    datetime.fromisoformat(pending["start"]))
                agent["known"][str(info.id)] = {"start": info.start_at.isoformat(), "reason": info.reason,
                                                "reference": info.reference}
                agent["phone"] = pending["phone"]
                reply = prompts.BOOKING_CONFIRMED.format(when=self._say(info.start_at),
                                                         reference=speech.say_reference(info.reference))
                route, outcome = "booking_saved", "booking_saved"
            elif kind == "reschedule":
                info = self._bookings.reschedule(pending["booking_id"], datetime.fromisoformat(pending["start"]))
                agent["known"][str(info.id)] = {"start": info.start_at.isoformat(), "reason": info.reason,
                                                "reference": info.reference}
                reply = prompts.CHANGE_DONE.format(when=self._say_on(info.start_at),
                                                   reference=speech.say_reference(info.reference))
                route, outcome = "booking_changed", "booking_changed"
            else:
                info = self._bookings.cancel(pending["booking_id"])
                agent["known"].pop(str(info.id), None)
                reply = prompts.CANCEL_DONE.format(when=self._say_on(info.start_at))
                route, outcome = "booking_cancelled_existing", "booking_cancelled"
        except SlotTaken:                                        # someone else took the time while we were talking
            agent["offered"], agent["offered_text"] = {}, {}
            return self._done(state, "booking_slot_taken", prompts.AGENT_SLOT_GONE, agent=agent)
        except LookupError:
            return self._done(state, "booking_save_error", prompts.AGENT_CHANGE_GONE, outcome="handed_off", agent=None)
        except Exception:
            log.exception("booking_commit_failed", extra={"kind": kind})
            return self._done(state, "booking_save_error", prompts.BOOKING_SAVE_ERROR, outcome="handed_off", agent=None)
        agent["offered"], agent["offered_text"] = {}, {}         # those times are used up or out of date
        log.info("booking_committed", extra={"kind": kind, "booking_id": info.id, "reference": info.reference})
        return self._done(state, route, reply, outcome=outcome, agent=agent)

    # ==================================================================
    @staticmethod
    def _done(state: CallState, route: str, reply: str, outcome: str | None = None, **changes) -> dict:
        return {"route": route, "reply": reply, "outcome": rules.merge_outcome(state.get("outcome"), outcome), **changes}
