"""The shape every tool must follow, and the checks that stop the LLM doing something unsafe.

The LLM decides WHICH tool to call and fills in the arguments. The code decides whether the call is allowed:

- Times: book/reschedule take a slot id (A, B, C) that check_availability returned in this call. The LLM cannot
  invent a time, and cannot book one that was never offered.
- Bookings: reschedule/cancel take a booking id that find_bookings returned (the caller proved who they are by the
  phone number) or that was made in this call.
- Phone numbers: only a number the caller really said during this call is accepted.
- Writes need a spoken yes: book/reschedule/cancel only STAGE the change and read it back. The change is made by
  the code (agent_confirm) when the caller says yes, never by the LLM.
- Facts are spoken by the code: a successful tool returns the exact words for Ava to say, so the LLM cannot garble
  a time, a price or a reference number.

A tool that is refused returns an error the LLM can read ("the id must be one of A, B, C"), so it can ask the caller
for what is missing instead of guessing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime

from ..agent import rules, speech
from ..config import Settings
from ..llm.client import LLM
from ..rag.retriever import Retriever
from ..services.booking_service import BookingService


@dataclass
class ToolContext:
    """Everything a tool may use. Tools get it as an argument, so they can be tested with fakes."""

    settings: Settings
    bookings: BookingService
    retriever: Retriever
    llm: LLM

    def today(self) -> date:
        return self.bookings.today()

    def say(self, dt: datetime) -> str:
        """A time as words: "Saturday the seventh of November at nine o'clock in the morning"."""
        return speech.say_slot(dt, self.today())

    def say_on(self, dt: datetime) -> str:
        """The same, with the right small word in front: "on Saturday ..." but "tomorrow at ..."."""
        return speech.say_slot_on(dt, self.today())


@dataclass
class ToolResult:
    ok: bool
    say: str | None = None                  # the exact words for Ava to say (only set when ok)
    error: str | None = None                # what went wrong, for the LLM to read
    staged: dict | None = None              # a change waiting for the caller's yes
    agent: dict = field(default_factory=dict)       # updates to the agent's memory
    turn: dict = field(default_factory=dict)        # updates to this turn's state (route, similarity score, ...)
    outcome: str | None = None
    data: dict = field(default_factory=dict)        # facts for the LLM to carry on with (e.g. booking ids)
    chain: bool = False                             # the LLM may take another step; if it has none, say `say`

    def for_llm(self) -> str:
        """What the LLM is shown when it needs to carry on (a refusal, or a step that leads to another)."""
        return json.dumps({"ok": self.ok, **({"error": self.error} if self.error else {}),
                           **({"result": self.say} if self.say else {}), **self.data})


class Tool:
    """Subclass this: set name, description and parameters (a JSON schema), and write run()."""

    name: str = ""
    description: str = ""
    parameters: dict = {}

    def spec(self) -> dict:
        """The tool as the LLM is told about it. Keep descriptions short: they are sent on every turn."""
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}

    def run(self, args: dict, agent: dict, ctx: ToolContext) -> ToolResult:
        raise NotImplementedError


# ---------------------------------------------------------------------- checks shared by several tools
def heard_phone(raw, agent: dict) -> tuple[str | None, str | None]:
    """A phone number the caller really said during this call. The LLM must not make one up or guess."""
    phone = rules.extract_phone(str(raw or ""))
    if phone is None:
        return None, "That is not a full phone number. Ask the caller to say it, digit by digit."
    if phone not in agent.get("heard_phones", []):
        return None, "The caller has not said that number. Ask for their phone number and use exactly what they say."
    return phone, None


def offered_slot(args: dict, agent: dict) -> tuple[str | None, str | None]:
    """The ISO time for a slot id that was offered in this call, or (None, an error for the LLM)."""
    slot_id = str(args.get("slot_id") or "").strip().upper()
    if slot_id not in agent.get("offered", {}):
        options = ", ".join(agent.get("offered", {})) or "none yet"
        return None, (f"slot_id must be one of the times on offer ({options}). If no times are on offer, call "
                      "check_availability first, then let the caller choose.")
    return agent["offered"][slot_id], None


def known_booking(args: dict, agent: dict) -> tuple[str | None, str | None]:
    """A booking id that was found for this caller, or (None, an error for the LLM)."""
    booking_id = str(args.get("booking_id") or "").strip()
    if booking_id not in agent.get("known", {}):
        return None, ("booking_id must be one found for this caller. Ask for the phone number the booking was "
                      "made with and call find_bookings first.")
    return booking_id, None


def mask_args(args: dict) -> str:
    """The arguments as stored in the audit trail: no full phone numbers, names or reasons."""
    masked = {}
    for key, value in (args or {}).items():
        text = str(value)
        if key == "phone":
            masked[key] = "…" + "".join(c for c in text if c.isdigit())[-4:]
        elif key == "caller_name":
            masked[key] = text[:1].upper() + "."
        elif key == "reason":
            masked[key] = "…"
        else:
            masked[key] = text[:80]
    return json.dumps(masked, ensure_ascii=False)
