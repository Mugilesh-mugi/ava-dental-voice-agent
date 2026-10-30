"""Tool: book an appointment. It only STAGES the booking and reads it back; the caller's spoken yes saves it."""
from __future__ import annotations

from datetime import datetime

from ..agent import prompts, rules, speech
from .base import Tool, ToolContext, ToolResult, heard_phone, offered_slot


class BookAppointment(Tool):
    name = "book_appointment"
    description = ("Book one of the times on offer. Needs the full name, reason and phone number. "
                   "The caller is asked to confirm before it is saved.")
    parameters = {"type": "object", "properties": {
        "caller_name": {"type": "string"}, "reason": {"type": "string"},
        "slot_id": {"type": "string", "description": "A, B or C from the times on offer"},
        "phone": {"type": "string", "description": "Digits only"}},
        "required": ["caller_name", "reason", "slot_id", "phone"]}

    def run(self, args: dict, agent: dict, ctx: ToolContext) -> ToolResult:
        name = rules.clean_name(str(args.get("caller_name") or ""))
        reason = rules.clean_answer(str(args.get("reason") or ""))
        phone, phone_error = heard_phone(args.get("phone"), agent)
        problems = []
        if len(name) < 2:
            problems.append("the caller's full name is missing")
        if not reason:
            problems.append("the reason for the visit is missing")
        if phone_error:
            problems.append(phone_error)
        start_iso, slot_error = offered_slot(args, agent)
        if slot_error:
            problems.append(slot_error)
        if problems:
            return ToolResult(False, error="Not booked: " + "; ".join(problems) + ". Ask the caller for what is missing.")
        start = datetime.fromisoformat(start_iso)
        say = prompts.BOOKING_READBACK.format(name=name, reason=reason, when=ctx.say_on(start),
                                              phone=speech.say_digits(phone))
        staged = {"kind": "book", "caller_name": name, "reason": reason, "phone": phone, "start": start_iso,
                  "said_time": next((t for i, t in agent["offered_text"].items() if agent["offered"][i] == start_iso), "")}
        return ToolResult(True, say=say, staged=staged, agent={"phone": phone}, turn={"route": "tool_book_appointment"})
