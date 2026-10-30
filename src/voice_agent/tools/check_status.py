"""Tool: look up the caller's upcoming appointments."""
from __future__ import annotations

from ..agent import prompts, speech
from .base import Tool, ToolContext, ToolResult, heard_phone


class FindBookings(Tool):
    name = "find_bookings"
    description = "Look up the caller's upcoming appointments. Needs the phone number they booked with."
    parameters = {"type": "object", "properties": {"phone": {"type": "string", "description": "Digits only"}},
                  "required": ["phone"]}

    def run(self, args: dict, agent: dict, ctx: ToolContext) -> ToolResult:
        phone, phone_error = heard_phone(args.get("phone"), agent)
        if phone_error:
            return ToolResult(False, error=phone_error)
        infos = ctx.bookings.upcoming_for_phone(phone)
        if not infos:
            return ToolResult(True, say=prompts.MANAGE_NOT_FOUND, outcome="answered", agent={"phone": phone, "known": {}},
                              turn={"route": "tool_find_bookings"})
        first = infos[0]
        say = prompts.STATUS_FOUND.format(when=ctx.say_on(first.start_at), reason=first.reason,
                                          reference=speech.say_reference(first.reference))
        if len(infos) > 1:
            say += prompts.STATUS_MORE.format(count=speech.number_words(len(infos) - 1))
        known = {str(i.id): {"start": i.start_at.isoformat(), "reason": i.reason, "reference": i.reference}
                 for i in infos}
        # chain: "move it to Tuesday" needs this lookup AND the next step, so the LLM gets to continue. If it has
        # nothing more to do, the caller simply hears `say`.
        data = {"bookings": [{"booking_id": i.id, "when": ctx.say_on(i.start_at), "reason": i.reason} for i in infos],
                "next": "If the caller wanted to move or cancel it, do that now: check_availability for the new "
                        "time, or cancel_appointment. If they only wanted to know about it, reply 'OK'."}
        return ToolResult(True, say=say, outcome="answered", agent={"phone": phone, "known": known},
                          turn={"route": "tool_find_bookings"}, data=data, chain=True)
