"""Tool: move a booking to another time. It only STAGES the move; the caller's spoken yes makes it."""
from __future__ import annotations

from datetime import datetime

from ..agent import prompts
from .base import Tool, ToolContext, ToolResult, known_booking, offered_slot


class RescheduleAppointment(Tool):
    name = "reschedule_appointment"
    description = "Move a found booking to one of the times on offer. The caller is asked to confirm first."
    parameters = {"type": "object", "properties": {"booking_id": {"type": "integer"}, "slot_id": {"type": "string"}},
                  "required": ["booking_id", "slot_id"]}

    def run(self, args: dict, agent: dict, ctx: ToolContext) -> ToolResult:
        booking_id, booking_error = known_booking(args, agent)
        start_iso, slot_error = offered_slot(args, agent)
        if booking_error or slot_error:
            return ToolResult(False, error=" ".join(e for e in (booking_error, slot_error) if e))
        old = datetime.fromisoformat(agent["known"][booking_id]["start"])
        say = prompts.CHANGE_CONFIRM.format(old=ctx.say(old), new=ctx.say(datetime.fromisoformat(start_iso)))
        return ToolResult(True, say=say, staged={"kind": "reschedule", "booking_id": int(booking_id), "start": start_iso},
                          turn={"route": "tool_reschedule_appointment"})
