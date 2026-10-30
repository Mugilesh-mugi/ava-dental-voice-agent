"""Tool: cancel a booking. It only STAGES the cancellation; the caller's spoken yes makes it."""
from __future__ import annotations

from datetime import datetime, timedelta

from ..agent import prompts
from .base import Tool, ToolContext, ToolResult, known_booking


class CancelAppointment(Tool):
    name = "cancel_appointment"
    description = "Cancel a found booking. The caller is asked to confirm first."
    parameters = {"type": "object", "properties": {"booking_id": {"type": "integer"}}, "required": ["booking_id"]}

    def run(self, args: dict, agent: dict, ctx: ToolContext) -> ToolResult:
        booking_id, booking_error = known_booking(args, agent)
        if booking_error:
            return ToolResult(False, error=booking_error)
        start = datetime.fromisoformat(agent["known"][booking_id]["start"])
        say = prompts.CANCEL_ASK.format(when=ctx.say_on(start))
        if start - ctx.bookings.now() < timedelta(hours=24):        # the clinic charges for late cancellations
            say += prompts.CANCEL_LATE_NOTE
        return ToolResult(True, say=say, staged={"kind": "cancel", "booking_id": int(booking_id)},
                          turn={"route": "tool_cancel_appointment"})
