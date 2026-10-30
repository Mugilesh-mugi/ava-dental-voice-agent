"""Tool: find free appointment times (a lookup in the calendar, never a guess)."""
from __future__ import annotations

from string import ascii_uppercase

from ..agent import prompts, speech, timeparse
from .base import Tool, ToolContext, ToolResult


class CheckAvailability(Tool):
    name = "check_availability"
    description = ("Find free appointment times. Only when the caller names a day or time, or asks what is free. "
                   "If they merely say they want an appointment, ask their name first instead.")
    parameters = {"type": "object", "properties": {"when": {
        "type": "string", "description": "The caller's words, e.g. 'Saturday morning'. Empty for the soonest."}},
        "required": ["when"]}

    def run(self, args: dict, agent: dict, ctx: ToolContext) -> ToolResult:
        when = str(args.get("when") or "")
        request = timeparse.parse_time_request(when, ctx.today()) if when.strip() else None
        suggestion = ctx.bookings.suggest(request)              # None means "whenever is soonest"
        if not suggestion.slots:
            return ToolResult(True, say=prompts.AVAILABILITY_NONE, outcome="answered")
        ids = ascii_uppercase[:len(suggestion.slots)]
        tail = prompts.SLOTS_TAIL_ONE if len(ids) == 1 else prompts.SLOTS_TAIL_MANY
        template = prompts.SLOTS_NEAREST if suggestion.fallback else prompts.SLOTS_OFFER
        options = speech.say_options(suggestion.slots, ctx.today())
        return ToolResult(True, say=template.format(options=options, tail=tail), outcome="answered",
                          agent={"offered": {i: s.isoformat() for i, s in zip(ids, suggestion.slots)},
                                 "offered_text": {i: ctx.say(s) for i, s in zip(ids, suggestion.slots)}},
                          turn={"route": "tool_check_availability"})
