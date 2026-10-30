"""The list of tools the agent may use, and the one place a tool is run.

Running a tool through the registry (never directly) means every call is counted in /metrics, written to the audit
trail (tool_calls table, with names and numbers masked), and cannot crash the call: a tool that blows up becomes
an error the LLM is told about. To add a tool: write a class in this package and add it to default_tools().
"""
from __future__ import annotations

import logging
import time

from .. import monitoring
from ..llm.client import LLMError
from .base import Tool, ToolContext, ToolResult, mask_args
from .cancel_booking import CancelAppointment
from .check_availability import CheckAvailability
from .check_status import FindBookings
from .create_booking import BookAppointment
from .reschedule_booking import RescheduleAppointment
from .search_knowledge import SearchKnowledge

log = logging.getLogger(__name__)


def default_tools() -> list[Tool]:
    return [CheckAvailability(), SearchKnowledge(), FindBookings(), BookAppointment(), RescheduleAppointment(),
            CancelAppointment()]


class ToolRegistry:
    def __init__(self, ctx: ToolContext, tools: list[Tool] | None = None):
        self._ctx = ctx
        self._tools = {tool.name: tool for tool in (tools if tools is not None else default_tools())}

    @property
    def names(self) -> set[str]:
        return set(self._tools)

    def specs(self) -> list[dict]:
        """The tools as the LLM is told about them."""
        return [tool.spec() for tool in self._tools.values()]

    def run(self, name: str, args: dict, agent: dict, call_id: str) -> ToolResult:
        start = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            result = ToolResult(False, error=f"There is no tool called {name}.")
        else:
            try:
                result = tool.run(args or {}, agent, self._ctx)
            except LLMError:
                raise                                       # the LLM itself is down: the agent handles that
            except Exception:
                log.exception("tool_failed", extra={"tool": name})
                result = ToolResult(False, error="That tool failed. Apologise and offer to pass the caller to the team.")
        ms = (time.perf_counter() - start) * 1000
        monitoring.TOOL_CALLS.labels(name, "ok" if result.ok else "refused").inc()
        log.info("tool_called", extra={"tool": name, "ok": result.ok, "ms": round(ms), "error": result.error})
        self._ctx.bookings.record_tool_call(call_id, name, mask_args(args), result.ok, ms,
                                            (result.say or result.error or "")[:200])
        return result
