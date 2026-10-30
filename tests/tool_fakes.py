"""A stand-in for the tool-choosing LLM: it plays back what a test tells it to decide, one decision per call."""
from __future__ import annotations

from fakes import FakeLLM
from voice_agent.llm.client import ToolCallRequest, ToolReply


def says(text: str) -> ToolReply:
    """The LLM answers in words."""
    return ToolReply(content=text)


def calls(name: str, **arguments) -> ToolReply:
    """The LLM asks for a tool."""
    return ToolReply(tool_calls=[ToolCallRequest(f"call_{name}", name, arguments)])


class ScriptedToolLLM(FakeLLM):
    """chat_tools() returns the next scripted decision (or raises it, if it is an exception).
    chat() still answers FAQ questions like FakeLLM, because get_clinic_info uses it."""

    def __init__(self, *script):
        super().__init__()
        self.script = list(script)
        self.requests: list[dict] = []          # what the agent sent each time: messages and tool_choice

    def chat_tools(self, messages, tools, tool_choice="auto"):
        self.requests.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        if not self.script:
            raise AssertionError("the agent asked the LLM for more decisions than the test scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
