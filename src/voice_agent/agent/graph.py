"""The conversation flow as a LangGraph graph.

LangGraph lets us draw the flow as boxes (nodes) and arrows (edges). Every turn runs through the graph once:

    START -> classify --severe--------> severe_emergency --> END      (fixed reply, no LLM)
                      --urgent--------> urgent_dental ----> END       (fixed reply, no LLM)
                      --handoff-------> handoff ----------> END       (fixed reply, no LLM)
                      --agent_confirm-> agent_confirm ----> END       (the caller answers "shall I book that?")
                      --thanks--------> thanks -----------> END
                      --greeting------> greeting ---------> END
                      --advice--------> advice -----------> END
                      --unclear-------> unclear ----------> END
                      --agent---------> tool_agent -------> END       (the LLM chooses a tool, or answers)

The ORDER of checks inside `classify` (rules.decide_intent) is the safety design: emergencies are always checked
first, before the LLM is ever involved.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import AgentNodes
from .state import TURN_DEFAULTS, CallState

INTENT_TO_NODE = {
    "severe": "severe_emergency",
    "urgent": "urgent_dental",
    "handoff": "handoff",
    "agent_confirm": "agent_confirm",
    "thanks": "thanks",
    "greeting": "greeting",
    "advice": "advice",
    "unclear": "unclear",
    "agent": "tool_agent",
}


def build_graph(nodes: AgentNodes):
    graph = StateGraph(CallState)
    graph.add_node("classify", nodes.classify)
    for name in INTENT_TO_NODE.values():
        graph.add_node(name, getattr(nodes, name))
        graph.add_edge(name, END)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", lambda s: INTENT_TO_NODE[s["intent"]], list(INTENT_TO_NODE.values()))
    return graph.compile()


class VoiceAgent:
    """Runs one caller message through the graph and returns the new call state."""

    def __init__(self, nodes: AgentNodes):
        self._graph = build_graph(nodes)

    def run_turn(self, state: CallState, caller_text: str) -> CallState:
        turn_input: CallState = {**state, **TURN_DEFAULTS, "caller_text": caller_text or ""}
        return self._graph.invoke(turn_input)

    def draw(self) -> str:
        """The graph as a Mermaid diagram (paste into https://mermaid.live to see it)."""
        return self._graph.get_graph().draw_mermaid()
