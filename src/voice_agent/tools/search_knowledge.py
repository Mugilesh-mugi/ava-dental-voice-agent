"""Tool: answer a question about the clinic from its documents (RAG)."""
from __future__ import annotations

import time

from ..agent import guardrails, prompts
from .base import Tool, ToolContext, ToolResult


class SearchKnowledge(Tool):
    name = "get_clinic_info"
    description = ("Answer a question about the clinic: hours, prices, insurance, payment, parking, location, "
                   "children, cancellation fees, emergency slots, new patients.")
    parameters = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}

    def run(self, args: dict, agent: dict, ctx: ToolContext) -> ToolResult:
        question = str(args.get("question") or "").strip()
        if not question:
            return ToolResult(False, error="Ask the caller what they would like to know.")
        found = ctx.retriever.search(question)
        turn = {"similarity_score": round(found.best_score, 4)}
        if not found.answerable:            # nothing in the documents answers it: say so, without asking the LLM
            return ToolResult(True, say=prompts.FALLBACK, outcome="answered", turn={**turn, "route": "unanswered"})
        texts = [chunk.page_content for chunk, _ in found.matches]
        start = time.perf_counter()
        reply = ctx.llm.chat(prompts.build_llm_messages(question, texts, []))       # may raise LLMError
        turn.update(faq_texts=texts, faq_titles=[chunk.metadata["title"] for chunk, _ in found.matches],
                    llm_ms=(time.perf_counter() - start) * 1000, route="faq_answered")
        if guardrails.looks_like_advice(reply):
            return ToolResult(True, say=prompts.NO_ADVICE, outcome="answered", turn={**turn, "route": "advice_blocked"})
        return ToolResult(True, say=reply, outcome="answered", turn=turn)
