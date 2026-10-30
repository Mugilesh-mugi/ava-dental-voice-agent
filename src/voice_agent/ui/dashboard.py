"""A simple dashboard tab: how calls went, how fast each stage is, and what Ava couldn't answer."""
from __future__ import annotations

import gradio as gr

from ..services.call_service import CallService


def _fmt_time(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "-"


def render(service: CallService) -> None:
    refresh = gr.Button("Refresh", variant="primary")
    summary = gr.Markdown()
    with gr.Row():
        outcomes = gr.Dataframe(headers=["Outcome", "Calls"], label="Calls by outcome", interactive=False)
        stages = gr.Dataframe(headers=["Stage", "Average time (ms)"], label="Average time per stage",
                              interactive=False)
    appointments = gr.Dataframe(headers=["When", "Reference", "Name", "Reason", "Phone", "Sent to clinic"],
                                label="Upcoming appointments (soonest first)", interactive=False, wrap=True)
    tool_calls = gr.Dataframe(headers=["When", "Call", "Tool", "Result", "ms", "Arguments (masked)", "Said / error"],
                              label="Tools the LLM called (newest first). 'refused' = the safety checks said no",
                              interactive=False, wrap=True)
    unanswered = gr.Dataframe(headers=["When", "Question", "Best score"], label="Unanswered questions (newest first)",
                              interactive=False, wrap=True)
    gr.Markdown("### Conversation viewer")
    call_picker = gr.Dropdown(label="Choose a call", choices=[], interactive=True)
    conversation = gr.Dataframe(headers=["Who", "Text", "Route", "Score", "STT ms", "Brain ms", "TTS ms"],
                                interactive=False, wrap=True)

    def load():
        st = service.dashboard_stats()
        text = (f"**Total calls:** {st['total_calls']} &nbsp;|&nbsp; "
                f"**Handoff rate:** {st['handoff_rate']:.0%} &nbsp;|&nbsp; "
                f"**Emergencies caught:** {st['emergencies_caught']} "
                f"({st['severe_caught']} severe, {st['urgent_caught']} urgent) &nbsp;|&nbsp; "
                f"**Appointments booked:** {st['bookings']}")
        outcome_rows = sorted(st["outcomes"].items(), key=lambda kv: -kv[1]) or [["(no calls yet)", 0]]
        stage_rows = [["Speech-to-text", round(st["avg_stt_ms"])], ["Brain (rules + search + LLM)", round(st["avg_llm_ms"])],
                      ["Text-to-speech", round(st["avg_tts_ms"])]]
        unanswered_rows = [[_fmt_time(q.created_at), q.question, round(q.best_score, 3)]
                           for q in service.unanswered_questions()]
        calls = [(f"{_fmt_time(c.started_at)}  ·  {c.outcome or 'in progress'}  ·  {c.id[:8]}", c.id)
                 for c in service.recent_calls()]
        return (text, [list(r) for r in outcome_rows], stage_rows, service.upcoming_appointments(), service.recent_tool_calls(),
                unanswered_rows,
                gr.update(choices=calls))

    def show_call(call_id):
        if not call_id:
            return []
        return [[m.role, m.text, m.route, "" if m.similarity_score is None else round(m.similarity_score, 3),
                 round(m.stt_ms), round(m.llm_ms), round(m.tts_ms)] for m in service.conversation(call_id)]

    refresh.click(load, None, [summary, outcomes, stages, appointments, tool_calls, unanswered, call_picker])
    call_picker.change(show_call, call_picker, conversation)
