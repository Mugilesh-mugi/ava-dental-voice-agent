"""The Gradio voice page: one browser session = one call.

Press "Start call" to hear Ava's greeting, then record your voice. Ava's reply is
shown and spoken, and the panel on the right shows how the turn was handled
(route, FAQ similarity score, and the time each stage took).
There is also a text box, as a backup for a live demo if the microphone fails.
"""
from __future__ import annotations

import gradio as gr

from ..agent import prompts
from ..services.call_service import CallService, CallSession, TurnResult

WELCOME_INFO = "Press **Start call**, then record your question. Ava's reply will play automatically."


def _info(session: CallSession | None, result: TurnResult | None = None) -> str:
    if session is None:
        return WELCOME_INFO
    lines = [f"**Call ID:** `{session.call_id[:8]}`  ", f"**Call outcome so far:** {session.state.get('outcome') or '-'}"]
    if result is not None:
        score = "-" if result.similarity_score is None else f"{result.similarity_score:.3f}"
        total = result.timings_detail.get("total_ms", result.stt_ms + result.llm_ms + result.tts_ms)
        lines += [
            "",
            f"**Route:** `{result.route}`  ",
            f"**FAQ similarity score:** {score}  ",
            "",
            "| Stage | Time |",
            "|---|---|",
            f"| Ears (speech-to-text) | {result.stt_ms:,.0f} ms |",
            f"| Brain (rules + search + LLM) | {result.llm_ms:,.0f} ms |",
            f"| &nbsp;&nbsp;of which LLM | {result.timings_detail.get('llm_call_ms', 0):,.0f} ms |",
            f"| Mouth (text-to-speech) | {result.tts_ms:,.0f} ms |",
            f"| **Total** | **{total:,.0f} ms** |",
        ]
    return "\n".join(lines)


def render(service: CallService) -> None:
    """Draw the demo page inside the current Gradio Blocks/Tab."""
    session_state = gr.State(None)       # the CallSession for this browser tab

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(label="Conversation", height=420)
            mic = gr.Audio(sources=["microphone"], type="numpy", label="Your voice (record, then stop)")
            with gr.Row():
                typed = gr.Textbox(label="Or type a message", placeholder="e.g. What time do you open on Saturday?",
                                   scale=4)
                send = gr.Button("Send", scale=1)
            ava_voice = gr.Audio(label="Ava's voice", autoplay=True, interactive=False)
        with gr.Column(scale=2):
            with gr.Row():
                start_btn = gr.Button("Start call", variant="primary")
                end_btn = gr.Button("End call", variant="stop", interactive=False)
            info = gr.Markdown(WELCOME_INFO)

    def start_call(session, history):
        if session is not None and not session.ended:
            service.end_call(session)
        session = service.start_call()
        text, audio, _ = service.greeting()
        return (session, [{"role": "assistant", "content": text}], audio, _info(session),
                gr.update(interactive=False), gr.update(interactive=True))

    def take_turn(session, history, audio=None, text=None):
        history = list(history or [])
        if session is None or session.ended:     # caller spoke without pressing Start: start a call silently
            session = service.start_call()
            history = []
        result = service.handle_turn(session, audio=audio, text=text)
        history += [{"role": "user", "content": result.caller_text or "(nothing heard)"},
                    {"role": "assistant", "content": result.reply}]
        return session, history, result.audio, _info(session, result)

    def on_audio(audio, session, history):
        if audio is None:                       # the recording was cleared, not finished
            return session, history, None, _info(session), None, gr.update(), gr.update()
        session, history, voice, panel = take_turn(session, history, audio=audio)
        return session, history, voice, panel, None, gr.update(interactive=False), gr.update(interactive=True)

    def on_text(text, session, history):
        if not (text or "").strip():
            return session, history, None, _info(session), "", gr.update(), gr.update()
        session, history, voice, panel = take_turn(session, history, text=text)
        return session, history, voice, panel, "", gr.update(interactive=False), gr.update(interactive=True)

    def end_call(session, history):
        history = list(history or [])
        if session is not None and not session.ended:
            outcome = service.end_call(session)
            history.append({"role": "assistant", "content": f"{prompts.CALL_ENDED} (Outcome: {outcome})"})
        return None, history, None, WELCOME_INFO, gr.update(interactive=True), gr.update(interactive=False)

    turn_outputs = [session_state, chat, ava_voice, info]
    start_btn.click(start_call, [session_state, chat], turn_outputs + [start_btn, end_btn])
    mic.stop_recording(on_audio, [mic, session_state, chat], turn_outputs + [mic, start_btn, end_btn])
    send.click(on_text, [typed, session_state, chat], turn_outputs + [typed, start_btn, end_btn])
    typed.submit(on_text, [typed, session_state, chat], turn_outputs + [typed, start_btn, end_btn])
    end_btn.click(end_call, [session_state, chat], turn_outputs + [start_btn, end_btn])
