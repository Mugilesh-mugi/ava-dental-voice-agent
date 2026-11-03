"""The Gradio voice page: one browser session = one call.

Press "Start call" to hear Ava's greeting, then record your voice. Ava's reply is shown and spoken.
There is also a text box, as a backup for a live demo if the microphone fails.
"""
from __future__ import annotations

import gradio as gr

from ..agent import prompts
from ..services.call_service import CallService


def render(service: CallService) -> None:
    """Draw the call page inside the current Gradio Blocks."""
    session_state = gr.State(None)       # the CallSession for this browser tab

    with gr.Row():
        start_btn = gr.Button("Start call", variant="primary")
        end_btn = gr.Button("End call", variant="stop", interactive=False)
    chat = gr.Chatbot(label="Conversation", height=420)
    mic = gr.Audio(sources=["microphone"], type="numpy", label="Your voice (record, then stop)")
    with gr.Row():
        typed = gr.Textbox(label="Or type a message", placeholder="e.g. What time do you open on Saturday?", scale=4)
        send = gr.Button("Send", scale=1)
    ava_voice = gr.Audio(label="Ava's voice", autoplay=True, interactive=False)

    def start_call(session, history):
        if session is not None and not session.ended:
            service.end_call(session)
        session = service.start_call()
        text, audio, _ = service.greeting()
        return (session, [{"role": "assistant", "content": text}], audio,
                gr.update(interactive=False), gr.update(interactive=True))

    def take_turn(session, history, audio=None, text=None):
        history = list(history or [])
        if session is None or session.ended:     # caller spoke without pressing Start: start a call silently
            session = service.start_call()
            history = []
        result = service.handle_turn(session, audio=audio, text=text)
        history += [{"role": "user", "content": result.caller_text or "(nothing heard)"},
                    {"role": "assistant", "content": result.reply}]
        return session, history, result.audio

    def on_audio(audio, session, history):
        if audio is None:                       # the recording was cleared, not finished
            return session, history, None, None, gr.update(), gr.update()
        session, history, voice = take_turn(session, history, audio=audio)
        return session, history, voice, None, gr.update(interactive=False), gr.update(interactive=True)

    def on_text(text, session, history):
        if not (text or "").strip():
            return session, history, None, "", gr.update(), gr.update()
        session, history, voice = take_turn(session, history, text=text)
        return session, history, voice, "", gr.update(interactive=False), gr.update(interactive=True)

    def end_call(session, history):
        history = list(history or [])
        if session is not None and not session.ended:
            outcome = service.end_call(session)
            history.append({"role": "assistant", "content": f"{prompts.CALL_ENDED} (Outcome: {outcome})"})
        return None, history, None, gr.update(interactive=True), gr.update(interactive=False)

    turn_outputs = [session_state, chat, ava_voice]
    start_btn.click(start_call, [session_state, chat], turn_outputs + [start_btn, end_btn])
    mic.stop_recording(on_audio, [mic, session_state, chat], turn_outputs + [mic, start_btn, end_btn])
    send.click(on_text, [typed, session_state, chat], turn_outputs + [typed, start_btn, end_btn])
    typed.submit(on_text, [typed, session_state, chat], turn_outputs + [typed, start_btn, end_btn])
    end_btn.click(end_call, [session_state, chat], turn_outputs + [start_btn, end_btn])
