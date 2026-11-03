"""Start Ava, the Sunrise Dental voice receptionist.

    python app.py

Then open http://127.0.0.1:7860 in your browser.

What runs:
    /            the Gradio page (the voice call)
    /health      is the app working? (database reachable, models loaded)
    /metrics     live numbers for monitoring tools (Prometheus format)
    /api/mock-consultation/bookings   what the pretend clinic booking system has received

This file only wires things together. The work is in src/voice_agent: llm/ (the LLM), rag/ (the knowledge search),
tools/ (the actions), agent/ (the conversation and its safety rules), memory/ (the database), api/ (the endpoints).
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # harmless Windows warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from voice_agent import __version__  # noqa: E402
from voice_agent.agent.graph import VoiceAgent  # noqa: E402
from voice_agent.agent.nodes import AgentNodes  # noqa: E402
from voice_agent.api.routes import build_router  # noqa: E402
from voice_agent.config import Settings, get_settings, warn_about_missing_settings  # noqa: E402
from voice_agent.llm.client import build_llm  # noqa: E402
from voice_agent.logging_setup import setup_logging  # noqa: E402
from voice_agent.memory.database import Database  # noqa: E402
from voice_agent.rag import load_chunks  # noqa: E402
from voice_agent.rag.embedder import build_embedder  # noqa: E402
from voice_agent.rag.retriever import build_retriever  # noqa: E402
from voice_agent.services.booking_service import BookingService  # noqa: E402
from voice_agent.services.call_service import CallService  # noqa: E402
from voice_agent.services.mock_clinic import MockConsultationTool  # noqa: E402
from voice_agent.services.scheduling import Scheduler  # noqa: E402
from voice_agent.ui import demo_page  # noqa: E402
from voice_agent.voice.stt import ModelLoadError, build_speech_to_text, model_load_error  # noqa: E402
from voice_agent.voice.tts import TextToSpeech  # noqa: E402

log = logging.getLogger("app")


@dataclass
class Components:
    db: Database
    call_service: CallService
    consultation_tool: MockConsultationTool
    model_names: dict[str, str]


def build_components(settings: Settings) -> Components:
    """Load every model and connect everything. Exits with a clear message if a model fails to load."""
    print("Loading models...", flush=True)
    try:
        print(f"  - Knowledge search embeddings: {settings.embed_model}", flush=True)
        try:
            embedder = build_embedder(settings.embed_model)
        except Exception as e:
            raise model_load_error("embedding", settings.embed_model, e) from e
        retriever = build_retriever(settings, embedder, load_chunks(settings.docs_dir))
        print(f"  - Speech-to-text ({settings.stt_engine}): {settings.stt_model}", flush=True)
        stt = build_speech_to_text(settings.stt_engine, settings.stt_model, settings.stt_prompt or None)
        print(f"  - Text-to-speech: {settings.tts_model}", flush=True)
        tts = TextToSpeech(settings.tts_model, settings.tts_voice, settings.tts_speed, settings.tts_dir)
    except ModelLoadError as e:
        print(f"\nERROR: {e}\n", file=sys.stderr, flush=True)
        sys.exit(1)
    llm = build_llm(settings)

    db = Database(settings.database_url)
    db.create_tables()
    clinic = MockConsultationTool()
    scheduler = Scheduler(settings.clinic_timezone, settings.slot_minutes, settings.booking_horizon_days,
                          settings.min_notice_minutes)
    bookings = BookingService(db, clinic, scheduler)
    agent = VoiceAgent(AgentNodes(settings, retriever, llm, bookings))
    service = CallService(settings, agent, db, stt=stt, tts=tts)
    service.close_stale_calls()
    names = {"stt": settings.stt_model, "tts": settings.tts_model, "embed": settings.embed_model,
             "llm": settings.llm_model}
    return Components(db, service, clinic, names)


def build_ui(service: CallService) -> gr.Blocks:
    with gr.Blocks(title="Sunrise Dental - Ava") as ui:
        gr.Markdown("## Sunrise Dental - Ava, the AI voice receptionist")
        demo_page.render(service)
    return ui


def create_app(settings: Settings, components: Components) -> FastAPI:
    app = FastAPI(title="Sunrise Dental voice agent", version=__version__)
    app.include_router(build_router(settings, components.db, components.consultation_tool, components.model_names))
    password = settings.dashboard_password_value
    return gr.mount_gradio_app(app, build_ui(components.call_service), path="/",
                               auth=(settings.dashboard_user, password) if password else None)


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir, settings.log_json)
    warn_about_missing_settings(settings)
    components = build_components(settings)
    app = create_app(settings, components)
    print(f"Ready. Open http://{settings.host}:{settings.port} in your browser.", flush=True)
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
