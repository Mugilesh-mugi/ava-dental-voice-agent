"""Shared test setup: a full agent wired with fakes and a throwaway SQLite database."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fakes import FakeSTT, FakeTTS, KeywordEmbedder, RecordingTool
from voice_agent.agent.graph import VoiceAgent
from voice_agent.agent.nodes import AgentNodes
from voice_agent.config import Settings
from voice_agent.memory.database import Database
from voice_agent.rag import load_chunks
from voice_agent.rag.retriever import Retriever
from voice_agent.services.booking_service import BookingService
from voice_agent.services.call_service import CallService
from tool_fakes import ScriptedToolLLM
from voice_agent.services.scheduling import Scheduler

# "Now" in every test: a Monday morning. London is on GMT in November, so local time equals UTC.
NOW = datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, match_threshold=0.25,
                    database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}", log_dir=tmp_path / "logs")


@pytest.fixture
def db(settings) -> Database:
    database = Database(settings.database_url)
    database.create_tables()
    yield database
    database.engine.dispose()       # release the file so Windows can delete tmp_path


@pytest.fixture
def faq_store(settings) -> Retriever:
    return Retriever(load_chunks(settings.docs_dir), KeywordEmbedder(), settings.match_threshold, settings.top_k)


@pytest.fixture
def tool() -> RecordingTool:
    return RecordingTool()


class Clock:
    """A clock tests can move: `clock.now = ...`."""

    def __init__(self):
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


def make_scheduler(settings: Settings) -> Scheduler:
    return Scheduler(settings.clinic_timezone, settings.slot_minutes, settings.booking_horizon_days,
                     settings.min_notice_minutes)


@pytest.fixture
def bookings(settings, db, tool, clock) -> BookingService:
    return BookingService(db, tool, make_scheduler(settings), clock=clock)


@pytest.fixture
def build_service(settings, db, faq_store, bookings, clock):
    """A CallService with the tool agent. Give it the LLM (usually a ScriptedToolLLM that plays back what the test
    wants the LLM to decide), a different speech-to-text, a different clinic system, or settings to change."""

    def _build(llm=None, stt=None, consultation_tool=None, **settings_overrides):
        call_settings = settings.model_copy(update=settings_overrides) if settings_overrides else settings
        service_bookings = (BookingService(db, consultation_tool, make_scheduler(settings), clock=clock)
                            if consultation_tool else bookings)
        agent = VoiceAgent(AgentNodes(call_settings, faq_store, llm or ScriptedToolLLM(), service_bookings))
        return CallService(call_settings, agent, db, stt=stt or FakeSTT(), tts=FakeTTS(), clock=clock)

    return _build


@pytest.fixture
def service(build_service) -> CallService:
    return build_service()
