"""The web app: health check, monitoring metrics, mock booking inbox, and the Gradio page mount."""
import pytest
from fastapi.testclient import TestClient

import app as app_module
from tool_fakes import ScriptedToolLLM, calls
from voice_agent.services.mock_clinic import MockConsultationTool


@pytest.fixture
def client(settings, db, service, tmp_path):
    components = app_module.Components(db=db, call_service=service,
                                       consultation_tool=MockConsultationTool(tmp_path / "inbox.jsonl"),
                                       model_names={"stt": "fake", "tts": "fake", "embed": "fake", "llm": "fake"})
    with TestClient(app_module.create_app(settings, components)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] is True


def test_metrics_show_turns(client, service):
    session = service.start_call()
    service.handle_turn(session, text="I can't breathe")
    text = client.get("/metrics").text
    assert 'voice_emergencies_total{tier="severe"}' in text
    assert "voice_stage_seconds_bucket" in text


def test_demo_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "gradio" in r.text.lower()


def test_mock_tool_rejects_incomplete_bookings(tmp_path):
    tool = MockConsultationTool(tmp_path / "inbox.jsonl")
    with pytest.raises(ValueError, match="phone"):
        tool.submit({"booking_id": 1, "caller_name": "A", "reason": "b", "preferred_time": "c"})
    ref = tool.submit({"booking_id": 7, "caller_name": "A", "reason": "b", "preferred_time": "c", "phone": "0"})
    assert ref == "SD-000007" and tool.received()[0]["reference"] == ref


def test_demo_page_handlers_run_a_whole_call(build_service, db):
    """Drive the Gradio button handlers directly, as the browser would."""
    service = build_service(llm=ScriptedToolLLM(calls("get_clinic_info", question="What time do you open on Saturday?")))
    ui = app_module.build_ui(service)
    handlers = {f.name: f.fn for f in ui.fns.values()}

    session, chat, audio, *_ = handlers["start_call"](None, [])
    assert chat[0]["role"] == "assistant" and "Sunrise Dental" in chat[0]["content"]
    assert audio is not None

    session, chat, audio, typed, *_ = handlers["on_text"]("I can't breathe", session, chat)
    assert chat[-2] == {"role": "user", "content": "I can't breathe"}
    assert "999" in chat[-1]["content"] and typed == ""

    session, chat, *_ = handlers["on_audio"]((16_000, [0] * 16_000), session, chat)   # FakeSTT hears an FAQ question
    assert chat[-2]["content"] == "What time do you open on Saturday?"

    session, chat, *_ = handlers["end_call"](session, chat)
    assert session is None and "emergency_999" in chat[-1]["content"]
