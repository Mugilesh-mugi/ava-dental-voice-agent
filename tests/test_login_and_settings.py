"""The login on the web page and the bookings API, and the new settings."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

import app as app_module
from voice_agent.services.mock_clinic import MockConsultationTool
from voice_agent.config import Settings, warn_about_missing_settings

BOOKING = {"booking_id": 1, "caller_name": "Priya", "reason": "check-up", "preferred_time": "Saturday", "phone": "07700900321"}


def make_client(settings, db, service, tmp_path, password=None):
    tool = MockConsultationTool(tmp_path / "inbox.jsonl")
    tool.submit(BOOKING)
    configured = settings.model_copy(update={"dashboard_password": SecretStr(password) if password else None})
    components = app_module.Components(db=db, call_service=service, consultation_tool=tool,
                                       model_names={"stt": "fake", "tts": "fake", "embed": "fake", "llm": "fake"})
    return TestClient(app_module.create_app(configured, components))


# ---------- Login ----------
def test_the_bookings_api_needs_the_password_when_one_is_set(settings, db, service, tmp_path):
    with make_client(settings, db, service, tmp_path, password="s3cret") as client:
        refused = client.get("/api/mock-consultation/bookings")
        assert refused.status_code == 401 and refused.headers["www-authenticate"] == "Basic"
        assert client.get("/api/mock-consultation/bookings", auth=("admin", "wrong")).status_code == 401
        assert client.get("/api/mock-consultation/bookings", auth=("someone", "s3cret")).status_code == 401
        allowed = client.get("/api/mock-consultation/bookings", auth=("admin", "s3cret"))
        assert allowed.status_code == 200 and allowed.json()[0]["caller_name"] == "Priya"


def test_without_a_password_the_api_stays_open(settings, db, service, tmp_path):
    with make_client(settings, db, service, tmp_path) as client:
        assert client.get("/api/mock-consultation/bookings").status_code == 200


def test_health_stays_open_and_says_whether_a_login_is_needed(settings, db, service, tmp_path):
    with make_client(settings, db, service, tmp_path, password="s3cret") as client:
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["login_required"] is True
        assert "caller_name" not in str(body)                       # nothing personal on the open page


def test_the_web_page_needs_a_login(settings, db, service, tmp_path):
    with make_client(settings, db, service, tmp_path, password="s3cret") as client:
        assert client.get("/config").status_code == 401 and client.get("/gradio_api/info").status_code == 401
        assert client.post("/login", data={"username": "admin", "password": "nope"}).status_code == 400
        assert client.post("/login", data={"username": "admin", "password": "s3cret"}).status_code == 200
        assert client.get("/config").status_code == 200              # logged in: the client keeps the cookie


def test_the_web_page_is_open_without_a_password(settings, db, service, tmp_path):
    with make_client(settings, db, service, tmp_path) as client:
        assert client.get("/config").status_code == 200


# ---------- Settings ----------
def test_new_defaults_are_safe():
    s = Settings(_env_file=None)
    assert s.dashboard_password_value is None and s.llm_api_key_value is None and s.agent_history_turns == 6


def test_the_llm_key_is_a_setting_that_is_never_printed():
    s = Settings(_env_file=None, llm_api_key="gsk_secret", llm_base_url="https://x/v1")
    assert s.llm_api_key_value == "gsk_secret" and "gsk_secret" not in repr(s)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_max_tool_rounds=0)


def test_warnings_for_settings_that_would_not_work_or_not_be_safe():
    warnings = warn_about_missing_settings(Settings(_env_file=None))
    assert any("LLM_API_KEY is not set" in w for w in warnings)
    warnings = warn_about_missing_settings(Settings(_env_file=None, llm_api_key="k", host="0.0.0.0"))
    assert any("without a password" in w for w in warnings) and not any("LLM_API_KEY" in w for w in warnings)
    clean = warn_about_missing_settings(Settings(_env_file=None, llm_api_key="k", host="0.0.0.0", dashboard_password="x"))
    assert not any("password" in w for w in clean)
    local = warn_about_missing_settings(Settings(_env_file=None, llm_base_url="http://127.0.0.1:11434/v1"))
    assert not any("LLM_API_KEY" in w for w in local)           # a local Ollama needs no key
