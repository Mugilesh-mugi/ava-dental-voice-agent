"""Checks that settings have the right defaults and can be changed with environment variables."""
import json
import logging

import pytest
from pydantic import ValidationError

from voice_agent.config import Settings, warn_about_missing_settings
from voice_agent.logging_setup import JsonFormatter, call_id_var, setup_logging


def make_settings(**env_overrides) -> Settings:
    # _env_file=None: ignore any real .env file so tests give the same result on every machine.
    return Settings(_env_file=None, **env_overrides)


def test_defaults_match_the_spec(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "DATABASE_URL", "DASHBOARD_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    s = make_settings()
    assert s.match_threshold == 0.35
    assert s.stt_model == "openai/whisper-small"
    assert s.llm_model == "qwen/qwen3.8-27b" and s.llm_base_url == "https://api.groq.com/openai/v1"
    assert s.database_url.startswith("postgresql+psycopg://")
    assert (s.agent_history_turns, s.agent_max_tool_rounds, s.max_unclear_turns) == (6, 3, 2)
    assert s.dashboard_password_value is None and s.use_prebuilt_index and s.docs_dir.name == "raw"


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("MATCH_THRESHOLD", "0.4")
    monkeypatch.setenv("LLM_MODEL", "some/other-model")
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@localhost/db")
    monkeypatch.setenv("MAX_UNCLEAR_TURNS", "3")
    s = make_settings()
    assert s.match_threshold == 0.4
    assert s.llm_model == "some/other-model"
    assert s.database_url == "mysql+pymysql://u:p@localhost/db"
    assert s.max_unclear_turns == 3


def test_bad_values_fail_at_startup(monkeypatch):
    monkeypatch.setenv("MATCH_THRESHOLD", "not-a-number")
    with pytest.raises(ValidationError):
        make_settings()


def test_keys_are_hidden_when_printed(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "gsk_secret123")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "s3cret-password")
    s = make_settings()
    assert "gsk_secret123" not in repr(s) and "s3cret-password" not in repr(s)
    assert s.llm_api_key_value == "gsk_secret123" and s.dashboard_password_value == "s3cret-password"


def test_missing_key_gives_clear_warning(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    warnings = warn_about_missing_settings(make_settings())
    assert any("LLM_API_KEY" in w for w in warnings)


def test_log_lines_are_json_with_call_id(tmp_path):
    setup_logging("INFO", log_dir=tmp_path)
    token = call_id_var.set("call-42")
    try:
        logging.getLogger("test").info("turn_finished", extra={"route": "faq", "llm_ms": 812})
    finally:
        call_id_var.reset(token)
    for handler in logging.getLogger().handlers:
        handler.flush()
    line = (tmp_path / "voice_agent.log").read_text(encoding="utf-8").strip().splitlines()[-1]
    entry = json.loads(line)
    assert entry["event"] == "turn_finished"
    assert entry["call_id"] == "call-42"
    assert entry["route"] == "faq" and entry["llm_ms"] == 812
    setup_logging("INFO", log_dir=None)  # close the file so tmp_path can be deleted on Windows


def test_json_formatter_includes_errors():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("x", logging.ERROR, "", 0, "llm_failed", None, exc_info=__import__("sys").exc_info())
    assert "boom" in json.loads(JsonFormatter().format(record))["error"]
