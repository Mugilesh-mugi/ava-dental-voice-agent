"""All settings for the voice agent, in one place.

Every setting has a sensible default, and every setting can be changed with an
environment variable of the same name (in capitals), without touching the code.
For example, in PowerShell:

    $env:MATCH_THRESHOLD = "0.30"

You can also put settings in a file called `.env` in the project folder
(copy `.env.example` to start). Real environment variables win over `.env`.

We use `pydantic-settings` for this. It reads the environment variables and
checks their types for us, so a typo like MATCH_THRESHOLD="abc" fails at
startup with a clear message instead of causing a strange bug later.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The project folder (the one that holds app.py). Used to build default paths.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",          # ignore unrelated variables in .env
        case_sensitive=False,    # MATCH_THRESHOLD and match_threshold both work
    )

    # ---------- Models (names of models on the Hugging Face Hub) ----------
    stt_model: str = "openai/whisper-small"            # speech-to-text
    # "transformers" (Hugging Face) or "faster_whisper" (several times faster on a CPU; use STT_MODEL=distil-small.en)
    stt_engine: Literal["transformers", "faster_whisper"] = "transformers"
    tts_model: str = "facebook/mms-tts-eng"            # text-to-speech
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # turns text into vectors for search
    llm_model: str = "qwen/qwen3.8-27b"                # the LLM that runs the conversation (see LLM_BASE_URL)
    # A sentence a caller might say. Whisper uses it as a hint, so it mishears these words less. Empty = no hint.
    # A natural sentence works better than a word list (a list made it write "cook an appointment").
    stt_prompt: str = ("Hello, Sunrise Dental. I'd like to book an appointment for a check-up or a cleaning, "
                       "maybe on Tuesday morning.")

    # The LLM: any service that speaks the OpenAI chat API and can choose tools. Groq (the default), xAI Grok, Gemini,
    # or a local Ollama at http://127.0.0.1:11434/v1 (a small local model is much less reliable at choosing tools).
    # SecretStr hides the key if the settings are ever printed or logged.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: SecretStr | None = None
    # How hard a "thinking" model thinks before answering: "low" for gpt-oss, "none" for Qwen3. A caller is
    # waiting, so blank picks a quick default for the model.
    llm_reasoning_effort: str | None = None

    # ---------- How the agent works ----------
    # The LLM chooses tools (check_availability, book_appointment, ...) and the code enforces the rules.
    agent_history_turns: int = Field(default=6, ge=1, le=12)     # past exchanges the agent can see
    agent_max_tool_rounds: int = Field(default=3, ge=1, le=6)    # tool calls allowed within one caller turn

    # ---------- LLM behaviour ----------
    llm_max_tokens: int = Field(default=120, ge=16, le=1024)    # longest reply the LLM may write
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0) # low = more predictable answers
    llm_timeout_s: float = Field(default=20.0, gt=0)            # give up on the LLM after this many seconds

    # ---------- FAQ search (RAG) ----------
    # Similarity score goes from -1 to 1. If the best FAQ match scores below this,
    # Ava says "I don't have that information" instead of asking the LLM.
    match_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    top_k: int = Field(default=2, ge=1, le=10)                  # how many FAQ matches the LLM sees
    docs_dir: Path = PROJECT_ROOT / "data" / "raw"                  # the clinic's documents (the FAQ and any notes)
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"       # the saved search index (python scripts/ingest.py)
    use_prebuilt_index: bool = True                                 # use that index when it matches the documents

    # ---------- Conversation rules ----------
    # How many unclear messages in a row before Ava hands over to a human.
    max_unclear_turns: int = Field(default=2, ge=1)

    # ---------- Appointments ----------
    # The clinic opens Mon-Fri 8am-6pm and Saturday 9am-1pm (see scheduling.OPENING_HOURS), with one dentist,
    # so one appointment per time slot. Slots are worked out from these hours, not stored.
    clinic_timezone: str = "Europe/London"
    slot_minutes: int = Field(default=30, ge=10, le=120)             # length of one appointment slot
    booking_horizon_days: int = Field(default=28, ge=1, le=120)      # how far ahead callers can book
    min_notice_minutes: int = Field(default=60, ge=0)                # no booking for a time that is too soon

    # ---------- Database ----------
    # PostgreSQL is the default production database. SQLite is only kept for local tests or demos.
    # Example local Postgres: postgresql+psycopg://postgres:postgres@localhost:5432/voice_agent
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/voice_agent"

    # ---------- Web server ----------
    host: str = "127.0.0.1"
    port: int = Field(default=7860, ge=1, le=65535)
    # Login for the web page, the dashboard and the bookings API. They show callers' names and numbers, so set a
    # password before the app is reachable from any other computer. Blank = no login (fine on 127.0.0.1 only).
    dashboard_user: str = "admin"
    dashboard_password: SecretStr | None = None

    # ---------- Logging and monitoring ----------
    log_level: str = "INFO"                  # DEBUG, INFO, WARNING, ERROR
    log_dir: Path = PROJECT_ROOT / "logs"    # where the log file is written
    log_json: bool = True                    # JSON lines in the log file (easy for tools to read)
    # Caller words are personal data. When False, logs keep routes and timings but not what was said.
    log_transcripts: bool = False

    @property
    def llm_api_key_value(self) -> str | None:
        return self.llm_api_key.get_secret_value() if self.llm_api_key else None

    @property
    def dashboard_password_value(self) -> str | None:
        return self.dashboard_password.get_secret_value() if self.dashboard_password else None


@lru_cache
def get_settings() -> Settings:
    """Load the settings once and reuse them everywhere (lru_cache remembers the result)."""
    return Settings()


def warn_about_missing_settings(settings: Settings) -> list[str]:
    """Log a clear warning for settings that will make parts of the app fail. Returns the warnings."""
    warnings = []
    if not settings.llm_api_key_value and "127.0.0.1" not in settings.llm_base_url \
            and "localhost" not in settings.llm_base_url:
        warnings.append("LLM_API_KEY is not set, so the LLM will not work: Ava will apologise and offer a human "
                        "for every question. Put the key in the .env file.")
    if not settings.dashboard_password_value and settings.host not in ("127.0.0.1", "localhost", "::1"):
        warnings.append(f"The app is listening on {settings.host} without a password: anyone who can reach it can "
                        "read callers' names and numbers. Set DASHBOARD_PASSWORD.")
    if not settings.docs_dir.exists():
        warnings.append(f"Documents folder not found at {settings.docs_dir}. Check the DOCS_DIR setting.")
    for message in warnings:
        log.warning(message)
    return warnings
