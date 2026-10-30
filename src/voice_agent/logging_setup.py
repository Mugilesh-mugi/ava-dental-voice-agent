"""Logging setup: readable lines on screen, and JSON lines in a log file.

What is a "JSON line"? Each log entry is written as one line of JSON, for example:
    {"ts": "...", "level": "INFO", "event": "turn_finished", "call_id": "abc", "route": "faq", "llm_ms": 812}
Tools like Azure Monitor, Grafana Loki, or a simple script can search and count
these easily, which is much harder with free-form text.

Every log entry automatically carries the current `call_id`, so you can pull out
everything that happened in one call. Set it with `call_id_var.set(...)` when a
call starts.
"""
from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path

# A "context variable" holds a value for the current piece of work (here: the current call),
# so we don't have to pass call_id into every function just for logging.
call_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("call_id", default="-")

# Built-in attributes every log record has. Anything else was passed with `extra={...}`.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime", "call_id"}


class _CallIdFilter(logging.Filter):
    """Adds the current call_id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.call_id = call_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Writes each log record as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "call_id": getattr(record, "call_id", "-"),
        }
        # Include any extra fields, e.g. log.info("turn_finished", extra={"route": "faq"})
        entry.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO", log_dir: Path | None = None, json_file: bool = True) -> None:
    """Configure logging for the whole app. Safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    call_filter = _CallIdFilter()

    screen = logging.StreamHandler(sys.stdout)
    screen.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(call_id)s] %(name)s: %(message)s",
                                          datefmt="%H:%M:%S"))
    screen.addFilter(call_filter)
    root.addHandler(screen)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        # A "rotating" file: when it reaches 5 MB it is renamed and a new one starts (5 old files kept),
        # so logs can never fill the disk.
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "voice_agent.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter() if json_file else screen.formatter)
        file_handler.addFilter(call_filter)
        root.addHandler(file_handler)

    # Third-party libraries are very chatty at INFO level; only show their warnings.
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "transformers", "sentence_transformers",
                  "gradio", "multipart", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
