"""Runs calls: start a call, handle each turn (hear -> think -> speak), log everything, end the call.

One turn:
    audio --STT--> caller text --agent graph--> reply text --TTS--> reply audio
Each stage is timed. Both sides of the turn are saved to the database, along with
the route taken and the FAQ similarity score, and summarised in one log line.

A failure in one stage never crashes the call:
- speech-to-text fails -> Ava asks the caller to repeat
- LLM fails            -> handled inside the agent (apology + offer a human)
- text-to-speech fails -> the reply is still shown as text
- database fails       -> the conversation carries on; the error is logged
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import func, select

from .. import monitoring
from ..agent import prompts, speech
from ..agent.graph import VoiceAgent
from ..agent.state import CallState, new_call_state
from ..config import Settings
from ..memory.database import Database
from zoneinfo import ZoneInfo

from ..memory.models import Booking, Call, Message, ToolCall, UnansweredQuestion, utcnow
from ..logging_setup import call_id_var
from ..voice.stt import SpeechToText
from ..voice.tts import TextToSpeech

log = logging.getLogger(__name__)

EMERGENCY_ROUTES = {"emergency_999": "severe", "urgent_dental": "urgent"}


@dataclass
class CallSession:
    """One call. The demo page keeps one of these per browser session."""

    call_id: str
    state: CallState
    ended: bool = False
    turns: int = 0


@dataclass
class TurnResult:
    caller_text: str
    reply: str
    route: str
    similarity_score: float | None
    stt_ms: float
    llm_ms: float                  # "brain" time: rules + FAQ search + LLM
    tts_ms: float
    outcome: str | None
    audio: tuple[int, np.ndarray] | None = None
    timings_detail: dict = field(default_factory=dict)


class CallService:
    def __init__(self, settings: Settings, agent: VoiceAgent, db: Database,
                 stt: SpeechToText | None = None, tts: TextToSpeech | None = None,
                 clock: Callable[[], datetime] = utcnow):
        self._s = settings
        self._agent = agent
        self._db = db
        self._stt = stt
        self._tts = tts
        self._clock = clock

    # ======================================================================
    # Call lifecycle
    # ======================================================================
    def start_call(self) -> CallSession:
        with self._db.session() as s:
            call = Call()
            s.add(call)
            s.flush()
            call_id = call.id
        call_id_var.set(call_id)
        monitoring.CALLS_STARTED.inc()
        monitoring.ACTIVE_CALLS.inc()
        log.info("call_started")
        return CallSession(call_id=call_id, state=new_call_state(call_id))

    def end_call(self, session: CallSession) -> str:
        """Close the call and store its final outcome. Safe to call twice."""
        if session.ended:
            return session.state.get("outcome") or "abandoned"
        call_id_var.set(session.call_id)
        outcome = session.state.get("outcome") or "abandoned"
        if (session.state.get("agent") or {}).get("pending") and outcome == "answered":
            outcome = "abandoned"            # hung up while a booking was waiting for the caller's yes
        try:
            with self._db.session() as s:
                call = s.get(Call, session.call_id)
                call.ended_at = utcnow()
                call.outcome = outcome
        except Exception:
            log.exception("call_end_save_failed")
            monitoring.ERRORS.labels("database").inc()
        session.ended = True
        monitoring.CALLS_ENDED.labels(outcome).inc()
        monitoring.ACTIVE_CALLS.dec()
        log.info("call_ended", extra={"outcome": outcome, "turns": session.turns})
        return outcome

    def close_stale_calls(self, older_than: timedelta = timedelta(hours=1)) -> int:
        """Mark calls left open (browser closed without 'End call') as abandoned."""
        cutoff = datetime.now(timezone.utc) - older_than
        with self._db.session() as s:
            stale = s.scalars(select(Call).where(Call.ended_at.is_(None), Call.started_at < cutoff)).all()
            for call in stale:
                call.ended_at = utcnow()
                call.outcome = call.outcome or "abandoned"
        if stale:
            log.info("stale_calls_closed", extra={"count": len(stale)})
        return len(stale)

    # ======================================================================
    # One turn
    # ======================================================================
    def greeting(self) -> tuple[str, tuple[int, np.ndarray] | None, float]:
        audio, tts_ms = self.speak(prompts.GREETING)
        return prompts.GREETING, audio, tts_ms

    def speak(self, text: str) -> tuple[tuple[int, np.ndarray] | None, float]:
        if self._tts is None:
            return None, 0.0
        start = time.perf_counter()
        try:
            audio = self._tts.speak(speech.for_voice(text))      # digits spelled out; the screen keeps the original text
        except Exception:
            log.exception("tts_failed")
            monitoring.ERRORS.labels("tts").inc()
            audio = None
        tts_ms = (time.perf_counter() - start) * 1000
        monitoring.STAGE_SECONDS.labels("tts").observe(tts_ms / 1000)
        return audio, tts_ms

    def handle_turn(self, session: CallSession, *, audio: tuple[int, np.ndarray] | None = None,
                    text: str | None = None) -> TurnResult:
        """Handle one caller turn, from audio (spoken) or text (typed)."""
        call_id_var.set(session.call_id)
        turn_start = time.perf_counter()

        # 1. Ears: speech-to-text
        stt_ms = 0.0
        stt_failed = False
        if audio is not None:
            start = time.perf_counter()
            try:
                text = self._stt.transcribe(*audio)
            except Exception:
                log.exception("stt_failed")
                monitoring.ERRORS.labels("stt").inc()
                text, stt_failed = "", True
            stt_ms = (time.perf_counter() - start) * 1000
            monitoring.STAGE_SECONDS.labels("stt").observe(stt_ms / 1000)
        caller_text = (text or "").strip()

        # 2. Brain: rules -> FAQ search -> LLM (the LangGraph flow)
        start = time.perf_counter()
        if stt_failed:
            new_state = {**session.state, "caller_text": "", "route": "stt_error", "reply": prompts.STT_ERROR,
                         "similarity_score": None, "llm_ms": 0.0}
        else:
            new_state = self._agent.run_turn(session.state, caller_text)
        brain_ms = (time.perf_counter() - start) * 1000
        monitoring.STAGE_SECONDS.labels("brain").observe(brain_ms / 1000)
        if new_state.get("llm_ms"):
            monitoring.STAGE_SECONDS.labels("llm").observe(new_state["llm_ms"] / 1000)

        # 3. Mouth: text-to-speech
        reply, route = new_state["reply"], new_state["route"]
        reply_audio, tts_ms = self.speak(reply)

        # 4. Remember the exchange for the LLM's context on later turns
        if caller_text:
            history = new_state.get("history", []) + [{"role": "user", "content": caller_text},
                                                      {"role": "assistant", "content": reply}]
            new_state["history"] = history[-2 * self._s.agent_history_turns:]
        session.state = new_state
        session.turns += 1

        result = TurnResult(caller_text=caller_text, reply=reply, route=route,
                            similarity_score=new_state.get("similarity_score"), stt_ms=stt_ms, llm_ms=brain_ms,
                            tts_ms=tts_ms, outcome=new_state.get("outcome"), audio=reply_audio,
                            timings_detail={"llm_call_ms": new_state.get("llm_ms", 0.0),
                                            "total_ms": (time.perf_counter() - turn_start) * 1000})
        self._record(session, result)
        return result

    # ======================================================================
    # Logging and storage
    # ======================================================================
    def _record(self, session: CallSession, r: TurnResult) -> None:
        monitoring.TURNS.labels(r.route).inc()
        if r.route in EMERGENCY_ROUTES:
            monitoring.EMERGENCIES.labels(EMERGENCY_ROUTES[r.route]).inc()
        if r.route == "unanswered":
            monitoring.UNANSWERED.inc()
        if r.route == "llm_error":
            monitoring.ERRORS.labels("llm").inc()
        if r.similarity_score is not None:
            monitoring.SIMILARITY.observe(r.similarity_score)

        extra = {"turn": session.turns, "route": r.route, "intent": session.state.get("intent"),
                 "similarity_score": r.similarity_score, "faq_titles": session.state.get("faq_titles"),
                 "stt_ms": round(r.stt_ms), "llm_ms": round(r.llm_ms), "tts_ms": round(r.tts_ms),
                 "llm_call_ms": round(r.timings_detail["llm_call_ms"]), "total_ms": round(r.timings_detail["total_ms"]),
                 "outcome": r.outcome}
        if self._s.log_transcripts:
            extra |= {"caller_text": r.caller_text, "reply": r.reply}
        log.info("turn_finished", extra=extra)

        try:
            with self._db.session() as s:
                s.add(Message(call_id=session.call_id, role="caller", text=r.caller_text, route=r.route,
                              similarity_score=r.similarity_score, stt_ms=r.stt_ms))
                s.add(Message(call_id=session.call_id, role="agent", text=r.reply, route=r.route,
                              similarity_score=r.similarity_score, stt_ms=r.stt_ms, llm_ms=r.llm_ms,
                              tts_ms=r.tts_ms))
                if r.route == "unanswered":
                    s.add(UnansweredQuestion(call_id=session.call_id, question=r.caller_text,
                                             best_score=r.similarity_score or 0.0))
                call = s.get(Call, session.call_id)
                call.outcome = r.outcome     # keep the running outcome, in case the call is never ended
        except Exception:
            log.exception("turn_save_failed")
            monitoring.ERRORS.labels("database").inc()

    # ======================================================================
    # Read-only queries for the dashboard
    # ======================================================================
    def dashboard_stats(self) -> dict:
        with self._db.session() as s:
            # Group by the plain column and name the calls with no outcome yet in Python: PostgreSQL rejects a
            # GROUP BY on an expression that contains a query parameter (SQLite accepts it, so tests alone miss it).
            outcomes: dict[str, int] = {}
            for outcome, count in s.execute(select(Call.outcome, func.count()).group_by(Call.outcome)).all():
                outcomes[outcome or "in_progress"] = outcomes.get(outcome or "in_progress", 0) + count
            total_calls = sum(outcomes.values())
            averages = s.execute(select(func.avg(Message.stt_ms), func.avg(Message.llm_ms), func.avg(Message.tts_ms))
                                 .where(Message.role == "agent")).one()
            routes = dict(s.execute(select(Message.route, func.count()).where(Message.role == "agent")
                                    .group_by(Message.route)).all())
            bookings = s.scalar(select(func.count()).select_from(Booking).where(Booking.status == "confirmed")) or 0
        return {
            "total_calls": total_calls,
            "outcomes": outcomes,
            "avg_stt_ms": averages[0] or 0.0,
            "avg_llm_ms": averages[1] or 0.0,
            "avg_tts_ms": averages[2] or 0.0,
            "handoff_rate": outcomes.get("handed_off", 0) / total_calls if total_calls else 0.0,
            "emergencies_caught": routes.get("emergency_999", 0) + routes.get("urgent_dental", 0),
            "severe_caught": routes.get("emergency_999", 0),
            "urgent_caught": routes.get("urgent_dental", 0),
            "routes": routes,
            "bookings": bookings,
        }

    def upcoming_appointments(self, limit: int = 50) -> list[list[str]]:
        """Confirmed appointments that haven't happened yet, soonest first, ready to show staff.
        The phone number is partly hidden: the dashboard is for an overview, not for reading out numbers."""
        tz = ZoneInfo(self._s.clinic_timezone)
        with self._db.session() as s:
            rows = s.scalars(select(Booking).where(Booking.status == "confirmed", Booking.start_at >= self._clock())
                             .order_by(Booking.start_at).limit(limit)).all()
        def local(dt):
            return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(tz)
        return [[local(b.start_at).strftime("%a %d %b %Y, %H:%M"), b.reference or "-", b.caller_name, b.reason,
                 f"…{b.phone[-4:]}", b.sync_status] for b in rows]

    def recent_tool_calls(self, limit: int = 50) -> list[list]:
        """The latest tools the LLM called, newest first, ready to show staff (the arguments are already masked)."""
        with self._db.session() as s:
            rows = s.scalars(select(ToolCall).order_by(ToolCall.id.desc()).limit(limit)).all()
        return [[r.created_at.strftime("%d %b %H:%M:%S"), r.call_id[:8], r.tool, "ok" if r.ok else "refused",
                 round(r.ms), r.args, r.result] for r in rows]

    def unanswered_questions(self, limit: int = 50) -> list[UnansweredQuestion]:
        with self._db.session() as s:
            return list(s.scalars(select(UnansweredQuestion).order_by(UnansweredQuestion.id.desc()).limit(limit)))

    def recent_calls(self, limit: int = 50) -> list[Call]:
        with self._db.session() as s:
            return list(s.scalars(select(Call).order_by(Call.started_at.desc()).limit(limit)))

    def conversation(self, call_id: str) -> list[Message]:
        with self._db.session() as s:
            return list(s.scalars(select(Message).where(Message.call_id == call_id).order_by(Message.id)))
