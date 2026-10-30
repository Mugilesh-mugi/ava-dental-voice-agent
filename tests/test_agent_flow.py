"""End-to-end conversation tests through the LangGraph flow and the call service. The LLM is faked (see tool_fakes.py), so
these run in seconds and use no LLM requests. Booking, moving and cancelling are in test_tool_agent.py."""
import json

from sqlalchemy import select

from fakes import FakeSTT
from tool_fakes import ScriptedToolLLM, calls, says
from voice_agent.agent import prompts
from voice_agent.memory.models import Call, Message


def say(service, session, *texts):
    return [service.handle_turn(session, text=t) for t in texts]


# ---------- Safety and small talk: fixed replies, the LLM is never asked ----------
def test_emergencies_never_reach_the_llm(build_service):
    llm = ScriptedToolLLM()
    service = build_service(llm=llm)
    session = service.start_call()
    severe, urgent = say(service, session, "I can't breathe", "I knocked my tooth out")
    assert severe.route == "emergency_999" and severe.reply == prompts.SEVERE_EMERGENCY
    assert urgent.route == "urgent_dental" and "01632 960 111" in urgent.reply
    assert llm.requests == []
    assert service.end_call(session) == "emergency_999"


def test_greeting_thanks_and_advice_never_reach_the_llm(build_service):
    llm = ScriptedToolLLM()
    service = build_service(llm=llm)
    session = service.start_call()
    greeting, advice, thanks = say(service, session, "hi", "what should I do about my toothache", "thank you")
    assert (greeting.route, advice.route, thanks.route) == ("greeting", "advice_refused", "thanks")
    assert advice.reply == prompts.NO_ADVICE and greeting.similarity_score is None and llm.requests == []


def test_handoff(service):
    assert say(service, service.start_call(), "can I speak to a real person")[0].route == "handed_off"


def test_two_unclear_turns_hand_off(service):
    session = service.start_call()
    first, second = say(service, session, "uh", "hmm")
    assert first.route == "unclear" and first.reply == prompts.UNCLEAR
    assert second.route == "unclear_handoff"
    assert service.end_call(session) == "handed_off"


def test_unclear_counter_resets_after_a_clear_turn(build_service):
    service = build_service(llm=ScriptedToolLLM(calls("get_clinic_info", question="What time do you open on Saturday?")))
    session = service.start_call()
    results = say(service, session, "uh", "what time do you open on Saturday", "um")
    assert [r.route for r in results] == ["unclear", "faq_answered", "unclear"]


# ---------- Failures ----------
def test_stt_failure_asks_to_repeat(build_service):
    service = build_service(stt=FakeSTT(fail=True))
    result = service.handle_turn(service.start_call(), audio=(16_000, [0.0] * 16_000))
    assert result.route == "stt_error" and result.reply == prompts.STT_ERROR


# ---------- Logging and storage ----------
HOURS = calls("get_clinic_info", question="What time do you open on Saturday?")


def test_every_turn_is_stored_with_timings(build_service, db):
    service = build_service(llm=ScriptedToolLLM(HOURS))
    session = service.start_call()
    service.handle_turn(session, audio=(16_000, [0.0] * 16_000))     # FakeSTT hears a question about opening hours
    service.end_call(session)
    with db.session() as s:
        call = s.get(Call, session.call_id)
        messages = s.scalars(select(Message).order_by(Message.id)).all()
    assert call.ended_at is not None and call.outcome == "answered"
    assert [m.role for m in messages] == ["caller", "agent"]
    agent = messages[1]
    assert agent.route == "faq_answered" and agent.similarity_score is not None
    assert agent.stt_ms >= 0 and agent.llm_ms > 0 and agent.tts_ms >= 0


def test_call_with_no_useful_turns_is_abandoned(service):
    session = service.start_call()
    say(service, session, "hello")
    assert service.end_call(session) == "abandoned"


def test_a_booking_waiting_for_yes_when_the_caller_hangs_up_is_abandoned(build_service):
    from test_tool_agent import BOOK_SCRIPT, BOOK_TURNS
    service = build_service(llm=ScriptedToolLLM(*BOOK_SCRIPT))
    session = service.start_call()
    say(service, session, *BOOK_TURNS)
    assert service.end_call(session) == "abandoned"


def test_turn_is_logged_as_json(build_service, settings):
    from voice_agent.logging_setup import setup_logging

    service = build_service(llm=ScriptedToolLLM(calls("get_clinic_info", question="Do you sell train tickets to London?")))
    setup_logging("INFO", settings.log_dir)
    try:
        session = service.start_call()
        service.handle_turn(session, text="Do you sell train tickets to London?")
    finally:
        setup_logging("INFO", None)
    lines = [json.loads(line) for line in (settings.log_dir / "voice_agent.log").read_text(encoding="utf-8").splitlines()]
    turn = next(line for line in lines if line["event"] == "turn_finished")
    assert turn["call_id"] == session.call_id and turn["route"] == "unanswered"
    assert "caller_text" not in turn          # caller words are not logged by default (personal data)


def test_dashboard_stats(service):
    session = service.start_call()
    say(service, session, "I can't breathe")
    service.end_call(session)
    stats = service.dashboard_stats()
    assert stats["total_calls"] == 1 and stats["severe_caught"] == 1
    assert stats["outcomes"] == {"emergency_999": 1}


def test_dashboard_counts_calls_still_in_progress(service):
    finished, open_call = service.start_call(), service.start_call()
    say(service, finished, "I can't breathe")
    service.end_call(finished)
    stats = service.dashboard_stats()
    assert stats["outcomes"] == {"emergency_999": 1, "in_progress": 1} and stats["total_calls"] == 2
