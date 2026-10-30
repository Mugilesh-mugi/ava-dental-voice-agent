"""The tool-using agent: the LLM chooses tools, and the code decides whether the call is safe.

"Now" is Monday 2 November 2026, 09:00. "Saturday morning" therefore offers Saturday 7 November at
9:00 (A), 9:30 (B) and 10:00 (C). The LLM is scripted: each test says exactly what it decides.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from tool_fakes import ScriptedToolLLM, calls, says
from voice_agent.agent import prompts
from voice_agent.memory.models import Booking, ToolCall, UnansweredQuestion
from voice_agent.llm.client import LLMRateLimited
from voice_agent.llm.client import LLMError
from voice_agent.services.scheduling import to_utc

SAT_9 = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)
PHONE = "07700 900321"


def talk(service, session, *texts):
    return [service.handle_turn(session, text=t) for t in texts]


def all_bookings(db):
    with db.session() as s:
        return s.scalars(select(Booking).order_by(Booking.id)).all()


def make_booking(bookings, db, phone="07700900123", start=SAT_9, name="Existing Caller"):
    from voice_agent.memory.models import Call
    with db.session() as s:
        call = Call()
        s.add(call)
        s.flush()
        call_id = call.id
    return bookings.create(call_id, {"caller_name": name, "reason": "check-up", "phone": phone}, start)


BOOK_SCRIPT = [says("What is your full name?"), says("And the reason for your visit?"),
               says("What day and time would suit you?"), calls("check_availability", when="Saturday morning"),
               says("What phone number can we reach you on?"),
               calls("book_appointment", caller_name="Priya", reason="check-up", slot_id="A", phone=PHONE)]
BOOK_TURNS = ["I need an appointment", "Priya", "a check-up", "Saturday morning", "the first one", PHONE]


# ---------- Booking ----------
def test_a_whole_booking_through_tools(build_service, db, tool):
    llm = ScriptedToolLLM(*BOOK_SCRIPT)
    service = build_service(llm)
    session = service.start_call()
    results = talk(service, session, *BOOK_TURNS)
    assert [r.route for r in results] == ["agent_reply", "agent_reply", "agent_reply", "tool_check_availability",
                                          "agent_reply", "tool_book_appointment"]
    assert "nine o'clock, half past nine or ten o'clock in the morning" in results[3].reply
    readback = results[5].reply                                    # the code spoke this, not the LLM
    assert "Priya" in readback and "Saturday the seventh of November at nine o'clock" in readback
    assert "0 7 7 0 0 9 0 0 3 2 1" in readback and readback.endswith("Shall I book that for you?")
    assert all_bookings(db) == []                                  # nothing is saved until the caller says yes

    done = talk(service, session, "yes")[0]
    assert done.route == "booking_saved" and "S D 0 0 0 0 0 1" in done.reply
    assert llm.script == [] and len(llm.requests) == 6             # the yes did not use the LLM
    booking, = all_bookings(db)
    assert (booking.caller_name, booking.reason, booking.phone) == ("Priya", "check-up", "07700900321")
    assert to_utc(booking.start_at) == SAT_9 and booking.status == "confirmed"
    assert tool.sent[0]["event"] == "created"
    assert service.end_call(session) == "booking_saved"


def test_the_llm_is_told_which_times_are_on_offer(build_service):
    llm = ScriptedToolLLM(*BOOK_SCRIPT[:5])
    service = build_service(llm)
    talk(service, service.start_call(), *BOOK_TURNS[:5])
    prompt = llm.requests[4]["messages"][0]["content"]               # what it saw when the caller said "the first one"
    assert "Times on offer: A = Saturday the seventh of November at nine o'clock in the morning" in prompt
    assert "B = " in prompt and "C = " in prompt and "today is Monday 2 November 2026" in prompt
    assert llm.requests[4]["messages"][-1] == {"role": "user", "content": "the first one"}
    assert any(m["content"] == "Priya" for m in llm.requests[4]["messages"])      # it remembers the conversation


@pytest.mark.parametrize("answer", ["no", "No thanks", "nope"])
def test_saying_no_saves_nothing(build_service, db, answer):
    service = build_service(ScriptedToolLLM(*BOOK_SCRIPT))
    session = service.start_call()
    talk(service, session, *BOOK_TURNS)
    declined = talk(service, session, answer)[0]
    assert declined.route == "agent_declined" and declined.reply == prompts.AGENT_DECLINED
    assert all_bookings(db) == [] and session.state["agent"]["pending"] is None


def test_saying_something_else_instead_of_yes_is_a_new_request(build_service, db):
    llm = ScriptedToolLLM(*BOOK_SCRIPT, says("Of course. What day suits you better?"))
    service = build_service(llm)
    session = service.start_call()
    talk(service, session, *BOOK_TURNS)
    other = talk(service, session, "actually I'd rather come another day")[0]
    assert other.route == "agent_reply" and all_bookings(db) == []


def test_yes_but_with_a_new_time_is_not_a_yes(build_service, db):
    llm = ScriptedToolLLM(*BOOK_SCRIPT, calls("check_availability", when="Tuesday"))
    service = build_service(llm)
    session = service.start_call()
    talk(service, session, *BOOK_TURNS)
    result = talk(service, session, "yes but make it Tuesday")[0]
    assert result.route == "tool_check_availability" and all_bookings(db) == []


def test_a_time_taken_in_the_meantime_is_not_double_booked(build_service, bookings, db):
    service = build_service(ScriptedToolLLM(*BOOK_SCRIPT))
    session = service.start_call()
    talk(service, session, *BOOK_TURNS)
    make_booking(bookings, db)                                       # someone else books 9:00 first
    taken = talk(service, session, "yes")[0]
    assert taken.route == "booking_slot_taken" and taken.reply == prompts.AGENT_SLOT_GONE
    assert len(all_bookings(db)) == 1


# ---------- The checks that stop the LLM doing something unsafe ----------
def test_a_time_that_was_never_offered_is_refused(build_service, db):
    llm = ScriptedToolLLM(calls("book_appointment", caller_name="Priya", reason="check-up", slot_id="Z", phone=PHONE),
                          says("Which of the times would you like?"))
    service = build_service(llm)
    result = talk(service, service.start_call(), "book me in")[0]
    assert result.reply == "Which of the times would you like?" and all_bookings(db) == []
    told_to_the_llm = llm.requests[1]["messages"][-1]["content"]      # the refusal the LLM was shown
    assert "slot_id must be one of the times on offer" in told_to_the_llm and '"ok": false' in told_to_the_llm
    assert llm.requests[1]["tool_choice"] == "none"                  # after a refusal it must ask the caller, not try another tool


def test_missing_details_are_refused_and_asked_for(build_service, db):
    llm = ScriptedToolLLM(calls("check_availability", when="Saturday morning"),
                          calls("book_appointment", caller_name="", reason="", slot_id="A", phone="123"),
                          says("Could I have your full name first?"))
    service = build_service(llm)
    results = talk(service, service.start_call(), "Saturday morning please", "the first one")
    assert results[1].reply == "Could I have your full name first?" and all_bookings(db) == []
    refusal = llm.requests[2]["messages"][-1]["content"]
    assert "full name is missing" in refusal and "reason" in refusal and "phone number" in refusal


def test_an_invented_booking_id_cannot_be_cancelled(build_service, bookings, db):
    existing = make_booking(bookings, db)
    llm = ScriptedToolLLM(calls("cancel_appointment", booking_id=existing.id), says("What number did you book with?"))
    service = build_service(llm)
    result = talk(service, service.start_call(), "cancel my appointment")[0]
    assert result.reply == "What number did you book with?"
    assert all_bookings(db)[0].status == "confirmed"                  # the id was not found for THIS caller


def test_someone_elses_booking_cannot_be_moved(build_service, bookings, db):
    make_booking(bookings, db, phone="07700900123", start=SAT_9)
    theirs = make_booking(bookings, db, phone="07700900999", start=datetime(2026, 11, 7, 9, 30, tzinfo=timezone.utc))
    llm = ScriptedToolLLM(says("What phone number did you book with?"), calls("find_bookings", phone="07700 900123"),
                          says("Here you are."), calls("check_availability", when="Tuesday morning"),
                          calls("reschedule_appointment", booking_id=theirs.id, slot_id="A"),
                          says("I can only change your own appointment."))
    service = build_service(llm)
    results = talk(service, service.start_call(), "I want to move my appointment", "07700 900123", "Tuesday morning",
                   "move the other one please")
    assert results[3].reply == "I can only change your own appointment."
    assert {to_utc(b.start_at) for b in all_bookings(db)} == {SAT_9, datetime(2026, 11, 7, 9, 30, tzinfo=timezone.utc)}


def test_an_unknown_tool_is_refused(build_service):
    llm = ScriptedToolLLM(calls("delete_everything"), says("Sorry, I can't do that."))
    service = build_service(llm)
    assert talk(service, service.start_call(), "please do something odd")[0].reply == "Sorry, I can't do that."
    assert "no tool called delete_everything" in llm.requests[1]["messages"][-1]["content"]


def test_a_time_stated_without_a_tool_is_not_allowed(build_service):
    llm = ScriptedToolLLM(says("We have Saturday at 9:00 available."), calls("check_availability", when="Saturday"))
    service = build_service(llm)
    result = talk(service, service.start_call(), "anything on Saturday?")[0]
    assert result.route == "tool_check_availability" and "nine o'clock" in result.reply     # the code's words, not the LLM's
    assert llm.requests[1]["tool_choice"] == "required"


def test_a_price_stated_without_a_tool_is_not_allowed(build_service):
    llm = ScriptedToolLLM(says("A check-up is 45 pounds."), calls("get_clinic_info", question="How much is a check-up?"))
    service = build_service(llm)
    result = talk(service, service.start_call(), "how much is a check-up?")[0]
    assert result.route in ("faq_answered", "unanswered") and llm.requests[1]["tool_choice"] == "required"


def test_the_llm_cannot_loop_forever(build_service):
    refused = calls("cancel_appointment", booking_id=1)
    llm = ScriptedToolLLM(refused, refused, refused)
    service = build_service(llm)
    result = talk(service, service.start_call(), "please cancel my booking now")[0]
    assert result.reply == prompts.AGENT_CONFUSED and llm.script == []


def test_advice_in_the_llms_words_is_blocked(build_service):
    llm = ScriptedToolLLM(says("Try rinse with warm salt water and take ibuprofen."))
    service = build_service(llm)
    assert talk(service, service.start_call(), "my gum feels odd")[0].reply == prompts.NO_ADVICE


# ---------- Looking up, moving and cancelling ----------
def test_look_up_then_cancel_by_phone_number(build_service, bookings, db, tool):
    existing = make_booking(bookings, db)
    llm = ScriptedToolLLM(says("What phone number did you book with?"), calls("find_bookings", phone="07700 900123"),
                          says("Found it."), calls("cancel_appointment", booking_id=existing.id))
    service = build_service(llm)
    session = service.start_call()
    ask, found, staged = talk(service, session, "cancel my appointment", "07700 900123", "yes cancel it please")
    assert "Saturday the seventh of November" in found.reply and "S D 0 0 0 0 0 1" in found.reply
    assert staged.reply.startswith("I found your appointment on Saturday the seventh") and "cancel it?" in staged.reply
    assert all_bookings(db)[0].status == "confirmed"                  # still not cancelled
    done = talk(service, session, "yes")[0]
    assert done.route == "booking_cancelled_existing" and "I've cancelled" in done.reply
    assert all_bookings(db)[0].status == "cancelled" and tool.sent[-1]["event"] == "cancelled"
    assert service.end_call(session) == "booking_cancelled"


def test_move_an_appointment(build_service, bookings, db, tool):
    existing = make_booking(bookings, db)
    llm = ScriptedToolLLM(says("What phone number did you book with?"), calls("find_bookings", phone="07700900123"),
                          says("Found it."), calls("check_availability", when="Tuesday morning"),
                          calls("reschedule_appointment", booking_id=existing.id, slot_id="B"))
    service = build_service(llm)
    session = service.start_call()
    _, _, offer, staged = talk(service, session, "I want to move my appointment", "07700900123",
                               "move it to Tuesday morning", "the second one")
    assert "tomorrow at eight o'clock" in offer.reply
    assert staged.reply == ("Shall I move your appointment from Saturday the seventh of November at nine o'clock in "
                            "the morning to tomorrow at half past eight in the morning?")
    done = talk(service, session, "yes")[0]
    assert done.route == "booking_changed"
    assert to_utc(all_bookings(db)[0].start_at) == datetime(2026, 11, 3, 8, 30, tzinfo=timezone.utc)
    assert tool.sent[-1]["event"] == "rescheduled"


def test_a_booking_made_in_this_call_can_be_cancelled_without_a_phone_number(build_service, db):
    service = build_service(ScriptedToolLLM(*BOOK_SCRIPT, calls("cancel_appointment", booking_id=1)))
    session = service.start_call()
    talk(service, session, *BOOK_TURNS, "yes")
    ask = talk(service, session, "actually cancel it")[0]
    assert ask.route == "tool_cancel_appointment" and talk(service, session, "yes")[0].route == "booking_cancelled_existing"
    assert all_bookings(db)[0].status == "cancelled"


def test_a_phone_number_with_no_bookings(build_service):
    llm = ScriptedToolLLM(says("What phone number did you book with?"), calls("find_bookings", phone="07700 111 222"))
    service = build_service(llm)
    results = talk(service, service.start_call(), "when is my appointment", "07700 111 222")
    assert results[1].reply == prompts.MANAGE_NOT_FOUND


def test_an_incomplete_phone_number_is_refused(build_service):
    llm = ScriptedToolLLM(calls("find_bookings", phone="0770"), says("Could you say the whole number?"))
    service = build_service(llm)
    assert talk(service, service.start_call(), "my number is 0770")[0].reply == "Could you say the whole number?"
    assert "not a full phone number" in llm.requests[1]["messages"][-1]["content"]


# ---------- Questions about the clinic (RAG as a tool) ----------
def test_clinic_questions_are_answered_from_the_faq(build_service):
    llm = ScriptedToolLLM(calls("get_clinic_info", question="What time do you open on Saturday?"))
    service = build_service(llm)
    result = talk(service, service.start_call(), "what time do you open on Saturday")[0]
    assert result.route == "faq_answered" and "Saturday" in result.reply and result.similarity_score >= 0.25


def test_a_question_the_faq_cannot_answer_is_logged(build_service, db):
    llm = ScriptedToolLLM(calls("get_clinic_info", question="Do you sell train tickets to London?"))
    service = build_service(llm)
    result = talk(service, service.start_call(), "do you sell train tickets to London")[0]
    assert result.route == "unanswered" and result.reply == prompts.FALLBACK
    with db.session() as s:
        assert s.scalars(select(UnansweredQuestion)).one().question == "do you sell train tickets to London"


def test_availability_without_a_day_gives_the_soonest_times(build_service):
    llm = ScriptedToolLLM(calls("check_availability", when=""))
    service = build_service(llm)
    result = talk(service, service.start_call(), "what's your next free slot")[0]
    assert "I can offer you today at ten o'clock" in result.reply        # it is 9:00 and one hour's notice is needed


# ---------- Safety and routing stay with the rules ----------
def test_emergencies_never_reach_the_llm(build_service):
    llm = ScriptedToolLLM()
    service = build_service(llm)
    session = service.start_call()
    severe, urgent = talk(service, session, "I can't breathe", "my face is really swollen")
    assert severe.route == "emergency_999" and urgent.route == "urgent_dental" and llm.requests == []


def test_an_emergency_cancels_a_change_that_was_waiting_for_yes(build_service, db):
    service = build_service(ScriptedToolLLM(*BOOK_SCRIPT))
    session = service.start_call()
    talk(service, session, *BOOK_TURNS)
    assert session.state["agent"]["pending"] is not None
    assert talk(service, session, "I can't breathe")[0].route == "emergency_999"
    assert session.state["agent"] is None and all_bookings(db) == []


def test_advice_greetings_and_thanks_use_no_llm_request(build_service):
    llm = ScriptedToolLLM()
    service = build_service(llm)
    routes = [r.route for r in talk(service, service.start_call(), "hello", "what should I do about my toothache",
                                    "thank you")]
    assert routes == ["greeting", "advice_refused", "thanks"] and llm.requests == []


def test_short_real_messages_go_to_the_agent(build_service):
    llm = ScriptedToolLLM(says("What is your full name?"), says("Thanks Priya. What is the reason for your visit?"),
                          calls("check_availability", when="Tuesday morning"))
    service = build_service(llm)
    first, name, when = talk(service, service.start_call(), "I need an appointment", "Priya", "Tuesday morning")
    assert name.route == "agent_reply" and when.route == "tool_check_availability"      # not "say that again"


@pytest.mark.parametrize("noise", ["hmm", "uh", "um erm", "...", "?!"])
def test_noise_is_unclear_and_never_reaches_the_llm(build_service, noise):
    llm = ScriptedToolLLM()
    service = build_service(llm)
    assert talk(service, service.start_call(), noise)[0].route == "unclear" and llm.requests == []


# ---------- When the LLM is not available ----------
def test_llm_down_apologises_then_hands_over(build_service):
    llm = ScriptedToolLLM(LLMError("network down"), LLMError("network down"))
    service = build_service(llm)
    session = service.start_call()
    first, second = talk(service, session, "I need an appointment", "I need an appointment")
    assert first.route == "llm_error" and first.reply == prompts.LLM_ERROR
    assert second.route == "unclear_handoff" and service.end_call(session) == "handed_off"


def test_rate_limit_asks_the_caller_to_wait(build_service):
    llm = ScriptedToolLLM(LLMRateLimited("too many", 20.0), says("What is your full name?"))
    service = build_service(llm)
    limited, recovered = talk(service, service.start_call(), "I need an appointment", "I need an appointment")
    assert limited.route == "llm_rate_limited" and limited.reply == prompts.AGENT_RATE_LIMITED
    assert recovered.route == "agent_reply"                              # one failure, then it worked: no handoff


def test_a_rate_limit_when_answering_from_the_faq(build_service):
    class LimitedFaq(ScriptedToolLLM):
        def chat(self, messages):
            raise LLMRateLimited("too many", 20.0)

    service = build_service(LimitedFaq(calls("get_clinic_info", question="What time do you open on Saturday?")))
    result = talk(service, service.start_call(), "what time do you open on Saturday")[0]
    assert result.route == "llm_rate_limited"


# ---------- The audit trail ----------
def test_every_tool_call_is_recorded_with_personal_details_masked(build_service, db):
    service = build_service(ScriptedToolLLM(*BOOK_SCRIPT, calls("cancel_appointment", booking_id=99), says("Hm.")))
    session = service.start_call()
    talk(service, session, *BOOK_TURNS, "cancel booking 99")
    with db.session() as s:
        rows = s.scalars(select(ToolCall).order_by(ToolCall.id)).all()
    assert [(r.tool, r.ok) for r in rows] == [("check_availability", True), ("book_appointment", True),
                                              ("cancel_appointment", False)]
    book = rows[1].args
    assert "07700900321" not in book and "07700 900321" not in book and "…0321" in book
    assert "Priya" not in book and '"P."' in book and "check-up" not in book
    assert all(r.call_id == session.call_id for r in rows)
    shown = service.recent_tool_calls()
    assert shown[0][2] == "cancel_appointment" and shown[0][3] == "refused"


# ---------- Looking up and then acting, in one turn ----------
def test_a_lookup_can_lead_straight_to_the_next_step(build_service, bookings, db):
    existing = make_booking(bookings, db)
    llm = ScriptedToolLLM(calls("find_bookings", phone="07700900123"),
                          calls("reschedule_appointment", booking_id=existing.id, slot_id="A"))
    service = build_service(llm)
    session = service.start_call()
    # Nothing is on offer yet, so the move is refused and the LLM must ask the caller for a time.
    llm.script.append(says("What day and time would you like instead?"))
    result = talk(service, session, "move my appointment, my number is 07700 900123")[0]
    assert result.reply == "What day and time would you like instead?" and len(llm.requests) == 3
    seen_by_llm = llm.requests[1]["messages"][-1]["content"]          # the lookup result it carried on with
    assert f'"booking_id": {existing.id}' in seen_by_llm and "Saturday the seventh of November" in seen_by_llm
    assert llm.requests[1]["tool_choice"] == "auto"


def test_a_plain_lookup_is_spoken_by_the_code_not_the_llm(build_service, bookings, db):
    make_booking(bookings, db)
    llm = ScriptedToolLLM(says("What number?"), calls("find_bookings", phone="07700900123"),
                          says("Sure thing, your appointment is on Tuesday at noon!"))     # the LLM gets the facts wrong
    service = build_service(llm)
    _, found = talk(service, service.start_call(), "when is my appointment", "07700900123")
    assert found.route == "tool_find_bookings" and "Saturday the seventh of November" in found.reply
    assert "Tuesday" not in found.reply


def test_a_phone_number_the_caller_never_said_is_refused(build_service, bookings, db):
    make_booking(bookings, db, phone="07700900999")
    llm = ScriptedToolLLM(says("What number did you book with?"), calls("find_bookings", phone="07700 900999"),
                          says("Could you tell me the number you booked with?"))
    service = build_service(llm)
    results = talk(service, service.start_call(), "when is my appointment", "I don't remember it")
    assert results[1].reply == "Could you tell me the number you booked with?"
    assert "has not said that number" in llm.requests[2]["messages"][-1]["content"]


def test_booking_with_a_phone_number_the_caller_never_said_is_refused(build_service, db):
    llm = ScriptedToolLLM(calls("check_availability", when="Saturday morning"),
                          calls("book_appointment", caller_name="Priya", reason="check-up", slot_id="A",
                                phone="07700 900321"),
                          says("What phone number can we reach you on?"))
    service = build_service(llm)
    results = talk(service, service.start_call(), "Saturday morning please, I'm Priya, for a check-up",
                   "the first one")
    assert results[1].reply == "What phone number can we reach you on?" and all_bookings(db) == []
    assert service.start_call().state.get("agent") is None


def test_the_code_tells_the_llm_which_offered_time_the_caller_chose(build_service):
    llm = ScriptedToolLLM(calls("check_availability", when="Saturday morning"), says("What phone number can we reach you on?"),
                          says("And your phone number?"), says("Could you say that again?"))
    service = build_service(llm)
    talk(service, service.start_call(), "Saturday morning please", "the second one", "ten o'clock", "what do you mean")
    seen = [r["messages"][0]["content"] for r in llm.requests]
    assert "chooses time" not in seen[0]                              # nothing on offer yet
    assert "latest message chooses time B: use slot_id B" in seen[1]
    assert "latest message chooses time C: use slot_id C" in seen[2]
    assert "Times on offer" in seen[3] and "chooses time" not in seen[3]       # still on offer, but no choice made
