"""The tools: their definitions, the checks that refuse unsafe calls, and the registry that runs them.

"Now" is Monday 2 November 2026, 09:00, so "Saturday morning" offers Saturday 7 November at 9:00 (A), 9:30 (B), 10:00 (C).
"""
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from fakes import FakeLLM
from voice_agent import monitoring
from voice_agent.agent import prompts
from voice_agent.agent.state import new_agent_state
from voice_agent.llm.client import LLMError
from voice_agent.memory.models import Call, ToolCall
from voice_agent.tools import Tool, ToolContext, ToolRegistry, ToolResult, default_tools, mask_args
from voice_agent.tools.base import heard_phone, known_booking, offered_slot

SAT_9 = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)
PHONE = "07700900321"


@pytest.fixture
def call_id(db):
    with db.session() as s:
        call = Call()
        s.add(call)
        s.flush()
        return call.id


@pytest.fixture
def ctx(settings, bookings, faq_store):
    return ToolContext(settings, bookings, faq_store, FakeLLM())


@pytest.fixture
def registry(ctx):
    return ToolRegistry(ctx)


def agent_with(**memory):
    agent = new_agent_state()
    agent.update(memory)
    return agent


OFFERED = {"offered": {"A": SAT_9.isoformat(), "B": "2026-11-07T09:30:00+00:00"},
           "offered_text": {"A": "Saturday at nine", "B": "Saturday at half past nine"}}


# ---------- The definitions ----------
def test_the_agent_has_exactly_these_tools(registry):
    assert registry.names == {"check_availability", "get_clinic_info", "find_bookings", "book_appointment",
                              "reschedule_appointment", "cancel_appointment"}


def test_every_tool_is_well_formed():
    for tool in default_tools():
        spec = tool.spec()
        function = spec["function"]
        assert spec["type"] == "function" and function["name"] == tool.name and function["description"]
        properties = function["parameters"]["properties"]
        assert function["parameters"]["required"] and set(function["parameters"]["required"]) <= set(properties)


def test_the_tool_definitions_stay_small(registry):
    # They are sent on every turn and hosted free tiers count tokens (about 4 characters each).
    assert len(json.dumps(registry.specs())) < 3800


def test_masked_arguments_hide_personal_details():
    masked = json.loads(mask_args({"caller_name": "Priya Shah", "reason": "my tooth hurts", "phone": "07700 900321",
                                   "slot_id": "A"}))
    assert masked == {"caller_name": "P.", "reason": "…", "phone": "…0321", "slot_id": "A"}
    assert json.loads(mask_args({"phone": "0 7 7 0 0 9 0 0 1 2 3"})) == {"phone": "…0123"}
    assert mask_args(None) == "{}"


# ---------- The checks that stop the LLM making things up ----------
def test_a_phone_number_must_be_one_the_caller_said():
    agent = agent_with(heard_phones=[PHONE])
    assert heard_phone("07700 900321", agent) == (PHONE, None)
    assert heard_phone("+44 7700 900321", agent) == (PHONE, None)              # the same number, written differently
    assert heard_phone("07700 111222", agent)[0] is None and "has not said" in heard_phone("07700 111222", agent)[1]
    assert "not a full phone number" in heard_phone("0770", agent)[1]
    assert heard_phone(None, agent)[0] is None


def test_a_slot_id_must_be_one_that_was_offered():
    agent = agent_with(**OFFERED)
    assert offered_slot({"slot_id": " b "}, agent) == ("2026-11-07T09:30:00+00:00", None)
    iso, error = offered_slot({"slot_id": "Z"}, agent)
    assert iso is None and "one of the times on offer (A, B)" in error
    assert "none yet" in offered_slot({"slot_id": "A"}, new_agent_state())[1]


def test_a_booking_id_must_be_one_that_was_found():
    agent = agent_with(known={"7": {"start": SAT_9.isoformat(), "reason": "x", "reference": "SD-000007"}})
    assert known_booking({"booking_id": 7}, agent) == ("7", None)
    assert known_booking({"booking_id": "7"}, agent) == ("7", None)
    assert "found for this caller" in known_booking({"booking_id": 8}, agent)[1]


# ---------- The registry ----------
def test_an_unknown_tool_is_refused_and_recorded(registry, db, call_id):
    result = registry.run("delete_everything", {}, new_agent_state(), call_id)
    assert not result.ok and "no tool called delete_everything" in result.error
    with db.session() as s:
        row = s.scalars(select(ToolCall)).one()
    assert (row.tool, row.ok, row.call_id) == ("delete_everything", False, call_id)


def test_a_tool_that_crashes_does_not_crash_the_call(ctx, call_id):
    class Broken(Tool):
        name, description = "broken", "Always fails."
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, args, agent, ctx):
            raise ValueError("boom")

    result = ToolRegistry(ctx, [Broken()]).run("broken", {}, new_agent_state(), call_id)
    assert not result.ok and "That tool failed" in result.error


def test_an_llm_outage_inside_a_tool_reaches_the_agent(ctx, call_id):
    class NeedsLlm(Tool):
        name, description = "needs_llm", "Calls the LLM."
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, args, agent, ctx):
            raise LLMError("network down")

    with pytest.raises(LLMError):
        ToolRegistry(ctx, [NeedsLlm()]).run("needs_llm", {}, new_agent_state(), call_id)


def test_a_new_tool_only_needs_a_class_and_the_registry(ctx, call_id):
    class Hello(Tool):
        name, description = "say_hello", "Says hello."
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, args, agent, ctx):
            return ToolResult(True, say="Hello!")

    registry = ToolRegistry(ctx, [Hello()])
    assert registry.names == {"say_hello"} and registry.specs()[0]["function"]["name"] == "say_hello"
    assert registry.run("say_hello", {}, new_agent_state(), call_id).say == "Hello!"


def test_every_call_is_counted_and_audited_with_masked_arguments(registry, db, call_id):
    ok, refused = monitoring.TOOL_CALLS.labels("find_bookings", "ok"), monitoring.TOOL_CALLS.labels("find_bookings", "refused")
    before = (ok._value.get(), refused._value.get())
    registry.run("find_bookings", {"phone": "07700 111222"}, agent_with(heard_phones=["07700111222"]), call_id)
    registry.run("find_bookings", {"phone": "07700 999888"}, new_agent_state(), call_id)           # never said: refused
    assert (ok._value.get() - before[0], refused._value.get() - before[1]) == (1, 1)
    with db.session() as s:
        rows = s.scalars(select(ToolCall).order_by(ToolCall.id)).all()
    assert [(r.tool, r.ok) for r in rows] == [("find_bookings", True), ("find_bookings", False)]
    assert all("07700" not in r.args and r.args.startswith('{"phone": "…') for r in rows)


# ---------- Each tool ----------
def test_check_availability_offers_ids_and_remembers_them(registry, call_id):
    result = registry.run("check_availability", {"when": "Saturday morning"}, new_agent_state(), call_id)
    assert result.ok and "nine o'clock, half past nine or ten o'clock in the morning" in result.say
    assert list(result.agent["offered"]) == ["A", "B", "C"] and result.turn["route"] == "tool_check_availability"
    assert result.agent["offered_text"]["B"] == "Saturday the seventh of November at half past nine in the morning"
    assert not result.chain and result.staged is None


def test_check_availability_falls_back_to_the_nearest_times(registry, call_id):
    result = registry.run("check_availability", {"when": "Sunday"}, new_agent_state(), call_id)
    assert result.say.startswith("I'm sorry, I don't have anything free then.")


def test_find_bookings_remembers_what_it_found_and_lets_the_llm_continue(registry, bookings, db, call_id):
    info = bookings.create(call_id, {"caller_name": "A", "reason": "check-up", "phone": PHONE}, SAT_9)
    result = registry.run("find_bookings", {"phone": PHONE}, agent_with(heard_phones=[PHONE]), call_id)
    assert result.ok and result.chain and "S D 0 0 0 0 0 1" in result.say
    assert str(info.id) in result.agent["known"] and result.data["bookings"][0]["booking_id"] == info.id
    nothing = registry.run("find_bookings", {"phone": "07700111222"}, agent_with(heard_phones=["07700111222"]), call_id)
    assert nothing.say == prompts.MANAGE_NOT_FOUND and not nothing.chain and nothing.agent["known"] == {}


def test_book_appointment_only_stages_the_booking(registry, bookings, db, call_id):
    agent = agent_with(heard_phones=[PHONE], **OFFERED)
    result = registry.run("book_appointment", {"caller_name": "my name is priya shah", "reason": "a check-up",
                                               "slot_id": "A", "phone": PHONE}, agent, call_id)
    assert result.ok and result.staged == {"kind": "book", "caller_name": "Priya Shah", "reason": "a check-up",
                                           "phone": PHONE, "start": SAT_9.isoformat(), "said_time": "Saturday at nine"}
    assert "Your name is Priya Shah" in result.say and result.say.endswith("Shall I book that for you?")
    assert bookings.upcoming_for_phone(PHONE) == []                 # nothing was saved


@pytest.mark.parametrize("args, complaint", [
    ({"caller_name": "", "reason": "x", "slot_id": "A", "phone": PHONE}, "full name is missing"),
    ({"caller_name": "Priya", "reason": " ", "slot_id": "A", "phone": PHONE}, "reason for the visit is missing"),
    ({"caller_name": "Priya", "reason": "x", "slot_id": "A", "phone": "07700111222"}, "has not said that number"),
    ({"caller_name": "Priya", "reason": "x", "slot_id": "Q", "phone": PHONE}, "slot_id must be one of"),
])
def test_book_appointment_says_exactly_what_is_missing(registry, call_id, args, complaint):
    result = registry.run("book_appointment", args, agent_with(heard_phones=[PHONE], **OFFERED), call_id)
    assert not result.ok and result.staged is None and complaint in result.error


def test_reschedule_and_cancel_only_stage_a_change(registry, call_id):
    agent = agent_with(known={"7": {"start": SAT_9.isoformat(), "reason": "x", "reference": "SD-000007"}}, **OFFERED)
    moved = registry.run("reschedule_appointment", {"booking_id": 7, "slot_id": "B"}, agent, call_id)
    assert moved.staged == {"kind": "reschedule", "booking_id": 7, "start": "2026-11-07T09:30:00+00:00"}
    cancelled = registry.run("cancel_appointment", {"booking_id": 7}, agent, call_id)
    assert cancelled.staged == {"kind": "cancel", "booking_id": 7} and "cancel it?" in cancelled.say
    assert not registry.run("cancel_appointment", {"booking_id": 8}, agent, call_id).ok
    assert not registry.run("reschedule_appointment", {"booking_id": 7, "slot_id": "Z"}, agent, call_id).ok


def test_a_late_cancellation_mentions_the_fee(registry, clock, call_id):
    agent = agent_with(known={"7": {"start": SAT_9.isoformat(), "reason": "x", "reference": "SD-000007"}})
    assert "twenty four hours" not in registry.run("cancel_appointment", {"booking_id": 7}, agent, call_id).say
    clock.now = datetime(2026, 11, 7, 6, 0, tzinfo=timezone.utc)            # three hours before
    assert "twenty four hours" in registry.run("cancel_appointment", {"booking_id": 7}, agent, call_id).say


def test_get_clinic_info_answers_from_the_documents(registry, call_id):
    result = registry.run("get_clinic_info", {"question": "What time do you open on Saturday?"}, new_agent_state(), call_id)
    assert result.ok and "Saturday" in result.say and result.turn["route"] == "faq_answered"
    assert result.turn["similarity_score"] >= 0.25 and result.turn["faq_titles"][0] == "Opening hours"


def test_get_clinic_info_does_not_ask_the_llm_when_the_documents_have_no_answer(ctx, call_id):
    class NeverCalled(FakeLLM):
        def chat(self, messages):
            raise AssertionError("the LLM must not be asked")

    ctx.llm = NeverCalled()
    result = ToolRegistry(ctx).run("get_clinic_info", {"question": "Do you sell train tickets to London?"}, new_agent_state(), call_id)
    assert result.say == prompts.FALLBACK and result.turn["route"] == "unanswered"
    assert not ToolRegistry(ctx).run("get_clinic_info", {"question": "  "}, new_agent_state(), call_id).ok


def test_get_clinic_info_blocks_advice_in_the_llms_answer(ctx, call_id):
    ctx.llm = FakeLLM(reply="You should take 400mg of ibuprofen.")
    result = ToolRegistry(ctx).run("get_clinic_info", {"question": "What time do you open on Saturday?"}, new_agent_state(), call_id)
    assert result.say == prompts.NO_ADVICE and result.turn["route"] == "advice_blocked"
