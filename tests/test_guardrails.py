"""The safety rules: emergencies, handoff and advice before the LLM, and checks on what the LLM wrote.

These are the tests that matter most. If one of them fails, a caller in trouble might not be told to call 999.
"""
import pytest

from voice_agent.agent import guardrails, rules


# ---------- Normalising ----------
def test_normalise_lowercases_trims_and_straightens_apostrophes():
    assert guardrails.normalise("  I CAN’T   Breathe!! ") == "i can't breathe"


def test_curly_and_missing_apostrophes_still_match():
    assert guardrails.is_severe_emergency("I can’t breathe")
    assert guardrails.is_severe_emergency("i cant breath")


# ---------- Severe emergencies (999) ----------
@pytest.mark.parametrize("text", [
    "I can't breathe", "I cant breath", "my son is having trouble breathing", "I can't brite",  # Whisper mishearing
    "it's hard to swallow", "my throat is swelling up", "my eye is swollen shut",
    "the bleeding won't stop", "there's a lot of blood", "my husband is unconscious", "she passed out",
    "I have chest pain", "I think it's an allergic reaction",
])
def test_severe_emergencies_are_caught(text):
    assert guardrails.is_severe_emergency(text)
    assert rules.decide_intent(text) == "severe"


# ---------- Urgent dental ----------
@pytest.mark.parametrize("text", [
    "my face is really swollen", "my tooth got knocked out", "my son knocked his front tooth out", "I broke my tooth", "I've cracked a molar",
    "I think I have an abscess", "I'm in severe pain", "my gum is bleeding", "the pain is unbearable",
    "I have a really bad toothache",
])
def test_urgent_dental_is_caught(text):
    assert not guardrails.is_severe_emergency(text)
    assert rules.decide_intent(text) == "urgent"


@pytest.mark.parametrize("text", ["I have bad breath, can I book a cleaning?", "Do you treat bad breath?", "Chester Smith"])
def test_ordinary_words_are_not_emergencies(text):
    assert not guardrails.is_severe_emergency(text)
    assert rules.decide_intent(text) == "agent"         # an ordinary request: the tool agent handles it


def test_severe_wins_over_urgent():
    # Swelling near the eye is severe even though "swollen" alone is only urgent.
    assert rules.decide_intent("my face and eye are swollen") == "severe"


def test_emergency_beats_a_booking_waiting_for_yes():
    assert rules.decide_intent("I can't breathe", agent_pending=True) == "severe"


@pytest.mark.parametrize("text", ["this is a question about prices", "do you have parking", "what time is it",
                                  "I have a toothache, can I book in", "a brief question please"])
def test_ordinary_questions_are_not_emergencies(text):
    assert not guardrails.is_severe_emergency(text)


# ---------- Handoff ----------
@pytest.mark.parametrize("text", ["can I speak to a person", "I want a real person", "put me through to reception",
                                  "talk to someone please", "are you human"])
def test_handoff(text):
    assert guardrails.wants_human(text) or rules.decide_intent(text) == "handoff"


def test_handoff_phrase_routes_to_handoff():
    assert rules.decide_intent("can I speak to a person please") == "handoff"


# ---------- Advice ----------
@pytest.mark.parametrize("text", ["what should I do about my toothache", "should I take ibuprofen",
                                  "do you think I need antibiotics"])
def test_advice_requests(text):
    assert rules.decide_intent(text) == "advice"


def test_clinic_logistics_are_not_advice():
    assert not guardrails.asks_for_advice("what should I do to cancel my appointment")


def test_advice_in_llm_output_is_detected():
    assert guardrails.looks_like_advice("You should take 400mg of ibuprofen.")
    assert guardrails.looks_like_advice("It sounds like you have an infection.")
    assert not guardrails.looks_like_advice("We are open Monday to Friday, 8am to 6pm.")




def test_greeting_is_never_an_emergency():
    for greeting in ["hi", "hello", "hey", "this is"]:
        assert not guardrails.is_severe_emergency(greeting)
        assert not guardrails.is_urgent_dental(greeting)


# ---------- Checks on what the LLM wrote ----------
@pytest.mark.parametrize("reply", ["We open at 9:30", "A check-up is 45 pounds", "How about Saturday at 9 am", "half past nine",
                                   "That costs £60", "see you at ten o'clock", "quarter to four", "$20",
                                   "We are located at 12 High Street, Springfield.", "It's 24 Harbour Road, near the post office",
                                   "You can find us at BW1 2AB", "We're based in the town centre", "Call us on the main line",
                                   "Email us at info@sunrise.example"])
def test_a_time_or_price_in_the_llms_own_words_is_flagged(reply):
    assert guardrails.states_unverified_fact(reply)


@pytest.mark.parametrize("reason,refused", [
    ("fever", True), ("I have a high temperature", True), ("flu", True), ("stomach ache", True), ("back pain", True),
    ("a check-up", False), ("toothache", False), ("fever from a tooth abscess", False), ("sensitive gums", False),
    ("scale and polish", False), ("", False)])
def test_a_plainly_non_dental_reason_is_recognised(reason, refused):
    assert guardrails.is_not_dental(reason) is refused


@pytest.mark.parametrize("reply", ["What is your full name?", "Which day suits you?", "We have free parking behind the building.",
                                   "I'm sorry, I don't have that information.", "Thank you, goodbye."])
def test_ordinary_replies_are_not_flagged(reply):
    assert not guardrails.states_unverified_fact(reply)


# ---------- The order of the checks is the safety design ----------
@pytest.mark.parametrize("agent_pending", [False, True])
def test_emergencies_come_before_everything_else(agent_pending):
    for text, expected in [("I can't breathe", "severe"), ("my face is really swollen", "urgent"),
                           ("let me speak to a real person", "handoff")]:
        assert rules.decide_intent(text, agent_pending=agent_pending) == expected


def test_a_change_waiting_for_yes_does_not_hide_an_emergency():
    assert rules.decide_intent("yes but I can't breathe", agent_pending=True) == "severe"
