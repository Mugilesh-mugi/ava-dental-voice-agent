"""Conversation rules: which branch handles a message, greetings, thanks, noise, yes/no, phone numbers, names, outcomes."""
import pytest

from voice_agent.agent import rules


# ---------- Which branch handles a message ----------
@pytest.mark.parametrize("text, intent", [
    ("I can't breathe", "severe"), ("my face is really swollen", "urgent"), ("can I speak to a real person", "handoff"),
    ("hello", "greeting"), ("thank you", "thanks"), ("what should I do about my toothache", "advice"),
    ("uh", "unclear"), ("hmm", "unclear"),
    ("I'd like to book an appointment", "agent"), ("what time do you open on Saturday", "agent"),
    ("cancel my appointment", "agent"), ("Priya", "agent"), ("Tuesday morning", "agent"), ("07700 900123", "agent"),
])
def test_routing(text, intent):
    assert rules.decide_intent(text) == intent


def test_a_change_waiting_for_yes_takes_the_next_answer():
    assert rules.decide_intent("yes please", agent_pending=True) == "agent_confirm"
    assert rules.decide_intent("no", agent_pending=True) == "agent_confirm"
    assert rules.decide_intent("I can't breathe", agent_pending=True) == "severe"        # but emergencies still come first


# ---------- Greetings and thanks ----------
@pytest.mark.parametrize("text", ["hi", "Hello!", "hey there", "good morning", "Hiya Ava", "um hello", "hii", "heyyy", "helloo"])
def test_greetings(text):
    assert rules.is_greeting(text)
    assert rules.decide_intent(text) == "greeting"


@pytest.mark.parametrize("text", ["this", "hi, what time do you open?", "good"])
def test_not_greetings(text):
    # "hi" must not match inside "this"; a greeting followed by a question is a question.
    assert not rules.is_greeting(text)


@pytest.mark.parametrize("text", ["thank you", "thanks", "ok thanks", "thank you very much", "many thanks"])
def test_thanks(text):
    assert rules.is_thanks(text) and rules.decide_intent(text) == "thanks"


# ---------- Noise ----------
@pytest.mark.parametrize("noise", ["", "   ", "hmm", "uh", "um erm", "...", "?!"])
def test_noise_is_unclear(noise):
    assert rules.is_noise(noise) and rules.decide_intent(noise) == "unclear"


@pytest.mark.parametrize("text", ["Priya", "Tuesday morning", "10", "07700 900321", "yes", "what", "no"])
def test_short_real_messages_are_not_noise(text):
    assert not rules.is_noise(text)                 # the tool agent understands them in context


# ---------- Yes and no ----------
@pytest.mark.parametrize("text, yes, no", [
    ("yes", True, False), ("yes please", True, False), ("yeah that's right", True, False), ("go ahead", True, False),
    ("no problem", True, False), ("no", False, True), ("nope", False, True), ("no thanks", False, True),
    ("not really", False, True), ("maybe", False, False),
])
def test_yes_and_no(text, yes, no):
    assert (rules.is_yes(text), rules.is_no(text)) == (yes, no)


# ---------- Phone numbers and names ----------
@pytest.mark.parametrize("text,expected", [
    ("07700 900123", "07700900123"),
    ("my number is 0 7 7 0 0 9 0 0 1 2 3", "07700900123"),
    ("oh seven seven double oh nine oh oh one two three", "07700900123"),
    ("+44 7700 900123", "07700900123"),
    ("I don't know", None),
    ("12345", None),
    # a time or date said earlier in the same message must not be glued onto the number
    ("book an appointment at monday 8, my name is Sam and my mobile number is 1234567899", "1234567899"),
    ("book monday at 8 my number is 07700900123", "07700900123"),
    ("on the 12th, my number is 07700 900123", "07700900123"),
])
def test_extract_phone(text, expected):
    assert rules.extract_phone(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("my name is jane doe", "Jane Doe"), ("Jane", "Jane"), ("sorry, Mugilesh", "Mugilesh"),
    ("sorry my name is mugilesh", "Mugilesh"), ("my name is mugilesh not illish", "Mugilesh"),
    ("actually call me Sam please", "Sam"), ("no, my name is Jane Doe", "Jane Doe"),
])
def test_clean_name(text, expected):
    assert rules.clean_name(text) == expected


def test_clean_answer_tidies_spaces_and_full_stops():
    assert rules.clean_answer("  a   check-up. ") == "a check-up"
    assert rules.clean_answer(None) == ""


# ---------- Outcomes ----------
def test_more_important_outcome_wins():
    assert rules.merge_outcome("answered", "emergency_999") == "emergency_999"
    assert rules.merge_outcome("emergency_999", "booking_saved") == "emergency_999"
    assert rules.merge_outcome(None, "answered") == "answered"
    assert rules.merge_outcome("answered", None) == "answered"
