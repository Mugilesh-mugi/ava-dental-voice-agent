"""Rules about the CONVERSATION (safety is in guardrails.py): which branch handles a message (decide_intent), plus
greetings, thanks, yes/no, phone numbers and names.

Pure functions: they only look at text. No models, no network, no database.
"""
from __future__ import annotations

import re

from . import guardrails
from .guardrails import normalise, prepare


def decide_intent(text: str, agent_pending: bool = False) -> str:
    """Which branch handles this message. The ORDER is the safety design: emergencies first, before any LLM.

    Returns one of: severe, urgent, handoff, agent_confirm, thanks, greeting, advice, unclear, agent.
    Small talk and refusals are fixed replies (no LLM request used); everything else goes to the tool agent.
    """
    if guardrails.is_severe_emergency(text):
        return "severe"
    if guardrails.is_urgent_dental(text):
        return "urgent"
    if guardrails.wants_human(text):
        return "handoff"
    if agent_pending:
        return "agent_confirm"         # the caller is answering "shall I book that?"
    if is_thanks(text):
        return "thanks"
    if is_greeting(text):
        return "greeting"
    if guardrails.asks_for_advice(text):
        return "advice"
    if is_noise(text):
        return "unclear"
    return "agent"


# --- Greetings. Matched as WHOLE words only. Never add these to an emergency list. ---
GREETING_WORDS = {"hi", "hello", "hey", "hiya", "morning", "afternoon", "evening", "good", "there",
                  "ava", "yo", "howdy", "greetings"}


# Small filler words that may appear around a greeting ("um hello there").
FILLER_WORDS = {"um", "uh", "erm", "er", "ah", "oh", "so", "yes", "yeah", "ok", "okay", "please", "thanks",
                "thank", "you"}


THANKS_PATTERNS = [
    r"^(?:\s*(?:oh|um|uh|erm|er|ah|so|yes|yeah|ok|okay|please)\s+)?thanks?\s*(?:you)?\s*$",
    r"^(?:\s*(?:oh|um|uh|erm|er|ah|so|yes|yeah|ok|okay|please)\s+)?thank\s+you(?:\s+very\s+much)?\s*$",
    r"^(?:\s*(?:oh|um|uh|erm|er|ah|so|yes|yeah|ok|okay|please)\s+)?many\s+thanks\s*$",
]


NUMBER_WORDS = {"zero": "0", "oh": "0", "o": "0", "nought": "0", "one": "1", "two": "2", "three": "3",
                "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}


NAME_PREFIXES = ["my name is", "my name's", "the name is", "name is", "it's", "it is", "i'm", "i am",
                 "this is", "call me"]


# Call outcomes, from most to least important. A call keeps its most important outcome.
OUTCOME_PRIORITY = ["emergency_999", "urgent_dental", "handed_off", "booking_saved", "booking_changed",
                    "booking_cancelled", "answered", "abandoned"]


def is_thanks(text: str) -> bool:
    """True for short gratitude phrases like "thank you" or "thanks"."""
    t = normalise(text)
    return any(re.fullmatch(p, t) for p in THANKS_PATTERNS)


def _squash(word: str) -> str:
    """Collapse repeated letters, so stretched words match: "hii" -> "hi", "heyyy" -> "hey"."""
    return re.sub(r"(.)\1+", r"\1", word)


_GREETING_SQUASHED = {_squash(w) for w in GREETING_WORDS}


_REAL_GREETING_SQUASHED = {_squash(w) for w in GREETING_WORDS - {"good", "there"}}


_FILLER_SQUASHED = {_squash(w) for w in FILLER_WORDS}


def is_greeting(text: str) -> bool:
    """True only if the message is JUST a greeting ("hi", "hii", "good morning Ava"), not "hi, what time do you open?"."""
    words = [_squash(w) for w in normalise(text).split()]
    if not words or not any(w in _REAL_GREETING_SQUASHED for w in words):
        return False
    return all(w in _GREETING_SQUASHED or w in _FILLER_SQUASHED for w in words)


_YES = re.compile(r"^(?:yes|yeah|yep|yup|sure|ok|okay|correct|right|confirm|confirmed|absolutely|definitely|perfect|"
                  r"fine|great|please do|go ahead|that's right|that is right|that's correct|sounds good|that works|"
                  r"do it|book it|no problem)\b")


_NO = re.compile(r"^(?:no|nope|nah|not really|wrong|incorrect|don't|do not|never mind|nevermind|change|different)\b")


def is_yes(text: str) -> bool:
    return bool(_YES.search(prepare(text)))


def is_no(text: str) -> bool:
    t = prepare(text)
    return bool(_NO.search(t)) and not _YES.search(t)


AGENT_NOISE_WORDS = {"um", "umm", "uh", "uhh", "erm", "er", "hmm", "hm", "mm", "mhm", "ah", "oh", "huh", "eh"}


def is_noise(text: str) -> bool:
    """Nothing the LLM could use: silence turned into text, or only "um" and "hmm". Short real messages such as
    "Tuesday morning", "Priya" or a phone number are NOT noise: the tool agent understands them in context."""
    words = normalise(text).split()
    has_content = any(re.search(r"[a-z]{2,}|[0-9]", w) for w in words)       # digits count: a phone number, "10"
    return not has_content or all(w in AGENT_NOISE_WORDS for w in words)


def extract_phone(text: str) -> str | None:
    """Pull a UK-style phone number (10 or 11 digits) out of text. Understands spoken digits
    ("oh seven seven double one ..."). Returns the digits, or None if there are not enough."""
    words = normalise(text).split()
    digits, repeat = [], 1
    for w in words:
        if w in ("double", "triple"):
            repeat = 2 if w == "double" else 3
            continue
        if w.isdigit():
            digits.append(w * repeat if len(w) == 1 else w)
        elif w in NUMBER_WORDS:
            digits.append(NUMBER_WORDS[w] * repeat)
        repeat = 1
    number = "".join(digits)
    if number.startswith("44") and len(number) == 12:
        number = "0" + number[2:]
    return number if 10 <= len(number) <= 11 else None


# "my name is X" anywhere in a message, e.g. "sorry, my name is Mugilesh". Longest phrases first.
_NAME_CORRECTION = re.compile(r"\b(?:my name is|my name's|the name is|name is|call me|my name)\s+(.+)$")


# Words people say before or after their name that are not part of it.
NAME_LEAD_WORDS = {"sorry", "oh", "um", "uh", "erm", "er", "ok", "okay", "so", "yes", "yeah", "no", "actually",
                   "well", "sure", "hi", "hello"}


NAME_TRAIL_WORDS = {"please", "thanks", "thank", "you", "sorry"}


def _tidy_name(t: str) -> str:
    t = re.split(r"\s+not\s+", t)[0]                  # "mugilesh not illish" -> "mugilesh"
    words = t.split()
    while words and words[0] in NAME_LEAD_WORDS:
        words.pop(0)
    while words and words[-1] in NAME_TRAIL_WORDS:
        words.pop()
    return " ".join(w.capitalize() for w in words[:4])


def name_correction(text: str) -> str | None:
    """The name, if the caller says "my name is ..." anywhere in the message. None otherwise."""
    m = _NAME_CORRECTION.search(normalise(text))
    return (_tidy_name(m.group(1)) or None) if m else None


def clean_name(text: str) -> str:
    """Remove lead-in words like "my name is" or "sorry," and keep the name."""
    corrected = name_correction(text)
    if corrected:
        return corrected
    t = normalise(text)
    for prefix in NAME_PREFIXES:
        if t.startswith(prefix + " "):
            t = t[len(prefix) + 1:]
            break
    return _tidy_name(t)


def clean_answer(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip(" .")


def merge_outcome(current: str | None, new: str | None) -> str | None:
    """Keep whichever outcome is more important (an emergency beats a booking, and so on)."""
    if new is None:
        return current
    if current is None:
        return new
    return min(current, new, key=OUTCOME_PRIORITY.index)
