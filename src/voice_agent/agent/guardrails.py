"""Safety rules that run before and after the LLM.

BEFORE the LLM (on what the caller said), in this order, so safety never depends on the LLM behaving:
    1. severe emergency   -> "call 999"
    2. urgent dental      -> the urgent line (and NHS 111 if closed)
    3. asks for a human   -> hand over
    4. asks for advice    -> "I can't give dental or medical advice"
AFTER the LLM (on what it wrote):
    - advice in its words is blocked            (looks_like_advice)
    - a time or price in its own words is not   (states_unverified_fact): those must come from a tool

Pure functions: they only look at text. No models, no network, no database. That makes them fast, predictable and
easy to test, and the tests in tests/test_guardrails.py are the proof that no emergency phrase is missed.

How matching works
------------------
1. The text is "normalised": lowercase, trimmed, curly apostrophes made straight, punctuation removed.
2. Each word list holds lowercase STEMS (the start of a word). A stem matches at the start of any word:
   "breath" matches "breathe" and "breathing", but never the middle of a word.
3. Whisper (speech-to-text) can mishear. It once wrote "brite" for "can't breathe". So the emergency lists are
   deliberately broad and include known mishearings. A false alarm is acceptable. A missed emergency is not.
"""
from __future__ import annotations

import re

# ======================================================================
# Word lists (all lowercase)
# ======================================================================

# --- Tier 1: SEVERE emergency -> tell the caller to call 999. ---
# Trouble breathing or swallowing, swelling near the throat or eye, bleeding that
# will not stop, unconscious, chest pain.
SEVERE_STEMS = [
    # breathing (plus mishearings Whisper has produced, like "brite").
    # Not the bare stem "breath": "bad breath" is a normal dental question, not an emergency.
    "breathe", "breathing", "breathless", "can't breath", "cannot breath", "short of breath",
    "shortness of breath", "out of breath", "breth", "breeth", "brite", "suffocat", "choking", "choke",
    "wheez", "gasping",
    # swallowing
    "swallow", "swolow",
    # unconscious / collapse
    "unconscious", "unconcious", "passed out", "pass out", "passing out", "fainted", "fainting",
    "collapsed", "unresponsive", "not responding", "not waking", "won't wake", "can't wake",
    # heart ("chest" is in SEVERE_WHOLE_WORDS below)
    "heart attack",
    # serious allergic reaction
    "anaphyla", "allergic reaction",
]
# Severe words that must match as a WHOLE word, so the name "Chester" does not count as "chest".
SEVERE_WHOLE_WORDS = ["chest"]
# Swelling only counts as severe when it is near the throat, neck, or eye.
SWELLING_STEMS = ["swell", "swollen", "swolen", "puff"]
SWELLING_DANGER_AREAS = ["throat", "neck", "eye", "tongue"]
# Bleeding only counts as severe when it will not stop or is heavy.
BLEEDING_STEMS = ["bleed", "blood", "bled"]
BLEEDING_SEVERE_WORDS = ["won't stop", "wont stop", "will not stop", "not stopping", "doesn't stop",
                         "does not stop", "can't stop", "cannot stop", "stop bleeding", "nonstop", "non stop",
                         "heav", "lot of blood", "lots of blood", "pouring", "gushing", "everywhere"]

# --- Tier 2: URGENT dental -> give the urgent line (and 111 if closed). ---
# Swollen face, knocked-out tooth, broken tooth, abscess, severe pain, bleeding.
URGENT_STEMS = [
    "swell", "swollen", "swolen",
    "knock", "tooth fell out", "tooth came out", "lost a tooth", "lost my tooth",
    "brok", "crack", "chipped", "snapped",          # "brok" also catches misspellings like "brokn"
    "abscess", "absess", "abcess", "abses", "infect",
    "bleed", "blood", "bled",
    "agony", "unbearable", "excruciating", "can't sleep", "cannot sleep",
]
# "pain" on its own is common ("I have a toothache, can I book?"), so it is only urgent
# with a strong word next to it.
PAIN_STEMS = ["pain", "ache", "toothache", "tooth ache", "aching", "hurt", "throb", "sore"]
PAIN_SEVERE_WORDS = ["severe", "really bad", "very bad", "terrible", "awful", "worst", "intense", "extreme",
                     "killing me", "so much"]

# --- Handoff: the caller asks for a human. ---
HANDOFF_STEMS = [
    "speak to a person", "speak to someone", "speak to somebody", "speak to a human", "speak with someone",
    "speak with a person", "talk to a person", "talk to someone", "talk to somebody", "talk to a human",
    "real person", "actual person", "human", "receptionist", "operator", "member of staff", "staff member",
    "put me through", "transfer me",
]

# --- Requests for advice (Ava never gives dental or medical advice) ---
ADVICE_STEMS = ["what should i do", "what can i do about", "should i take", "what can i take",
                "what should i take", "what medicine", "which medicine", "painkiller", "antibiotic",
                "is it normal", "is this normal", "what's wrong", "what is wrong", "is it serious",
                "diagnos", "how do i treat", "how to treat", "home remed", "ibuprofen", "paracetamol",
                "what could it be", "what might it be"]
# ...unless the question is clearly about the clinic itself ("what should I do to cancel?").
ADVICE_EXCEPTION_STEMS = ["cancel", "book", "appointment", "pay", "park", "insurance", "register", "open"]

# Words or patterns that should never appear in an LLM answer (they look like advice).
ADVICE_OUTPUT_PATTERNS = [
    r"\b\d+\s?(mg|ml|milligrams?)\b", r"\byou should take\b", r"\btry taking\b", r"\bibuprofen\b",
    r"\bparacetamol\b", r"\bantibiotics?\b", r"\bpainkillers?\b", r"\bsounds like (you have|an?|it'?s)\b",
    r"\bprobably (an?|a case of)\b", r"\brinse with\b",
]


# ======================================================================
# Helpers
# ======================================================================

def normalise(text: str | None) -> str:
    """Lowercase, trim, straighten curly apostrophes, drop punctuation, squash spaces."""
    if not text:
        return ""
    text = text.lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compile_stems(stems: list[str]) -> re.Pattern:
    # \b at the start only: the stem must begin a word, but may continue ("breath" -> "breathing").
    # Spaces inside a phrase may be one or more spaces.
    parts = [r"\s+".join(map(re.escape, s.split())) for s in stems]
    return re.compile(r"\b(?:" + "|".join(parts) + ")")


_SEVERE = compile_stems(SEVERE_STEMS)
_SEVERE_WHOLE = re.compile(r"\b(?:" + "|".join(map(re.escape, SEVERE_WHOLE_WORDS)) + r")\b")
_SWELLING = compile_stems(SWELLING_STEMS)
_SWELLING_AREA = compile_stems(SWELLING_DANGER_AREAS)
_BLEEDING = compile_stems(BLEEDING_STEMS)
_BLEEDING_SEVERE = compile_stems(BLEEDING_SEVERE_WORDS)
_URGENT = compile_stems(URGENT_STEMS)
_PAIN = compile_stems(PAIN_STEMS)
_PAIN_SEVERE = compile_stems(PAIN_SEVERE_WORDS)
_HANDOFF = compile_stems(HANDOFF_STEMS)
_ADVICE = compile_stems(ADVICE_STEMS)
_ADVICE_EXCEPTION = compile_stems(ADVICE_EXCEPTION_STEMS)
_ADVICE_OUTPUT = [re.compile(p, re.IGNORECASE) for p in ADVICE_OUTPUT_PATTERNS]


def prepare(text: str) -> str:
    """normalise(), then add back apostrophes people leave out ("cant" -> "can't") so stems still match."""
    t = normalise(text)
    return re.sub(r"\b(can|won|don|doesn|isn|didn|couldn)t\b", r"\1't", t)


# ======================================================================
# Checks on what the caller said
# ======================================================================

def is_severe_emergency(text: str) -> bool:
    t = prepare(text)
    if _SEVERE.search(t) or _SEVERE_WHOLE.search(t):
        return True
    if _SWELLING.search(t) and _SWELLING_AREA.search(t):
        return True
    return bool(_BLEEDING.search(t) and _BLEEDING_SEVERE.search(t))


def is_urgent_dental(text: str) -> bool:
    t = prepare(text)
    if _URGENT.search(t):
        return True
    return bool(_PAIN.search(t) and _PAIN_SEVERE.search(t))


def wants_human(text: str) -> bool:
    return bool(_HANDOFF.search(prepare(text)))


def asks_for_advice(text: str) -> bool:
    t = prepare(text)
    return bool(_ADVICE.search(t)) and not _ADVICE_EXCEPTION.search(t)


_BOOKING_REQUEST = re.compile(
    r"\b(?:(?:want|need|like|wanna|looking|trying)\b.{0,15}\b(?:book|appointment|schedule|reserve)"
    r"|(?:can|could|may) (?:i|you) (?:please )?(?:book|make|schedule)|book (?:me|an?|my|us)\b|make (?:me )?an? appointment)")


def wants_to_book(text: str) -> bool:
    """The caller is asking to make an appointment (not asking a question about appointments)."""
    return bool(_BOOKING_REQUEST.search(prepare(text)))


def looks_like_advice(llm_reply: str) -> bool:
    """Safety net for LLM output: True if the reply seems to give dental or medical advice."""
    return any(p.search(llm_reply) for p in _ADVICE_OUTPUT)


# ======================================================================
# Checks on what the LLM wrote
# ======================================================================

# A time, a price or an address has to come from a tool (the calendar, the FAQ). If the LLM writes one itself, it is guessing.
_STREETS = r"(?:street|st|road|rd|lane|ln|avenue|ave|drive|close|court|way|place|square|crescent|terrace|gardens|hill)"
_UNVERIFIED_FACT = re.compile(
    r"\d{1,2}:\d{2}|\b\d{1,2}\s?(?:am|pm)\b|o'clock|half past|quarter (?:past|to)|"
    r"[£$]\s?\d|\b\d+\s?pounds?\b|\bpounds?\s?\d|"
    rf"\b\d{{1,4}}[a-z]?\s+(?:[a-z']+\s+){{0,3}}{_STREETS}\b|"                  # "12 High Street"
    r"\b[a-z]{1,2}\d[a-z\d]?\s?\d[a-z]{2}\b|"                                   # a UK postcode, "BW1 2AB"
    r"\b(?:we(?:'re| are) (?:located|based|found)|our address|you can find us|find us at)\b|"
    r"\b(?:call|ring|phone|email|contact) us (?:on|at)\b|\bour (?:phone )?number is\b|\w@\w+\.\w", re.IGNORECASE)


# A visit reason that is plainly not dental ("fever"). Conservative on purpose: a dental word anywhere in the reason
# ("fever from a tooth abscess", "sensitive to cold") means the clinic may well be the right place, so it is not refused.
_NOT_DENTAL = re.compile(r"\b(?:fever|flu|influenza|covid|cough(?:ing)?|high temperature|stomach|vomit(?:ing)?|nausea|"
                         r"diarrh?oea|back pain|knee|rash|migraine|period pain|pregnan\w*)\b", re.IGNORECASE)
_DENTAL = re.compile(r"\b(?:teeth|tooth|toothache|gums?|jaw|mouth|dental|dentist|filling|crown|denture|braces|hygien\w*|"
                     r"check-?up|clean(?:ing)?|whitening|scale|polish|x-?ray|extraction|wisdom|root canal|cavity|abscess|"
                     r"orthodont\w*|bite|smile|lip|tongue)\b", re.IGNORECASE)


def is_not_dental(reason: str) -> bool:
    """True if the reason for a visit is plainly something a dental clinic does not treat."""
    return bool(_NOT_DENTAL.search(reason)) and not _DENTAL.search(reason)


_REFUSAL = re.compile(r"\bcan(?:'|no)?t (?:share|give|provide|tell|disclose|help|say|answer)|\bunable to\b|\bnot able to\b|"
                      r"\bnot (?:allowed|permitted)\b|\bdon'?t have (?:that|this)\b|\bcan(?:'|no)?t do that\b", re.IGNORECASE)


def refuses(llm_reply: str) -> bool:
    """True if the LLM is declining to answer ("I can't share that"). For a clinic question that is wrong: the clinic's
    own phone number, address and hours are public, and the documents have them."""
    return bool(_REFUSAL.search(llm_reply))


def states_unverified_fact(llm_reply: str) -> bool:
    """True if the LLM wrote a clock time, a price, an address or a contact detail in its own words."""
    return bool(_UNVERIFIED_FACT.search(llm_reply))
