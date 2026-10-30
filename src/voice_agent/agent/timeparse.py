"""Understand WHEN the caller wants an appointment: "Saturday morning", "tomorrow at 3pm", "the 12th".

Pure functions: text in, a TimeRequest out. No database, no clock (today's date is passed in), so they
are easy to test. Rules rather than an LLM, because dates and times must be exact and a small local
LLM is not reliable at them. Whisper usually writes times as digits ("3 p.m.", "10:30"), which is
what this expects, plus a few spoken forms ("half past ten", "three o'clock").
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Sequence

WEEKDAYS = {"monday": 0, "tuesday": 1, "tues": 1, "wednesday": 2, "weds": 2, "thursday": 3, "thurs": 3,
            "friday": 4, "saturday": 5, "sunday": 6}
MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
          "september": 9, "sept": 9, "october": 10, "november": 11, "december": 12,
          "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
          "nov": 11, "dec": 12}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
# Parts of the day, as (earliest slot start, latest slot start + 1 minute).
PARTS_OF_DAY = {"morning": (time(0), time(12)), "afternoon": (time(12), time(17)), "evening": (time(17), time(23, 59)),
                "lunchtime": (time(12), time(14)), "lunch": (time(12), time(14))}
ASAP_PHRASES = ("asap", "as soon as", "earliest", "soonest", "first available", "next available", "any time",
                "anytime", "whenever", "any day", "next free")

_NUM = "|".join(NUMBER_WORDS)
_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
_DAY = "|".join(WEEKDAYS)
START_OF_DAY, END_OF_DAY = time(0), time(23, 59)


@dataclass(frozen=True)
class TimeRequest:
    """What the caller asked for. Empty `days` means "any day"."""

    days: tuple[date, ...] = ()
    start: time = START_OF_DAY         # earliest acceptable slot start
    end: time = END_OF_DAY             # slots must start before this
    exact: time | None = None          # a specific time was named, so prefer slots closest to it
    asap: bool = False                 # "as soon as possible", "earliest"

    @property
    def has_window(self) -> bool:
        return self.start != START_OF_DAY or self.end != END_OF_DAY


def _prep(text: str) -> str:
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", t)                   # "p.m." -> "pm"
    t = re.sub(r"\b(\d{1,2}(?::\d{2})?)\s+([ap])\s?m\b", r"\1\2m", t)    # "3 p m" -> "3pm"
    t = re.sub(r"[^a-z0-9:./'\s-]", " ", t)
    t = re.sub(rf"\bhalf past ({_NUM})\b", lambda m: f"{NUMBER_WORDS[m.group(1)]}:30", t)
    t = re.sub(rf"\bquarter past ({_NUM})\b", lambda m: f"{NUMBER_WORDS[m.group(1)]}:15", t)
    t = re.sub(rf"\bquarter to ({_NUM})\b", lambda m: f"{(NUMBER_WORDS[m.group(1)] - 2) % 12 + 1}:45", t)
    t = re.sub(rf"\b({_NUM})\s+(thirty|fifteen|forty[ -]five)\b",
               lambda m: f"{NUMBER_WORDS[m.group(1)]}:{ {'thirty': 30, 'fifteen': 15}.get(m.group(2), 45):02d}", t)
    t = re.sub(rf"\b({_NUM})(?=\s*(?:am|pm|o'?\s?clock)\b)", lambda m: str(NUMBER_WORDS[m.group(1)]), t)
    return re.sub(r"\s+", " ", t).strip()


def _to_24h(hour: int, minute: int, suffix: str | None) -> time | None:
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    elif suffix is None and 1 <= hour <= 7:       # the clinic is never open at 3am, so "3" means 3pm
        hour += 12
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def _find_time(t: str) -> tuple[time | None, tuple[int, int] | None]:
    """The time of day named in the text, and where it was found (so it can be cut out)."""
    patterns = [
        (r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?\b", True),
        (r"\b(\d{1,2})\s*(am|pm)\b", False),
        (r"\b(\d{1,2})\s*o'?\s?clock\b", False),
        (r"\b(?:at|around|about|for|by|after|before|from)\s+(\d{1,2})\b(?!\s*(?:st|nd|rd|th|/|-|people|person))", False),
    ]
    for i, (pattern, has_minutes) in enumerate(patterns):
        m = re.search(pattern, t)
        if not m:
            continue
        if has_minutes:
            hour, minute, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
        elif i == 1:
            hour, minute, suffix = int(m.group(1)), 0, m.group(2)
        else:
            hour, minute, suffix = int(m.group(1)), 0, None
        found = _to_24h(hour, minute, suffix)
        if found is not None:
            return found, m.span()
    m = re.search(r"\b(noon|midday)\b", t)
    if m:
        return time(12, 0), m.span()
    return None, None


def _next_weekday(today: date, weekday: int, strict: bool = False) -> date:
    ahead = (weekday - today.weekday()) % 7
    if strict and ahead == 0:
        ahead = 7
    return today + timedelta(days=ahead)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _find_dates(t: str, today: date) -> list[date]:
    """Dates named in the text, in order: "8th of November", "November 8", "8/11", "the 12th", weekdays..."""
    found: list[date] = []

    def future(month: int, day: int) -> date | None:
        d = _safe_date(today.year, month, day)
        if d is not None and d < today:
            d = _safe_date(today.year + 1, month, day)
        return d

    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\b", t):
        found.append(future(MONTHS[m.group(2)], int(m.group(1))))
    for m in re.finditer(rf"\b({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", t):
        found.append(future(MONTHS[m.group(1)], int(m.group(2))))
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})\b", t):                 # UK order: day/month
        found.append(future(int(m.group(2)), int(m.group(1))))
    if not found:
        for m in re.finditer(r"\b(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\b", t):   # "the 12th": this month or next
            day = int(m.group(1))
            d = _safe_date(today.year, today.month, day)
            if d is None or d < today:
                nxt_year, nxt_month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
                d = _safe_date(nxt_year, nxt_month, day)
            found.append(d)
    found = [d for d in found if d is not None]
    if found:
        return found

    if "day after tomorrow" in t:
        return [today + timedelta(days=2)]
    if re.search(r"\btomorrow\b", t):
        return [today + timedelta(days=1)]
    if re.search(r"\b(today|tonight|this evening|this afternoon|this morning)\b", t):
        return [today]
    monday_this_week = today - timedelta(days=today.weekday())
    if re.search(r"\bnext week\b", t):
        start = monday_this_week + timedelta(days=7)
        return [start + timedelta(days=i) for i in range(6)]         # Monday to Saturday
    if re.search(r"\bthis week\b", t):
        return [today + timedelta(days=i) for i in range(6 - today.weekday() + 1) if today.weekday() + i <= 5]
    days = []
    for m in re.finditer(rf"\b(next\s+|this\s+|on\s+)?({_DAY})\b", t):
        weekday = WEEKDAYS[m.group(2)]
        if (m.group(1) or "").strip() == "next":
            d = _next_weekday(today, weekday, strict=True)
            if d.isocalendar()[:2] == today.isocalendar()[:2]:       # "next Tuesday" said on a Monday = next week
                d += timedelta(days=7)
        else:
            d = _next_weekday(today, weekday)
        days.append(d)
    if re.search(r"\bweekend\b", t) and not days:
        days.append(_next_weekday(today, 5))
    return sorted(set(days))


def parse_time_request(text: str, today: date) -> TimeRequest | None:
    """What day and time the caller wants, or None if the text names no day, time, or part of the day."""
    t = _prep(text)
    at, span = _find_time(t)
    modifier = ""
    if span:
        before = t[max(0, span[0] - 14):span[0]]
        for word in ("after", "from", "before", "by", "until"):
            if re.search(rf"\b{word}\s*$", before):
                modifier = word
        t = (t[:span[0]] + " " + t[span[1]:]).strip()
    days = _find_dates(t, today)
    part = next((p for p in PARTS_OF_DAY if re.search(rf"\b{p}\b", t)), None)
    asap = any(p in t for p in ASAP_PHRASES)
    if not (at or days or part or asap):
        return None

    start, end, exact = START_OF_DAY, END_OF_DAY, None
    if part:
        start, end = PARTS_OF_DAY[part]
    if at:
        if modifier in ("after", "from"):
            start = at
        elif modifier in ("before", "by", "until"):
            end = at
        else:
            exact = at
    return TimeRequest(days=tuple(days), start=start, end=end, exact=exact, asap=asap)


# ----------------------------------------------------------------------
# Picking one of the times Ava offered
# ----------------------------------------------------------------------
_ORDINALS = [(r"\b(first|1st)\b", 0), (r"\b(second|2nd)\b", 1), (r"\b(third|3rd)\b", 2)]


def pick_offered(text: str, offered: Sequence[datetime], today: date) -> int | None:
    """Which of the offered slots did the caller choose? An index, or None if it can't tell.

    Understands "the first one", "the last one", "10 o'clock", "half past nine", "Tuesday", and a plain
    "yes" or "that's fine" when only one time was offered.
    """
    if not offered:
        return None
    t = _prep(text)
    if len(offered) == 1 and re.match(r"^(?:yes|yeah|yep|yup|sure|ok|okay|fine|perfect|great|that|please)\b", t):
        return 0
    if re.fullmatch(r"\d{1,2}(?::\d{2})?", t.strip(" .")):                # just "10" or "9:30"
        t = f"at {t.strip(' .')}"
    m = re.fullmatch(r"(?:the\s+)?(first|1st|second|2nd|third|3rd)(?:\s+(?:one|option|time|slot))?", t.strip(" ."))
    if m:                                                                   # "1st" is a choice, not the 1st of the month
        index = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}[m.group(1)]
        return index if index < len(offered) else None
    req = parse_time_request(t, today)
    if req and (req.exact or req.days):
        named = {WEEKDAYS[w] for w in re.findall(rf"\b({_DAY})\b", t)}       # "Monday" means the Monday on offer
        matches = [i for i, s in enumerate(offered)
                   if (not req.exact or s.time() == req.exact)
                   and (not req.days or s.date() in req.days or s.weekday() in named)]
        if len(matches) == 1:
            return matches[0]
        if matches and req.days and not req.exact:
            return matches[0]                                              # "Tuesday": the earliest that day
        return None
    for pattern, index in _ORDINALS:
        if re.search(pattern, t) and index < len(offered):
            return index
    if re.search(r"\b(last|latest)\b", t):
        return len(offered) - 1
    if re.search(r"\b(earliest|earlier one|soonest)\b", t):
        return 0
    return None
