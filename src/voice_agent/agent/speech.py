"""Turn dates, times and numbers into words that sound right when read aloud.

The text-to-speech voice reads digits badly ("9:30", "8th"), so Ava's replies spell them out:
"Saturday the eighth of November at half past nine in the morning".
"""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime
from typing import Sequence

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
_ORDINALS = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
             "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
             "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third",
             "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth",
             "twenty-ninth", "thirtieth", "thirty-first"]


def number_words(n: int) -> str:
    """45 -> "forty-five", 500 -> "five hundred", 1250 -> "one thousand two hundred and fifty"."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = n - n % 10, n % 10
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    if n < 1000:
        rest = n % 100
        return f"{_ONES[n // 100]} hundred" + (f" and {number_words(rest)}" if rest else "")
    if n < 1_000_000:
        rest = n % 1000
        joiner = " and " if 0 < rest < 100 else " "
        return f"{number_words(n // 1000)} thousand" + (f"{joiner}{number_words(rest)}" if rest else "")
    return " ".join(_ONES[int(c)] for c in str(n))


def say_date(d: date, today: date) -> str:
    if d == today:
        return "today"
    if (d - today).days == 1:
        return "tomorrow"
    return f"{calendar.day_name[d.weekday()]} the {_ORDINALS[d.day]} of {calendar.month_name[d.month]}"


def _period(dt: datetime) -> str:
    return "morning" if dt.hour < 12 else "afternoon" if dt.hour < 17 else "evening"


def say_clock(dt: datetime) -> str:
    """The time without the part of the day: "half past nine", "ten o'clock"."""
    hour = dt.hour % 12 or 12
    if dt.minute == 0:
        return f"{number_words(hour)} o'clock"
    if dt.minute == 30:
        return f"half past {number_words(hour)}"
    if dt.minute == 15:
        return f"quarter past {number_words(hour)}"
    if dt.minute == 45:
        return f"quarter to {number_words(hour % 12 + 1)}"
    return f"{number_words(hour)} {'oh ' + number_words(dt.minute) if dt.minute < 10 else number_words(dt.minute)}"


def say_slot(dt: datetime, today: date) -> str:
    """One appointment time: "Saturday the eighth of November at half past nine in the morning"."""
    return f"{say_date(dt.date(), today)} at {say_clock(dt)} in the {_period(dt)}"


def say_slot_on(dt: datetime, today: date) -> str:
    """Like say_slot, with the right small word in front: "on Saturday the seventh ..." but "tomorrow at ..."."""
    text = say_slot(dt, today)
    return text if text.startswith(("today", "tomorrow")) else f"on {text}"


def say_options(slots: Sequence[datetime], today: date) -> str:
    """Several times, grouped by day: "Saturday the eighth of November at nine, half past nine or ten in the morning"."""
    groups: list[list[datetime]] = []
    for slot in slots:
        if groups and groups[-1][0].date() == slot.date():
            groups[-1].append(slot)
        else:
            groups.append([slot])
    parts = []
    for group in groups:
        periods = {_period(s) for s in group}
        if len(periods) == 1:
            times = [say_clock(s) for s in group]
            suffix = f" in the {periods.pop()}"
        else:
            times = [f"{say_clock(s)} in the {_period(s)}" for s in group]
            suffix = ""
        joined = times[0] if len(times) == 1 else ", ".join(times[:-1]) + f" or {times[-1]}"
        parts.append(f"{say_date(group[0].date(), today)} at {joined}{suffix}")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f", or {parts[-1]}"


def say_digits(digits: str) -> str:
    """A phone number read one digit at a time: "0 7 7 0 0 9 0 0 1 2 3"."""
    return " ".join(digits)


def say_reference(reference: str) -> str:
    """"SD-000012" read out as "S D 0 0 0 0 1 2"."""
    return "S D " + " ".join(c for c in reference if c.isdigit())


# ======================================================================
# Everything Ava says goes through for_voice() just before the voice reads it
# ======================================================================
_PHONE = re.compile(r"(?<![\w£$:.])\+?\d[\d \-]{5,}\d(?![\w:])")                  # 01632 960 000
_SERVICE_NUMBER = re.compile(r"\b(999|111|112|101)\b")                           # read one digit at a time
_POSTCODE = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?)\s?(\d[A-Z]{2})\b")              # BW1 2AB
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s?([ap]m)\b)?", re.IGNORECASE)            # 9:30, 9:30am
_HOUR = re.compile(r"\b(\d{1,2})\s?([ap])m\b", re.IGNORECASE)                # 8am, 6 pm
_MONEY = re.compile(r"([£$])\s?(\d+)(?:\.(\d{2}))?")                              # £45, £12.50
_ORDINAL = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)             # 8th
_NUMBER = re.compile(r"\d+")


def _phone_words(match: re.Match) -> str:
    groups = re.split(r"[ \-]+", match.group().lstrip("+"))
    if sum(len(g) for g in groups) < 7:
        return match.group()                                                      # not a phone number, leave it
    if all(len(g) == 1 for g in groups):
        return " ".join(groups)                                                   # already one digit at a time
    return ", ".join(" ".join(g) for g in groups)                                 # a pause between the groups


def _minutes(m: int) -> str:
    return "" if m == 0 else f"oh {number_words(m)}" if m < 10 else number_words(m)


def _clock_words(match: re.Match) -> str:
    hour, minute, ampm = int(match.group(1)), int(match.group(2)), match.group(3)
    words = number_words(hour % 12 or 12 if ampm else hour)
    spoken = f"{words} {_minutes(minute)}" if minute else (words if ampm else f"{words} o'clock")
    return spoken + (f" {ampm[0].lower()} m" if ampm else "")


def _money_words(match: re.Match) -> str:
    unit = "pounds" if match.group(1) == "£" else "dollars"
    text = f"{number_words(int(match.group(2)))} {unit}"
    return text + (f" {number_words(int(match.group(3)))}" if match.group(3) and match.group(3) != "00" else "")


def for_voice(text: str) -> str:
    """Spell out digits so the text-to-speech voice does not garble them: phone numbers one digit at a time with a
    pause between the groups, "8am" as "eight a m", "45 pounds" as "forty-five pounds", "NHS 111" as "N H S one one one".
    The text on screen stays as written; only what is sent to the voice changes."""
    text = _PHONE.sub(_phone_words, text)
    text = _SERVICE_NUMBER.sub(lambda m: " ".join(m.group(1)), text)
    text = _POSTCODE.sub(lambda m: f"{' '.join(m.group(1))}, {' '.join(m.group(2))}", text)
    text = _CLOCK.sub(_clock_words, text)
    text = _HOUR.sub(lambda m: f"{number_words(int(m.group(1)))} {m.group(2).lower()} m", text)
    text = _MONEY.sub(_money_words, text)
    text = _ORDINAL.sub(lambda m: _ORDINALS[int(m.group(1))] if 1 <= int(m.group(1)) <= 31 else m.group(0), text)
    text = text.replace("NHS", "N H S")
    return _NUMBER.sub(lambda m: _digits_to_words(m), text)


def _digits_to_words(match: re.Match) -> str:
    digits = match.group()
    if (len(digits) > 1 and digits.startswith("0")) or len(digits) > 9:
        return " ".join(_ONES[int(c)] for c in digits)
    return number_words(int(digits))
