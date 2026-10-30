"""Turn dates, times and numbers into words that sound right when read aloud.

The text-to-speech voice reads digits badly ("9:30", "8th"), so Ava's replies spell them out:
"Saturday the eighth of November at half past nine in the morning".
"""
from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Sequence

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}
_ORDINALS = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
             "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
             "eighteenth", "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third",
             "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth",
             "twenty-ninth", "thirtieth", "thirty-first"]


def number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = n - n % 10, n % 10
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


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
