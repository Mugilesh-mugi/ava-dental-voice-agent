"""Understanding when the caller wants to come: days, times, parts of the day, and picking an offered time."""
from datetime import date, datetime, time

import pytest

from voice_agent.agent import speech
from voice_agent.agent.timeparse import parse_time_request, pick_offered

TODAY = date(2026, 11, 2)           # a Monday


def d(day):
    return date(2026, 11, day)


@pytest.mark.parametrize("text, days", [
    ("Saturday morning", (d(7),)), ("tomorrow", (d(3),)), ("today", (d(2),)), ("day after tomorrow", (d(4),)),
    ("next Tuesday", (d(10),)), ("next Monday", (d(9),)), ("this Friday", (d(6),)), ("Tuesday or Thursday", (d(3), d(5))),
    ("the 12th", (d(12),)), ("the 1st", (date(2026, 12, 1),)), ("5th of November", (d(5),)), ("November 9", (d(9),)),
    ("nov 9", (d(9),)), ("9/11", (d(9),)), ("this weekend", (d(7),)),
    ("3rd January", (date(2027, 1, 3),)),
])
def test_days(text, days):
    assert parse_time_request(text, TODAY).days[:len(days)] == days


def test_next_week_is_monday_to_saturday_of_next_week():
    assert parse_time_request("sometime next week", TODAY).days == tuple(d(i) for i in range(9, 15))


@pytest.mark.parametrize("text, hour, minute", [
    ("tomorrow at 3pm", 15, 0), ("3 p.m.", 15, 0), ("10:30", 10, 30), ("10.30 am", 10, 30), ("half past ten", 10, 30),
    ("ten thirty", 10, 30), ("quarter past nine", 9, 15), ("quarter to four", 15, 45), ("three o'clock", 15, 0),
    ("at 9", 9, 0), ("at 3", 15, 0), ("around noon", 12, 0), ("12 pm", 12, 0), ("8 am", 8, 0),
])
def test_times(text, hour, minute):
    expected = time(hour, minute)
    assert parse_time_request(text, TODAY).exact == expected


@pytest.mark.parametrize("text, start, end", [
    ("Saturday morning", time(0), time(12)), ("Friday afternoon", time(12), time(17)), ("evening", time(17), time(23, 59)),
    ("after 4pm", time(16), time(23, 59)), ("before 11 am", time(0), time(11)),
])
def test_windows(text, start, end):
    request = parse_time_request(text, TODAY)
    assert (request.start, request.end) == (start, end) and request.exact is None


@pytest.mark.parametrize("text", ["as soon as possible", "the earliest you have", "first available", "any time"])
def test_asap(text):
    assert parse_time_request(text, TODAY).asap


@pytest.mark.parametrize("text", ["cleaning", "check-up", "banana", "", "my name is Jane", "can you give me suggestions"])
def test_text_with_no_time_in_it(text):
    assert parse_time_request(text, TODAY) is None


def test_words_that_look_like_months_or_days_are_not_dates():
    assert parse_time_request("may I book a check-up", TODAY) is None
    assert parse_time_request("I sat in the chair", TODAY) is None


def test_a_date_that_has_passed_means_next_year():
    assert parse_time_request("1st of October", TODAY).days == (date(2027, 10, 1),)


def test_invalid_dates_are_ignored():
    assert parse_time_request("31st of February", TODAY) is None


OFFERED = [datetime(2026, 11, 7, 9, 0), datetime(2026, 11, 7, 9, 30), datetime(2026, 11, 7, 10, 0)]


@pytest.mark.parametrize("text, index", [
    ("the first one", 0), ("1st", 0), ("second", 1), ("the third one", 2), ("the last one", 2), ("the earliest", 0),
    ("9:30", 1), ("half past nine", 1), ("ten o'clock", 2), ("10", 2), ("at 9", 0), ("Saturday at 10", 2),
    ("banana", None), ("none of those", None), ("9:45", None), ("Tuesday", None), ("hmm", None),
])
def test_picking_an_offered_time(text, index):
    assert pick_offered(text, OFFERED, TODAY) == index


def test_yes_picks_the_only_time_offered():
    assert pick_offered("yes please", OFFERED[:1], TODAY) == 0
    assert pick_offered("yes please", OFFERED, TODAY) is None          # which one?


def test_naming_a_day_picks_the_earliest_that_day():
    mixed = [datetime(2026, 11, 7, 9, 0), datetime(2026, 11, 9, 8, 0), datetime(2026, 11, 9, 8, 30)]
    assert pick_offered("Monday", mixed, TODAY) == 1
    assert pick_offered("saturday", mixed, TODAY) == 0


# ---------- Reading things aloud ----------
@pytest.mark.parametrize("dt, text", [
    (datetime(2026, 11, 7, 9, 0), "Saturday the seventh of November at nine o'clock in the morning"),
    (datetime(2026, 11, 3, 15, 30), "tomorrow at half past three in the afternoon"),
    (datetime(2026, 11, 2, 12, 0), "today at twelve o'clock in the afternoon"),
    (datetime(2026, 11, 21, 8, 15), "Saturday the twenty-first of November at quarter past eight in the morning"),
    (datetime(2026, 11, 21, 17, 45), "Saturday the twenty-first of November at quarter to six in the evening"),
    (datetime(2026, 11, 21, 9, 5), "Saturday the twenty-first of November at nine oh five in the morning"),
])
def test_say_slot(dt, text):
    assert speech.say_slot(dt, TODAY) == text


def test_say_options_groups_by_day_and_shares_the_part_of_the_day():
    slots = [datetime(2026, 11, 7, 9, 0), datetime(2026, 11, 7, 9, 30), datetime(2026, 11, 9, 8, 0)]
    assert speech.say_options(slots, TODAY) == ("Saturday the seventh of November at nine o'clock or half past nine "
                                                "in the morning, or Monday the ninth of November at eight o'clock "
                                                "in the morning")
    mixed = [datetime(2026, 11, 7, 11, 30), datetime(2026, 11, 7, 12, 0)]
    assert speech.say_options(mixed, TODAY) == ("Saturday the seventh of November at half past eleven in the morning "
                                                "or twelve o'clock in the afternoon")


def test_say_digits_and_reference():
    assert speech.say_digits("07700900123") == "0 7 7 0 0 9 0 0 1 2 3"
    assert speech.say_reference("SD-000012") == "S D 0 0 0 0 1 2"
