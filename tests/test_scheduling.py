"""The calendar: which times exist and which are free. Pure logic, no database."""
from datetime import date, datetime, timezone

import pytest

from voice_agent.agent.timeparse import TimeRequest, parse_time_request
from voice_agent.services.scheduling import Scheduler, to_utc

NOW = datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)           # Monday, 9am
TODAY = date(2026, 11, 2)
scheduler = Scheduler()


def utc(day, hour, minute=0):
    return datetime(2026, 11, day, hour, minute, tzinfo=timezone.utc)


def starts(suggestion):
    return [(s.day, s.hour, s.minute) for s in suggestion.slots]


def test_opening_hours_make_the_slots():
    monday, saturday = scheduler.day_slots(date(2026, 11, 2)), scheduler.day_slots(date(2026, 11, 7))
    assert (len(monday), monday[0].hour, monday[-1].hour, monday[-1].minute) == (20, 8, 17, 30)    # 8:00 to 17:30
    assert (len(saturday), saturday[0].hour, saturday[-1].hour, saturday[-1].minute) == (8, 9, 12, 30)
    assert scheduler.day_slots(date(2026, 11, 8)) == []                                            # Sunday: closed


def test_slot_length_is_a_setting():
    assert len(Scheduler(slot_minutes=60).day_slots(date(2026, 11, 2))) == 10


def test_9am_stays_9am_when_the_clocks_change():
    winter, summer = scheduler.day_slots(date(2026, 3, 27))[0], scheduler.day_slots(date(2026, 3, 30))[0]
    assert to_utc(winter).hour == 8 and to_utc(summer).hour == 7        # BST began on 29 March 2026


def test_nothing_is_offered_inside_the_minimum_notice():
    first = scheduler.suggest(TimeRequest(asap=True), set(), NOW).slots[0]
    assert (first.hour, first.minute) == (10, 0)                        # it is 9:00 now and one hour's notice is needed


def test_booked_times_are_not_offered():
    request = parse_time_request("Saturday morning", TODAY)
    assert starts(scheduler.suggest(request, set(), NOW)) == [(7, 9, 0), (7, 9, 30), (7, 10, 0)]
    assert starts(scheduler.suggest(request, {utc(7, 9, 0), utc(7, 10, 0)}, NOW)) == [(7, 9, 30), (7, 10, 30), (7, 11, 0)]


def test_a_named_time_offers_the_closest_times_first():
    request = parse_time_request("Tuesday at 3pm", TODAY)
    assert starts(scheduler.suggest(request, set(), NOW)) == [(3, 15, 0), (3, 14, 30), (3, 15, 30)]


def test_several_days_give_one_time_per_day():
    request = parse_time_request("next week in the morning", TODAY)
    assert starts(scheduler.suggest(request, set(), NOW)) == [(9, 8, 0), (10, 8, 0), (11, 8, 0)]


def test_a_full_day_falls_back_to_the_nearest_times():
    request = parse_time_request("Saturday morning", TODAY)
    everything = {to_utc(s) for s in scheduler.day_slots(date(2026, 11, 7))}
    suggestion = scheduler.suggest(request, everything, NOW)
    assert suggestion.fallback and starts(suggestion)[0] == (9, 8, 0)    # next Monday morning


def test_a_closed_day_falls_back():
    suggestion = scheduler.suggest(parse_time_request("Sunday", TODAY), set(), NOW)
    assert suggestion.fallback and starts(suggestion)[0][0] == 9


def test_nothing_beyond_the_booking_horizon():
    far = parse_time_request("1st of March", TODAY)
    suggestion = scheduler.suggest(far, set(), NOW)
    assert suggestion.fallback                              # too far ahead, so the nearest times instead
    assert all(s <= datetime(2026, 11, 30, 9, 0, tzinfo=timezone.utc) for s in suggestion.slots)


def test_everything_booked_gives_no_times():
    everything = {to_utc(s) for i in range(40) for s in scheduler.day_slots(date.fromordinal(TODAY.toordinal() + i))}
    assert scheduler.suggest(TimeRequest(asap=True), everything, NOW).slots == []


@pytest.mark.parametrize("start, booked, ok", [
    (utc(7, 9, 0), set(), True), (utc(7, 9, 0), {utc(7, 9, 0)}, False), (utc(7, 9, 10), set(), False),
    (utc(8, 9, 0), set(), False), (utc(7, 13, 0), set(), False), (utc(2, 9, 30), set(), False),
    (utc(2, 10, 0), set(), True), (datetime(2027, 1, 9, 9, 0, tzinfo=timezone.utc), set(), False),
])
def test_is_bookable(start, booked, ok):
    assert scheduler.is_bookable(start, booked, NOW) is ok


def test_times_without_a_time_zone_are_treated_as_utc():
    assert to_utc(datetime(2026, 11, 7, 9, 0)) == utc(7, 9, 0)
