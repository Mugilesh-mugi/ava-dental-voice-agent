"""Which appointment times exist, and which are free.

The calendar is WORKED OUT from the clinic's opening hours rather than stored, so there are no slot
rows to keep up to date. A slot is free when no confirmed booking has that start time. Pure logic with
no database access: the booked times are passed in, so it is easy to test.

Clinic hours are in the clinic's own time zone (so "9am" stays 9am when the clocks change), while the
database stores times in UTC.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from ..agent.timeparse import START_OF_DAY, TimeRequest

# Monday = 0 ... Sunday = 6. A day that is missing is closed. Keep in step with faq.md.
OPENING_HOURS: dict[int, tuple[time, time]] = {
    0: (time(8), time(18)), 1: (time(8), time(18)), 2: (time(8), time(18)), 3: (time(8), time(18)),
    4: (time(8), time(18)), 5: (time(9), time(13)),
}


@dataclass(frozen=True)
class Suggestion:
    slots: list[datetime]      # clinic-local times, best first
    fallback: bool = False     # True if nothing was free when asked, so these are the nearest instead


def to_utc(dt: datetime) -> datetime:
    """Treat a naive time (some databases drop the time zone) as UTC."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class Scheduler:
    def __init__(self, tz_name: str = "Europe/London", slot_minutes: int = 30, horizon_days: int = 28,
                 min_notice_minutes: int = 60, opening_hours: dict[int, tuple[time, time]] | None = None):
        self.tz = ZoneInfo(tz_name)
        self.slot = timedelta(minutes=slot_minutes)
        self.horizon_days = horizon_days
        self.min_notice = timedelta(minutes=min_notice_minutes)
        self.opening_hours = OPENING_HOURS if opening_hours is None else opening_hours

    def local(self, dt: datetime) -> datetime:
        return to_utc(dt).astimezone(self.tz)

    def today(self, now: datetime) -> date:
        return self.local(now).date()

    def day_slots(self, day: date) -> list[datetime]:
        """Every appointment start time on one day (clinic-local), in order. Empty if the clinic is closed."""
        hours = self.opening_hours.get(day.weekday())
        if hours is None:
            return []
        opens, closes = (datetime.combine(day, t, tzinfo=self.tz) for t in hours)
        slots, start = [], opens
        while start + self.slot <= closes:
            slots.append(start)
            start += self.slot
        return slots

    def bookable_window(self, now: datetime) -> tuple[datetime, datetime]:
        """The earliest and latest times (UTC) a caller may book."""
        now = to_utc(now)
        return now + self.min_notice, now + timedelta(days=self.horizon_days)

    def is_bookable(self, start: datetime, booked: set[datetime], now: datetime) -> bool:
        """Is `start` a real, free slot that is not too soon or too far ahead?"""
        earliest, latest = self.bookable_window(now)
        start_utc = to_utc(start)
        local = self.local(start)
        return (earliest <= start_utc <= latest and start_utc not in booked
                and any(local == s for s in self.day_slots(local.date())))

    def suggest(self, request: TimeRequest | None, booked: set[datetime], now: datetime, limit: int = 3) -> Suggestion:
        """Up to `limit` free times that fit the request. If nothing fits, the nearest free times instead."""
        first = self._search(request, booked, now, limit)
        if first:
            return Suggestion(first)
        if request is not None and (request.days or request.has_window):
            start_day = min(request.days) if request.days else None
            if start_day is not None and start_day > self.local(self.bookable_window(now)[1]).date():
                start_day = None                       # asked for a day beyond the booking horizon
            for relaxed in (TimeRequest(start=request.start, end=request.end, exact=request.exact),
                            TimeRequest(exact=request.exact)):
                found = self._search(relaxed, booked, now, limit, from_day=start_day)
                if found:
                    return Suggestion(found, fallback=True)
        return Suggestion([])

    def _search(self, request: TimeRequest | None, booked: set[datetime], now: datetime, limit: int,
                from_day: date | None = None) -> list[datetime]:
        request = request or TimeRequest()
        earliest, latest = self.bookable_window(now)
        today = self.today(now)
        specific = list(request.days)
        days = specific or [max(today, from_day or today) + timedelta(days=i) for i in range(self.horizon_days + 1)]
        # One day asked for (or "as soon as possible"): show several times from it.
        # Several days, or "any morning": show one time per day so the caller sees a spread.
        per_day = limit if (len(specific) == 1 or request.asap or not (specific or request.has_window)) else 1
        found: list[datetime] = []
        for day in days:
            if day < today:
                continue
            free = [s for s in self.day_slots(day)
                    if earliest <= to_utc(s) <= latest and to_utc(s) not in booked
                    and request.start <= s.time() < request.end]
            if request.exact is not None:
                mins = request.exact.hour * 60 + request.exact.minute
                free.sort(key=lambda s: abs(s.hour * 60 + s.minute - mins))
            found.extend(free[:per_day])
            if len(found) >= limit:
                break
        return found[:limit]
