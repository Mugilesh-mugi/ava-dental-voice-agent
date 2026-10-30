"""Appointments: find free times, book, look up, move, and cancel. Everything that touches the bookings table.

Each change is saved FIRST, then the clinic's booking system (the "consultation tool") is told about
it. If telling the clinic fails, nothing is lost: the booking stays saved with sync_status 'pending' so
staff, or a later retry, can pick it up.

Two callers can't take the same slot: the database has a unique index on the start time of confirmed
bookings, so the second attempt fails with SlotTaken even if both arrive at the same instant.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .. import monitoring
from ..agent.timeparse import TimeRequest
from ..memory.database import Database
from ..memory.models import Booking, ToolCall, utcnow
from .scheduling import Scheduler, Suggestion, to_utc

log = logging.getLogger(__name__)


class SlotTaken(Exception):
    """The time is no longer available (someone else just booked it, or it is not a real slot)."""


class ConsultationTool(Protocol):
    def submit(self, booking: dict) -> str:
        """Send a booking event to the clinic's system and return its reference."""


@dataclass(frozen=True)
class BookingInfo:
    id: int
    reference: str
    start_at: datetime         # clinic-local time
    reason: str
    caller_name: str
    phone: str
    status: str


class BookingService:
    def __init__(self, db: Database, consultation_tool: ConsultationTool, scheduler: Scheduler | None = None,
                 clock: Callable[[], datetime] = utcnow):
        self._db = db
        self._tool = consultation_tool
        self.scheduler = scheduler or Scheduler()
        self._clock = clock                  # a function so tests can freeze time

    # ---------- Time ----------
    def now(self) -> datetime:
        return to_utc(self._clock())

    def today(self) -> date:
        return self.scheduler.today(self.now())

    # ---------- Availability ----------
    def _booked_starts(self) -> set[datetime]:
        earliest, latest = self.scheduler.bookable_window(self.now())
        with self._db.session() as s:
            rows = s.scalars(select(Booking.start_at).where(Booking.status == "confirmed",
                                                            Booking.start_at >= earliest - self.scheduler.slot,
                                                            Booking.start_at <= latest)).all()
        return {to_utc(r) for r in rows}

    def suggest(self, request: TimeRequest | None, limit: int = 3) -> Suggestion:
        """Free times that fit what the caller asked for (nearest ones if nothing fits)."""
        return self.scheduler.suggest(request, self._booked_starts(), self.now(), limit)

    # ---------- Reading ----------
    def _info(self, b: Booking) -> BookingInfo:
        start = self.scheduler.local(b.start_at) if b.start_at else None
        return BookingInfo(b.id, b.reference or f"SD-{b.id:06d}", start, b.reason, b.caller_name, b.phone, b.status)

    def get(self, booking_id: int) -> BookingInfo | None:
        with self._db.session() as s:
            b = s.get(Booking, booking_id)
            return self._info(b) if b else None

    def upcoming_for_phone(self, phone: str) -> list[BookingInfo]:
        """Confirmed bookings that haven't happened yet for this phone number, soonest first."""
        with self._db.session() as s:
            rows = s.scalars(select(Booking).where(Booking.phone == phone, Booking.status == "confirmed",
                                                   Booking.start_at >= self.now())
                             .order_by(Booking.start_at)).all()
            return [self._info(b) for b in rows]

    # ---------- Changing ----------
    def create(self, call_id: str, details: dict, start: datetime) -> BookingInfo:
        """Book `start` for the caller. Raises SlotTaken if it is no longer free."""
        self._require_free(start)
        try:
            with self._db.session() as s:
                booking = Booking(call_id=call_id, caller_name=details["caller_name"], reason=details["reason"],
                                  preferred_time=(details.get("said_time") or "")[:200], phone=details["phone"],
                                  start_at=to_utc(start), status="confirmed", sync_status="pending")
                s.add(booking)
                s.flush()                                    # ask the database for the new id
                booking.reference = f"SD-{booking.id:06d}"
                info = self._info(booking)
        except IntegrityError as e:                          # the unique index said no: someone got there first
            monitoring.BOOKINGS.labels("slot_conflict").inc()
            raise SlotTaken(str(start)) from e
        monitoring.BOOKINGS.labels("saved").inc()
        self._tell_clinic(info, "created")
        return info

    def reschedule(self, booking_id: int, start: datetime) -> BookingInfo:
        """Move a booking to a new time. The old time is only released if the new one is taken successfully."""
        self._require_free(start)
        try:
            with self._db.session() as s:
                booking = s.get(Booking, booking_id)
                if booking is None or booking.status != "confirmed":
                    raise SlotTaken("that booking can no longer be changed")
                booking.start_at = to_utc(start)
                booking.sync_status = "pending"
                s.flush()
                info = self._info(booking)
        except IntegrityError as e:
            monitoring.BOOKINGS.labels("slot_conflict").inc()
            raise SlotTaken(str(start)) from e
        monitoring.BOOKINGS.labels("rescheduled").inc()
        self._tell_clinic(info, "rescheduled")
        return info

    def cancel(self, booking_id: int) -> BookingInfo:
        with self._db.session() as s:
            booking = s.get(Booking, booking_id)
            if booking is None:
                raise LookupError(f"no booking {booking_id}")
            booking.status = "cancelled"
            booking.sync_status = "pending"
            s.flush()
            info = self._info(booking)
        monitoring.BOOKINGS.labels("cancelled").inc()
        self._tell_clinic(info, "cancelled")
        return info

    # ---------- Audit trail ----------
    def record_tool_call(self, call_id: str, tool: str, args: str, ok: bool, ms: float, result: str) -> None:
        """Remember that the LLM called a tool (arguments already masked). Never raises: a lost audit row
        must not break a call."""
        try:
            with self._db.session() as s:
                s.add(ToolCall(call_id=call_id, tool=tool, ok=ok, ms=ms, args=args[:1000], result=result[:500]))
        except Exception:
            log.exception("tool_call_audit_failed")

    # ---------- Helpers ----------
    def _require_free(self, start: datetime) -> None:
        if not self.scheduler.is_bookable(start, self._booked_starts(), self.now()):
            monitoring.BOOKINGS.labels("slot_conflict").inc()
            raise SlotTaken(str(start))

    def _tell_clinic(self, info: BookingInfo, event: str) -> None:
        """Tell the clinic's booking system. A failure is logged and counted, never raised: the booking is saved."""
        payload = {"booking_id": info.id, "reference": info.reference, "event": event,
                   "caller_name": info.caller_name, "reason": info.reason, "phone": info.phone,
                   "preferred_time": info.start_at.strftime("%A %d %B %Y at %H:%M") if info.start_at else "",
                   "start_at": info.start_at.isoformat() if info.start_at else None}
        try:
            self._tool.submit(payload)
        except Exception:
            log.exception("booking_send_failed", extra={"booking_id": info.id, "event": event})
            monitoring.BOOKINGS.labels("send_failed").inc()
            monitoring.ERRORS.labels("consultation_tool").inc()
            return
        try:
            with self._db.session() as s:
                s.get(Booking, info.id).sync_status = "sent"
        except Exception:
            log.exception("booking_sync_status_failed", extra={"booking_id": info.id})
            return
        monitoring.BOOKINGS.labels("sent").inc()
        log.info("booking_sent", extra={"booking_id": info.id, "reference": info.reference, "event": event})
