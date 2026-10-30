"""Database tables, defined with SQLAlchemy.

SQLAlchemy is an "ORM" (object-relational mapper): each table is a Python class,
each row is an object. The same code works on SQLite and MySQL.

The CheckConstraints make the database itself refuse bad values (for example an
unknown outcome), so a bug cannot quietly store rubbish.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

OUTCOMES = ("answered", "booking_saved", "booking_changed", "booking_cancelled", "handed_off", "emergency_999",
            "urgent_dental", "abandoned")
ROLES = ("caller", "agent")
# requested = an old booking that only has the caller's words, no real appointment slot
BOOKING_STATUSES = ("requested", "confirmed", "cancelled")
SYNC_STATUSES = ("pending", "sent")      # has the clinic's booking system been told about it?


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    pass


class Call(Base):
    __tablename__ = "calls"
    __table_args__ = (CheckConstraint(f"outcome IS NULL OR {_in('outcome', OUTCOMES)}", name="ck_call_outcome"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[str | None] = mapped_column(String(20), index=True)

    messages: Mapped[list[Message]] = relationship(back_populates="call", order_by="Message.id")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (CheckConstraint(_in("role", ROLES), name="ck_message_role"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"), index=True)
    role: Mapped[str] = mapped_column(String(10))
    text: Mapped[str] = mapped_column(Text)
    route: Mapped[str] = mapped_column(String(30), index=True)
    similarity_score: Mapped[float | None] = mapped_column(Float)
    stt_ms: Mapped[float] = mapped_column(Float, default=0.0)
    llm_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tts_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    call: Mapped[Call] = relationship(back_populates="messages")


class Booking(Base):
    __tablename__ = "bookings"
    __table_args__ = (
        CheckConstraint(_in("status", BOOKING_STATUSES), name="ck_booking_status"),
        CheckConstraint(_in("sync_status", SYNC_STATUSES), name="ck_booking_sync"),
        # The database itself refuses two confirmed bookings for the same slot (one dentist), even if
        # two calls try at the same moment. Cancelled bookings don't count, so their slot is free again.
        Index("uq_booking_active_slot", "start_at", unique=True,
              postgresql_where=text("status = 'confirmed'"), sqlite_where=text("status = 'confirmed'")),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"), index=True)
    caller_name: Mapped[str] = mapped_column(String(200))
    reason: Mapped[str] = mapped_column(Text)
    preferred_time: Mapped[str] = mapped_column(String(200))     # what the caller said, e.g. "Saturday morning"
    phone: Mapped[str] = mapped_column(String(20), index=True)
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)   # the real slot (UTC)
    reference: Mapped[str | None] = mapped_column(String(20), unique=True, index=True)       # e.g. SD-000012
    status: Mapped[str] = mapped_column(String(12), default="confirmed")
    sync_status: Mapped[str] = mapped_column(String(10), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=utcnow)


class ToolCall(Base):
    """Audit trail: every tool the LLM called, whether the code accepted it, and how long it took.
    Names, reasons and phone numbers are masked before they are stored."""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"), index=True)
    tool: Mapped[str] = mapped_column(String(40), index=True)
    ok: Mapped[bool] = mapped_column(Boolean)
    ms: Mapped[float] = mapped_column(Float, default=0.0)
    args: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UnansweredQuestion(Base):
    __tablename__ = "unanswered_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    best_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
