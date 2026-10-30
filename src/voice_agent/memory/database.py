"""Database connection. The address comes from DATABASE_URL, so moving between PostgreSQL
and SQLite is a settings change, not a code change."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from .models import BOOKING_STATUSES, OUTCOMES, Base, _in

log = logging.getLogger(__name__)


class Database:
    def __init__(self, url: str):
        parsed = make_url(url)
        is_sqlite = parsed.get_backend_name() == "sqlite"
        if is_sqlite and parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            url,
            # SQLite: allow use from Gradio's worker threads. MySQL: check connections are alive before use.
            connect_args={"check_same_thread": False} if is_sqlite else {},
            pool_pre_ping=not is_sqlite,
        )
        if is_sqlite:
            event.listen(self.engine, "connect", _sqlite_pragmas)
        # expire_on_commit=False: objects stay readable after the session closes.
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)
        log.info("database_connected", extra={"backend": parsed.get_backend_name()})

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)
        self._upgrade_existing_tables()

    def _upgrade_existing_tables(self) -> None:
        """create_all() only makes MISSING tables. If an older version of the app already made the
        bookings table, add what the appointment feature needs, keeping the old rows. Safe to run
        every start: each step is skipped when it has already been done."""
        inspector = inspect(self.engine)
        if "bookings" not in inspector.get_table_names():
            return
        columns = {c["name"] for c in inspector.get_columns("bookings")}
        postgres = self.engine.dialect.name == "postgresql"
        if "start_at" not in columns and not postgres:
            raise RuntimeError("This SQLite file was made by an older version of the app. Delete it (it is "
                               "only demo data) or point DATABASE_URL at PostgreSQL, then start again.")
        with self.engine.begin() as conn:
            if "start_at" not in columns:
                log.info("database_upgrade", extra={"step": "bookings: add appointment columns"})
                for ddl in ("ALTER TABLE bookings ADD COLUMN start_at TIMESTAMP WITH TIME ZONE",
                            "ALTER TABLE bookings ADD COLUMN reference VARCHAR(20)",
                            "ALTER TABLE bookings ADD COLUMN sync_status VARCHAR(10) NOT NULL DEFAULT 'pending'",
                            "ALTER TABLE bookings ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE",
                            "ALTER TABLE bookings ALTER COLUMN status TYPE VARCHAR(12)",
                            "ALTER TABLE bookings DROP CONSTRAINT IF EXISTS ck_booking_status"):
                    conn.execute(text(ddl))
                # Old rows: status was new/sent (sent to the clinic or not). They have no real slot.
                conn.execute(text("UPDATE bookings SET sync_status = CASE WHEN status = 'sent' THEN 'sent' "
                                  "ELSE 'pending' END, status = 'requested', "
                                  "reference = 'SD-' || LPAD(id::text, 6, '0')"))
                conn.execute(text(f"ALTER TABLE bookings ADD CONSTRAINT ck_booking_status CHECK "
                                  f"({_in('status', BOOKING_STATUSES)})"))
                conn.execute(text("ALTER TABLE bookings ADD CONSTRAINT ck_booking_sync CHECK "
                                  "(sync_status IN ('pending', 'sent'))"))
            if postgres:   # the list of allowed call outcomes grew (booking_changed, booking_cancelled)
                current = conn.execute(text("SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                                            "WHERE conname = 'ck_call_outcome'")).scalar()
                if current is not None and not all(f"'{o}'" in current for o in OUTCOMES):
                    log.info("database_upgrade", extra={"step": "calls: allow new outcomes"})
                    conn.execute(text("ALTER TABLE calls DROP CONSTRAINT ck_call_outcome"))
                    conn.execute(text(f"ALTER TABLE calls ADD CONSTRAINT ck_call_outcome CHECK "
                                      f"(outcome IS NULL OR {_in('outcome', OUTCOMES)})"))
            for ddl in ("CREATE INDEX IF NOT EXISTS ix_bookings_start_at ON bookings (start_at)",
                        "CREATE INDEX IF NOT EXISTS ix_bookings_phone ON bookings (phone)",
                        "CREATE UNIQUE INDEX IF NOT EXISTS ix_bookings_reference ON bookings (reference)",
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_booking_active_slot ON bookings (start_at) "
                        "WHERE status = 'confirmed'"):
                conn.execute(text(ddl))

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Use with `with db.session() as s:`. Saves on success, undoes everything on error."""
        s = self._sessions()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            log.exception("database_ping_failed")
            return False


def _sqlite_pragmas(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")      # SQLite ignores foreign keys unless asked
    cursor.execute("PRAGMA busy_timeout=5000")    # wait up to 5 s if another thread is writing
    cursor.close()
