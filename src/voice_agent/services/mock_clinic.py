"""A pretend version of the clinic's booking system (its "consultation tool").

In a real clinic, a finished booking request would be sent to their practice
management software. Here we just check the booking and append it to a file,
data/consultation_inbox.jsonl, one JSON line per booking.

To see what it has received:
    python -m voice_agent.services.mock_clinic     (from the src folder)
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_INBOX = Path(__file__).resolve().parents[3] / "data" / "consultation_inbox.jsonl"
REQUIRED_FIELDS = ("booking_id", "caller_name", "reason", "preferred_time", "phone")


class MockConsultationTool:
    def __init__(self, inbox_path: Path = DEFAULT_INBOX):
        self.inbox_path = Path(inbox_path)
        self._lock = threading.Lock()

    def submit(self, booking: dict) -> str:
        """Accept a booking and return a reference number. Raises ValueError if fields are missing."""
        missing = [f for f in REQUIRED_FIELDS if not str(booking.get(f, "")).strip()]
        if missing:
            raise ValueError(f"booking is missing: {', '.join(missing)}")
        reference = f"SD-{int(booking['booking_id']):06d}"
        record = {**booking, "reference": reference, "received_at": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self.inbox_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.inbox_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return reference

    def received(self) -> list[dict]:
        if not self.inbox_path.exists():
            return []
        with open(self.inbox_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    items = MockConsultationTool().received()
    print(f"{len(items)} booking(s) received")
    for item in items:
        print(f"  {item['reference']}  {item['received_at'][:19]}  {item['caller_name']}  "
              f"| {item['reason']} | {item['preferred_time']} | {item['phone']}")
