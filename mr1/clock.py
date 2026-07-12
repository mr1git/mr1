"""
Time seam.

Every autonomous subsystem that reasons about wall-clock time — approval
TTLs, consent-grant TTLs, supervisor scheduling, failure backoff — reads
time through a `Clock` instead of calling `datetime.now()` directly, so a
`VirtualClock` can drive months of simulated uptime in milliseconds.

`SystemClock` is the production default and preserves existing behaviour
exactly: UTC-aware wall clock, `time.monotonic()` for durations.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol, runtime_checkable


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into an aware UTC datetime, or None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@runtime_checkable
class Clock(Protocol):
    """The time surface every governed subsystem is allowed to depend on."""

    def now(self) -> datetime:
        """Current wall-clock time as an aware UTC datetime."""

    def now_iso(self) -> str:
        """Current wall-clock time as an ISO-8601 string."""

    def monotonic(self) -> float:
        """Monotonic seconds; only differences are meaningful."""

    def sleep(self, seconds: float) -> None:
        """Block for `seconds`."""

    def wait(self, event: threading.Event, timeout: float) -> bool:
        """Wait on `event` for at most `timeout` seconds; True if it was set."""


class SystemClock:
    """Real time. The production default."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def now_iso(self) -> str:
        return self.now().isoformat()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def wait(self, event: threading.Event, timeout: float) -> bool:
        return event.wait(timeout)


class VirtualClock:
    """
    Deterministic time under test.

    Time only moves when `advance()` is called, so a soak can run 10 000
    supervisor ticks across simulated weeks without waiting. `sleep()` and
    `wait()` advance time instead of blocking, which keeps any production
    code path that sleeps from stalling a simulation.
    """

    def __init__(
        self,
        start: Optional[datetime] = None,
        *,
        monotonic_start: float = 0.0,
    ):
        base = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        self._lock = threading.RLock()
        self._now = base.astimezone(timezone.utc)
        self._monotonic = float(monotonic_start)

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def now_iso(self) -> str:
        return self.now().isoformat()

    def monotonic(self) -> float:
        with self._lock:
            return self._monotonic

    def advance(self, seconds: float) -> datetime:
        """Move time forward. Negative advances are rejected — time is a ratchet."""
        if seconds < 0:
            raise ValueError("VirtualClock cannot move backwards")
        with self._lock:
            self._now = self._now + timedelta(seconds=seconds)
            self._monotonic += float(seconds)
            return self._now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.advance(seconds)

    def wait(self, event: threading.Event, timeout: float) -> bool:
        if event.is_set():
            return True
        self.sleep(timeout)
        return event.is_set()


_DEFAULT_CLOCK = SystemClock()


def default_clock() -> Clock:
    return _DEFAULT_CLOCK
