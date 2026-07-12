"""A0 — the clock seam."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from mr1.clock import Clock, SystemClock, VirtualClock, default_clock, parse_iso


def test_system_clock_satisfies_protocol():
    clock = SystemClock()
    assert isinstance(clock, Clock)
    assert isinstance(default_clock(), Clock)


def test_system_clock_now_is_utc_aware():
    now = SystemClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_system_clock_monotonic_is_non_decreasing():
    clock = SystemClock()
    first = clock.monotonic()
    second = clock.monotonic()
    assert second >= first


def test_virtual_clock_satisfies_protocol():
    assert isinstance(VirtualClock(), Clock)


def test_virtual_clock_is_frozen_until_advanced():
    clock = VirtualClock(start=datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert clock.now() == datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert clock.now() == datetime(2026, 3, 1, tzinfo=timezone.utc)
    assert clock.now_iso() == "2026-03-01T00:00:00+00:00"


def test_virtual_clock_advance_moves_wall_and_monotonic_time():
    clock = VirtualClock(start=datetime(2026, 3, 1, tzinfo=timezone.utc))
    before_monotonic = clock.monotonic()
    clock.advance(90)
    assert clock.now() == datetime(2026, 3, 1, 0, 1, 30, tzinfo=timezone.utc)
    assert clock.monotonic() - before_monotonic == pytest.approx(90.0)


def test_virtual_clock_advance_accumulates():
    clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    for _ in range(10):
        clock.advance(60)
    assert clock.now() == datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc)


def test_virtual_clock_rejects_backwards_time():
    clock = VirtualClock()
    with pytest.raises(ValueError):
        clock.advance(-1)


def test_virtual_clock_sleep_advances_instead_of_blocking():
    clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    clock.sleep(3600)
    assert clock.now() == datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)


def test_virtual_clock_wait_advances_and_reports_event_state():
    clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    event = threading.Event()
    assert clock.wait(event, 30) is False
    assert clock.now() == datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
    event.set()
    assert clock.wait(event, 30) is True
    # An already-set event must not burn simulated time.
    assert clock.now() == datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)


def test_virtual_clock_naive_start_is_treated_as_utc():
    clock = VirtualClock(start=datetime(2026, 5, 5))
    assert clock.now() == datetime(2026, 5, 5, tzinfo=timezone.utc)


def test_parse_iso_round_trips_virtual_clock_output():
    clock = VirtualClock(start=datetime(2026, 7, 4, 12, tzinfo=timezone.utc))
    assert parse_iso(clock.now_iso()) == clock.now()


def test_parse_iso_handles_zulu_and_naive_and_junk():
    assert parse_iso("2026-01-01T00:00:00Z") == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_iso("2026-01-01T00:00:00") == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_iso("not-a-timestamp") is None
    assert parse_iso(None) is None
    assert parse_iso("") is None
