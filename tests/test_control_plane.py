"""A2 — control.json, the modes, and the service singleton lock."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from mr1.autonomy.control import (
    MODE_HALTED,
    MODE_PAUSED,
    MODE_RUNNING,
    MODE_STOPPING,
    ControlPlane,
    ServiceLock,
    ServiceLockError,
)
from mr1.clock import VirtualClock
from mr1.event_log import EventLog


def _control(tmp_path) -> ControlPlane:
    return ControlPlane(tmp_path, clock=VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc)))


def test_missing_control_file_means_running(tmp_path):
    state = _control(tmp_path).read()
    assert state.mode == MODE_RUNNING
    assert state.planning_allowed is True
    assert state.draining is True


def test_set_mode_round_trips_through_disk(tmp_path):
    control = _control(tmp_path)
    control.set_mode(MODE_PAUSED, reason="deploying", requested_by="marwan")

    reread = ControlPlane(tmp_path).read()
    assert reread.mode == MODE_PAUSED
    assert reread.reason == "deploying"
    assert reread.requested_by == "marwan"
    assert reread.requested_at.startswith("2026-01-01")


def test_mode_semantics(tmp_path):
    control = _control(tmp_path)

    control.set_mode(MODE_RUNNING)
    state = control.read()
    assert (state.planning_allowed, state.draining, state.should_exit) == (True, True, False)

    control.set_mode(MODE_PAUSED)
    state = control.read()
    assert (state.planning_allowed, state.draining, state.should_exit) == (False, True, False)

    control.set_mode(MODE_STOPPING)
    state = control.read()
    assert (state.planning_allowed, state.draining, state.should_exit) == (False, True, True)

    control.set_mode(MODE_HALTED)
    state = control.read()
    assert (state.planning_allowed, state.draining, state.should_exit) == (False, False, True)


def test_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _control(tmp_path).set_mode("sideways")


def test_corrupt_control_file_fails_closed_to_paused(tmp_path):
    control = _control(tmp_path)
    control.set_mode(MODE_RUNNING)
    control.path.write_text("{ this is not json", encoding="utf-8")

    state = control.read()
    assert state.mode == MODE_PAUSED
    assert state.corrupt is True
    assert state.planning_allowed is False


def test_unknown_mode_on_disk_fails_closed_to_paused(tmp_path):
    control = _control(tmp_path)
    control.path.write_text(json.dumps({"mode": "yolo"}), encoding="utf-8")

    state = control.read()
    assert state.mode == MODE_PAUSED
    assert state.corrupt is True


def test_mode_change_emits_a_timeline_event(tmp_path):
    control = _control(tmp_path)
    control.set_mode(MODE_PAUSED, reason="lunch", requested_by="marwan")

    events = EventLog(tmp_path / "events").filter_events(event_type="control_mode_changed")
    assert len(events) == 1
    assert events[0].metadata["previous_mode"] == MODE_RUNNING
    assert events[0].metadata["mode"] == MODE_PAUSED
    assert events[0].metadata["reason"] == "lunch"


def test_repeating_a_mode_does_not_re_emit(tmp_path):
    control = _control(tmp_path)
    control.set_mode(MODE_PAUSED)
    control.set_mode(MODE_PAUSED)

    events = EventLog(tmp_path / "events").filter_events(event_type="control_mode_changed")
    assert len(events) == 1


def test_service_lock_is_a_singleton(tmp_path):
    first = ServiceLock(tmp_path)
    pid = first.acquire()
    assert pid > 0
    try:
        with pytest.raises(ServiceLockError, match="already running"):
            ServiceLock(tmp_path).acquire()
    finally:
        first.release()

    # Released — a second supervisor may now start.
    second = ServiceLock(tmp_path)
    second.acquire()
    second.release()


def test_service_lock_reports_liveness(tmp_path):
    lock = ServiceLock(tmp_path)
    assert lock.is_held_by_live_process() is False
    lock.acquire()
    try:
        assert ServiceLock(tmp_path).is_held_by_live_process() is True
        assert ServiceLock(tmp_path).read_pid() == lock.read_pid()
    finally:
        lock.release()
    assert ServiceLock(tmp_path).is_held_by_live_process() is False
