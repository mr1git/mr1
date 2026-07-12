"""A1 — a wedged background loop must be distinguishable from an idle one."""

from __future__ import annotations

import threading
import time

import pytest

from mr1.event_log import EventLog
from mr1.kazi_runner import MockRunner
from mr1.scheduler import Scheduler
from mr1.workflow_store import WorkflowStore


class _ExplodingQueries:
    def __init__(self, exc: Exception):
        self._exc = exc

    def active_workflows(self, _live_handles):
        raise self._exc


def _scheduler(tmp_path, *, sink=None, auto_tick=False, tick_interval_s=0.01):
    store = WorkflowStore(root=tmp_path / "workflows")
    return Scheduler(
        store,
        MockRunner(),
        auto_tick=auto_tick,
        tick_interval_s=tick_interval_s,
        workspace_root=tmp_path,
        runtime_error_sink=sink,
    )


def test_tick_metrics_start_clean(tmp_path):
    scheduler = _scheduler(tmp_path)
    metrics = scheduler.metrics()
    assert metrics["tick_count"] == 0
    assert metrics["tick_errors"] == 0
    assert metrics["last_tick_error"] is None


def test_healthy_ticks_increment_the_counter_and_record_duration(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler.tick()
    scheduler.tick()
    metrics = scheduler.metrics()
    assert metrics["tick_count"] == 2
    assert metrics["tick_errors"] == 0
    assert metrics["last_tick_at"]
    assert metrics["last_tick_duration_ms"] >= 0.0


def test_tick_still_raises_for_direct_callers(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler._workflow_queries = _ExplodingQueries(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        scheduler.tick()


def test_loop_failure_is_counted_persisted_and_emitted(tmp_path):
    recorded: list[tuple] = []

    def sink(source, error, *, details=None):
        recorded.append((source, error, details))

    scheduler = _scheduler(tmp_path, sink=sink)
    scheduler._workflow_queries = _ExplodingQueries(RuntimeError("tick exploded"))

    try:
        scheduler.tick()
    except RuntimeError:
        pass
    scheduler._record_tick_failure(RuntimeError("tick exploded"))

    metrics = scheduler.metrics()
    assert metrics["tick_errors"] == 1
    assert metrics["consecutive_tick_errors"] == 1
    assert "tick exploded" in metrics["last_tick_error"]
    assert metrics["last_tick_error_at"]

    assert recorded and recorded[0][0] == "scheduler_tick"
    assert "tick exploded" in recorded[0][1]
    assert recorded[0][2]["tick_errors"] == 1
    assert "traceback" in recorded[0][2]

    events = EventLog(tmp_path / "events").filter_events(event_type="scheduler_tick_failed")
    assert len(events) == 1
    assert events[0].severity == "ERROR"
    assert events[0].metadata["error_type"] == "RuntimeError"


def test_running_loop_survives_a_crashing_tick_and_reports_it(tmp_path):
    recorded: list[tuple] = []

    def sink(source, error, *, details=None):
        recorded.append((source, error, details))

    scheduler = _scheduler(tmp_path, sink=sink, auto_tick=True, tick_interval_s=0.01)
    scheduler._workflow_queries = _ExplodingQueries(RuntimeError("wedged"))
    try:
        deadline = time.monotonic() + 5.0
        while scheduler.metrics()["tick_errors"] < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        metrics = scheduler.metrics()
        assert metrics["tick_errors"] >= 2, "loop stopped reporting failures"
        assert metrics["consecutive_tick_errors"] >= 2
        assert recorded, "no runtime error was persisted"
        assert scheduler._thread.is_alive(), "the loop died instead of failing loud"
    finally:
        scheduler.shutdown()

    events = EventLog(tmp_path / "events").filter_events(event_type="scheduler_tick_failed")
    assert len(events) >= 2


def test_consecutive_error_counter_resets_after_a_clean_tick(tmp_path):
    scheduler = _scheduler(tmp_path)
    scheduler._record_tick_failure(RuntimeError("one"))
    scheduler._record_tick_failure(RuntimeError("two"))
    assert scheduler.metrics()["consecutive_tick_errors"] == 2

    scheduler.tick()

    metrics = scheduler.metrics()
    assert metrics["consecutive_tick_errors"] == 0
    assert metrics["tick_errors"] == 2, "the lifetime error count must not be reset"


def test_error_sink_failure_cannot_kill_the_loop(tmp_path):
    def hostile_sink(source, error, *, details=None):
        raise ValueError("sink is broken too")

    scheduler = _scheduler(tmp_path, sink=hostile_sink)
    scheduler._record_tick_failure(RuntimeError("boom"))
    assert scheduler.metrics()["tick_errors"] == 1
