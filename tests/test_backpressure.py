"""
B4 — backpressure and adaptive degradation.

Phase A capped what MR1 could *spend*. This is about what MR1 does when the
machine underneath it is in trouble: the disk is filling, a loop is crashing
repeatedly, the health checks are failing.

The rule every test here enforces is the same one: **backpressure stops MR1
creating work, and never stops it finishing work.** A runtime that sheds
in-flight tasks when the disk gets tight has converted a resource problem into a
correctness problem — half-written workflows, tasks that ran but whose results
were never persisted. So under every kind of pressure: no new plans, and the
scheduler keeps draining.

The second rule is that a refusal you cannot see is indistinguishable from a
bug. But a refusal you see 1 440 times a day is indistinguishable from noise. So
signals are emitted on their edges.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mr1.autonomy.backpressure import (
    CONCURRENCY_CAP,
    DISK_PRESSURE,
    HEALTH_DEGRADED,
    SCHEDULER_DEGRADED,
    SUPERVISOR_DEGRADED,
    BackpressureLimits,
    BackpressureReporter,
    BackpressureSignal,
    RuntimePressure,
    evaluate_backpressure,
)
from mr1.autonomy.budget import BudgetLedger, BudgetLimits
from mr1.autonomy.objectives import KIND_RECURRING, ObjectiveStore
from mr1.autonomy.service import Supervisor, SupervisorConfig
from mr1.clock import VirtualClock
from mr1.worker_runner import MockRunner, RunStatus
from mr1.scoped_agents import AgentStore


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
GIB = 1024**3
MIB = 1024**2


# ---------------------------------------------------------------------------
# The pure evaluator
# ---------------------------------------------------------------------------


def test_a_healthy_runtime_has_no_backpressure():
    signals = evaluate_backpressure(
        RuntimePressure(active_workflows=1, disk_free_bytes=50 * GIB),
        BackpressureLimits(),
    )
    assert signals == []


def test_disk_pressure_stops_planning_before_the_disk_is_full():
    """
    The threshold is not where MR1 dies. It is where MR1 stops digging — with
    room left to finish what is running, persist the results, and run retention.
    """
    limits = BackpressureLimits(min_disk_free_bytes=512 * MIB)

    fine = evaluate_backpressure(
        RuntimePressure(disk_free_bytes=600 * MIB),
        limits,
    )
    assert fine == []

    tight = evaluate_backpressure(
        RuntimePressure(disk_free_bytes=100 * MIB),
        limits,
    )
    assert [signal.code for signal in tight] == [DISK_PRESSURE]
    assert tight[0].observed == 100 * MIB
    assert tight[0].limit == 512 * MIB


def test_an_unreadable_volume_is_not_treated_as_a_full_one():
    """
    `disk_free_bytes` returns -1 when statvfs fails. Reading "unknown" as "no
    space" would wedge planning on any transient OSError — a monitoring failure
    becoming an outage.
    """
    signals = evaluate_backpressure(
        RuntimePressure(disk_free_bytes=-1),
        BackpressureLimits(min_disk_free_bytes=512 * MIB),
    )
    assert signals == []


def test_repeated_loop_failures_enter_degraded_mode():
    limits = BackpressureLimits(
        max_consecutive_supervisor_errors=3,
        max_consecutive_scheduler_errors=5,
    )

    assert evaluate_backpressure(
        RuntimePressure(consecutive_supervisor_errors=2, consecutive_scheduler_errors=4),
        limits,
    ) == []

    degraded = evaluate_backpressure(
        RuntimePressure(consecutive_supervisor_errors=3, consecutive_scheduler_errors=5),
        limits,
    )
    codes = {signal.code for signal in degraded}
    assert codes == {SUPERVISOR_DEGRADED, SCHEDULER_DEGRADED}


def test_every_reason_is_reported_not_just_the_first():
    """An operator with two problems should be told about two problems."""
    signals = evaluate_backpressure(
        RuntimePressure(
            active_workflows=10,
            disk_free_bytes=1 * MIB,
            consecutive_supervisor_errors=99,
            health_status="error",
        ),
        BackpressureLimits(max_concurrent_workflows=4),
    )
    codes = {signal.code for signal in signals}
    assert codes == {HEALTH_DEGRADED, DISK_PRESSURE, SUPERVISOR_DEGRADED, CONCURRENCY_CAP}


def test_limits_are_validated():
    with pytest.raises(ValueError):
        BackpressureLimits(max_concurrent_workflows=0).validate()
    with pytest.raises(ValueError):
        BackpressureLimits(min_disk_free_bytes=-1).validate()


# ---------------------------------------------------------------------------
# Observability without flooding
# ---------------------------------------------------------------------------


def test_a_sustained_signal_is_emitted_once_not_once_per_tick():
    """
    A 60-second tick under six hours of disk pressure would otherwise write 360
    identical events. That is not observability; it is a wall of text that
    guarantees the one line that mattered is missed.
    """
    emitted: list[tuple[str, dict]] = []

    def emit(event_type, *, status, summary, metadata):
        emitted.append((event_type, metadata))

    reporter = BackpressureReporter(emit=emit)
    signal = BackpressureSignal(code=DISK_PRESSURE, detail="disk is low")

    for _ in range(100):
        reporter.observe([signal])

    assert len(emitted) == 1, "one edge, not a hundred ticks"
    assert emitted[0][0] == "backpressure_applied"
    assert reporter.active_codes == {DISK_PRESSURE}


def test_a_lifted_signal_says_so():
    emitted: list[str] = []

    def emit(event_type, *, status, summary, metadata):
        emitted.append(event_type)

    reporter = BackpressureReporter(emit=emit)
    signal = BackpressureSignal(code=DISK_PRESSURE, detail="disk is low")

    reporter.observe([signal])
    reporter.observe([signal])
    reporter.observe([])          # pressure released
    reporter.observe([])

    assert emitted == ["backpressure_applied", "backpressure_lifted"]
    assert reporter.active_codes == set()


def test_a_new_signal_during_an_existing_one_is_still_reported():
    emitted: list[tuple[str, str]] = []

    def emit(event_type, *, status, summary, metadata):
        emitted.append((event_type, metadata["code"]))

    reporter = BackpressureReporter(emit=emit)
    disk = BackpressureSignal(code=DISK_PRESSURE, detail="d")
    degraded = BackpressureSignal(code=SUPERVISOR_DEGRADED, detail="s")

    reporter.observe([disk])
    reporter.observe([disk, degraded])

    assert emitted == [
        ("backpressure_applied", DISK_PRESSURE),
        ("backpressure_applied", SUPERVISOR_DEGRADED),
    ]


def test_a_broken_emitter_never_breaks_the_loop():
    def emit(*_args, **_kwargs):
        raise RuntimeError("the timeline is on fire")

    reporter = BackpressureReporter(emit=emit)
    reporter.observe([BackpressureSignal(code=DISK_PRESSURE, detail="d")])
    assert reporter.active_codes == {DISK_PRESSURE}


# ---------------------------------------------------------------------------
# Through the real supervisor
# ---------------------------------------------------------------------------


class _CountingPlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, *_args, **_kwargs):
        self.calls += 1
        return {
            "title": "work",
            "tasks": [{
                "label": "t",
                "title": "T",
                "task_kind": "agent",
                "agent_type": "worker",
                "prompt": "do it",
            }],
        }


def _runtime(tmp_path, clock, planner, runner, **config):
    root = tmp_path / "runtime"
    agents = AgentStore(root=root / "agents")
    objectives = ObjectiveStore(root, clock=clock)
    objectives.create(
        title="Recurring",
        statement="keep doing the thing",
        kind=KIND_RECURRING,
        trigger={"type": "interval", "interval_s": 60},
        owner_agent_id=agents.root_agent_id,
    )
    supervisor = Supervisor(
        root,
        config=SupervisorConfig(
            tick_interval_s=60,
            retention_interval_s=0,
            **config,
        ),
        clock=clock,
        runner=runner,
        auto_scheduler_tick=False,
        planner=planner,
    )
    return root, supervisor


def test_disk_pressure_stops_new_plans_but_the_scheduler_keeps_draining(tmp_path, monkeypatch):
    """
    The load-bearing test of the checkpoint.

    Under disk pressure MR1 must create nothing and finish everything. A runtime
    that abandoned its in-flight task here would leave a workflow that ran but
    whose result was never written — exactly the corruption the pressure was
    supposed to prevent.
    """
    clock = VirtualClock(start=START)
    planner = _CountingPlanner()
    runner = MockRunner()
    _root, supervisor = _runtime(
        tmp_path, clock, planner, runner,
        min_disk_free_bytes=512 * MIB,
    )

    # Plenty of room: it plans, and the work starts.
    monkeypatch.setattr("mr1.autonomy.service.disk_free_bytes", lambda _p: 50 * GIB)
    supervisor.tick()
    assert planner.calls == 1
    supervisor.scheduler.tick()
    workflows = supervisor.scheduler.list_workflows()
    assert len(workflows) == 1
    task_id = next(iter(workflows[0].tasks))

    # The disk fills while that workflow is in flight.
    monkeypatch.setattr("mr1.autonomy.service.disk_free_bytes", lambda _p: 100 * MIB)
    clock.advance(600)
    outcome = supervisor.tick()

    assert outcome["gate"] == "draining"
    assert planner.calls == 1, "no new work may be created under disk pressure"
    assert DISK_PRESSURE in supervisor.backpressure.active_codes

    # And the in-flight task still finishes.
    runner.complete(task_id, RunStatus.SUCCEEDED, summary="done")
    supervisor.scheduler.tick()
    from mr1.workflow_models import WorkflowStatus

    assert supervisor.scheduler.get_workflow(
        workflows[0].workflow_id
    ).status is WorkflowStatus.SUCCEEDED, "draining must still drain"

    # Space is freed: planning resumes on its own.
    monkeypatch.setattr("mr1.autonomy.service.disk_free_bytes", lambda _p: 50 * GIB)
    clock.advance(600)
    supervisor.tick()
    assert planner.calls == 2
    assert supervisor.backpressure.active_codes == set()
    supervisor.shutdown()


def test_the_pressure_is_visible_on_the_timeline_exactly_twice(tmp_path, monkeypatch):
    from mr1.event_log import EventLog

    clock = VirtualClock(start=START)
    root, supervisor = _runtime(
        tmp_path, clock, _CountingPlanner(), MockRunner(),
        min_disk_free_bytes=512 * MIB,
    )

    monkeypatch.setattr("mr1.autonomy.service.disk_free_bytes", lambda _p: 100 * MIB)
    for _ in range(20):
        clock.advance(60)
        supervisor.tick()

    monkeypatch.setattr("mr1.autonomy.service.disk_free_bytes", lambda _p: 50 * GIB)
    clock.advance(60)
    supervisor.tick()

    events = [
        event.event_type
        for event in EventLog(root / "events").list_events()
        if event.event_type.startswith("backpressure_")
    ]
    assert events == ["backpressure_applied", "backpressure_lifted"], (
        "21 ticks under pressure must produce 2 events, not 21"
    )
    supervisor.shutdown()


def test_a_repeatedly_crashing_supervisor_degrades_instead_of_planning(tmp_path):
    clock = VirtualClock(start=START)
    planner = _CountingPlanner()
    _root, supervisor = _runtime(
        tmp_path, clock, planner, MockRunner(),
        max_consecutive_supervisor_errors=2,
    )

    # Force the reconcile phase to blow up, so the tick records a failure.
    def explode(*_args, **_kwargs):
        raise RuntimeError("state is unreadable")

    supervisor._reconcile = explode

    for _ in range(2):
        clock.advance(60)
        supervisor.tick()
    assert supervisor._consecutive_tick_errors >= 2

    # Repair the fault: the loop works again, but MR1 has been failing and must
    # not resume creating work in state it just proved it cannot read.
    supervisor._reconcile = lambda *_a, **_k: None
    clock.advance(60)
    outcome = supervisor.tick()

    assert outcome["gate"] == "draining"
    assert SUPERVISOR_DEGRADED in supervisor.backpressure.active_codes
    assert planner.calls == 0, "a degraded supervisor plans nothing"
    supervisor.shutdown()


def test_the_concurrency_cap_is_read_from_shared_disk_state(tmp_path):
    """
    The cap counts workflows in the store, not in memory — so it holds across
    processes, and it holds across a restart that forgot everything.
    """
    clock = VirtualClock(start=START)
    planner = _CountingPlanner()
    _root, supervisor = _runtime(
        tmp_path, clock, planner, MockRunner(),
        max_concurrent_workflows=1,
    )

    supervisor.tick()
    assert planner.calls == 1
    supervisor.scheduler.tick()

    clock.advance(600)
    outcome = supervisor.tick()

    assert outcome["gate"] == "draining"
    assert CONCURRENCY_CAP in supervisor.backpressure.active_codes
    assert planner.calls == 1
    supervisor.shutdown()


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------


def test_a_restart_does_not_reset_the_spending_windows(tmp_path):
    """
    Budgets are the enforcement that must survive a restart. If they did not, an
    objective that kept crashing MR1 would get a fresh hour of planning budget
    every time it did — a crash loop that pays for itself.
    """
    clock = VirtualClock(start=START)
    root = tmp_path / "runtime"
    limits = BudgetLimits(max_plans_per_hour=3)

    first = BudgetLedger(root, clock=clock, limits=limits)
    for _ in range(3):
        assert first.try_consume_plan().allowed is True
    assert first.try_consume_plan().allowed is False

    # A brand-new process, same runtime root.
    second = BudgetLedger(root, clock=clock, limits=limits)
    assert second.try_consume_plan().allowed is False, (
        "a restart must not hand back a spent budget"
    )
    assert second.snapshot()["plans_this_hour"] == 3

    # The window rolls on wall-clock time, not on process lifetime.
    clock.advance(3601)
    assert second.try_consume_plan().allowed is True


def test_budget_windows_use_the_injected_clock(tmp_path):
    clock = VirtualClock(start=START)
    ledger = BudgetLedger(
        tmp_path / "runtime",
        clock=clock,
        limits=BudgetLimits(max_plans_per_hour=1),
    )

    assert ledger.try_consume_plan().allowed is True
    assert ledger.try_consume_plan().allowed is False

    clock.advance(3599)
    assert ledger.try_consume_plan().allowed is False, "still inside the hour"

    clock.advance(2)
    assert ledger.try_consume_plan().allowed is True, "the window rolled"
