"""A2 — the headless host: heartbeat, health.json, gating, drain, halt."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mr1.autonomy.control import (
    MODE_HALTED,
    MODE_PAUSED,
    MODE_RUNNING,
    MODE_STOPPING,
    ControlPlane,
)
from mr1.autonomy.health import HealthReporter, heartbeat_age_s, heartbeat_is_stale, read_health
from mr1.autonomy.service import Supervisor, SupervisorConfig
from mr1.clock import VirtualClock
from mr1.kazi_runner import MockRunner, RunResult, RunStatus
from mr1.workflow_models import Provenance, WorkflowStatus
from mr1.workflow_store import WorkflowStore


class _FakeDoctor:
    def __init__(self, status: str = "ok"):
        self.status = status
        self.summary = {"checks_run": 0}
        self.calls = 0

    def __call__(self, _runtime_root):
        self.calls += 1
        return self


def _supervisor(tmp_path, *, clock=None, doctor=None, runner=None, config=None):
    clock = clock or VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    runtime_root = tmp_path / "runtime"
    health = HealthReporter(
        runtime_root,
        clock=clock,
        doctor_fn=doctor,
        doctor_interval_s=0.0,
    )
    supervisor = Supervisor(
        runtime_root,
        config=config or SupervisorConfig(tick_interval_s=60.0),
        clock=clock,
        runner=runner or MockRunner(on_poll=lambda _h: RunResult(status=RunStatus.SUCCEEDED, exit_code=0, summary="ok")),
        health_reporter=health,
        workspace_root=tmp_path,
        auto_scheduler_tick=False,
    )
    return supervisor, clock, runtime_root


def _spec(title="autonomy"):
    return {"title": title, "tasks": [{"label": "one", "prompt": "work"}]}


def test_tick_writes_heartbeat_and_health_json(tmp_path):
    supervisor, clock, runtime_root = _supervisor(tmp_path, doctor=_FakeDoctor("ok"))

    supervisor.tick()

    payload = read_health(runtime_root)
    assert payload is not None
    assert payload["supervisor_heartbeat_at"].startswith("2026-01-01")
    assert payload["mode"] == MODE_RUNNING
    assert payload["doctor_status"] == "ok"
    assert payload["gauges"]["active_workflows"] == 0
    assert payload["gauges"]["supervisor_tick_count"] == 1
    assert heartbeat_age_s(payload, clock=clock) == 0.0


def test_heartbeat_goes_stale_when_the_supervisor_stops_ticking(tmp_path):
    supervisor, clock, runtime_root = _supervisor(tmp_path, doctor=_FakeDoctor())
    supervisor.tick()
    payload = read_health(runtime_root)

    assert heartbeat_is_stale(payload, tick_interval_s=60.0, clock=clock) is False
    clock.advance(60 * 4)
    assert heartbeat_is_stale(payload, tick_interval_s=60.0, clock=clock) is True
    assert heartbeat_age_s(payload, clock=clock) == 240.0


def test_missing_health_file_reads_as_stale(tmp_path):
    assert read_health(tmp_path) is None
    assert heartbeat_is_stale(None, tick_interval_s=60.0) is True


def test_steady_state_ticks_are_cheap_and_quiet(tmp_path):
    supervisor, _clock, runtime_root = _supervisor(tmp_path, doctor=_FakeDoctor())
    for _ in range(10):
        supervisor.tick()

    from mr1.event_log import EventLog

    events = EventLog(runtime_root / "events").list_events()
    # An idle tick must not append to the timeline; the pulse lives in health.json.
    assert [event for event in events if event.event_type == "supervisor_tick"] == []
    assert supervisor.metrics()["supervisor_tick_count"] == 10
    assert supervisor.metrics()["supervisor_tick_errors"] == 0


def test_gate_allows_planning_when_running_and_healthy(tmp_path):
    supervisor, _clock, _root = _supervisor(tmp_path, doctor=_FakeDoctor("ok"))
    assert supervisor.tick()["gate"] == "planning"


def test_paused_gates_planning_but_keeps_draining(tmp_path):
    supervisor, _clock, runtime_root = _supervisor(tmp_path, doctor=_FakeDoctor())
    ControlPlane(runtime_root).set_mode(MODE_PAUSED, reason="test")

    outcome = supervisor.tick()

    assert outcome["gate"] == "draining"
    assert supervisor.exit_requested is False


def test_degraded_health_gates_planning(tmp_path):
    supervisor, _clock, _root = _supervisor(tmp_path, doctor=_FakeDoctor("error"))

    outcome = supervisor.tick()

    assert outcome["gate"] == "draining"
    assert any(item["source"] == "supervisor_health" for item in supervisor.runtime_errors)


def test_concurrency_cap_gates_planning(tmp_path):
    config = SupervisorConfig(tick_interval_s=60.0, max_concurrent_workflows=1)
    supervisor, _clock, _root = _supervisor(tmp_path, doctor=_FakeDoctor(), config=config)
    supervisor.scheduler.submit_workflow(_spec(), Provenance(type="user", id="test"))

    assert supervisor.tick()["gate"] == "draining"


def test_stopping_drains_in_flight_work_then_exits(tmp_path):
    runner = MockRunner()
    supervisor, _clock, runtime_root = _supervisor(tmp_path, doctor=_FakeDoctor(), runner=runner)
    workflow_id = supervisor.scheduler.submit_workflow(_spec(), Provenance(type="user", id="test"))
    supervisor.scheduler.tick()

    ControlPlane(runtime_root).set_mode(MODE_STOPPING, reason="test")

    # Work is still in flight: stopping drains, it does not exit.
    assert supervisor.tick()["gate"] == "draining"
    assert supervisor.exit_requested is False

    store = WorkflowStore(root=runtime_root / "workflows")
    task_id = next(iter(store.load_workflow(workflow_id).tasks))
    runner.complete(task_id, RunStatus.SUCCEEDED, exit_code=0, summary="done")
    supervisor.scheduler.tick()
    supervisor.scheduler.tick()

    assert supervisor.tick()["gate"] == "exit"
    assert supervisor.exit_requested is True
    assert store.load_workflow(workflow_id).status is WorkflowStatus.SUCCEEDED


def test_halt_cancels_in_flight_work_and_exits(tmp_path):
    runner = MockRunner()
    supervisor, _clock, runtime_root = _supervisor(tmp_path, doctor=_FakeDoctor(), runner=runner)
    workflow_id = supervisor.scheduler.submit_workflow(_spec(), Provenance(type="user", id="test"))
    supervisor.scheduler.tick()

    ControlPlane(runtime_root).set_mode(MODE_HALTED, reason="emergency")

    outcome = supervisor.tick()

    assert outcome["gate"] == "exit"
    assert supervisor.exit_requested is True
    store = WorkflowStore(root=runtime_root / "workflows")
    assert store.load_workflow(workflow_id).status is WorkflowStatus.CANCELLED


def test_supervisor_survives_a_failing_tick_and_reports_it(tmp_path):
    supervisor, _clock, runtime_root = _supervisor(tmp_path, doctor=_FakeDoctor())

    def explode(_state):
        raise RuntimeError("sweep exploded")

    supervisor.sweep = explode

    outcome = supervisor.tick()

    assert "sweep exploded" in outcome["error"]
    metrics = supervisor.metrics()
    assert metrics["supervisor_tick_errors"] == 1
    assert metrics["supervisor_consecutive_tick_errors"] == 1
    assert any(item["source"] == "supervisor_tick" for item in supervisor.runtime_errors)

    from mr1.event_log import EventLog

    events = EventLog(runtime_root / "events").filter_events(event_type="supervisor_tick_failed")
    assert len(events) == 1
    # The heartbeat is still written on a failing tick — a wedged supervisor
    # that stopped beating and one that is erroring must be distinguishable.
    assert read_health(runtime_root)["supervisor_heartbeat_at"]


def test_serve_drains_a_workflow_with_no_repl(tmp_path):
    """The A2 acceptance property: submitted work advances without a human."""
    runner = MockRunner(on_poll=lambda _h: RunResult(status=RunStatus.SUCCEEDED, exit_code=0, summary="ok"))
    supervisor, _clock, runtime_root = _supervisor(tmp_path, doctor=_FakeDoctor(), runner=runner)
    workflow_id = supervisor.scheduler.submit_workflow(_spec(), Provenance(type="user", id="test"))

    # The scheduler is the executor; the supervisor only hosts it.
    for _ in range(3):
        supervisor.scheduler.tick()
    supervisor.tick()

    store = WorkflowStore(root=runtime_root / "workflows")
    assert store.load_workflow(workflow_id).status is WorkflowStatus.SUCCEEDED
    assert read_health(runtime_root)["gauges"]["active_workflows"] == 0


def test_doctor_runs_on_its_own_cadence_not_every_tick(tmp_path):
    clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
    doctor = _FakeDoctor()
    runtime_root = tmp_path / "runtime"
    health = HealthReporter(runtime_root, clock=clock, doctor_fn=doctor, doctor_interval_s=300.0)
    supervisor = Supervisor(
        runtime_root,
        config=SupervisorConfig(tick_interval_s=60.0),
        clock=clock,
        runner=MockRunner(),
        health_reporter=health,
        workspace_root=tmp_path,
        auto_scheduler_tick=False,
    )

    for _ in range(4):
        supervisor.tick()
        clock.advance(60)

    assert doctor.calls == 1

    clock.advance(300)
    supervisor.tick()
    assert doctor.calls == 2


def test_health_json_is_valid_json_on_disk(tmp_path):
    supervisor, _clock, runtime_root = _supervisor(tmp_path, doctor=_FakeDoctor())
    supervisor.tick()
    payload = json.loads((runtime_root / "health.json").read_text(encoding="utf-8"))
    assert payload["pid"] == 0  # not serving; pid is set by serve()
    assert "gauges" in payload
