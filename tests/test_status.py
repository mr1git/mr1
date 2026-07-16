"""
B3 — operator status and health.

The gauges all existed. What did not exist was an answer to the only question an
operator ever actually asks: *is anything stuck, and do I have to do something?*

So these tests are written against states, not fields. Each one puts the runtime
into a condition an operator would recognise — paused, halted, wedged, waiting
on an approval nobody answered, out of disk, quarantined — and asserts that
`mr1 status` rolls up to the right severity, names the problem in a sentence,
and exits with a code a cron job can branch on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mr1.autonomy.consent import ConsentGrantStore
from mr1.autonomy.control import MODE_HALTED, MODE_PAUSED, ControlPlane
from mr1.autonomy.health import HealthReporter
from mr1.autonomy.objectives import KIND_ONCE, ObjectiveStore
from mr1.autonomy.ownership import ROLE_SERVICE, ExecutionOwnership
from mr1.autonomy.status import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_WARNING,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_WARNING,
    StatusThresholds,
    collect_status,
)
from mr1.clock import VirtualClock
from mr1.scoped_agents import PersistentAgentStore


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir(parents=True)
    (root / "workflows").mkdir()
    PersistentAgentStore(root=root / "agents")
    return root


def _clock(**delta) -> VirtualClock:
    return VirtualClock(start=START + timedelta(**delta))


def _beat(root, clock, *, mode="running", doctor="ok", gauges=None) -> None:
    """Write a heartbeat as a live supervisor would."""
    reporter = HealthReporter(root, clock=clock, doctor_fn=None)
    reporter._last_doctor_status = doctor  # the doctor is a disk scan; pin it
    reporter.write(
        pid=4321,
        mode=mode,
        started_at=clock.now_iso(),
        uptime_s=100.0,
        gauges=gauges or {},
    )


def _codes(report) -> set[str]:
    return {finding.code for finding in report.findings}


# ---------------------------------------------------------------------------
# Healthy
# ---------------------------------------------------------------------------


def test_a_healthy_runtime_says_nothing_needs_you(runtime):
    clock = _clock()
    _beat(runtime, clock)

    report = collect_status(runtime, clock=clock)

    assert report.health == STATUS_OK
    assert report.exit_code == EXIT_OK
    assert report.findings == []
    assert report.schema_version == 1
    assert report.service["mode"] == "running"


def test_the_payload_schema_is_stable(runtime):
    """Automation reads this. The shape must not depend on what happens to exist."""
    report = collect_status(runtime, clock=_clock())
    payload = report.to_dict()

    for section in (
        "schema_version", "generated_at", "health", "findings",
        "service", "scheduler", "objectives", "workflows",
        "approvals", "grants", "budgets", "autonomy", "storage", "triage",
    ):
        assert section in payload, f"missing top-level section: {section}"

    assert isinstance(payload["findings"], list)
    assert payload["service"]["mode"] == "running"
    # Empty runtime: the sections exist and are empty, not absent.
    assert payload["objectives"]["total"] == 0
    assert payload["approvals"]["pending"] == 0


# ---------------------------------------------------------------------------
# Control-plane states
# ---------------------------------------------------------------------------


def test_paused_is_a_warning_not_a_failure(runtime):
    clock = _clock()
    ControlPlane(runtime, clock=clock).set_mode(MODE_PAUSED, reason="operator went to lunch")
    _beat(runtime, clock, mode=MODE_PAUSED)

    report = collect_status(runtime, clock=clock)

    assert report.health == STATUS_WARNING
    assert report.exit_code == EXIT_WARNING
    assert "paused" in _codes(report)
    assert "lunch" in report.findings[0].detail
    assert "mr1 resume" in report.findings[0].action


def test_halted_is_an_error_and_says_authority_is_gone(runtime):
    clock = _clock()
    ControlPlane(runtime, clock=clock).set_mode(MODE_HALTED, reason="emergency")
    _beat(runtime, clock, mode=MODE_HALTED)

    report = collect_status(runtime, clock=clock)

    assert report.health == STATUS_ERROR
    assert report.exit_code == EXIT_ERROR
    assert "halted" in _codes(report)
    assert "consent is revoked" in report.findings[0].detail.lower()


def test_a_corrupt_control_file_is_an_error_and_reads_as_paused(runtime):
    clock = _clock()
    (runtime / "control.json").write_text("{not json", encoding="utf-8")

    report = collect_status(runtime, clock=clock)

    assert report.health == STATUS_ERROR
    assert "control_file_corrupt" in _codes(report)
    assert report.service["mode"] == MODE_PAUSED, "fail closed"


# ---------------------------------------------------------------------------
# Stale heartbeat — the signal nothing else provides
# ---------------------------------------------------------------------------


def test_a_stale_heartbeat_is_detected_when_the_process_is_alive_but_wedged(runtime):
    """
    The most important check in the file: a supervisor holding its lock but not
    ticking looks exactly like a healthy one from the outside. Nothing but the
    heartbeat age can tell them apart.
    """
    clock = _clock()
    _beat(runtime, clock)

    lock = _hold_service_lock(runtime)
    try:
        later = _clock(seconds=600)  # ten minutes, tick interval 60s
        report = collect_status(
            runtime,
            clock=later,
            thresholds=StatusThresholds(tick_interval_s=60, max_missed_heartbeats=3),
        )

        assert report.health == STATUS_ERROR
        assert "stale_heartbeat" in _codes(report)
        assert report.service["heartbeat_age_s"] == pytest.approx(600, abs=1)
    finally:
        lock.release()


def test_a_stale_heartbeat_from_a_dead_supervisor_is_not_an_error(runtime):
    """
    A stopped MR1 is not a broken MR1. Only a *running* supervisor that has gone
    quiet is wedged; one that exited cleanly just is not there, and shouting
    about it every time you check status is how alerts get ignored.
    """
    clock = _clock()
    _beat(runtime, clock)

    report = collect_status(runtime, clock=_clock(seconds=6000))

    assert "stale_heartbeat" not in _codes(report)
    assert report.service["supervisor_running"] is False


def _hold_service_lock(runtime):
    from mr1.autonomy.control import ServiceLock

    # `is_held_by_live_process` probes the flock, so an actual holder is needed —
    # a fabricated pidfile would be exactly the stale-PID trap B8 avoids.
    import subprocess
    import sys
    import textwrap
    import time

    script = textwrap.dedent(
        """
        import sys, time
        from pathlib import Path
        from mr1.autonomy.control import ServiceLock
        lock = ServiceLock(Path(sys.argv[1]))
        lock.acquire()
        Path(sys.argv[2]).write_text("held")
        time.sleep(30)
        """
    )
    path = runtime / "holder.py"
    path.write_text(script, encoding="utf-8")
    flag = runtime / "held.flag"

    import os

    env = dict(os.environ)
    repo = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [sys.executable, str(path), str(runtime), str(flag)],
        cwd=repo,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 20
    while not flag.exists() and time.time() < deadline:
        time.sleep(0.02)

    class _Holder:
        def release(self):
            proc.kill()
            proc.wait(timeout=10)

    assert flag.exists(), "the holder process never took the service lock"
    return _Holder()


# ---------------------------------------------------------------------------
# Blocked work
# ---------------------------------------------------------------------------


def test_an_unanswered_approval_escalates_from_warning_to_error(runtime):
    from mr1.capability_policy import (
        CapabilityApprovalStore,
        CapabilityRequest,
        PolicyEngine,
        ScopeContext,
        build_approval_request,
        maybe_route_approval_request,
        metadata_for_capability,
    )
    from mr1.messages import MessageStore

    clock = _clock()
    _beat(runtime, clock)

    agents = PersistentAgentStore(root=runtime / "agents")
    approvals = CapabilityApprovalStore(runtime / "capability_approvals", clock=clock)
    request = CapabilityRequest(
        actor_id=agents.root_agent_id,
        actor_type="mr1",
        actor_clearance=0.99,
        invocation_mode="workflow",
        capability_name="shell_command",
        args={"argv": ["pytest"], "cwd": str(runtime)},
        scope=ScopeContext(allowed_roots=[runtime], workspace_root=runtime),
        workflow_id="wf-1",
        task_id="tk-1",
    )
    metadata = metadata_for_capability("shell_command", "tool")
    decision = PolicyEngine().evaluate(request, metadata)
    maybe_route_approval_request(
        build_approval_request(request, metadata, decision),
        approval_store=approvals,
        message_store=MessageStore(root=runtime / "messages", scoped_agent_store=agents),
        scoped_agent_store=agents,
    )

    # Fresh: nothing to say.
    assert collect_status(runtime, clock=clock).health == STATUS_OK

    # Two hours later: worth mentioning.
    warn = collect_status(runtime, clock=_clock(hours=2))
    assert warn.health == STATUS_WARNING
    assert "approval_pending" in _codes(warn)

    # Two days later: MR1 has been stalled for two days.
    err = collect_status(runtime, clock=_clock(days=2))
    assert err.health == STATUS_ERROR
    assert "approval_stalled" in _codes(err)
    assert err.approvals["pending"] == 1
    assert err.approvals["oldest_pending"]["capability"] == "shell_command"
    assert "mr1 approvals approve" in err.findings[0].action


def test_a_quarantined_objective_is_an_error_and_a_waiting_one_is_a_warning(runtime):
    clock = _clock()
    _beat(runtime, clock)
    agents = PersistentAgentStore(root=runtime / "agents")
    store = ObjectiveStore(runtime, clock=clock)

    waiting = store.create(
        title="Needs consent",
        statement="do the thing",
        kind=KIND_ONCE,
        owner_agent_id=agents.root_agent_id,
    )
    store.set_status(waiting.objective_id, "waiting_human", reason="needs shell consent")

    warn = collect_status(runtime, clock=clock)
    assert warn.health == STATUS_WARNING
    assert "objectives_waiting_human" in _codes(warn)

    gave_up = store.create(
        title="Gave up",
        statement="do the impossible",
        kind=KIND_ONCE,
        owner_agent_id=agents.root_agent_id,
    )
    store.set_status(gave_up.objective_id, "quarantined", reason="budget exhausted")

    err = collect_status(runtime, clock=clock)
    assert err.health == STATUS_ERROR
    assert "objectives_quarantined" in _codes(err)
    assert err.objectives["quarantined"] == 1
    assert err.objectives["waiting_human"] == 1
    assert {item["objective_id"] for item in err.objectives["blocked"]} == {
        waiting.objective_id,
        gave_up.objective_id,
    }


# ---------------------------------------------------------------------------
# Degradation and pressure
# ---------------------------------------------------------------------------


def test_disk_pressure_warns_then_errors(runtime, monkeypatch):
    clock = _clock()
    _beat(runtime, clock)

    def free(_path, value=[0]):
        return value[0]

    monkeypatch.setattr("mr1.autonomy.health.disk_free_bytes", lambda _p: free.value)

    free.value = 10 * 1024**3
    assert collect_status(runtime, clock=clock).health == STATUS_OK

    free.value = 1 * 1024**3  # 1 GiB — under the 2 GiB warn line
    warn = collect_status(runtime, clock=clock)
    assert warn.health == STATUS_WARNING
    assert "disk_low" in _codes(warn)

    free.value = 100 * 1024**2  # 100 MiB — MR1 will stop creating work
    err = collect_status(runtime, clock=clock)
    assert err.health == STATUS_ERROR
    assert "disk_critical" in _codes(err)
    assert err.storage["disk_free_bytes"] == 100 * 1024**2


def test_a_doctor_error_is_surfaced_as_degraded_health(runtime):
    clock = _clock()
    _beat(runtime, clock, doctor="error")

    report = collect_status(runtime, clock=clock)

    assert report.health == STATUS_ERROR
    assert "doctor_error" in _codes(report)


def test_scheduler_tick_errors_are_surfaced(runtime):
    clock = _clock()
    _beat(
        runtime,
        clock,
        gauges={"scheduler_tick_errors": 4, "scheduler_last_tick_error": "OSError: boom"},
    )

    report = collect_status(runtime, clock=clock)

    assert report.health == STATUS_WARNING
    assert "scheduler_tick_errors" in _codes(report)
    assert report.scheduler["tick_errors"] == 4


def test_an_expiring_grant_warns_before_the_objective_stalls(runtime):
    clock = _clock()
    _beat(runtime, clock)
    agents = PersistentAgentStore(root=runtime / "agents")
    grants = ConsentGrantStore(runtime, clock=clock, scoped_agent_store=agents)

    grants.create(
        objective_id="obj-1",
        capability_name="shell_command",
        scope_roots=[str(runtime)],
        max_risk=1.0,
        granted_by=agents.root_agent_id,
        ttl_s=3600,  # expires in an hour
        reason="genesis",
    )

    report = collect_status(runtime, clock=clock)

    assert report.health == STATUS_WARNING
    assert "grants_expiring" in _codes(report)
    assert report.grants["active"] == 1
    assert report.grants["expiring_soon"] == 1


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def test_status_reports_who_owns_execution(runtime):
    clock = _clock()
    _beat(runtime, clock)

    assert collect_status(runtime, clock=clock).service["execution_owned"] is False

    ownership = ExecutionOwnership(runtime, role=ROLE_SERVICE)
    ownership.acquire()
    try:
        report = collect_status(runtime, clock=clock)
        assert report.service["execution_owned"] is True
        assert report.service["execution_owner"]["role"] == ROLE_SERVICE
    finally:
        ownership.release()


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_status_never_dies_on_a_corrupt_store(runtime):
    """
    A status command that crashes on a broken runtime fails exactly when it is
    needed. Every section degrades to a recorded error instead.
    """
    clock = _clock()
    _beat(runtime, clock)
    (runtime / "objectives").mkdir(parents=True, exist_ok=True)
    (runtime / "objectives" / "obj-broken.json").write_text("{not json", encoding="utf-8")

    report = collect_status(runtime, clock=clock)

    assert report.to_dict(), "it still produces a payload"
    assert report.health in {STATUS_OK, STATUS_WARNING, STATUS_ERROR}
