"""
B5 — the real-time soak harness must itself be trustworthy.

A soak tool that reports PASSED without checking anything is worse than no soak
tool: it manufactures confidence. So these tests do two things.

They run a genuinely real (if very short) soak — real `SystemClock`, real
scheduler, real subprocesses, real consent gate, real disk — and assert it
produces samples, a report, and a verdict.

And they feed the analyser samples describing runtimes that were *broken* —
leaking memory, wedged heartbeat, planning on idle ticks, executing without
consent — and assert it says so. An invariant check that cannot fail is not an
invariant check.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.soak.realtime import (
    REPORT_NAME,
    SAMPLES_NAME,
    Sample,
    SoakHarness,
    analyse,
    load_samples,
    parse_duration,
    render_report,
)


def _sample(**overrides) -> Sample:
    base = Sample(
        at="2026-01-01T00:00:00+00:00",
        elapsed_s=0.0,
        rss_bytes=40 * 1024**2,
        cpu_s=1.0,
        threads=3,
        open_fds=30,
        runtime_bytes=1024,
        events_live_bytes=1024,
        events_archive_bytes=0,
        archive_segments=0,
        total_events=10,
        workflows_total=2,
        workflows_active=0,
        objective_runs=2,
        objective_status="active",
        brain_calls=2,
        grant_uses=2,
        supervisor_ticks=50,
        supervisor_tick_errors=0,
        scheduler_ticks=100,
        scheduler_tick_errors=0,
        tick_latency_ms=2.0,
        heartbeat_age_s=5.0,
        health="ok",
    )
    return replace(base, **overrides)


def _analyse(samples, tmp_path) -> dict:
    return analyse(
        samples,
        runtime_root=tmp_path / "runtime",
        duration_s=60.0,
        planner="fake",
    )


# ---------------------------------------------------------------------------
# The analyser catches what it claims to catch
# ---------------------------------------------------------------------------


def test_a_clean_run_passes(tmp_path):
    report = _analyse([_sample(elapsed_s=0), _sample(elapsed_s=60)], tmp_path)
    assert report["passed"] is True
    assert report["failures"] == []


def test_a_memory_leak_is_caught(tmp_path):
    report = _analyse(
        [
            _sample(elapsed_s=0, rss_bytes=100 * 1024**2),
            _sample(elapsed_s=3600, rss_bytes=900 * 1024**2),
        ],
        tmp_path,
    )
    assert report["passed"] is False
    assert any("RSS grew" in item for item in report["failures"])


def test_a_file_descriptor_leak_is_caught(tmp_path):
    report = _analyse(
        [_sample(elapsed_s=0, open_fds=30), _sample(elapsed_s=3600, open_fds=400)],
        tmp_path,
    )
    assert report["passed"] is False
    assert any("file descriptors" in item for item in report["failures"])


def test_a_stale_heartbeat_is_caught(tmp_path):
    report = _analyse(
        [_sample(elapsed_s=0), _sample(elapsed_s=600, heartbeat_age_s=900.0)],
        tmp_path,
    )
    assert report["passed"] is False
    assert any("heartbeat went stale" in item for item in report["failures"])


def test_a_stuck_objective_is_caught(tmp_path):
    for status in ("quarantined", "waiting_human"):
        report = _analyse([_sample(objective_status=status)], tmp_path)
        assert report["passed"] is False
        assert any(status in item for item in report["failures"])


def test_brain_calls_on_idle_ticks_are_caught(tmp_path):
    """
    The cost invariant of the whole autonomy design: an idle tick is free. If
    the brain were called per tick rather than per state change, a 24h soak
    would cost thousands of calls — and this is the check that would say so.
    """
    report = _analyse(
        [_sample(supervisor_ticks=100, brain_calls=140, objective_runs=100)],
        tmp_path,
    )
    assert report["passed"] is False
    assert any("idle ticks" in item for item in report["failures"])


def test_runaway_planning_is_caught(tmp_path):
    report = _analyse(
        [_sample(supervisor_ticks=1000, brain_calls=90, objective_runs=2)],
        tmp_path,
    )
    assert report["passed"] is False
    assert any("running away" in item for item in report["failures"])


def test_a_duplicate_launch_is_caught(tmp_path):
    """
    Counted from the timeline, not from the filesystem.

    The 24h soak taught this the hard way: the check used to count workflow
    *directories*, and retention archives terminal workflows while the soak is
    still running. It reported a duplicate launch that had never happened. The
    event log is complete history, so it cannot be moved out from under a count.
    """
    from mr1.event_log import EventLog

    runtime_root = tmp_path / "runtime"
    log = EventLog(runtime_root / "events")
    for i in range(3):
        log.emit(
            event_type="workflow_created",
            actor_id="MR1",
            actor_type="root_orchestrator",
            target_id=f"wf-{i}",
            target_type="workflow",
            status="pending",
            summary=f"wf {i}",
            workflow_id=f"wf-{i}",
        )

    report = analyse(
        [_sample(workflows_total=3, grant_uses=9)],
        runtime_root=runtime_root,
        duration_s=60.0,
        planner="fake",
    )
    assert report["passed"] is False
    assert any("ran more than once" in item for item in report["failures"])


def test_archived_workflows_do_not_read_as_a_duplicate_launch(tmp_path):
    """
    The exact false positive the 24-hour soak produced.

    48 workflows ran, every task exactly once. Retention archived 3 of them
    mid-run, so 45 directories remained — and the analyser compared 48 grant
    uses against 45 workflows and cried duplicate. Two Phase B features
    colliding, in the one place built to catch collisions.
    """
    from mr1.event_log import EventLog

    runtime_root = tmp_path / "runtime"
    log = EventLog(runtime_root / "events")
    for i in range(48):
        log.emit(
            event_type="workflow_created",
            actor_id="MR1",
            actor_type="root_orchestrator",
            target_id=f"wf-{i}",
            target_type="workflow",
            status="pending",
            summary=f"wf {i}",
            workflow_id=f"wf-{i}",
        )
        log.emit(
            event_type="workflow_task_started",
            actor_id="MR1",
            actor_type="root_orchestrator",
            target_id=f"tk-{i}",
            target_type="task",
            status="running",
            summary=f"task {i}",
            workflow_id=f"wf-{i}",
            task_id=f"tk-{i}",
        )

    # What the live filesystem would have shown after retention ran: 45.
    report = analyse(
        [_sample(workflows_total=45, grant_uses=48, objective_runs=48, brain_calls=48)],
        runtime_root=runtime_root,
        duration_s=86_400.0,
        planner="fake",
    )
    assert report["passed"] is True, report["failures"]
    assert report["totals"]["workflows"] == 48
    assert report["totals"]["task_starts"] == 48


def test_tick_errors_are_caught(tmp_path):
    report = _analyse([_sample(supervisor_tick_errors=2)], tmp_path)
    assert report["passed"] is False
    assert any("supervisor tick error" in item for item in report["failures"])


def test_creeping_tick_latency_is_caught(tmp_path):
    """An O(n) tick is a runtime that gets slower for as long as it runs."""
    samples = (
        [_sample(elapsed_s=i, tick_latency_ms=5.0) for i in range(10)]
        + [_sample(elapsed_s=100 + i, tick_latency_ms=400.0) for i in range(10)]
    )
    report = _analyse(samples, tmp_path)
    assert report["passed"] is False
    assert any("latency crept" in item for item in report["failures"])


def test_authority_used_outside_consent_is_caught(tmp_path):
    """
    The governance invariant. A risk-1.0 shell command allowed with neither a
    consent grant nor a human approval means MR1 authorized itself — which is
    the one thing the whole consent design exists to make impossible.
    """
    from mr1.event_log import EventLog

    runtime_root = tmp_path / "runtime"
    log = EventLog(runtime_root / "events")
    log.emit(
        event_type="capability_requested",
        actor_id="MR1",
        actor_type="root_orchestrator",
        target_id="shell_command",
        target_type="capability",
        status="requested",
        summary="requested",
        workflow_id="wf-1",
        task_id="tk-1",
    )
    log.emit(
        event_type="capability_allowed",
        actor_id="MR1",
        actor_type="root_orchestrator",
        target_id="shell_command",
        target_type="capability",
        status="allowed",
        summary="allowed with no authority at all",
        workflow_id="wf-1",
        task_id="tk-1",
        metadata={"reason": "risk_ok"},  # no consent_grant_id, no approval
    )

    report = analyse(
        [_sample()],
        runtime_root=runtime_root,
        duration_s=60.0,
        planner="fake",
    )
    assert report["passed"] is False
    assert any("outside consent" in item for item in report["failures"])


def test_no_samples_is_a_failure_not_a_pass(tmp_path):
    """The most important one: an empty soak must never report success."""
    report = _analyse([], tmp_path)
    assert report["passed"] is False
    assert report["failures"] == ["no samples were collected"]


# ---------------------------------------------------------------------------
# A genuinely real (short) soak
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_real_short_soak_runs_and_reports(tmp_path):
    """
    Real clock, real scheduler, real subprocess, real consent gate, real disk —
    for eight seconds. Everything an 8h soak does, minus the eight hours.

    This is the `--planner fake` mode the harness exists to make cheap: it
    validates the harness end to end without an LLM and without a wait.
    """
    soak_dir = tmp_path / "soak"
    harness = SoakHarness(
        soak_dir,
        workspace=Path.cwd(),
        duration_s=8.0,
        sample_interval_s=2.0,
        tick_interval_s=1.0,
        objective_interval_s=3.0,
        retention_interval_s=4.0,
        planner="fake",
        events_max_live_bytes=8_000,
        events_keep_recent=10,
    )
    report = harness.run()

    # It really ran.
    assert report["duration_actual_s"] >= 7.0
    assert report["samples"] >= 3
    assert report["totals"]["supervisor_ticks"] > 0

    # It really did the work — under real consent, through the real gate.
    assert report["totals"]["objective_runs"] >= 1, "the objective must have run for real"
    assert report["totals"]["grant_uses"] >= 1, "and it must have used its consent grant"
    assert report["totals"]["capability_executions"] >= 1

    # And nothing was violated.
    assert report["passed"] is True, report["failures"]

    # The artifacts an operator would read afterwards.
    assert (soak_dir / SAMPLES_NAME).exists()
    assert (soak_dir / REPORT_NAME).exists()
    persisted = json.loads((soak_dir / REPORT_NAME).read_text())
    assert persisted["passed"] is True

    # Samples survive on disk, so an interrupted soak is still analysable.
    reloaded = load_samples(soak_dir)
    assert len(reloaded) == report["samples"]
    assert analyse(
        reloaded,
        runtime_root=harness.runtime_root,
        duration_s=8.0,
        planner="fake",
    )["passed"] is True

    assert "REAL-TIME SOAK" in render_report(report)


def test_the_soak_grant_admits_read_only_git_and_refuses_everything_else(tmp_path):
    """
    The grant the soak runs under, checked against what a real planner emits.

    The first 8h `--planner real` run parked 43 seconds in: the grant was pinned
    to `^git status`, and the compiler answered "report what changed" with
    `git log --oneline -10`. MR1 was *right* to refuse — the preflight caught it,
    escalated, and never self-authorized. The grant was over-fitted to the fake
    planner's one command.

    Widening it must not widen it into a write. That is the whole test.
    """
    from mr1.autonomy.consent import ConsentGrantStore
    from mr1.scoped_agents import AgentStore
    from tests.soak.realtime import SoakHarness

    runtime = tmp_path / "soak"
    harness = SoakHarness(
        runtime,
        workspace=tmp_path,
        duration_s=1.0,
        planner="fake",
    )
    harness.prepare()
    # `prepare()` builds a real Supervisor, which starts a real scheduler thread.
    # Leaving it running would have it scanning disk every second for the rest of
    # the pytest session — which is exactly what dragged the full suite from 47s
    # to 455s before this line existed.
    supervisor = harness._supervisor
    try:
        grants = ConsentGrantStore(
            harness.runtime_root,
            scoped_agent_store=AgentStore(root=harness.runtime_root / "agents"),
        )
        predicate = grants.require(harness._grant_id).arg_predicate
    finally:
        if supervisor is not None:
            supervisor.shutdown()

    import re

    pattern = predicate["argv"]["regex"]

    def allowed(*argv: str) -> bool:
        return re.search(pattern, " ".join(argv)) is not None

    # Read-only inspection — including the command that stalled the real run.
    assert allowed("git", "log", "--oneline", "-10")
    assert allowed("git", "status", "--short")
    assert allowed("git", "diff", "--stat")
    assert allowed("git", "show", "HEAD")
    assert allowed("git", "rev-parse", "HEAD")

    # Anything that writes, publishes, or touches the network. A soak grant that
    # let one of these through would hand an unattended MR1 real authority over
    # the repo for eight hours.
    assert not allowed("git", "push", "origin", "main")
    assert not allowed("git", "commit", "-m", "x")
    assert not allowed("git", "fetch")
    assert not allowed("git", "reset", "--hard")
    assert not allowed("git", "clean", "-fd")
    assert not allowed("rm", "-rf", "/")
    assert not allowed("git", "status-hack")   # not a bare prefix match


def test_the_real_planner_is_wrapped_so_its_calls_are_counted():
    """
    `--planner real` reported `brain_calls: 0` for an entire 8h run, because the
    sampler reads `planner.calls` and `CompilerPlanner` has no such attribute.

    A metric that is always zero cannot fail its invariant — so the cost check
    (LLM calls vs idle ticks), the single most important number in a soak, was
    silently not being checked at all.
    """
    from tests.soak.realtime import CountingPlanner

    class _Inner:
        def __init__(self):
            self.seen = 0

        def plan(self, objective, context):
            self.seen += 1
            return {"title": "x", "tasks": []}

    inner = _Inner()
    planner = CountingPlanner(inner)

    assert planner.calls == 0
    planner.plan(object(), {})
    planner.plan(object(), {})

    assert planner.calls == 2, "the sampler reads this attribute"
    assert inner.seen == 2, "and the real planner still did the work"
    assert getattr(planner, "seen") == 2, "attributes pass through to the wrapped planner"


def test_durations_parse_the_way_an_operator_writes_them():
    assert parse_duration("30s") == 30
    assert parse_duration("45m") == 2700
    assert parse_duration("8h") == 28_800
    assert parse_duration("24h") == 86_400
    assert parse_duration("7d") == 604_800
    assert parse_duration("90") == 90
