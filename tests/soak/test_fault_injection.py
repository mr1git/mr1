"""
A8 — fault injection.

Every failure path, driven to its end. The property under test is the one that
separates a bounded autonomous system from a runaway one:

    every failure reaches success, waiting-human, quarantine, or another
    explicit terminal state — and nothing retries forever.
"""

from __future__ import annotations

import pytest

from mr1.autonomy.budget import BudgetLimits
from mr1.autonomy.objectives import (
    PARKED_STATUSES,
    STATUS_QUARANTINED,
    STATUS_SATISFIED,
    STATUS_WAITING_HUMAN,
    TERMINAL_STATUSES,
)
from mr1.autonomy.recovery import FailurePolicy
from mr1.autonomy.service import SupervisorConfig
from tests.soak.harness import FakeBrain, FaultInjector, FaultProfile, SoakRuntime


def _runtime(
    tmp_path,
    profile,
    *,
    seed=99,
    planner=None,
    objectives=1,
    stuck_workflow_after_s=86_400.0,
    **objective_kwargs,
):
    runtime = SoakRuntime(
        tmp_path / "runtime",
        workspace_root=tmp_path,
        runner=FaultInjector(profile, seed=seed),
        planner=planner,
        config=SupervisorConfig(
            tick_interval_s=60.0,
            doctor_interval_s=86_400.0,
            max_concurrent_workflows=8,
            max_plans_per_hour=10_000,
            max_workflows_per_objective_per_day=10_000,
            stuck_workflow_after_s=stuck_workflow_after_s,
        ),
        limits=BudgetLimits(
            max_plans_per_hour=10_000,
            max_actions_per_hour=10_000,
            max_workflows_per_objective_per_day=10_000,
        ),
    )
    for index in range(objectives):
        runtime.create_objective(
            title=f"objective-{index}",
            statement=f"do useful work {index}",
            **objective_kwargs,
        )
    return runtime


def test_a_mixed_fault_storm_leaves_nothing_unresolved(tmp_path):
    """1 000 ticks, 10% failure across every class, 12 objectives."""
    profile = FaultProfile(
        timeout=0.03,
        infrastructure=0.02,
        merit_failure=0.03,
        identical_failure=0.02,
    )
    runtime = _runtime(tmp_path, profile, objectives=12)

    runtime.run(1_000)

    unresolved = runtime.unresolved_objectives()
    assert unresolved == [], (
        "objectives never reached an explicit end state: "
        + ", ".join(f"{item.objective_id}={item.status}" for item in unresolved)
    )
    for objective in runtime.objectives.list_objectives():
        assert objective.status in TERMINAL_STATUSES | PARKED_STATUSES
    assert runtime.supervisor.metrics()["supervisor_tick_errors"] == 0


def test_every_recovery_budget_is_respected(tmp_path):
    profile = FaultProfile(merit_failure=1.0)  # everything fails on its merits
    runtime = _runtime(
        tmp_path,
        profile,
        objectives=3,
        failure_policy=FailurePolicy(max_retries=2, max_replans=2),
    )

    runtime.run(200)

    for objective in runtime.objectives.list_objectives():
        assert objective.status == STATUS_QUARANTINED
        assert objective.retries_used <= 2
        assert objective.replans_used <= 2
        assert objective.consecutive_failures <= objective.failure_policy.max_consecutive_failures


def test_transient_failures_retry_then_stop(tmp_path):
    profile = FaultProfile(timeout=1.0)  # every run times out
    runtime = _runtime(
        tmp_path,
        profile,
        failure_policy=FailurePolicy(max_retries=3, max_replans=1),
    )

    runtime.run(300)

    objective = runtime.objectives.list_objectives()[0]
    assert objective.status == STATUS_QUARANTINED
    assert objective.retries_used <= 3
    assert runtime.events(event_type="objective_quarantined")
    # Nothing retried forever: the workflow count is bounded by the budgets.
    assert len(runtime.store.list_workflows()) <= 10


def test_the_same_failure_repeated_terminates(tmp_path):
    """The identical-failure detector: retrying will not help, so it stops."""
    profile = FaultProfile(identical_failure=1.0)
    runtime = _runtime(
        tmp_path,
        profile,
        failure_policy=FailurePolicy(
            max_retries=5,
            max_replans=5,
            max_identical_failures=3,
        ),
    )

    runtime.run(300)

    objective = runtime.objectives.list_objectives()[0]
    assert objective.status == STATUS_QUARANTINED
    assert objective.last_escalation["reason"] == "repeated_failure"
    assert objective.replans_used < 5, "it stopped before burning the whole replan budget"


def test_a_blocked_objective_escalates_and_never_self_authorizes(tmp_path):
    """Consent it does not hold: it asks, and it does not grant itself anything."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    planner = FakeBrain(spec_factory=lambda objective, context: {
        "title": "shell",
        "tasks": [{
            "label": "run",
            "task_kind": "tool",
            "tool_type": "shell_command",
            "tool_config": {"argv": ["echo", "hi"], "cwd": str(workspace)},
        }],
    })
    runtime = _runtime(tmp_path, FaultProfile(), planner=planner)

    runtime.run(50)

    objective = runtime.objectives.list_objectives()[0]
    assert objective.status == STATUS_WAITING_HUMAN
    assert runtime.supervisor.consent.list_active() == []
    assert runtime.store.list_workflows() == []
    # It asked exactly once, not once per tick.
    inbox = runtime.supervisor._message_store.list_inbox(runtime.agents.root_agent_id)
    assert len(inbox) == 1


def test_an_expired_consent_grant_stops_authorizing_mid_flight(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    planner = FakeBrain(spec_factory=lambda objective, context: {
        "title": "shell",
        "tasks": [{
            "label": "run",
            "task_kind": "tool",
            "tool_type": "shell_command",
            "tool_config": {"argv": ["echo", "hi"], "cwd": str(workspace)},
        }],
    })
    runtime = _runtime(
        tmp_path,
        FaultProfile(),
        planner=planner,
        kind="recurring",
        trigger={"type": "interval", "interval_s": 600},
    )
    objective = runtime.objectives.list_objectives()[0]
    runtime.supervisor.consent.create(
        objective_id=objective.objective_id,
        capability_name="shell_command",
        scope_roots=[workspace],
        max_risk=1.0,
        granted_by=runtime.agents.root_agent_id,
        ttl_s=3_600,
        arg_predicate={"argv": {"regex": r"^echo\b"}},
        reason="soak",
    )

    runtime.run(20)  # 20 minutes: runs unattended under the grant
    reloaded = runtime.objectives.require(objective.objective_id)
    assert reloaded.success_count >= 1

    runtime.run(80)  # the grant expires partway through

    final = runtime.objectives.require(objective.objective_id)
    assert final.status in PARKED_STATUSES, "an expired grant must stop autonomy, not be ignored"
    assert runtime.events(event_type="consent_grant_expired")
    inbox = runtime.supervisor._message_store.list_inbox(runtime.agents.root_agent_id)
    assert inbox, "the human must be told the standing consent lapsed"


def test_a_planner_that_keeps_crashing_escalates(tmp_path):
    planner = FakeBrain(fail_every=1)
    runtime = _runtime(tmp_path, FaultProfile(), planner=planner)

    runtime.run(50)

    objective = runtime.objectives.list_objectives()[0]
    assert objective.status in PARKED_STATUSES
    assert runtime.events(event_type="objective_plan_failed")


def test_a_task_that_vanishes_hangs_then_gets_escalated_as_stuck(tmp_path):
    """
    crash_mid_run: the task never reports again and no result is ever written.

    Nothing else in the runtime notices this. The supervisor's SWEEP does: past
    the stuck threshold, a workflow that is still "running" is not working, and
    the objective goes in front of a human rather than waiting forever.
    """
    profile = FaultProfile(crash_mid_run=1.0)
    runtime = _runtime(tmp_path, profile, stuck_workflow_after_s=3_600.0)

    runtime.run(30)  # 30 minutes: still inside the threshold, still in flight
    objective = runtime.objectives.list_objectives()[0]
    assert objective.status == "executing"

    runtime.run(40)  # past the threshold

    stuck = runtime.objectives.list_objectives()[0]
    assert stuck.status == STATUS_WAITING_HUMAN
    assert stuck.last_escalation["reason"] == "stuck"
    inbox = runtime.supervisor._message_store.list_inbox(runtime.agents.root_agent_id)
    assert inbox, "a task that vanished must not be silently forgotten"
    assert runtime.supervisor.metrics()["supervisor_tick_errors"] == 0


def test_a_fault_storm_never_wedges_the_supervisor(tmp_path):
    profile = FaultProfile(
        timeout=0.2,
        infrastructure=0.2,
        merit_failure=0.2,
        identical_failure=0.2,
        crash_mid_run=0.1,
    )
    runtime = _runtime(tmp_path, profile, objectives=6, stuck_workflow_after_s=3_600.0)

    runtime.run(500)

    assert runtime.supervisor.metrics()["supervisor_tick_errors"] == 0
    assert runtime.supervisor.metrics()["supervisor_consecutive_tick_errors"] == 0
    gauges = runtime.gauges()
    assert gauges["scheduler_tick_errors"] == 0
    # Everything reached a terminal state or a human — including the tasks that
    # simply vanished, which the stuck sweep escalated.
    assert runtime.unresolved_objectives() == []


def test_the_same_seed_reproduces_the_same_soak(tmp_path):
    profile = FaultProfile(timeout=0.1, merit_failure=0.1)

    first = _runtime(tmp_path / "a", profile, seed=4242, objectives=3)
    first.run(120)
    second = _runtime(tmp_path / "b", profile, seed=4242, objectives=3)
    second.run(120)

    assert first.runner.outcomes == second.runner.outcomes
    assert (
        sorted(item.status for item in first.objectives.list_objectives())
        == sorted(item.status for item in second.objectives.list_objectives())
    )
