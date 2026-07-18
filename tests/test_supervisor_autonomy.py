"""
A5 + A7 — the supervisor's objective lifecycle, end to end.

Drives a real objective through success, transient failure (retry), planning
failure (replan), budget exhaustion (quarantine), and a blocked failure
(escalate, no self-authorization) — with a fake brain, a virtual clock, and the
real scheduler, stores, policy engine, and inbox.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mr1.autonomy.budget import BudgetLedger, BudgetLimits
from mr1.autonomy.objectives import (
    KIND_ONCE,
    KIND_RECURRING,
    STATUS_ACTIVE,
    STATUS_EXECUTING,
    STATUS_PAUSED,
    STATUS_QUARANTINED,
    STATUS_RECOVERING,
    STATUS_SATISFIED,
    STATUS_WAITING_HUMAN,
)
from mr1.autonomy.planner import AmbiguousObjective, PlanningError
from mr1.autonomy.recovery import FailurePolicy
from mr1.autonomy.service import Supervisor, SupervisorConfig
from mr1.clock import VirtualClock
from mr1.event_log import EventLog
from mr1.worker_runner import MockRunner, RunResult, RunStatus
from mr1.workflow_models import TaskStatus, WorkflowStatus


class FakeBrain:
    """A planner that never calls an LLM — and counts how often it is asked."""

    def __init__(self, spec_factory=None):
        self.calls: list[dict] = []
        self._spec_factory = spec_factory or (lambda objective, context: {
            "title": f"plan for {objective.title}",
            "tasks": [{"label": "work", "prompt": "do the thing"}],
        })
        self.raise_next = None

    def plan(self, objective, context):
        self.calls.append(dict(context))
        if self.raise_next is not None:
            exc = self.raise_next
            self.raise_next = None
            raise exc
        return self._spec_factory(objective, context)

    @property
    def call_count(self) -> int:
        return len(self.calls)


class _FakeDoctor:
    def __init__(self, status="ok"):
        self.status = status
        self.summary = {}

    def __call__(self, _root):
        return self


class Harness:
    def __init__(self, tmp_path, *, planner=None, config=None, runner=None, limits=None):
        from mr1.autonomy.health import HealthReporter

        self.tmp_path = tmp_path
        self.runtime_root = tmp_path / "runtime"
        self.clock = VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.runner = runner or MockRunner()
        self.planner = planner or FakeBrain()
        self.doctor = _FakeDoctor()
        budget = BudgetLedger(
            self.runtime_root,
            clock=self.clock,
            limits=limits or BudgetLimits(),
        )
        self.supervisor = Supervisor(
            self.runtime_root,
            config=config or SupervisorConfig(tick_interval_s=60.0),
            clock=self.clock,
            runner=self.runner,
            planner=self.planner,
            budget=budget,
            health_reporter=HealthReporter(
                self.runtime_root,
                clock=self.clock,
                doctor_fn=self.doctor,
                doctor_interval_s=0.0,
            ),
            workspace_root=tmp_path,
            auto_scheduler_tick=False,
        )
        self.objectives = self.supervisor.objectives
        self.scheduler = self.supervisor.scheduler
        self.store = self.scheduler._store
        self.agents = self.scheduler._scoped_agents

    def objective(self, **overrides):
        payload = {
            "title": "Genesis",
            "statement": "run the weekly genesis cycle",
            "owner_agent_id": self.agents.root_agent_id,
            "kind": KIND_ONCE,
            "idempotent": True,
        }
        payload.update(overrides)
        return self.objectives.create(**payload)

    def reload(self, objective):
        return self.objectives.require(objective.objective_id)

    def run_workflow(self, *, status=RunStatus.SUCCEEDED, error=None, error_type=None):
        """Drive whatever the scheduler has in flight to a terminal state."""
        for _ in range(2):
            self.scheduler.tick()
        for workflow in self.store.list_active_workflows():
            for task in workflow.tasks.values():
                if task.status is TaskStatus.RUNNING:
                    self.runner.complete(
                        task.task_id,
                        status,
                        exit_code=0 if status is RunStatus.SUCCEEDED else 1,
                        summary="done" if status is RunStatus.SUCCEEDED else "failed",
                        error=error,
                        error_type=error_type,
                    )
        for _ in range(2):
            self.scheduler.tick()

    def events(self, **kwargs):
        return EventLog(self.runtime_root / "events").filter_events(**kwargs)

    def inbox(self):
        return self.supervisor._message_store.list_inbox(self.agents.root_agent_id)


@pytest.fixture
def hx(tmp_path):
    return Harness(tmp_path)


# -- the zero-token invariant ----------------------------------------------


def test_a_steady_state_tick_makes_zero_brain_calls(hx):
    """The property that makes continuous operation affordable."""
    for _ in range(50):
        hx.supervisor.tick()

    assert hx.planner.call_count == 0


def test_ticks_with_no_eligible_objective_make_zero_brain_calls(hx):
    satisfied = hx.objective(title="already done")
    hx.objectives.set_status(satisfied.objective_id, STATUS_SATISFIED)
    paused = hx.objective(title="paused")
    hx.objectives.set_status(paused.objective_id, STATUS_PAUSED)
    waiting = hx.objective(title="waiting")
    hx.objectives.set_status(waiting.objective_id, STATUS_WAITING_HUMAN)

    for _ in range(10):
        hx.supervisor.tick()

    assert hx.planner.call_count == 0


def test_an_executing_objective_is_not_replanned_every_tick(hx):
    hx.objective()
    hx.supervisor.tick()
    assert hx.planner.call_count == 1

    for _ in range(20):
        hx.supervisor.tick()

    assert hx.planner.call_count == 1, "the brain was called while work was already in flight"


# -- the happy path ---------------------------------------------------------


def test_an_objective_plans_submits_and_is_satisfied(hx):
    objective = hx.objective()

    hx.supervisor.tick()

    reloaded = hx.reload(objective)
    assert reloaded.status == STATUS_EXECUTING
    assert reloaded.current_workflow_id
    workflow = hx.store.load_workflow(reloaded.current_workflow_id)
    # Every autonomous workflow is attributable to its objective.
    assert workflow.metadata["objective_id"] == objective.objective_id
    assert workflow.metadata["autonomous"] is True

    hx.run_workflow(status=RunStatus.SUCCEEDED)
    hx.supervisor.tick()

    done = hx.reload(objective)
    assert done.status == STATUS_SATISFIED
    assert done.success_count == 1
    assert done.current_workflow_id is None
    assert hx.events(event_type="objective_planned")
    assert hx.events(event_type="objective_satisfied")


def test_a_recurring_objective_goes_back_to_active_and_waits(hx):
    objective = hx.objective(
        kind=KIND_RECURRING,
        trigger={"type": "interval", "interval_s": 604_800},
    )

    hx.supervisor.tick()
    hx.run_workflow()
    hx.supervisor.tick()

    reloaded = hx.reload(objective)
    assert reloaded.status == STATUS_ACTIVE
    assert reloaded.success_count == 1

    # Not due again for a week: no further plans.
    for _ in range(5):
        hx.supervisor.tick()
    assert hx.planner.call_count == 1

    hx.clock.advance(604_801)
    hx.supervisor.tick()
    assert hx.planner.call_count == 2


# -- transient failure -> retry with backoff, no brain call -----------------


def test_a_transient_failure_retries_after_backoff_without_calling_the_brain(hx):
    objective = hx.objective()

    hx.supervisor.tick()
    hx.run_workflow(status=RunStatus.TIMED_OUT, error="timed out", error_type="timeout")
    hx.supervisor.tick()

    recovering = hx.reload(objective)
    assert recovering.status == STATUS_RECOVERING
    assert recovering.retries_used == 1
    assert recovering.next_attempt_at is not None
    assert hx.planner.call_count == 1

    # Backoff has not elapsed: nothing is resubmitted.
    hx.supervisor.tick()
    assert hx.reload(objective).status == STATUS_RECOVERING

    hx.clock.advance(31)
    hx.supervisor.tick()

    retried = hx.reload(objective)
    assert retried.status == STATUS_EXECUTING
    assert retried.current_workflow_id
    # A retry replays the spec MR1 already has: it costs zero tokens.
    assert hx.planner.call_count == 1
    assert hx.events(event_type="objective_recovery")


def test_retries_are_bounded_and_then_escalate_to_a_replan(hx):
    objective = hx.objective(
        failure_policy=FailurePolicy(max_retries=1, max_replans=1),
    )

    hx.supervisor.tick()
    hx.run_workflow(status=RunStatus.TIMED_OUT, error_type="timeout")
    hx.supervisor.tick()
    assert hx.reload(objective).status == STATUS_RECOVERING

    hx.clock.advance(31)
    hx.supervisor.tick()
    hx.run_workflow(status=RunStatus.TIMED_OUT, error_type="timeout")
    hx.supervisor.tick()

    # Retry budget spent: the ladder escalates exactly one level, to a replan,
    # and PLAN (which runs after RECOVER in the same tick) acts on it at once.
    replanning = hx.reload(objective)
    assert replanning.replans_used == 1
    assert replanning.status == STATUS_EXECUTING
    assert hx.planner.call_count == 2, "the replan should have asked the brain for a new spec"


# -- planning failure -> replan --------------------------------------------


def test_a_planning_failure_replans(hx):
    objective = hx.objective()

    hx.supervisor.tick()
    hx.run_workflow(status=RunStatus.FAILED, error="assertion failed", error_type="unknown")
    hx.supervisor.tick()

    reloaded = hx.reload(objective)
    assert reloaded.replans_used == 1
    assert reloaded.last_failure["classification"] == "planning"
    # RECOVER decided to replan, and PLAN (later in the same tick) did it.
    assert reloaded.status == STATUS_EXECUTING
    assert hx.planner.call_count == 2
    # The brain is told why the last attempt failed.
    assert hx.planner.calls[-1]["last_failure"]["classification"] == "planning"


def test_repeated_planning_failures_quarantine_and_escalate(hx):
    objective = hx.objective(failure_policy=FailurePolicy(max_retries=0, max_replans=1))

    for _ in range(4):
        hx.supervisor.tick()
        hx.run_workflow(status=RunStatus.FAILED, error="assertion failed", error_type="unknown")
        hx.supervisor.tick()
        if hx.reload(objective).is_parked:
            break

    quarantined = hx.reload(objective)
    assert quarantined.status == STATUS_QUARANTINED
    assert hx.events(event_type="objective_quarantined")

    messages = hx.inbox()
    assert messages, "quarantine must reach a human"
    body = messages[0].body
    assert "WHAT MR1 TRIED" in body
    assert "WHAT MR1 NEEDS FROM YOU" in body
    assert "RECOMMENDED NEXT ACTION" in body
    assert objective.objective_id in body


# -- blocked -> escalate, never self-authorize ------------------------------


def test_a_blocked_failure_escalates_and_never_self_authorizes(tmp_path):
    """The load-bearing safety property of the whole phase."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    planner = FakeBrain(spec_factory=lambda objective, context: {
        "title": "shell work",
        "tasks": [{
            "label": "run",
            "task_kind": "tool",
            "tool_type": "shell_command",
            "tool_config": {"argv": ["echo", "hi"], "cwd": str(workspace)},
        }],
    })
    hx = Harness(tmp_path, planner=planner)
    objective = hx.objective(statement="shell out weekly")

    hx.supervisor.tick()

    reloaded = hx.reload(objective)
    # It never submitted: the preflight caught that it lacks the authority.
    assert reloaded.status == STATUS_WAITING_HUMAN
    assert reloaded.current_workflow_id is None
    assert hx.store.list_active_workflows() == []
    assert reloaded.last_escalation["reason"] == "consent_missing"

    # It did NOT grant itself the consent it is missing.
    assert hx.supervisor.consent.list_active() == []

    messages = hx.inbox()
    assert len(messages) == 1
    assert "shell_command" in messages[0].body
    assert "mr1 grant create" in messages[0].body
    assert hx.events(event_type="escalation_raised")


def test_the_same_objective_runs_unattended_once_consent_is_granted(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    planner = FakeBrain(spec_factory=lambda objective, context: {
        "title": "shell work",
        "tasks": [{
            "label": "run",
            "task_kind": "tool",
            "tool_type": "shell_command",
            "tool_config": {"argv": ["echo", "hi"], "cwd": str(workspace)},
        }],
    })
    hx = Harness(tmp_path, planner=planner)
    objective = hx.objective(statement="shell out weekly")

    hx.supervisor.consent.create(
        objective_id=objective.objective_id,
        capability_name="shell_command",
        scope_roots=[workspace],
        max_risk=1.0,
        granted_by=hx.agents.root_agent_id,
        ttl_s=7 * 86_400,
        arg_predicate={"argv": {"regex": r"^echo\b"}},
        reason="acceptance",
    )

    hx.supervisor.tick()
    for _ in range(4):
        hx.scheduler.tick()
    hx.supervisor.tick()

    done = hx.reload(objective)
    assert done.status == STATUS_SATISFIED, done.status_reason
    assert hx.inbox() == [], "nobody should have been asked"


def test_an_escalation_is_not_repeated_every_tick(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    planner = FakeBrain(spec_factory=lambda objective, context: {
        "title": "shell work",
        "tasks": [{
            "label": "run",
            "task_kind": "tool",
            "tool_type": "shell_command",
            "tool_config": {"argv": ["echo", "hi"], "cwd": str(workspace)},
        }],
    })
    hx = Harness(tmp_path, planner=planner)
    hx.objective()

    for _ in range(10):
        hx.supervisor.tick()

    assert len(hx.inbox()) == 1, "a 60s tick must not send 60 identical messages an hour"


# -- planner failures -------------------------------------------------------


def test_an_ambiguous_objective_escalates_instead_of_guessing(hx):
    objective = hx.objective()
    hx.planner.raise_next = AmbiguousObjective("not confident enough to run this")

    hx.supervisor.tick()

    reloaded = hx.reload(objective)
    assert reloaded.status == STATUS_WAITING_HUMAN
    assert reloaded.last_escalation["reason"] == "ambiguous_objective"
    assert hx.store.list_active_workflows() == []


def test_a_planner_crash_escalates_and_is_visible(hx):
    objective = hx.objective()
    hx.planner.raise_next = PlanningError("the compiler exploded")

    hx.supervisor.tick()

    reloaded = hx.reload(objective)
    assert reloaded.status == STATUS_WAITING_HUMAN
    assert reloaded.last_escalation["reason"] == "plan_failed"
    assert hx.events(event_type="objective_plan_failed")
    assert hx.inbox()


def test_an_invalid_spec_never_reaches_the_scheduler(hx):
    objective = hx.objective()
    hx.planner._spec_factory = lambda objective, context: {"tasks": []}

    hx.supervisor.tick()

    assert hx.reload(objective).status == STATUS_WAITING_HUMAN
    assert hx.store.list_workflows() == []


# -- backpressure -----------------------------------------------------------


def test_the_plan_rate_budget_stops_planning(tmp_path):
    hx = Harness(tmp_path, limits=BudgetLimits(max_plans_per_hour=1))
    first = hx.objective(title="first")
    second = hx.objective(title="second")

    hx.supervisor.tick()

    planned = [
        objective
        for objective in (hx.reload(first), hx.reload(second))
        if objective.status == STATUS_EXECUTING
    ]
    assert len(planned) == 1
    assert hx.planner.call_count == 1
    assert hx.events(event_type="budget_exhausted")


def test_the_daily_per_objective_workflow_cap_escalates(tmp_path):
    hx = Harness(
        tmp_path,
        limits=BudgetLimits(max_plans_per_hour=100, max_workflows_per_objective_per_day=1),
    )
    objective = hx.objective(kind=KIND_RECURRING, trigger={"type": "interval", "interval_s": 60})

    hx.supervisor.tick()
    hx.run_workflow()
    hx.supervisor.tick()
    hx.clock.advance(61)
    hx.supervisor.tick()

    reloaded = hx.reload(objective)
    assert reloaded.status == STATUS_QUARANTINED
    assert reloaded.last_escalation["reason"] == "budget_exhausted"
    assert hx.inbox()


def test_the_concurrency_cap_stops_new_plans_but_not_draining(tmp_path):
    config = SupervisorConfig(tick_interval_s=60.0, max_concurrent_workflows=1)
    hx = Harness(tmp_path, config=config)
    first = hx.objective(title="first")
    second = hx.objective(title="second")

    hx.supervisor.tick()          # plans one, submits it
    outcome = hx.supervisor.tick()  # now at the cap

    assert outcome["gate"] == "draining"
    statuses = {hx.reload(first).status, hx.reload(second).status}
    assert STATUS_EXECUTING in statuses
    assert STATUS_ACTIVE in statuses


def test_pause_stops_planning_but_lets_work_drain(hx):
    from mr1.autonomy.control import MODE_PAUSED, MODE_RUNNING

    objective = hx.objective()
    hx.supervisor.tick()
    assert hx.reload(objective).status == STATUS_EXECUTING

    hx.supervisor.control.set_mode(MODE_PAUSED, reason="test")
    second = hx.objective(title="second")

    hx.run_workflow()
    hx.supervisor.tick()

    # The in-flight objective still completed...
    assert hx.reload(objective).status == STATUS_SATISFIED
    # ...but no new work was created.
    assert hx.reload(second).status == STATUS_ACTIVE
    assert hx.planner.call_count == 1

    hx.supervisor.control.set_mode(MODE_RUNNING, reason="test")
    hx.supervisor.tick()
    assert hx.reload(second).status == STATUS_EXECUTING


def test_degraded_health_stops_planning_and_tells_a_human(hx):
    hx.objective()
    hx.doctor.status = "error"

    outcome = hx.supervisor.tick()

    assert outcome["gate"] == "draining"
    assert hx.planner.call_count == 0
    assert hx.inbox(), "a health failure that halts autonomy must be visible"


def test_halt_cancels_work_revokes_grants_and_pauses_objectives(hx):
    from mr1.autonomy.control import MODE_HALTED

    objective = hx.objective()
    hx.supervisor.consent.create(
        objective_id=objective.objective_id,
        capability_name="shell_command",
        scope_roots=[hx.tmp_path],
        max_risk=1.0,
        granted_by=hx.agents.root_agent_id,
        ttl_s=86_400,
        reason="test",
    )
    hx.supervisor.tick()
    workflow_id = hx.reload(objective).current_workflow_id
    assert workflow_id

    hx.supervisor.control.set_mode(MODE_HALTED, reason="emergency")
    outcome = hx.supervisor.tick()

    assert outcome["gate"] == "exit"
    assert hx.supervisor.exit_requested is True
    assert hx.supervisor.consent.list_active() == []
    assert hx.reload(objective).status == STATUS_PAUSED
    assert hx.store.load_workflow(workflow_id).status is WorkflowStatus.CANCELLED


# -- crash / restart --------------------------------------------------------


def test_an_objective_survives_a_supervisor_restart(tmp_path):
    hx = Harness(tmp_path)
    objective = hx.objective()
    hx.supervisor.tick()
    workflow_id = hx.reload(objective).current_workflow_id

    # A brand-new supervisor over the same runtime root.
    reborn = Harness(tmp_path, planner=hx.planner, runner=hx.runner)

    survivor = reborn.objectives.require(objective.objective_id)
    assert survivor.status == STATUS_EXECUTING
    assert survivor.current_workflow_id == workflow_id

    reborn.run_workflow()
    reborn.supervisor.tick()

    assert reborn.objectives.require(objective.objective_id).status == STATUS_SATISFIED


def test_a_grant_survives_a_restart_and_still_authorizes(tmp_path):
    hx = Harness(tmp_path)
    objective = hx.objective()
    grant = hx.supervisor.consent.create(
        objective_id=objective.objective_id,
        capability_name="shell_command",
        scope_roots=[tmp_path],
        max_risk=1.0,
        granted_by=hx.agents.root_agent_id,
        ttl_s=86_400,
        reason="test",
    )

    reborn = Harness(tmp_path)

    active = reborn.supervisor.consent.list_active(objective_id=objective.objective_id)
    assert [item.grant_id for item in active] == [grant.grant_id]
