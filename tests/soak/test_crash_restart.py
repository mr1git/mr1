"""
A8 — crash and restart.

The supervisor is killed mid-flight and a fresh one starts over the same disk,
fifty times. What must survive:

  * objectives persist, with their counters and history intact
  * the objective -> current workflow relationship stays coherent
  * in-flight work interrupted by the crash is handled by the recovery policy
    (transient, retried) rather than becoming a permanent failure
  * grants and approvals keep their correct status across the restart
"""

from __future__ import annotations

from mr1.autonomy.budget import BudgetLimits
from mr1.autonomy.objectives import (
    KIND_RECURRING,
    PARKED_STATUSES,
    STATUS_SATISFIED,
    TERMINAL_STATUSES,
)
from mr1.autonomy.recovery import FailureClass, classify
from mr1.autonomy.service import SupervisorConfig
from mr1.worker_runner import MockRunner, RunResult, RunStatus
from mr1.workflow_models import TaskStatus, WorkflowStatus
from tests.soak.harness import FakeBrain, FaultInjector, FaultProfile, SoakRuntime


def _runtime(tmp_path, *, runner=None, planner=None):
    return SoakRuntime(
        tmp_path / "runtime",
        workspace_root=tmp_path,
        runner=runner,
        planner=planner,
        config=SupervisorConfig(
            tick_interval_s=60.0,
            doctor_interval_s=86_400.0,
            max_concurrent_workflows=4,
            max_plans_per_hour=10_000,
            max_workflows_per_objective_per_day=10_000,
        ),
        limits=BudgetLimits(
            max_plans_per_hour=10_000,
            max_actions_per_hour=10_000,
            max_workflows_per_objective_per_day=10_000,
        ),
    )


def test_an_objective_and_its_workflow_survive_a_restart(tmp_path):
    runtime = _runtime(tmp_path)
    objective = runtime.create_objective(title="survivor")
    runtime.tick()

    before = runtime.objectives.require(objective.objective_id)
    assert before.status == "executing"
    workflow_id = before.current_workflow_id

    runtime.restart()

    after = runtime.objectives.require(objective.objective_id)
    assert after.status == "executing"
    assert after.current_workflow_id == workflow_id
    assert runtime.store.load_workflow(workflow_id) is not None

    # And it still finishes.
    runtime.run(3)
    assert runtime.objectives.require(objective.objective_id).status == STATUS_SATISFIED


def test_fifty_crashes_mid_flight_lose_nothing(tmp_path):
    """Kill it, restart it, fifty times, while work is genuinely in flight."""
    runtime = _runtime(tmp_path)
    objective = runtime.create_objective(
        title="resilient",
        kind=KIND_RECURRING,
        trigger={"type": "interval", "interval_s": 300},
    )

    for _ in range(50):
        runtime.run(4)
        runtime.restart()

    survivor = runtime.objectives.require(objective.objective_id)
    assert survivor.success_count > 10, "no progress survived the restarts"
    assert survivor.consecutive_failures == 0
    assert runtime.supervisor.metrics()["supervisor_tick_errors"] == 0
    assert runtime.stuck_objectives() == []
    # Exactly one objective file: restarts must not clone objectives.
    assert len(list((runtime.runtime_root / "objectives").glob("obj-*.json"))) == 1


def test_a_task_interrupted_by_a_crash_is_transient_not_permanent(tmp_path):
    """
    On restart the scheduler has no live handle for a task it persisted as
    RUNNING, and force-fails it as `infrastructure_failure`. If the ladder
    treated that as a real defect, every restart would convert in-flight work
    into permanent failure. It is classified transient, so it is retried.
    """
    # A runner whose tasks never report on their own: the work is genuinely
    # in flight when the process dies.
    runtime = _runtime(tmp_path, runner=MockRunner())
    objective = runtime.create_objective(title="interrupted", idempotent=True)

    runtime.tick(scheduler_ticks=1)  # plan + launch; the task is now RUNNING
    workflow_id = runtime.objectives.require(objective.objective_id).current_workflow_id
    workflow = runtime.store.load_workflow(workflow_id)
    assert any(task.status is TaskStatus.RUNNING for task in workflow.tasks.values())

    # The crash: the in-memory handle registry dies with the process.
    runtime.restart()
    runtime.scheduler._handles.clear()

    runtime.run(2)

    failed = runtime.store.load_workflow(workflow_id)
    assert failed.status is WorkflowStatus.FAILED
    signal = classify(failed)
    assert signal.classification is FailureClass.TRANSIENT
    assert signal.error_type == "infrastructure_failure"

    after = runtime.objectives.require(objective.objective_id)
    assert after.status in {"recovering", "executing"}, after.status
    assert after.retries_used == 1, "the interrupted work must be retried, not abandoned"

    # And with the runtime healthy again, the retry completes.
    runtime.clock.advance(60)
    runtime.runner = MockRunner(
        on_poll=lambda _handle: RunResult(status=RunStatus.SUCCEEDED, exit_code=0, summary="ok")
    )
    runtime.restart()
    runtime.run(4)
    assert runtime.objectives.require(objective.objective_id).status == STATUS_SATISFIED


def test_grants_keep_their_status_across_a_restart(tmp_path):
    runtime = _runtime(tmp_path)
    objective = runtime.create_objective(title="granted")
    live = runtime.supervisor.consent.create(
        objective_id=objective.objective_id,
        capability_name="shell_command",
        scope_roots=[tmp_path],
        max_risk=1.0,
        granted_by=runtime.agents.root_agent_id,
        ttl_s=7 * 86_400,
        reason="soak",
    )
    revoked = runtime.supervisor.consent.create(
        objective_id=objective.objective_id,
        capability_name="write_file",
        scope_roots=[tmp_path],
        max_risk=0.9,
        granted_by=runtime.agents.root_agent_id,
        ttl_s=7 * 86_400,
        reason="soak",
    )
    runtime.supervisor.consent.revoke(revoked.grant_id, revoked_by="operator")
    short = runtime.supervisor.consent.create(
        objective_id=objective.objective_id,
        capability_name="read_file",
        scope_roots=[tmp_path],
        max_risk=0.5,
        granted_by=runtime.agents.root_agent_id,
        ttl_s=60,
        reason="soak",
    )

    runtime.clock.advance(120)  # `short` has now expired
    runtime.restart()

    consent = runtime.supervisor.consent
    now = runtime.clock.now()
    assert consent.require(live.grant_id).status(now) == "active"
    assert consent.require(revoked.grant_id).status(now) == "revoked"
    assert consent.require(short.grant_id).status(now) == "expired"
    assert [item.grant_id for item in consent.list_active()] == [live.grant_id]


def test_a_pending_approval_keeps_its_deadline_across_a_restart(tmp_path):
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
    runtime = _runtime(tmp_path, planner=planner)
    objective = runtime.create_objective(title="needs approval")

    # Submit the shell workflow directly (bypassing the preflight escalation) so
    # the scheduler routes a real approval request.
    workflow_id = runtime.scheduler.submit_workflow(
        planner.plan(objective, {}),
        __import__("mr1.workflow_models", fromlist=["Provenance"]).Provenance(
            type="agent", id="supervisor",
        ),
        workflow_metadata={"objective_id": objective.objective_id},
    )
    runtime.run(2)

    approvals = runtime.supervisor._approval_store.list_requests()
    assert len(approvals) == 1
    assert approvals[0].status == "pending"
    expires_at = approvals[0].expires_at
    assert expires_at

    runtime.restart()

    reloaded = runtime.supervisor._approval_store.list_requests()
    assert reloaded[0].status == "pending"
    assert reloaded[0].expires_at == expires_at

    # The TTL still fires after the restart, and the objective is escalated.
    runtime.clock.advance(runtime.supervisor.config.approval_ttl_s + 60)
    runtime.run(2)

    assert runtime.supervisor._approval_store.list_requests()[0].status == "expired"
    assert runtime.objectives.require(objective.objective_id).status in PARKED_STATUSES
    inbox = runtime.supervisor._message_store.list_inbox(runtime.agents.root_agent_id)
    assert inbox, "an approval nobody answered must escalate, not park forever"


def test_restarts_under_a_fault_storm_still_terminate(tmp_path):
    runtime = _runtime(
        tmp_path,
        runner=FaultInjector(
            FaultProfile(timeout=0.15, merit_failure=0.15, infrastructure=0.1),
            seed=31337,
        ),
    )
    for index in range(5):
        runtime.create_objective(title=f"obj-{index}", statement=f"work {index}")

    for _ in range(25):
        runtime.run(8)
        runtime.restart()

    for objective in runtime.objectives.list_objectives():
        assert objective.status in TERMINAL_STATUSES | PARKED_STATUSES, (
            f"{objective.objective_id} ended at {objective.status}"
        )
    assert runtime.supervisor.metrics()["supervisor_tick_errors"] == 0
