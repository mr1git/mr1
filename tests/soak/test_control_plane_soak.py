"""
A8 — control-plane behaviour under load.

pause / resume / stop / halt, driven against a live objective set, plus the
property that matters most: halt revokes MR1's standing authority. A stop that
leaves consent in place has not stopped anything.
"""

from __future__ import annotations

from mr1.autonomy.budget import BudgetLimits
from mr1.autonomy.control import (
    MODE_HALTED,
    MODE_PAUSED,
    MODE_RUNNING,
    MODE_STOPPING,
)
from mr1.autonomy.objectives import (
    KIND_RECURRING,
    STATUS_ACTIVE,
    STATUS_EXECUTING,
    STATUS_PAUSED,
)
from mr1.autonomy.service import SupervisorConfig
from mr1.kazi_runner import MockRunner
from mr1.workflow_models import WorkflowStatus
from tests.soak.harness import SoakRuntime


def _runtime(tmp_path, *, runner=None):
    return SoakRuntime(
        tmp_path / "runtime",
        workspace_root=tmp_path,
        runner=runner,
        config=SupervisorConfig(
            tick_interval_s=60.0,
            doctor_interval_s=86_400.0,
            max_concurrent_workflows=8,
            max_plans_per_hour=10_000,
            max_workflows_per_objective_per_day=10_000,
        ),
        limits=BudgetLimits(
            max_plans_per_hour=10_000,
            max_actions_per_hour=10_000,
            max_workflows_per_objective_per_day=10_000,
        ),
    )


def _grant(runtime, objective):
    return runtime.supervisor.consent.create(
        objective_id=objective.objective_id,
        capability_name="shell_command",
        scope_roots=[runtime.workspace_root],
        max_risk=1.0,
        granted_by=runtime.agents.root_agent_id,
        ttl_s=7 * 86_400,
        reason="soak",
    )


def test_pause_holds_for_hundreds_of_ticks_then_resume_recovers(tmp_path):
    runtime = _runtime(tmp_path)
    for index in range(4):
        runtime.create_objective(
            title=f"obj-{index}",
            statement=f"work {index}",
            kind=KIND_RECURRING,
            trigger={"type": "interval", "interval_s": 600},
        )

    runtime.supervisor.control.set_mode(MODE_PAUSED, reason="soak")
    runtime.run(300)

    assert runtime.planner.call_count == 0, "paused must create no new work, ever"
    assert runtime.store.list_workflows() == []
    for objective in runtime.objectives.list_objectives():
        assert objective.status == STATUS_ACTIVE

    runtime.supervisor.control.set_mode(MODE_RUNNING, reason="soak")
    runtime.run(10)

    assert runtime.planner.call_count >= 4
    assert any(
        objective.success_count > 0
        for objective in runtime.objectives.list_objectives()
    )


def test_pause_lets_in_flight_work_drain(tmp_path):
    runtime = _runtime(tmp_path, runner=MockRunner())
    objective = runtime.create_objective(title="draining")
    runtime.tick(scheduler_ticks=1)
    workflow_id = runtime.objectives.require(objective.objective_id).current_workflow_id
    assert workflow_id

    runtime.supervisor.control.set_mode(MODE_PAUSED, reason="soak")

    # The in-flight task completes; pause must not have cancelled it.
    workflow = runtime.store.load_workflow(workflow_id)
    task_id = next(iter(workflow.tasks))
    from mr1.kazi_runner import RunStatus

    runtime.runner.complete(task_id, RunStatus.SUCCEEDED, exit_code=0, summary="ok")
    runtime.run(3)

    assert runtime.store.load_workflow(workflow_id).status is WorkflowStatus.SUCCEEDED
    assert runtime.objectives.require(objective.objective_id).status == "satisfied"


def test_stopping_drains_then_asks_to_exit(tmp_path):
    runtime = _runtime(tmp_path, runner=MockRunner())
    objective = runtime.create_objective(title="stopper")
    runtime.tick(scheduler_ticks=1)
    workflow_id = runtime.objectives.require(objective.objective_id).current_workflow_id

    runtime.supervisor.control.set_mode(MODE_STOPPING, reason="soak")

    outcome = runtime.supervisor.tick()
    assert outcome["gate"] == "draining"
    assert runtime.supervisor.exit_requested is False

    from mr1.kazi_runner import RunStatus

    workflow = runtime.store.load_workflow(workflow_id)
    runtime.runner.complete(
        next(iter(workflow.tasks)),
        RunStatus.SUCCEEDED,
        exit_code=0,
        summary="ok",
    )
    runtime.scheduler.tick()
    runtime.scheduler.tick()

    outcome = runtime.supervisor.tick()
    assert outcome["gate"] == "exit"
    assert runtime.supervisor.exit_requested is True


def test_halt_revokes_every_grant_and_pauses_every_objective(tmp_path):
    runtime = _runtime(tmp_path, runner=MockRunner())
    objectives = [
        runtime.create_objective(title=f"obj-{index}", statement=f"work {index}")
        for index in range(3)
    ]
    grants = [_grant(runtime, objective) for objective in objectives]
    runtime.tick(scheduler_ticks=1)

    in_flight = [
        runtime.objectives.require(objective.objective_id).current_workflow_id
        for objective in objectives
    ]
    assert any(in_flight)
    assert len(runtime.supervisor.consent.list_active()) == 3

    runtime.supervisor.control.set_mode(MODE_HALTED, reason="emergency")
    outcome = runtime.supervisor.tick()

    assert outcome["gate"] == "exit"
    assert runtime.supervisor.exit_requested is True
    # Authority is gone.
    assert runtime.supervisor.consent.list_active() == []
    for grant in grants:
        assert runtime.supervisor.consent.require(grant.grant_id).revoked is True
    # Every objective is parked.
    for objective in runtime.objectives.list_objectives():
        assert objective.status == STATUS_PAUSED
    # Every in-flight workflow is cancelled.
    for workflow_id in in_flight:
        if workflow_id:
            assert runtime.store.load_workflow(workflow_id).status is WorkflowStatus.CANCELLED


def test_a_halted_runtime_stays_halted_across_a_restart(tmp_path):
    runtime = _runtime(tmp_path)
    objective = runtime.create_objective(title="halted")
    _grant(runtime, objective)
    runtime.supervisor.control.set_mode(MODE_HALTED, reason="emergency")
    runtime.supervisor.tick()

    # A fresh supervisor picks the mode up from disk. It does not resurrect the
    # objectives, and it does not get its authority back.
    runtime.restart()
    runtime.run(20)

    assert runtime.supervisor.consent.list_active() == []
    assert runtime.objectives.require(objective.objective_id).status == STATUS_PAUSED
    assert runtime.planner.call_count == 0
    assert runtime.store.list_workflows() == []


def test_the_cli_halt_path_revokes_authority_with_no_supervisor_running(tmp_path):
    """A halt that only works when a supervisor happens to be alive is not a halt."""
    from mr1.cli.service import halt_runtime

    runtime = _runtime(tmp_path)
    objective = runtime.create_objective(title="halted by cli")
    _grant(runtime, objective)

    payload = halt_runtime(
        runtime.runtime_root,
        reason="operator pulled the cord",
        requested_by="marwan",
        clock=runtime.clock,
    )

    assert len(payload["revoked_grants"]) == 1
    assert payload["paused_objectives"] == [objective.objective_id]
    assert runtime.supervisor.consent.list_active() == []
    assert runtime.objectives.require(objective.objective_id).status == STATUS_PAUSED


def test_mode_flapping_never_wedges_the_supervisor(tmp_path):
    runtime = _runtime(tmp_path)
    for index in range(3):
        runtime.create_objective(
            title=f"obj-{index}",
            statement=f"work {index}",
            kind=KIND_RECURRING,
            trigger={"type": "interval", "interval_s": 300},
        )

    modes = [MODE_RUNNING, MODE_PAUSED, MODE_RUNNING, MODE_PAUSED, MODE_RUNNING]
    for cycle in range(60):
        runtime.supervisor.control.set_mode(modes[cycle % len(modes)], reason="flap")
        runtime.run(5)

    assert runtime.supervisor.metrics()["supervisor_tick_errors"] == 0
    assert runtime.stuck_objectives() == []
    assert any(
        objective.success_count > 0
        for objective in runtime.objectives.list_objectives()
    )
