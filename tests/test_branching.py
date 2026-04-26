from __future__ import annotations

import pytest

from mr1 import workflow_events as ev
from mr1.kazi_runner import MockRunner, RunStatus
from mr1.scheduler import Scheduler, replace_workflow_on_disk
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


PROV = Provenance(type="agent", id="MR1")


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def runner():
    return MockRunner()


@pytest.fixture
def scheduler(store, runner):
    sched = Scheduler(store, runner, auto_tick=False)
    yield sched
    sched.shutdown()


def _task(wf, label):
    return wf.task_by_label(label)


def _complete_check(runner: MockRunner, check_id: str, exit_code: int) -> None:
    runner.complete(
        check_id,
        RunStatus.SUCCEEDED,
        summary=f"exit {exit_code}",
        result_payload={"summary": f"exit {exit_code}", "data": {"exit_code": exit_code}},
    )


def _single_branch_spec() -> dict:
    return {
        "title": "Conditional branch",
        "tasks": [
            {
                "label": "check",
                "title": "Check",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "check",
            },
            {
                "label": "success_path",
                "title": "Success path",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["check"],
                "run_if": {
                    "ref": "check.result.data.exit_code",
                    "op": "eq",
                    "value": 0,
                },
                "prompt": "success",
            },
            {
                "label": "final",
                "title": "Final",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["success_path"],
                "prompt": "final",
            },
        ],
    }


def _join_spec() -> dict:
    return {
        "title": "Branch join",
        "tasks": [
            {
                "label": "check",
                "title": "Check",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "check",
            },
            {
                "label": "success_path",
                "title": "Success path",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["check"],
                "run_if": {
                    "ref": "check.result.data.exit_code",
                    "op": "eq",
                    "value": 0,
                },
                "prompt": "success",
            },
            {
                "label": "failure_path",
                "title": "Failure path",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["check"],
                "run_if": {
                    "ref": "check.result.data.exit_code",
                    "op": "ne",
                    "value": 0,
                },
                "prompt": "failure",
            },
            {
                "label": "final",
                "title": "Final",
                "task_kind": "agent",
                "agent_type": "kazi",
                "depends_on": ["success_path", "failure_path"],
                "dependency_policy": "any_succeeded",
                "prompt": "final",
            },
        ],
    }


def _join_with_branch_context_spec() -> dict:
    spec = _join_spec()
    spec["tasks"][3]["inputs"] = [
        {"name": "success_path_status", "from": "success_path.status"},
        {"name": "success_path_condition", "from": "success_path.condition_result"},
        {"name": "success_path_skip_reason", "from": "success_path.skip_reason"},
        {"name": "failure_path_status", "from": "failure_path.status"},
        {"name": "failure_path_condition", "from": "failure_path.condition_result"},
        {"name": "failure_path_skip_reason", "from": "failure_path.skip_reason"},
    ]
    return spec


class TestBranchingScheduler:
    def test_task_with_true_condition_runs(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(_single_branch_spec(), PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        _complete_check(runner, wf.label_to_task_id["check"], 0)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "success_path").status is TaskStatus.RUNNING

    def test_task_with_false_condition_becomes_skipped_and_creates_no_attempt(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(_single_branch_spec(), PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        _complete_check(runner, wf.label_to_task_id["check"], 1)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        branch = _task(wf, "success_path")
        assert branch.status is TaskStatus.SKIPPED
        assert branch.attempt_count == 0
        assert branch.current_attempt == 0
        assert branch.skip_reason
        assert branch.condition_result["passed"] is False
        assert store.load_task_output(wf_id, branch.task_id) is None

    def test_all_succeeded_with_skipped_dependency_skips_downstream(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(_single_branch_spec(), PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        _complete_check(runner, wf.label_to_task_id["check"], 1)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "success_path").status is TaskStatus.SKIPPED
        assert _task(wf, "final").status is TaskStatus.SKIPPED

    def test_any_succeeded_join_runs_after_one_branch_succeeds_and_one_skips(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(_join_spec(), PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        _complete_check(runner, wf.label_to_task_id["check"], 0)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "success_path").status is TaskStatus.RUNNING
        assert _task(wf, "failure_path").status is TaskStatus.SKIPPED
        assert _task(wf, "final").status is TaskStatus.WAITING

        runner.complete(_task(wf, "success_path").task_id, RunStatus.SUCCEEDED, summary="ok")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "final").status is TaskStatus.RUNNING

    def test_any_succeeded_join_with_branch_context_inputs_materializes_nullable_fields(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(_join_with_branch_context_spec(), PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        _complete_check(runner, wf.label_to_task_id["check"], 0)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "success_path").status is TaskStatus.RUNNING
        assert _task(wf, "failure_path").status is TaskStatus.SKIPPED

        runner.complete(_task(wf, "success_path").task_id, RunStatus.SUCCEEDED, summary="ok")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        final = _task(wf, "final")
        assert final.status is TaskStatus.RUNNING
        assert final.dataflow_error is None

        resolved_inputs = store.load_task_inputs(wf_id, final.task_id)
        assert resolved_inputs is not None
        values = {item.name: item.value for item in resolved_inputs}
        assert values["success_path_status"] == "succeeded"
        assert values["success_path_condition"] is None
        assert values["success_path_skip_reason"] is None
        assert values["failure_path_status"] == "skipped"
        assert values["failure_path_condition"]["passed"] is False
        assert isinstance(values["failure_path_skip_reason"], str)

    def test_any_succeeded_skips_if_no_dependency_succeeded(self, scheduler, store, runner):
        spec = _join_spec()
        spec["tasks"][1]["run_if"]["value"] = 10
        spec["tasks"][2]["run_if"]["op"] = "eq"
        spec["tasks"][2]["run_if"]["value"] = 11
        wf_id = scheduler.submit_workflow(spec, PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        _complete_check(runner, wf.label_to_task_id["check"], 0)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "success_path").status is TaskStatus.SKIPPED
        assert _task(wf, "failure_path").status is TaskStatus.SKIPPED
        assert _task(wf, "final").status is TaskStatus.SKIPPED

    def test_events_include_condition_evaluated_and_task_skipped(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(_single_branch_spec(), PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        _complete_check(runner, wf.label_to_task_id["check"], 1)
        scheduler.tick()

        events = store.load_events(wf_id)
        types = [event.event_type for event in events]
        assert ev.CONDITION_EVALUATED in types
        assert ev.TASK_SKIPPED in types
        condition_event = next(event for event in events if event.event_type == ev.CONDITION_EVALUATED)
        assert condition_event.metadata["op"] == "eq"
        assert condition_event.metadata["actual"] == 1
        assert condition_event.metadata["passed"] is False

    def test_rerun_skipped_task_re_evaluates_condition(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(_single_branch_spec(), PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        _complete_check(runner, wf.label_to_task_id["check"], 1)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "success_path").status is TaskStatus.SKIPPED

        scheduler.rerun_task(wf_id, "success_path")
        wf = store.load_workflow(wf_id)
        branch = _task(wf, "success_path")
        assert branch.status is TaskStatus.READY
        assert branch.skip_reason is None
        assert branch.condition_result is None

        scheduler.tick()
        wf = store.load_workflow(wf_id)
        assert _task(wf, "success_path").status is TaskStatus.SKIPPED

    def test_replace_skipped_task_allowed(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(_single_branch_spec(), PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)

        _complete_check(runner, wf.label_to_task_id["check"], 1)
        scheduler.tick()

        replace_workflow_on_disk(
            store,
            wf_id,
            "success_path",
            {
                "tasks": [
                    {
                        "label": "success_path",
                        "title": "Forced path",
                        "task_kind": "agent",
                        "agent_type": "kazi",
                        "depends_on": ["check"],
                        "prompt": "forced",
                    }
                ]
            },
            agent_id="test",
        )

        wf = store.load_workflow(wf_id)
        branch = _task(wf, "success_path")
        assert branch.prompt == "forced"
        assert branch.status is TaskStatus.READY

    def test_skipped_downstream_tasks_reopen_after_upstream_rerun(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(_single_branch_spec(), PROV)
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        check_id = wf.label_to_task_id["check"]

        _complete_check(runner, check_id, 1)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "success_path").status is TaskStatus.SKIPPED
        assert _task(wf, "final").status is TaskStatus.SKIPPED

        scheduler.rerun_task(wf_id, "check")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "check").status is TaskStatus.RUNNING
        assert _task(wf, "success_path").status is TaskStatus.WAITING
        assert _task(wf, "final").status is TaskStatus.WAITING

        _complete_check(runner, check_id, 0)
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "success_path").status is TaskStatus.RUNNING

        runner.complete(_task(wf, "success_path").task_id, RunStatus.SUCCEEDED, summary="ok")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert _task(wf, "final").status is TaskStatus.RUNNING
