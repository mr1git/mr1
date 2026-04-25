from __future__ import annotations

import pytest

from mr1.kazi_runner import MockRunner, RunStatus
from mr1.scheduler import Scheduler
from mr1.workflow_models import Provenance, TaskStatus
from mr1.workflow_store import WorkflowStore


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


class TestRerun:
    def test_rerun_after_failure_unblocks_downstream(self, scheduler, store, runner):
        spec = {
            "title": "Rerun unblock",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "x"},
                {"label": "b", "title": "B", "task_kind": "agent", "agent_type": "kazi", "prompt": "y", "depends_on": ["a"]},
            ],
        }
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        b_id = wf.label_to_task_id["b"]

        runner.complete(a_id, RunStatus.FAILED, error="boom")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[b_id].status is TaskStatus.BLOCKED

        scheduler.rerun_task(wf_id, "a")
        scheduler.tick()
        runner.complete(a_id, RunStatus.SUCCEEDED, summary="recovered")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].attempt_count == 2
        assert wf.tasks[b_id].status in {TaskStatus.READY, TaskStatus.RUNNING}

    def test_rerun_after_success_creates_new_attempt_and_preserves_output(self, scheduler, store, runner):
        spec = {
            "title": "Rerun success",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "x"},
            ],
        }
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task_id = wf.label_to_task_id["a"]

        runner.complete(task_id, RunStatus.SUCCEEDED, summary="ok", result_payload={"status": "succeeded"})
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        first_output = wf.tasks[task_id].output_path

        scheduler.rerun_task(wf_id, "a")
        scheduler.tick()
        runner.complete(task_id, RunStatus.FAILED, error="second failed")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        task = wf.tasks[task_id]
        assert task.attempt_count == 2
        assert task.status is TaskStatus.FAILED
        assert task.output_path == first_output
        assert task.last_error == "second failed"

    def test_explicitly_cancelled_downstream_stays_cancelled(self, scheduler, store, runner):
        spec = {
            "title": "Cancelled downstream",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "x"},
                {"label": "b", "title": "B", "task_kind": "agent", "agent_type": "kazi", "prompt": "y", "depends_on": ["a"]},
            ],
        }
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        b_id = wf.label_to_task_id["b"]

        runner.complete(a_id, RunStatus.FAILED, error="boom")
        scheduler.tick()
        scheduler.cancel_task(b_id)
        scheduler.rerun_task(wf_id, "a")
        scheduler.tick()
        runner.complete(a_id, RunStatus.SUCCEEDED, summary="fixed")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[b_id].status is TaskStatus.CANCELLED
