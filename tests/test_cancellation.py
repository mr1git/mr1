from __future__ import annotations

import pytest

from mr1 import workflow_events as ev
from mr1.kazi_runner import MockRunner
from mr1.scheduler import Scheduler
from mr1.workflow_models import Provenance, TaskStatus, WorkflowStatus
from mr1.workflow_store import WorkflowStore


SPEC = {
    "title": "Cancellation",
    "tasks": [
        {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "x"},
        {"label": "b", "title": "B", "task_kind": "agent", "agent_type": "kazi", "prompt": "y", "depends_on": ["a"]},
    ],
}


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


class TestCancellation:
    def test_cancel_running_task_marks_attempt_cancelled(self, scheduler, store):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task_id = wf.label_to_task_id["a"]

        cancelled = scheduler.cancel_task(task_id)
        assert cancelled == task_id
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        task = wf.tasks[task_id]
        assert task.status is TaskStatus.CANCELLED
        assert task.attempt_count == 1
        assert task.attempts[0].status is TaskStatus.CANCELLED
        assert task.attempts[0].error_type == "cancelled"
        assert task.last_error_type == "cancelled"

        events = store.load_events(wf_id, task_id=task_id)
        event_types = [event.event_type for event in events]
        assert ev.TASK_CANCELLED in event_types
        assert ev.TASK_ATTEMPT_FINISHED in event_types

    def test_cancel_waiting_task_does_not_create_attempt(self, scheduler, store):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task_id = wf.label_to_task_id["b"]

        scheduler.cancel_task(task_id)
        wf = store.load_workflow(wf_id)
        task = wf.tasks[task_id]
        assert task.status is TaskStatus.CANCELLED
        assert task.attempt_count == 0
        assert task.attempts == []

    def test_cancel_workflow_cancels_running_and_waiting_tasks(self, scheduler, store):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()

        assert scheduler.cancel_workflow(wf_id) is True
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.CANCELLED
        assert wf.tasks[wf.label_to_task_id["a"]].status is TaskStatus.CANCELLED
        assert wf.tasks[wf.label_to_task_id["b"]].status is TaskStatus.CANCELLED
        assert wf.tasks[wf.label_to_task_id["a"]].attempt_count == 1
        assert wf.tasks[wf.label_to_task_id["b"]].attempt_count == 0
