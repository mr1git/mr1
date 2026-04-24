"""
When a parent task fails, its descendants must transition to `blocked`
with `blocked_reason`, `blocked_by`, and `blocked_at` populated.
"""

import pytest

from mr1 import workflow_events as ev
from mr1.kazi_runner import MockRunner, RunStatus
from mr1.scheduler import Scheduler
from mr1.workflow_models import Provenance, TaskStatus, WorkflowStatus
from mr1.workflow_store import WorkflowStore


SPEC = {
    "title": "Failure cascade",
    "tasks": [
        {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "x"},
        {"label": "b", "title": "B", "task_kind": "agent", "agent_type": "kazi",
         "prompt": "x", "depends_on": ["a"]},
        {"label": "c", "title": "C", "task_kind": "agent", "agent_type": "kazi",
         "prompt": "x", "depends_on": ["a"]},
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
    s = Scheduler(store, runner, auto_tick=False)
    yield s
    s.shutdown()


class TestFailureCascade:
    def test_failed_parent_blocks_both_dependents(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        b_id = wf.label_to_task_id["b"]
        c_id = wf.label_to_task_id["c"]

        runner.complete(a_id, RunStatus.FAILED, error="boom")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].status is TaskStatus.FAILED
        assert wf.tasks[b_id].status is TaskStatus.BLOCKED
        assert wf.tasks[c_id].status is TaskStatus.BLOCKED

        b = wf.tasks[b_id]
        assert b.blocked_by == [a_id]
        assert "failed" in (b.blocked_reason or "")
        assert b.blocked_at is not None

    def test_timed_out_parent_also_blocks(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        a_id = wf.label_to_task_id["a"]
        b_id = wf.label_to_task_id["b"]

        runner.complete(a_id, RunStatus.TIMED_OUT, error="timeout")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.tasks[a_id].status is TaskStatus.TIMED_OUT
        assert wf.tasks[b_id].status is TaskStatus.BLOCKED
        assert "timed_out" in (wf.tasks[b_id].blocked_reason or "")

    def test_workflow_marked_failed(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        runner.complete(wf.label_to_task_id["a"], RunStatus.FAILED, error="boom")
        scheduler.tick()

        wf = store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.FAILED
        assert wf.finished_at is not None

    def test_blocked_event_emitted(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        runner.complete(wf.label_to_task_id["a"], RunStatus.FAILED, error="boom")
        scheduler.tick()

        events = store.load_events(wf_id)
        types = [e.event_type for e in events]
        assert ev.TASK_FAILED in types
        assert types.count(ev.TASK_BLOCKED) == 2
        assert ev.WORKFLOW_FAILED == types[-1]

    def test_cancel_workflow_marks_live_tasks_cancelled(self, scheduler, store, runner):
        wf_id = scheduler.submit_workflow(SPEC, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        ok = scheduler.cancel_workflow(wf_id)
        assert ok

        wf = store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.CANCELLED
        for t in wf.tasks.values():
            assert t.is_terminal()
