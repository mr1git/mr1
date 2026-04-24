"""
Scheduler discovers workflows written to disk by an external process
(the CLI) without needing a fresh submit_workflow() call.
"""

import pytest

from mr1.kazi_runner import MockRunner
from mr1.scheduler import Scheduler, submit_spec_to_disk
from mr1.workflow_models import Provenance, TaskStatus, WorkflowStatus
from mr1.workflow_store import WorkflowStore


SPEC = {
    "title": "CLI-submitted workflow",
    "tasks": [
        {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "x"},
        {"label": "b", "title": "B", "task_kind": "agent", "agent_type": "kazi",
         "prompt": "x", "depends_on": ["a"]},
    ],
}


def test_tick_claims_workflow_written_externally(tmp_path):
    # Simulate the CLI: write the workflow to disk without a scheduler.
    store = WorkflowStore(root=tmp_path / "workflows")
    wf_id = submit_spec_to_disk(SPEC, Provenance(type="user", id="cli"), store)

    # Confirm tasks landed as `created` on disk.
    wf = store.load_workflow(wf_id)
    assert wf.status is WorkflowStatus.PENDING
    for task in wf.tasks.values():
        assert task.status is TaskStatus.CREATED

    # Now MR1 starts a scheduler over the same store. One tick should
    # promote created → waiting/ready and launch the root task.
    runner = MockRunner()
    sched = Scheduler(store, runner, auto_tick=False)
    try:
        sched.tick()
    finally:
        sched.shutdown()

    wf = store.load_workflow(wf_id)
    a = wf.task_by_label("a")
    b = wf.task_by_label("b")
    assert a.status is TaskStatus.RUNNING
    assert b.status is TaskStatus.WAITING
    assert wf.status is WorkflowStatus.RUNNING
    assert runner.started_task_ids == [a.task_id]


def test_cli_can_submit_while_mr1_offline(tmp_path):
    """
    End-to-end shape of the MR1-offline case: CLI writes, no scheduler
    exists yet, workflow stays `created`. Later a scheduler attaches
    and drives it.
    """
    store = WorkflowStore(root=tmp_path / "workflows")
    wf_id = submit_spec_to_disk(SPEC, Provenance(type="user", id="cli"), store)
    # Nothing touches the workflow.
    for _ in range(3):
        wf = store.load_workflow(wf_id)
        assert wf.status is WorkflowStatus.PENDING

    runner = MockRunner()
    sched = Scheduler(store, runner, auto_tick=False)
    try:
        sched.tick()
    finally:
        sched.shutdown()

    wf = store.load_workflow(wf_id)
    assert wf.status is WorkflowStatus.RUNNING
