from __future__ import annotations

import pytest

from mr1.kazi_runner import MockRunner, RunStatus
from mr1.scheduler import (
    Scheduler,
    WorkflowSpecError,
    append_workflow_on_disk,
    insert_workflow_on_disk,
    replace_workflow_on_disk,
)
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


class TestWorkflowMutation:
    def test_append_workflow_adds_new_tasks(self, scheduler, store):
        spec = {
            "title": "Append",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "x"},
            ],
        }
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))

        append_workflow_on_disk(
            store,
            wf_id,
            {
                "tasks": [
                    {
                        "label": "b",
                        "title": "B",
                        "task_kind": "agent",
                        "agent_type": "kazi",
                        "prompt": "y",
                        "depends_on": ["a"],
                    }
                ]
            },
            agent_id="test",
        )

        wf = store.load_workflow(wf_id)
        assert "b" in wf.label_to_task_id
        b = wf.task_by_label("b")
        assert b.status is TaskStatus.CREATED
        assert b.depends_on == [wf.task_by_label("a").task_id]

    def test_insert_workflow_rewires_direct_children(self, scheduler, store):
        spec = {
            "title": "Insert",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "x"},
                {"label": "b", "title": "B", "task_kind": "agent", "agent_type": "kazi", "prompt": "y", "depends_on": ["a"]},
            ],
        }
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))

        insert_workflow_on_disk(
            store,
            wf_id,
            "a",
            {
                "tasks": [
                    {
                        "label": "x",
                        "title": "X",
                        "task_kind": "agent",
                        "agent_type": "kazi",
                        "prompt": "mid",
                    }
                ]
            },
            agent_id="test",
        )

        wf = store.load_workflow(wf_id)
        a = wf.task_by_label("a")
        b = wf.task_by_label("b")
        x = wf.task_by_label("x")
        assert x.depends_on == [a.task_id]
        assert b.depends_on == [x.task_id]

    def test_replace_workflow_preserves_task_id_and_attempt_history(self, scheduler, store, runner):
        spec = {
            "title": "Replace",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "old"},
            ],
        }
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task_id = wf.label_to_task_id["a"]
        runner.complete(task_id, RunStatus.FAILED, error="boom")
        scheduler.tick()

        replace_workflow_on_disk(
            store,
            wf_id,
            "a",
            {
                "tasks": [
                    {
                        "label": "a",
                        "title": "A2",
                        "task_kind": "agent",
                        "agent_type": "kazi",
                        "prompt": "new",
                    }
                ]
            },
            agent_id="test",
        )

        wf = store.load_workflow(wf_id)
        task = wf.task_by_label("a")
        assert task.task_id == task_id
        assert task.prompt == "new"
        assert task.attempt_count == 1
        assert len(task.attempts) == 1
        assert task.status is TaskStatus.CREATED

    def test_replace_rejects_succeeded_task(self, scheduler, store, runner):
        spec = {
            "title": "Replace reject",
            "tasks": [
                {"label": "a", "title": "A", "task_kind": "agent", "agent_type": "kazi", "prompt": "old"},
            ],
        }
        wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id="MR1"))
        scheduler.tick()
        wf = store.load_workflow(wf_id)
        task_id = wf.label_to_task_id["a"]
        runner.complete(task_id, RunStatus.SUCCEEDED, summary="ok")
        scheduler.tick()

        with pytest.raises(WorkflowSpecError):
            replace_workflow_on_disk(
                store,
                wf_id,
                "a",
                {
                    "tasks": [
                        {
                            "label": "a",
                            "title": "A2",
                            "task_kind": "agent",
                            "agent_type": "kazi",
                            "prompt": "new",
                        }
                    ]
                },
                agent_id="test",
            )
