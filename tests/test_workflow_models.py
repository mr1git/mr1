"""Tests for mr1.workflow_models"""

import re

import pytest

from mr1.workflow_models import (
    Provenance,
    Task,
    TaskStatus,
    Workflow,
    WorkflowEvent,
    WorkflowStatus,
    TERMINAL_TASK_STATUSES,
    FAILED_TASK_STATUSES,
    new_task_id,
    new_workflow_id,
)


class TestIDs:
    def test_workflow_id_format(self):
        wid = new_workflow_id()
        assert re.fullmatch(r"wf-\d{8}T\d{6}-[0-9a-f]{6}", wid)

    def test_task_id_format(self):
        tid = new_task_id()
        assert re.fullmatch(r"tk-\d{8}T\d{6}-[0-9a-f]{6}", tid)

    def test_ids_unique(self):
        assert len({new_workflow_id() for _ in range(50)}) == 50
        assert len({new_task_id() for _ in range(50)}) == 50


class TestProvenance:
    def test_roundtrip(self):
        p = Provenance(type="agent", id="MR1")
        assert Provenance.from_dict(p.to_dict()) == p


class TestTask:
    def _sample(self) -> Task:
        return Task(
            task_id="tk-1",
            workflow_id="wf-1",
            label="a",
            title="Task A",
            task_kind="agent",
            agent_type="kazi",
            prompt="Say hi",
            depends_on=["tk-0"],
            status=TaskStatus.WAITING,
            created_by=Provenance(type="agent", id="MR1"),
            blocked_by=[],
            timeout_s=60,
        )

    def test_roundtrip(self):
        t = self._sample()
        data = t.to_dict()
        assert data["status"] == "waiting"
        assert data["created_by"] == {"type": "agent", "id": "MR1"}
        t2 = Task.from_dict(data)
        assert t2.to_dict() == data

    def test_status_enum_preserved(self):
        t = self._sample()
        t.status = TaskStatus.BLOCKED
        data = t.to_dict()
        assert data["status"] == "blocked"
        assert Task.from_dict(data).status is TaskStatus.BLOCKED

    def test_is_terminal(self):
        t = self._sample()
        assert not t.is_terminal()
        for terminal in TERMINAL_TASK_STATUSES:
            t.status = terminal
            assert t.is_terminal()

    def test_failed_set_subset_of_terminal(self):
        assert FAILED_TASK_STATUSES <= TERMINAL_TASK_STATUSES


class TestWorkflow:
    def test_roundtrip(self):
        t = Task(
            task_id="tk-1",
            workflow_id="wf-1",
            label="a",
            title="Task A",
            task_kind="agent",
            agent_type="kazi",
            prompt="Do it",
        )
        wf = Workflow(
            workflow_id="wf-1",
            title="Demo",
            status=WorkflowStatus.RUNNING,
            created_by=Provenance(type="user", id="cli"),
            tasks={t.task_id: t},
            label_to_task_id={"a": t.task_id},
        )
        data = wf.to_dict()
        wf2 = Workflow.from_dict(data)
        assert wf2.to_dict() == data
        assert wf2.task_by_label("a") is not None
        assert wf2.task_by_label("a").title == "Task A"

    def test_missing_label_returns_none(self):
        wf = Workflow(workflow_id="wf-1", title="x")
        assert wf.task_by_label("missing") is None


class TestWorkflowEvent:
    def test_new_populates_timestamp(self):
        ev = WorkflowEvent.new(
            event_type="task_started",
            workflow_id="wf-1",
            task_id="tk-1",
            agent_id="scheduler",
            metadata={"extra": 1},
        )
        assert ev.event_type == "task_started"
        assert ev.workflow_id == "wf-1"
        assert ev.metadata == {"extra": 1}
        assert ev.timestamp

    def test_roundtrip(self):
        ev = WorkflowEvent.new(
            event_type="task_succeeded",
            workflow_id="wf-1",
            task_id="tk-1",
        )
        assert WorkflowEvent.from_dict(ev.to_dict()) == ev
