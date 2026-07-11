from __future__ import annotations

import pytest

from mr1.capability_policy import CapabilityApprovalDecision
from mr1.kazi_runner import MockRunner
from mr1.scheduler import Scheduler
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_models import Provenance, TaskStatus, WorkflowStatus
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def runner():
    return MockRunner()


@pytest.fixture
def scheduler(store, agent_store, runner, tmp_path):
    sched = Scheduler(
        store,
        runner,
        auto_tick=False,
        scoped_agent_store=agent_store,
        workspace_root=tmp_path,
    )
    yield sched
    sched.shutdown()


def _scoped_child(agent_store: PersistentAgentStore, tmp_path, *, clearance: float = 0.1):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research", security_clearance=clearance)
    child.scope_roots = [str(tmp_path)]
    agent_store.save_agent(child)
    return root, child


def _write_file_workflow_spec(target_path: str) -> dict:
    return {
        "title": "Scoped write workflow",
        "tasks": [{
            "label": "write",
            "title": "Write",
            "task_kind": "tool",
            "tool_type": "write_file",
            "tool_config": {
                "path": target_path,
                "content": "hello",
                "overwrite": True,
            },
        }],
    }


def test_workflow_risk_above_clearance_blocks_and_requests_approval(scheduler, store, agent_store, tmp_path):
    root, child = _scoped_child(agent_store, tmp_path, clearance=0.1)
    target = tmp_path / "blocked.txt"

    wf_id = scheduler.submit_workflow(
        _write_file_workflow_spec(str(target)),
        Provenance(type="agent", id=root.agent_id),
        owner_agent_id=child.agent_id,
        caller_agent_id=root.agent_id,
    )
    scheduler.tick()

    wf = store.load_workflow(wf_id)
    task = wf.task_by_label("write")
    approvals = scheduler._approval_store.list_requests()

    assert task.status is TaskStatus.BLOCKED
    assert task.last_error_type == "approval_required"
    assert target.exists() is False
    assert wf.status is WorkflowStatus.RUNNING
    assert len(approvals) == 1
    assert approvals[0].designated_approver_id == root.agent_id


def test_risk_1_0_workflow_requires_human_approval(scheduler, store, agent_store, tmp_path):
    root = agent_store.ensure_root_agent()
    wf_id = scheduler.submit_workflow(
        {
            "title": "Shell workflow",
            "tasks": [{
                "label": "shell",
                "title": "Shell",
                "task_kind": "tool",
                "tool_type": "shell_command",
                "tool_config": {
                    "argv": ["python", "--version"],
                    "cwd": str(tmp_path),
                },
            }],
        },
        Provenance(type="agent", id=root.agent_id),
        owner_agent_id=root.agent_id,
    )
    scheduler.tick()

    wf = store.load_workflow(wf_id)
    task = wf.task_by_label("shell")
    approval = scheduler._approval_store.list_requests()[0]

    assert task.status is TaskStatus.BLOCKED
    assert task.last_error_type == "approval_required"
    assert approval.risk_score == 1.0
    assert approval.designated_approver_id == root.agent_id


def test_approval_grant_reopens_blocked_workflow_task_automatically(scheduler, store, agent_store, tmp_path):
    root, child = _scoped_child(agent_store, tmp_path, clearance=0.1)
    target = tmp_path / "resume.txt"
    wf_id = scheduler.submit_workflow(
        _write_file_workflow_spec(str(target)),
        Provenance(type="agent", id=root.agent_id),
        owner_agent_id=child.agent_id,
        caller_agent_id=root.agent_id,
    )
    scheduler.tick()

    approval = scheduler._approval_store.list_requests()[0]
    scheduler._approval_store.apply_decision(
        approval.approval_request_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=approval.approval_request_id,
            decision="approved",
            decided_by=root.agent_id,
            reason="approved",
            timestamp=1.0,
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )

    wf = store.load_workflow(wf_id)
    task = wf.task_by_label("write")
    assert task.status is TaskStatus.READY

    scheduler.tick()
    wf = store.load_workflow(wf_id)
    task = wf.task_by_label("write")

    assert task.status is TaskStatus.SUCCEEDED
    assert task.attempt_count == 2
    assert target.read_text(encoding="utf-8") == "hello"


def test_approval_resume_is_not_clobbered_by_stale_scheduler_write(scheduler, store, agent_store, tmp_path):
    root = agent_store.ensure_root_agent()
    wf_id = scheduler.submit_workflow(
        {
            "title": "Approval resume race",
            "tasks": [
                {
                    "label": "shell",
                    "title": "Shell",
                    "task_kind": "tool",
                    "tool_type": "shell_command",
                    "tool_config": {
                        "argv": ["python", "--version"],
                        "cwd": str(tmp_path),
                    },
                },
                {
                    "label": "agent",
                    "title": "Agent",
                    "task_kind": "agent",
                    "agent_type": "kazi",
                    "prompt": "keep running",
                },
            ],
        },
        Provenance(type="agent", id=root.agent_id),
        owner_agent_id=root.agent_id,
    )
    scheduler.tick()

    stale_wf = store.load_workflow(wf_id)
    assert stale_wf is not None
    assert stale_wf.task_by_label("shell").status is TaskStatus.BLOCKED
    assert stale_wf.task_by_label("agent").status is TaskStatus.RUNNING

    approval = scheduler._approval_store.list_requests()[0]
    scheduler._approval_store.apply_decision(
        approval.approval_request_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=approval.approval_request_id,
            decision="approved",
            decided_by=root.agent_id,
            reason="approve",
            timestamp=1.0,
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )

    reopened = store.load_workflow(wf_id)
    assert reopened is not None
    assert reopened.task_by_label("shell").status is TaskStatus.READY

    scheduler._commit(
        stale_wf,
        stale_wf.task_by_label("agent"),
        TaskStatus.RUNNING,
        event=None,
        message="preserve reopened task",
    )

    preserved = store.load_workflow(wf_id)
    assert preserved is not None
    assert preserved.task_by_label("shell").status is TaskStatus.READY


def test_approval_denial_leaves_blocked_final_state(scheduler, store, agent_store, tmp_path):
    root, child = _scoped_child(agent_store, tmp_path, clearance=0.1)
    wf_id = scheduler.submit_workflow(
        _write_file_workflow_spec(str(tmp_path / "denied.txt")),
        Provenance(type="agent", id=root.agent_id),
        owner_agent_id=child.agent_id,
        caller_agent_id=root.agent_id,
    )
    scheduler.tick()

    approval = scheduler._approval_store.list_requests()[0]
    scheduler._approval_store.apply_decision(
        approval.approval_request_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=approval.approval_request_id,
            decision="denied",
            decided_by=root.agent_id,
            reason="no",
            timestamp=1.0,
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )
    scheduler.tick()

    wf = store.load_workflow(wf_id)
    task = wf.task_by_label("write")
    approval = scheduler._approval_store.require(approval.approval_request_id)

    assert approval.status == "denied"
    assert task.status is TaskStatus.BLOCKED
    assert task.last_error_type == "approval_required"
    assert wf.status is WorkflowStatus.FAILED


def test_cancelled_workflow_expires_pending_approvals(scheduler, store, agent_store, tmp_path):
    root, child = _scoped_child(agent_store, tmp_path, clearance=0.1)
    wf_id = scheduler.submit_workflow(
        _write_file_workflow_spec(str(tmp_path / "cancelled.txt")),
        Provenance(type="agent", id=root.agent_id),
        owner_agent_id=child.agent_id,
        caller_agent_id=root.agent_id,
    )
    scheduler.tick()

    approval = scheduler._approval_store.list_requests()[0]
    assert scheduler.cancel_workflow(wf_id) is True

    wf = store.load_workflow(wf_id)
    approval = scheduler._approval_store.require(approval.approval_request_id)

    assert wf.status is WorkflowStatus.CANCELLED
    assert approval.status == "expired"
