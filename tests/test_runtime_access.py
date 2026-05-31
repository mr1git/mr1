from __future__ import annotations

import json

import pytest

from mr1.capability_policy import (
    CapabilityApprovalRequest,
    CapabilityRequest,
    CapabilityApprovalStore,
    build_scope_context,
)
from mr1.dataflow import ResolvedTaskInput, TaskOutput
from mr1.event_log import EventLog
from mr1.messages import MessageStore
from mr1.runtime_access import RuntimeAccess
from mr1.scoped_agents import (
    PersistentAgentStore,
    build_assignment_packet,
    render_assignment_mission,
)
from mr1.workflow_models import Provenance, Task, TaskStatus, Workflow, WorkflowStatus
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def workflow_store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def message_store(tmp_path, agent_store):
    return MessageStore(root=tmp_path / "messages", scoped_agent_store=agent_store)


@pytest.fixture
def approval_store(tmp_path):
    return CapabilityApprovalStore(tmp_path / "capability_approvals")


@pytest.fixture
def event_log(tmp_path):
    return EventLog(tmp_path / "events")


@pytest.fixture
def runtime_access(workflow_store, agent_store, message_store, approval_store, event_log):
    return RuntimeAccess(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        approval_store=approval_store,
        event_log=event_log,
    )


def _make_workflow(child_id: str) -> Workflow:
    task = Task(
        task_id="tk-1",
        workflow_id="wf-1",
        label="analyze",
        title="Analyze",
        task_kind="agent",
        agent_type="kazi",
        prompt="Analyze the runtime failure.",
        status=TaskStatus.FAILED,
        last_error="x" * 600,
        last_error_type="policy_block",
        result_summary="y" * 400,
    )
    return Workflow(
        workflow_id="wf-1",
        title="Observation workflow",
        status=WorkflowStatus.FAILED,
        created_by=Provenance(type="agent", id="MR1"),
        owner_agent_id=child_id,
        owner_agent_title="Sentinel",
        metadata={"memory_refs_used": ["insight:test"]},
        tasks={task.task_id: task},
        label_to_task_id={task.label: task.task_id},
    )


def test_runtime_access_preview_and_full_reads(
    runtime_access: RuntimeAccess,
    workflow_store: WorkflowStore,
    agent_store: PersistentAgentStore,
    message_store: MessageStore,
    approval_store: CapabilityApprovalStore,
    event_log: EventLog,
    tmp_path,
):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "Sentinel")
    long_mission = "Mission " + ("alpha " * 120)
    long_parent_request = "Parent " + ("beta " * 120)
    long_last_action_detail = "gamma " * 160
    child.mission = long_mission
    child.parent_request = long_parent_request
    child.run_status = "waiting"
    child.last_action = {
        "action": "send_message",
        "reason": long_last_action_detail,
        "detail": long_last_action_detail,
        "next_status": "waiting",
    }
    agent_store.save_agent(child)
    assignment_packet = build_assignment_packet(
        root,
        child.title,
        long_parent_request,
        {
            "assigned_clearance": child.security_clearance,
            "agents": [root.agent_id, child.agent_id],
            "messages": [message.message_id for message in []],
            "workflows": ["wf-1"],
        },
    )
    child.assignment_packet = assignment_packet
    child.mission = render_assignment_mission(assignment_packet) or long_mission
    agent_store.save_agent(child)

    long_body = "Body " + ("delta" * 1100)
    message = message_store.create_message(
        from_agent_id=child.agent_id,
        to_agent_id=root.agent_id,
        kind="question",
        subject="Need review",
        body=long_body,
    )

    workflow = _make_workflow(child.agent_id)
    workflow_store.save_workflow(workflow)
    workflow_store.write_result(
        workflow.workflow_id,
        "tk-1",
        {
            "task_id": "tk-1",
            "workflow_id": workflow.workflow_id,
            "status": "failed",
            "summary": "z" * 300,
            "text": "Detailed runtime failure",
            "error": "policy block because full detail matters",
            "error_type": "policy_block",
            "failure_type": "policy_block",
        },
    )
    workflow_store.write_task_output(
        workflow.workflow_id,
        "tk-1",
        TaskOutput(
            task_id="tk-1",
            workflow_id=workflow.workflow_id,
            status="failed",
            summary="Detailed output summary",
            text="Complete synthesized findings",
        ),
    )
    workflow_store.write_task_inputs(
        workflow.workflow_id,
        "tk-1",
        [
            ResolvedTaskInput(
                name="message_body",
                source=f"{message.message_id}.body",
                resolved_task_id="tk-1",
                resolved_type="text",
                value=long_body,
            )
        ],
    )

    request = CapabilityRequest(
        actor_id=child.agent_id,
        actor_type="mrn",
        invocation_mode="direct",
        capability_name="read_file",
        args={"path": "README.md"},
        scope=build_scope_context(
            actor_id=child.agent_id,
            workspace_root=tmp_path,
            scoped_agent_store=agent_store,
        ),
        step_id=f"{child.agent_id}:1",
    )
    approval = CapabilityApprovalRequest(
        approval_request_id="cap_approval_runtime_access",
        requesting_actor_id=child.agent_id,
        capability_name="read_file",
        invocation_mode="direct",
        args={"path": "README.md"},
        risk_score=0.1,
        reason="outside_actor_scope",
        scope_summary=request.scope.to_dict(),
        original_request=request,
        original_step_id=request.step_id,
        designated_approver_id=root.agent_id,
        workflow_id=workflow.workflow_id,
        task_id="tk-1",
    )
    approval_store.save(approval)
    event_log.emit(
        event_type="workflow_task_failed",
        actor_id="MR1",
        actor_type="mr1",
        target_id="tk-1",
        target_type="task",
        status="failed",
        summary="Complete runtime failure detail for task tk-1",
        workflow_id=workflow.workflow_id,
        task_id="tk-1",
        message_id=message.message_id,
        approval_request_id=approval.approval_request_id,
        record_path=str(tmp_path / "workflows" / workflow.workflow_id / "tasks" / "tk-1" / "result.json"),
    )

    agent_preview = next(
        item for item in runtime_access.list_agents(caller_agent_id=root.agent_id)
        if item["agent_id"] == child.agent_id
    )
    assert agent_preview["lifecycle_status"] == "active"
    assert agent_preview["is_live"] is True
    assert agent_preview["is_terminal"] is False
    assert agent_preview["status_conflict"] is False
    assert agent_preview["mission_truncated"] is True
    assert agent_preview["mission_full_available"] is True
    assert agent_preview["parent_request_truncated"] is True
    assert agent_preview["last_action_truncated"] is True
    assert agent_preview["pending_parent_messages"][0]["message_id"] == message.message_id

    agent_detail = runtime_access.read_agent(child.agent_id, caller_agent_id=root.agent_id)
    assert agent_detail["lifecycle_status"] == "active"
    assert agent_detail["is_live"] is True
    assert agent_detail["is_terminal"] is False
    assert agent_detail["status_conflict"] is False
    assert agent_detail["mission"] == (render_assignment_mission(assignment_packet) or long_mission)
    assert agent_detail["parent_request"] == long_parent_request
    assert agent_detail["last_action"]["detail"] == long_last_action_detail
    assert agent_detail["assignment_packet"]["full_parent_request"] == long_parent_request
    assert agent_detail["assignment_packet"]["relevant_context"]["agents"] == [root.agent_id, child.agent_id]
    assert agent_detail["pending_parent_messages"][0]["message_id"] == message.message_id

    message_preview = next(
        item for item in runtime_access.list_messages(caller_agent_id=root.agent_id)
        if item["message_id"] == message.message_id
    )
    assert message_preview["body_truncated"] is True
    assert message_preview["body_full_available"] is True
    assert message_preview["body_preview"].endswith("...")

    message_detail = runtime_access.read_message(message.message_id, caller_agent_id=root.agent_id)
    assert message_detail["body"] == long_body
    assert message_detail["body_truncated"] is False

    workflow_preview = next(
        item for item in runtime_access.list_workflows(caller_agent_id=root.agent_id)
        if item["workflow_id"] == workflow.workflow_id
    )
    assert workflow_preview["recent_task_status_summary"][0]["summary_truncated"] is True
    assert workflow_preview["recent_task_status_summary"][0]["summary_full_available"] is True

    workflow_detail = runtime_access.read_workflow(workflow.workflow_id, caller_agent_id=root.agent_id)
    task_detail = workflow_detail["task_details"]["tk-1"]
    assert task_detail["task"]["last_error"] == "x" * 600
    assert task_detail["result_payload"]["error"] == "policy block because full detail matters"
    assert task_detail["output_payload"]["text"] == "Complete synthesized findings"
    assert task_detail["inputs_payload"][0]["value"] == long_body

    approvals = runtime_access.list_pending_approvals(caller_agent_id=root.agent_id)
    assert approvals[0]["approval_request_id"] == approval.approval_request_id
    assert approvals[0]["task_id"] == "tk-1"

    recent_errors = runtime_access.list_recent_errors(caller_agent_id=root.agent_id)
    assert recent_errors[0]["workflow_id"] == workflow.workflow_id
    assert recent_errors[0]["task_id"] == "tk-1"
    assert recent_errors[0]["message_id"] == message.message_id
    assert recent_errors[0]["approval_request_id"] == approval.approval_request_id
    assert recent_errors[0]["record_path"].endswith("result.json")


def test_runtime_access_marks_legacy_active_terminated_agent_as_terminal(
    runtime_access: RuntimeAccess,
    agent_store: PersistentAgentStore,
):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "Sentinel")
    child.run_status = "terminated"
    agent_store.save_agent(child)

    preview = next(
        item for item in runtime_access.list_agents(caller_agent_id=root.agent_id)
        if item["agent_id"] == child.agent_id
    )
    detail = runtime_access.read_agent(child.agent_id, caller_agent_id=root.agent_id)

    assert preview["status"] == "active"
    assert preview["run_status"] == "terminated"
    assert preview["lifecycle_status"] == "terminated"
    assert preview["is_live"] is False
    assert preview["is_terminal"] is True
    assert preview["status_conflict"] is True
    assert detail["lifecycle_status"] == "terminated"
    assert detail["is_live"] is False
    assert detail["is_terminal"] is True
    assert detail["status_conflict"] is True
