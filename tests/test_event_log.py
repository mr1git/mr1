from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mr1.capability_policy import CapabilityApprovalDecision
from mr1.capability_runner import CapabilityRunner
from mr1.event_log import EventLog, SystemEvent
from mr1.messages import MessageStore
from mr1.mrn_loop import MRnStepRunner
from mr1.scoped_agents import PersistentAgentStore
from mr1.scheduler import Scheduler
from mr1.kazi_runner import MockRunner, RunStatus
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


def _ts(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()


def _action(action: str, **extra) -> str:
    payload = {
        "action": action,
        "reason": "test reason",
        "next_status": extra.pop("next_status", "idle" if action == "idle" else "working"),
        "workflow_request": extra.pop("workflow_request", None),
        "workflow_context": extra.pop("workflow_context", None),
        "workflow_id": extra.pop("workflow_id", None),
        "report": extra.pop("report", None),
        "message_kind": extra.pop("message_kind", None),
        "message_subject": extra.pop("message_subject", None),
        "message_body": extra.pop("message_body", None),
        "to_agent_id": extra.pop("to_agent_id", None),
        "parent_request": extra.pop("parent_request", None),
    }
    payload.update(extra)
    return json.dumps(payload)


class FakeReasoner:
    def __init__(self, *responses: str):
        self._responses = list(responses)

    def __call__(self, agent, system_prompt: str, prompt: str) -> str:
        if not self._responses:
            raise AssertionError("no reasoner responses configured")
        return self._responses.pop(0)


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
def event_log(tmp_path):
    return EventLog(tmp_path / "events")


def test_append_and_trace_are_deterministic(event_log: EventLog):
    first = event_log.emit(
        event_type="workflow_created",
        actor_id="cli",
        actor_type="user",
        target_id="wf-1",
        target_type="workflow",
        status="pending",
        summary="created",
        workflow_id="wf-1",
        timestamp=_ts("2026-04-29T12:00:00.100000+00:00"),
    )
    duplicate = event_log.emit(
        event_type="workflow_created",
        actor_id="cli",
        actor_type="user",
        target_id="wf-1",
        target_type="workflow",
        status="pending",
        summary="created",
        workflow_id="wf-1",
        timestamp=_ts("2026-04-29T12:00:00.100000+00:00"),
    )
    second = event_log.emit(
        event_type="workflow_started",
        actor_id="scheduler",
        actor_type="scheduler",
        target_id="wf-1",
        target_type="workflow",
        status="running",
        summary="started",
        workflow_id="wf-1",
        correlation_id=first.correlation_id,
        timestamp=_ts("2026-04-29T11:59:59.000000+00:00"),
    )

    assert first.event_id == duplicate.event_id
    assert duplicate.event_index == 1
    trace = event_log.trace_by_correlation(first.correlation_id or "")
    assert [item.event_index for item in trace] == [1, 2]
    assert [item.event_type for item in trace] == ["workflow_created", "workflow_started"]


def test_parent_event_must_exist(event_log: EventLog):
    with pytest.raises(ValueError, match="parent event not found"):
        event_log.append_event(SystemEvent(
            event_id="evt-test",
            event_index=0,
            event_version=1,
            timestamp=_ts("2026-04-29T12:00:00+00:00"),
            event_type="message_read",
            event_kind="communication",
            actor_id="ag-1",
            actor_type="mrn",
            target_id="msg-1",
            target_type="message",
            status="read",
            severity="INFO",
            summary="read",
            parent_event_id="evt-missing",
            metadata={},
        ))


def test_invalid_kind_and_severity_are_rejected():
    with pytest.raises(ValueError, match="invalid event_kind"):
        SystemEvent(
            event_id="evt-test",
            event_index=0,
            event_version=1,
            timestamp=_ts("2026-04-29T12:00:00+00:00"),
            event_type="message_read",
            event_kind="bad",
            actor_id="ag-1",
            actor_type="mrn",
            target_id="msg-1",
            target_type="message",
            status="read",
            severity="INFO",
            summary="read",
            metadata={},
        )
    with pytest.raises(ValueError, match="invalid severity"):
        SystemEvent(
            event_id="evt-test",
            event_index=0,
            event_version=1,
            timestamp=_ts("2026-04-29T12:00:00+00:00"),
            event_type="message_read",
            event_kind="communication",
            actor_id="ag-1",
            actor_type="mrn",
            target_id="msg-1",
            target_type="message",
            status="read",
            severity="BAD",
            summary="read",
            metadata={},
        )


def test_capability_block_approval_and_consumption_emit_events(tmp_path, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child")
    allowed_dir = tmp_path / "allowed"
    denied_dir = tmp_path / "denied"
    allowed_dir.mkdir()
    denied_dir.mkdir()
    target = denied_dir / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    child.scope_roots = [str(allowed_dir)]
    agent_store.save_agent(child)

    runner = CapabilityRunner(
        scoped_agent_store=agent_store,
        workspace_root=tmp_path,
    )

    first = runner.run_capability(
        "read_file",
        {"path": str(target)},
        child.agent_id,
        step_id=f"{child.agent_id}:1",
    )
    assert first.status == "requires_approval"

    approval = runner._approval_store.apply_decision(
        first.approval_request_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=first.approval_request_id,
            decision="approved",
            decided_by=root.agent_id,
            reason="approve",
            timestamp=datetime.now(timezone.utc).timestamp(),
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )
    assert approval.status == "approved"

    second = runner.run_capability(
        "read_file",
        {"path": str(target)},
        child.agent_id,
        step_id=f"{child.agent_id}:1",
    )
    assert second.status == "succeeded"

    event_log = EventLog(tmp_path / "events")
    event_types = [event.event_type for event in event_log.list_events()]
    assert "capability_requested" in event_types
    assert "capability_blocked" in event_types
    assert "approval_requested" in event_types
    assert "approval_approved" in event_types
    assert "approval_consumed" in event_types


def test_scope_and_message_events_emit(tmp_path, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child")
    scope_dir = tmp_path / "scope"
    scope_dir.mkdir()

    agent_store.grant_scope(root.agent_id, child.agent_id, scope_dir, reason="allow")
    agent_store.revoke_scope(root.agent_id, child.agent_id, scope_dir)
    message = message_store.create_message(
        from_agent_id=root.agent_id,
        to_agent_id=child.agent_id,
        kind="status",
        subject="hello",
        body="body",
    )
    message_store.mark_read(message.message_id, actor_id=child.agent_id)

    event_types = [event.event_type for event in EventLog(tmp_path / "events").list_events()]
    assert "agent_created" in event_types
    assert "agent_scope_granted" in event_types
    assert "agent_scope_revoked" in event_types
    assert "message_sent" in event_types
    assert "message_read" in event_types


def test_workflow_lifecycle_events_emit(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    scheduler = Scheduler(
        workflow_store,
        MockRunner(),
        auto_tick=False,
        scoped_agent_store=agent_store,
    )
    spec = {
        "title": "Timeline workflow",
        "tasks": [
            {
                "label": "a",
                "title": "A",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "x",
            }
        ],
    }
    wf_id = scheduler.submit_workflow(spec, Provenance(type="agent", id=root.agent_id))
    scheduler.tick()
    wf = workflow_store.load_workflow(wf_id)
    task = wf.task_by_label("a")
    scheduler._runner.complete(task.task_id, RunStatus.SUCCEEDED, summary="ok")
    scheduler.tick()
    scheduler.shutdown()

    event_types = [event.event_type for event in EventLog(workflow_store.root.parent / "events").list_events()]
    assert "workflow_created" in event_types
    assert "workflow_started" in event_types
    assert "workflow_task_started" in event_types
    assert "workflow_task_completed" in event_types
    assert "workflow_completed" in event_types


def test_mrn_step_and_report_events_emit(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_action("write_report", report="## Findings", next_status="reporting")),
    )

    result = runner.step(child.agent_id)
    assert result.report_path is not None

    event_types = [event.event_type for event in EventLog(workflow_store.root.parent / "events").list_events()]
    assert "mrn_step_started" in event_types
    assert "mrn_step_completed" in event_types
    assert "mrn_reported" in event_types
