"""Tests for bounded persistent MRn step execution."""

from __future__ import annotations

import json

import pytest

from mr1.messages import MessageStore
from mr1.mrn_loop import MRnStepRunner
from mr1.scoped_agents import AgentScopeError, PersistentAgentStore
from mr1.scheduler import submit_spec_to_disk
from mr1.workflow_cli import _format_agent
from mr1.workflow_compiler import WorkflowCompilerClient
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


SPEC = {
    "title": "Owned workflow",
    "tasks": [
        {
            "label": "a",
            "title": "Task A",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "do it",
        }
    ],
}

COMPILE_SPEC = {
    "title": "Generated workflow",
    "tasks": [
        {
            "label": "read_notes",
            "title": "Read notes",
            "task_kind": "tool",
            "tool_type": "read_file",
            "tool_config": {"path": "notes.txt"},
        }
    ],
}


class FakeReasoner:
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, agent, system_prompt: str, prompt: str) -> str:
        self.calls.append((agent.agent_id, system_prompt, prompt))
        if not self._responses:
            raise AssertionError("no reasoner responses configured")
        return self._responses.pop(0)


class FakeCompiler:
    def __init__(self, *responses: str):
        self._responses = list(responses)

    def __call__(self, system_prompt: str, prompt: str) -> str:
        if not self._responses:
            raise AssertionError("no compiler responses configured")
        return self._responses.pop(0)


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


def _envelope(spec: dict) -> str:
    return json.dumps({
        "preview": "Generated preview",
        "spec": spec,
        "assumptions": [],
        "risks": [],
        "needs_confirmation": False,
        "confidence": "high",
    })


@pytest.fixture
def workflow_store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def message_store(tmp_path, agent_store):
    return MessageStore(root=tmp_path / "messages", scoped_agent_store=agent_store)


def test_assign_mission_persists_fields(agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    child.run_status = "waiting"
    child.current_iteration = 4
    child.last_step_at = "2026-01-01T00:00:00+00:00"
    child.last_action = {"action": "ask_parent"}
    child.parent_request = "old request"
    agent_store.save_agent(child)

    updated = agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate the repo")
    reloaded = agent_store.require_agent(child.agent_id)

    assert updated.mission == "Investigate the repo"
    assert reloaded.run_status == "idle"
    assert reloaded.current_iteration == 0
    assert reloaded.last_step_at is None
    assert reloaded.last_action is None
    assert reloaded.parent_request is None


def test_step_rejects_terminated_agent(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    agent_store.terminate_agent(root.agent_id, child.agent_id)
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(_action("idle")),
    )

    with pytest.raises(ValueError, match=f"agent terminated: {child.agent_id}"):
        runner.step(child.agent_id)


def test_step_rejects_out_of_scope_caller(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    agent_store.assign_mission(root.agent_id, right.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(_action("idle")),
    )

    with pytest.raises(AgentScopeError, match="access denied: agent not in scope"):
        runner.step(right.agent_id, caller_agent_id=left.agent_id)


def test_no_mission_assigned_error(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(_action("idle")),
    )

    with pytest.raises(ValueError, match=f"no mission assigned: {child.agent_id}"):
        runner.step(child.agent_id)


def test_valid_idle_action_updates_iteration_and_log(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(_action("idle")),
    )

    result = runner.step(child.agent_id)
    reloaded = agent_store.require_agent(child.agent_id)
    logs = agent_store.step_log_path(child.agent_id).read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(logs[-1])

    assert result.action == "idle"
    assert result.iteration == 1
    assert reloaded.current_iteration == 1
    assert reloaded.run_status == "idle"
    assert record["action"] == "idle"
    assert record["iteration"] == 1


def test_valid_write_report_writes_report_file_and_updates_status(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_action("write_report", report="## Findings\nReady.", next_status="reporting")),
    )

    result = runner.step(child.agent_id)
    reloaded = agent_store.require_agent(child.agent_id)
    content = agent_store.list_reports(child.agent_id)[0].read_text(encoding="utf-8")
    inbox = message_store.list_inbox(root.agent_id)

    assert result.report_path is not None
    assert result.message_id is not None
    assert reloaded.run_status == "reporting"
    assert "Investigate" in content
    assert "## Findings" in content
    assert inbox[0].kind == "report"
    assert "Report path:" in inbox[0].body


def test_ask_parent_stores_parent_request_and_waiting_status(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_action("ask_parent", parent_request="Need product clarification")),
    )

    result = runner.step(child.agent_id)
    reloaded = agent_store.require_agent(child.agent_id)
    inbox = message_store.list_inbox(root.agent_id)

    assert result.parent_request == "Need product clarification"
    assert result.message_id is not None
    assert reloaded.parent_request == "Need product clarification"
    assert reloaded.run_status == "waiting"
    assert inbox[0].kind == "question"
    assert inbox[0].subject == f"Parent request from {child.title}"


def test_send_message_action_creates_parent_message_and_persists_message_id(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_action(
            "send_message",
            message_kind="status",
            message_subject="Checkpoint",
            message_body="Need review.",
            next_status="waiting",
        )),
    )

    result = runner.step(child.agent_id)
    reloaded = agent_store.require_agent(child.agent_id)
    record = json.loads(agent_store.step_log_path(child.agent_id).read_text(encoding="utf-8").strip().splitlines()[-1])
    message = message_store.list_inbox(root.agent_id)[0]

    assert result.message_id == message.message_id
    assert reloaded.last_action["message_id"] == message.message_id
    assert record["message_id"] == message.message_id
    assert message.subject == "Checkpoint"
    assert message.kind == "status"


def test_send_message_action_can_target_descendant(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    parent = agent_store.create_child_agent(root.agent_id, "parent")
    child = agent_store.create_child_agent(parent.agent_id, "child")
    agent_store.assign_mission(root.agent_id, parent.agent_id, "Coordinate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_action(
            "send_message",
            message_kind="request",
            message_subject="Do task",
            message_body="Please handle this.",
            to_agent_id=child.agent_id,
        )),
    )

    result = runner.step(parent.agent_id)

    assert message_store.list_inbox(child.agent_id)[0].message_id == result.message_id


def test_send_message_action_to_sibling_blocks_agent(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    agent_store.assign_mission(root.agent_id, left.agent_id, "Coordinate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_action(
            "send_message",
            message_kind="request",
            message_subject="Nope",
            message_body="This should fail.",
            to_agent_id=right.agent_id,
        )),
    )

    result = runner.step(left.agent_id)

    assert result.status_after == "blocked"
    assert result.error == "access denied: recipient not in agent scope"
    assert message_store.list_inbox(right.agent_id) == []


def test_inspect_workflow_respects_scope_and_logs_summary(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    workflow_id = submit_spec_to_disk(
        SPEC,
        Provenance(type="user", id="cli"),
        workflow_store,
        owner_agent_id=child.agent_id,
        scoped_agent_store=agent_store,
    )
    agent_store.assign_mission(root.agent_id, child.agent_id, "Inspect workflows")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(_action("inspect_workflow", workflow_id=workflow_id)),
    )

    result = runner.step(child.agent_id)
    record = json.loads(agent_store.step_log_path(child.agent_id).read_text(encoding="utf-8").strip().splitlines()[-1])

    assert result.workflow_id == workflow_id
    assert record["workflow_summary"]["workflow_id"] == workflow_id
    assert record["workflow_summary"]["status"] == "pending"


def test_create_workflow_stamps_owner_agent_id_as_mrn(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Create a workflow")
    client = WorkflowCompilerClient(
        compiler=FakeCompiler(_envelope(COMPILE_SPEC)),
        scoped_agent_store=agent_store,
    )
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        workflow_compiler_client=client,
        reasoner=FakeReasoner(_action("create_workflow", workflow_request="Read notes")),
    )

    result = runner.step(child.agent_id)
    workflow = workflow_store.load_workflow(result.workflow_id)

    assert workflow is not None
    assert workflow.owner_agent_id == child.agent_id
    assert result.created_workflow_id == result.workflow_id
    assert result.created_workflow_status == "pending"
    assert result.workflow_submitted is True


def test_create_workflow_uses_default_compiler_client_when_only_compiler_is_provided(
    workflow_store,
    agent_store,
):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Create a workflow")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        workflow_compiler=FakeCompiler(_envelope(COMPILE_SPEC)),
        reasoner=FakeReasoner(_action("create_workflow", workflow_request="Read notes")),
    )

    result = runner.step(child.agent_id)
    workflow = workflow_store.load_workflow(result.workflow_id)

    assert workflow is not None
    assert workflow.owner_agent_id == child.agent_id


def test_create_workflow_forced_confirmation_writes_report_and_parent_message(
    workflow_store,
    agent_store,
    message_store,
):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Create a workflow")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        workflow_compiler=FakeCompiler(_envelope(COMPILE_SPEC)),
        reasoner=FakeReasoner(_action("create_workflow", workflow_request="Read notes")),
        require_confirmation_for_workflows=True,
    )

    result = runner.step(child.agent_id)
    inbox = message_store.list_inbox(root.agent_id)

    assert result.workflow_id is None
    assert result.confirmation_required is True
    assert result.workflow_submitted is False
    assert result.report_path is not None
    assert result.created_parent_message_id == result.message_id
    assert inbox[0].subject == f"Workflow confirmation needed from {child.title}"


def test_invalid_action_triggers_one_correction_pass(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    reasoner = FakeReasoner("not json", _action("idle"))
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=reasoner,
    )

    result = runner.step(child.agent_id)

    assert result.action == "idle"
    assert len(reasoner.calls) == 2


def test_step_prompt_includes_bounded_message_context(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    for index in range(6):
        message_store.create_message(
            from_agent_id=root.agent_id,
            to_agent_id=child.agent_id,
            kind="request",
            subject=f"Inbox {index}",
            body="body",
        )
    for index in range(6):
        message_store.create_message(
            from_agent_id=child.agent_id,
            to_agent_id=root.agent_id,
            kind="status",
            subject=f"Outbox {index}",
            body="body",
        )
    reasoner = FakeReasoner(_action("idle"))
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=reasoner,
    )

    runner.step(child.agent_id)
    payload = json.loads(reasoner.calls[0][2].split("Scoped context:\n", 1)[1])

    assert len(payload["unread_messages"]) == 5
    assert len(payload["recent_sent_messages"]) == 5
    assert payload["parent_messages"]
    assert payload["unread_messages"][0]["subject"] == "Inbox 5"


def test_second_invalid_action_blocks_agent(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner("not json", "still not json"),
    )

    result = runner.step(child.agent_id)
    reloaded = agent_store.require_agent(child.agent_id)
    record = json.loads(agent_store.step_log_path(child.agent_id).read_text(encoding="utf-8").strip().splitlines()[-1])

    assert result.status_after == "blocked"
    assert reloaded.run_status == "blocked"
    assert "invalid mrn action:" in (result.error or "")
    assert "invalid mrn action:" in (record["error"] or "")


def test_agent_inspection_shows_mission_status_iteration_and_message_counts(agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    child.mission = "Investigate repo"
    child.run_status = "waiting"
    child.current_iteration = 3
    child.last_run = {
        "run_id": "run-1",
        "stopped_reason": "waiting",
        "step_count": 2,
        "finished_at": "2026-01-01T00:00:00+00:00",
    }
    agent_store.save_agent(child)
    message_store.create_message(
        from_agent_id=root.agent_id,
        to_agent_id=child.agent_id,
        kind="request",
        subject="Need update",
        body="Status?",
    )
    message_store.create_message(
        from_agent_id=child.agent_id,
        to_agent_id=root.agent_id,
        kind="status",
        subject="Update",
        body="Working.",
    )

    formatted = _format_agent(
        child,
        reports=agent_store.list_reports(child.agent_id),
        message_store=message_store,
    )

    assert "mission:      Investigate repo" in formatted
    assert "run_status:   waiting" in formatted
    assert "iteration:    3" in formatted
    assert "latest_run:   run-1" in formatted
    assert "run_reason:   waiting" in formatted
    assert "unread_inbox: 1" in formatted
    assert "Need update" in formatted
    assert "Update" in formatted


# ---------------------------------------------------------------------------
# call_capability action (Phase 14)
# ---------------------------------------------------------------------------


from mr1.capability_runner import CapabilityResult, CapabilityRunner


class FakeCapabilityRunner:
    """Stub CapabilityRunner for mrn_loop tests."""

    def __init__(
        self,
        *,
        result: CapabilityResult | None = None,
        raises: str | None = None,
    ):
        self._result = result or CapabilityResult(
            status="succeeded",
            output={"value": "ok"},
            error=None,
            duration_ms=5,
            capability="read_file",
            decision={"status": "allowed"},
        )
        self._raises = raises
        self.calls: list[tuple[str, dict, str, str | None]] = []

    def run_capability(
        self,
        name: str,
        config: dict,
        caller_agent_id: str,
        mode: str = "direct",
        *,
        step_id: str | None = None,
    ) -> CapabilityResult:
        self.calls.append((name, config, caller_agent_id, step_id))
        if self._raises:
            raise ValueError(self._raises)
        return CapabilityResult(
            status=self._result.status,
            output=self._result.output,
            error=self._result.error,
            duration_ms=self._result.duration_ms,
            capability=name,
            decision=dict(self._result.decision),
        )


def _call_cap_action(capability: str, **extra) -> str:
    payload = {
        "action": "call_capability",
        "reason": "test",
        "next_status": "working",
        "capability": capability,
        "config": extra.pop("config", {}),
        "store_as": extra.pop("store_as", None),
        "workflow_request": None,
        "workflow_context": None,
        "workflow_id": None,
        "report": None,
        "message_kind": None,
        "message_subject": None,
        "message_body": None,
        "to_agent_id": None,
        "parent_request": None,
    }
    payload.update(extra)
    return json.dumps(payload)


def test_call_capability_parse_and_validate(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Check files")
    fake_runner = FakeCapabilityRunner()
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(_call_cap_action("read_file", config={"path": "/tmp/x"})),
        capability_runner=fake_runner,
    )

    result = runner.step(child.agent_id)

    assert result.action == "call_capability"
    assert result.capability_result is not None
    assert result.capability_result["status"] == "succeeded"
    assert len(fake_runner.calls) == 1
    name, config, caller, step_id = fake_runner.calls[0]
    assert name == "read_file"
    assert config == {"path": "/tmp/x"}
    assert caller == child.agent_id
    assert step_id is not None


def test_call_capability_store_as_persists_output_to_step_context(
    workflow_store, agent_store
):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Check files")
    fake_runner = FakeCapabilityRunner(
        result=CapabilityResult(
            status="succeeded",
            output={"exists": True},
            error=None,
            duration_ms=1,
            capability="file_exists",
            decision={"status": "allowed"},
        )
    )
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(
            _call_cap_action("file_exists", store_as="check_result")
        ),
        capability_runner=fake_runner,
    )

    result = runner.step(child.agent_id)
    reloaded = agent_store.require_agent(child.agent_id)

    assert result.stored_as == "check_result"
    assert reloaded.step_context["check_result"] == {"exists": True}


def test_call_capability_store_as_skipped_on_failure(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Check files")
    fake_runner = FakeCapabilityRunner(
        result=CapabilityResult(
            status="failed",
            output={},
            error="path does not exist",
            duration_ms=1,
            capability="read_file",
            decision={"status": "allowed"},
        )
    )
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(
            _call_cap_action("read_file", config={"path": "/missing"}, store_as="data")
        ),
        capability_runner=fake_runner,
    )

    result = runner.step(child.agent_id)
    reloaded = agent_store.require_agent(child.agent_id)

    assert result.stored_as is None
    assert "data" not in reloaded.step_context


def test_step_context_included_in_scoped_prompt(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Check files")
    child = agent_store.require_agent(child.agent_id)
    child.step_context["key1"] = {"value": 42}
    agent_store.save_agent(child)

    captured_prompts: list[str] = []

    def capturing_reasoner(agent, system_prompt, prompt):
        captured_prompts.append(prompt)
        return _action("idle")

    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=capturing_reasoner,
    )
    runner.step(child.agent_id)

    assert captured_prompts
    context_text = captured_prompts[0]
    assert "step_context" in context_text
    assert "key1" in context_text

def test_call_capability_preflight_error_produces_blocked_step(
    workflow_store, agent_store
):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Check files")
    fake_runner = FakeCapabilityRunner(
        result=CapabilityResult(
            status="denied",
            output={"status": "denied", "reason": "capability_not_allowed_in_direct_mode"},
            error=None,
            duration_ms=0,
            capability="write_file",
            decision={"status": "denied"},
        )
    )
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(_call_cap_action("write_file")),
        capability_runner=fake_runner,
    )

    result = runner.step(child.agent_id)

    assert result.action == "call_capability"
    assert result.status_after == "working"
    assert result.capability_result["status"] == "denied"
