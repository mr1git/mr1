"""Tests for bounded persistent MRn step execution."""

from __future__ import annotations

import json

import pytest

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


def test_valid_write_report_writes_report_file_and_updates_status(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(_action("write_report", report="## Findings\nReady.", next_status="reporting")),
    )

    result = runner.step(child.agent_id)
    reloaded = agent_store.require_agent(child.agent_id)
    content = agent_store.list_reports(child.agent_id)[0].read_text(encoding="utf-8")

    assert result.report_path is not None
    assert reloaded.run_status == "reporting"
    assert "Investigate" in content
    assert "## Findings" in content


def test_ask_parent_stores_parent_request_and_waiting_status(workflow_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    runner = MRnStepRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        reasoner=FakeReasoner(_action("ask_parent", parent_request="Need product clarification")),
    )

    result = runner.step(child.agent_id)
    reloaded = agent_store.require_agent(child.agent_id)

    assert result.parent_request == "Need product clarification"
    assert reloaded.parent_request == "Need product clarification"
    assert reloaded.run_status == "waiting"


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


def test_agent_inspection_shows_mission_status_and_iteration(agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    child.mission = "Investigate repo"
    child.run_status = "waiting"
    child.current_iteration = 3
    agent_store.save_agent(child)

    formatted = _format_agent(child, reports=agent_store.list_reports(child.agent_id))

    assert "mission:      Investigate repo" in formatted
    assert "run_status:   waiting" in formatted
    assert "iteration:    3" in formatted
