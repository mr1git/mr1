"""Tests for bounded MRn multi-step runs."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from mr1.messages import MessageStore
from mr1.mrn_loop import MRnStepResult
from mr1.mrn_run import MRnRunPolicy, MRnRunRunner
from mr1.scoped_agents import AgentScopeError, PersistentAgentStore
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


def _step_result(
    agent_id: str,
    iteration: int,
    action: str = "idle",
    status_after: str = "idle",
    **extra,
) -> MRnStepResult:
    payload = {
        "agent_id": agent_id,
        "iteration": iteration,
        "action": action,
        "status_before": extra.pop("status_before", "idle"),
        "status_after": status_after,
        "reason": extra.pop("reason", "test"),
        "message": extra.pop("message", "step"),
        "workflow_id": extra.pop("workflow_id", None),
        "report_path": extra.pop("report_path", None),
        "message_id": extra.pop("message_id", None),
        "parent_request": extra.pop("parent_request", None),
        "error": extra.pop("error", None),
        "created_workflow_id": extra.pop("created_workflow_id", None),
        "created_workflow_status": extra.pop("created_workflow_status", None),
        "created_parent_message_id": extra.pop("created_parent_message_id", None),
        "message_to_agent_id": extra.pop("message_to_agent_id", None),
        "confirmation_required": extra.pop("confirmation_required", False),
        "workflow_submitted": extra.pop("workflow_submitted", False),
    }
    payload.update(extra)
    return MRnStepResult(**payload)


class StubStepRunner:
    def __init__(self, agent_store: PersistentAgentStore, results: list[MRnStepResult]):
        self._agent_store = agent_store
        self._results = list(results)
        self.calls: list[tuple[str, str | None, str | None]] = []

    def step(
        self,
        agent_id: str,
        caller_agent_id: str | None = None,
        run_id: str | None = None,
    ) -> MRnStepResult:
        self.calls.append((agent_id, caller_agent_id, run_id))
        if not self._results:
            raise AssertionError("no step results configured")
        result = self._results.pop(0)
        agent = self._agent_store.require_agent(agent_id)
        agent.run_status = result.status_after
        agent.current_iteration = result.iteration
        self._agent_store.save_agent(agent)
        return result


def _child_agent(agent_store: PersistentAgentStore, *, mission: str = "Investigate"):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    agent_store.assign_mission(root.agent_id, child.agent_id, mission)
    return root, child


def test_run_advances_multiple_steps_until_max_steps(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    step_runner = StubStepRunner(agent_store, [
        _step_result(child.agent_id, 1, status_after="working"),
        _step_result(child.agent_id, 2, status_after="working"),
        _step_result(child.agent_id, 3, status_after="working"),
    ])
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=step_runner,
    )

    result = runner.run(child.agent_id, MRnRunPolicy(max_steps=3), caller_agent_id=root.agent_id)

    assert result.steps_completed == 3
    assert result.stopped_reason == "max_steps"
    assert result.status == "completed"
    assert len(step_runner.calls) == 3


def test_run_stops_on_waiting(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(child.agent_id, 1, status_after="waiting"),
        ]),
    )

    result = runner.run(child.agent_id, MRnRunPolicy(), caller_agent_id=root.agent_id)

    assert result.stopped_reason == "waiting"
    assert result.status == "stopped"


def test_run_stops_on_blocked(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(child.agent_id, 1, action="invalid", status_after="blocked", error="bad"),
        ]),
    )

    result = runner.run(child.agent_id, MRnRunPolicy(), caller_agent_id=root.agent_id)

    assert result.stopped_reason == "blocked"


def test_run_stops_on_parent_message(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(
                child.agent_id,
                1,
                action="send_message",
                status_after="working",
                message_id="msg-1",
                created_parent_message_id="msg-1",
                message_to_agent_id=root.agent_id,
            ),
        ]),
    )

    result = runner.run(child.agent_id, MRnRunPolicy(), caller_agent_id=root.agent_id)

    assert result.stopped_reason == "parent_message"
    assert result.messages_created == 1


def test_run_stops_on_workflow_running(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(
                child.agent_id,
                1,
                action="create_workflow",
                status_after="working",
                workflow_id="wf-1",
                created_workflow_id="wf-1",
                created_workflow_status="pending",
                workflow_submitted=True,
            ),
        ]),
    )

    result = runner.run(child.agent_id, MRnRunPolicy(), caller_agent_id=root.agent_id)

    assert result.stopped_reason == "workflow_running"
    assert result.workflows_created == 1


def test_run_stops_on_workflow_creation_limit(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(
                child.agent_id,
                1,
                action="create_workflow",
                status_after="working",
                workflow_id="wf-1",
                created_workflow_id="wf-1",
                created_workflow_status="succeeded",
                workflow_submitted=True,
            ),
        ]),
    )

    result = runner.run(
        child.agent_id,
        MRnRunPolicy(max_workflows_created=0, stop_on_workflow_running=False),
        caller_agent_id=root.agent_id,
    )

    assert result.stopped_reason == "workflow_limit"


def test_allowed_actions_blocks_disallowed_action(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(child.agent_id, 1, action="write_report", status_after="reporting"),
        ]),
    )

    result = runner.run(
        child.agent_id,
        MRnRunPolicy(allowed_actions=["idle"]),
        caller_agent_id=root.agent_id,
    )

    assert result.stopped_reason == "disallowed_action"


def test_max_runtime_stops_run(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(child.agent_id, 1, status_after="working"),
        ]),
    )

    with patch("mr1.mrn_run.time.monotonic", side_effect=[0.0, 0.0, 2.0]):
        result = runner.run(
            child.agent_id,
            MRnRunPolicy(max_runtime_s=1, stop_on_waiting=False),
            caller_agent_id=root.agent_id,
        )

    assert result.stopped_reason == "runtime_limit"


def test_run_log_written_with_policy_and_step_summaries(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(
                child.agent_id,
                1,
                status_after="working",
                prompt_artifact_path="/tmp/prompt-artifact.json",
            ),
        ]),
    )

    result = runner.run(child.agent_id, MRnRunPolicy(max_steps=1), caller_agent_id=root.agent_id)

    run_log = agent_store.run_log_path(child.agent_id, result.run_id)
    summary_log = agent_store.run_summary_log_path(child.agent_id)
    payload = json.loads(run_log.read_text(encoding="utf-8"))
    summary = json.loads(summary_log.read_text(encoding="utf-8").strip().splitlines()[-1])
    reloaded = agent_store.require_agent(child.agent_id)

    assert payload["policy"]["max_steps"] == 1
    assert payload["steps"][0]["iteration"] == 1
    assert payload["steps"][0]["prompt_artifact_path"] == "/tmp/prompt-artifact.json"
    assert summary["run_id"] == result.run_id
    assert reloaded.last_run["run_id"] == result.run_id


def test_confirmation_required_stops_run(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(
                child.agent_id,
                1,
                action="create_workflow",
                status_after="reporting",
                confirmation_required=True,
                message_id="msg-1",
                created_parent_message_id="msg-1",
            ),
        ]),
    )

    result = runner.run(child.agent_id, MRnRunPolicy(), caller_agent_id=root.agent_id)

    assert result.stopped_reason == "confirmation_required"


def test_no_mission_assigned_rejection(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )

    with pytest.raises(ValueError, match=f"no mission assigned: {child.agent_id}"):
        runner.run(child.agent_id, MRnRunPolicy(), caller_agent_id=root.agent_id)


def test_terminated_agent_rejection(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    agent_store.terminate_agent(root.agent_id, child.agent_id)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )

    with pytest.raises(ValueError, match=f"agent terminated: {child.agent_id}"):
        runner.run(child.agent_id, MRnRunPolicy(), caller_agent_id=root.agent_id)


def test_scope_denial(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    agent_store.assign_mission(root.agent_id, right.agent_id, "Investigate")
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )

    with pytest.raises(AgentScopeError, match="access denied: agent not in scope"):
        runner.run(right.agent_id, MRnRunPolicy(), caller_agent_id=left.agent_id)


def test_invalid_policy_validation(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )

    with pytest.raises(ValueError, match="unknown actions rejected deterministically"):
        runner.run(
            child.agent_id,
            MRnRunPolicy(allowed_actions=["not_real"]),
            caller_agent_id=root.agent_id,
        )


def test_policy_confirmation_flag_is_passed_to_default_step_runner(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    captured = {}

    def _recording_runner(**kwargs):
        captured.update(kwargs)
        return StubStepRunner(agent_store, [
            _step_result(child.agent_id, 1, status_after="working"),
        ])

    with patch("mr1.mrn_run.MRnStepRunner", side_effect=_recording_runner):
        runner = MRnRunRunner(
            workflow_store=workflow_store,
            scoped_agent_store=agent_store,
            message_store=message_store,
        )
        runner.run(
            child.agent_id,
            MRnRunPolicy(require_confirmation_for_workflows=False, max_steps=1),
            caller_agent_id=root.agent_id,
        )

    assert captured["require_confirmation_for_workflows"] is False


# ---------------------------------------------------------------------------
# call_capability in MRnRunPolicy (Phase 14)
# ---------------------------------------------------------------------------


def test_allowed_actions_accepts_call_capability(workflow_store, agent_store, message_store):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(child.agent_id, 1, action="call_capability", status_after="working"),
        ]),
    )

    result = runner.run(
        child.agent_id,
        MRnRunPolicy(allowed_actions=["call_capability", "idle"], max_steps=1),
        caller_agent_id=root.agent_id,
    )

    assert result.stopped_reason != "disallowed_action"


def test_allowed_actions_rejects_call_capability_when_omitted(
    workflow_store, agent_store, message_store
):
    root, child = _child_agent(agent_store)
    runner = MRnRunRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        step_runner=StubStepRunner(agent_store, [
            _step_result(child.agent_id, 1, action="call_capability", status_after="working"),
        ]),
    )

    result = runner.run(
        child.agent_id,
        MRnRunPolicy(allowed_actions=["idle"]),
        caller_agent_id=root.agent_id,
    )

    assert result.stopped_reason == "disallowed_action"


def test_call_capability_is_valid_allowed_action_name():
    from mr1.mrn_loop import ALLOWED_MRN_ACTIONS
    from mr1.mrn_run import MRnRunPolicy

    policy = MRnRunPolicy(allowed_actions=["call_capability"])

    assert "call_capability" in ALLOWED_MRN_ACTIONS
    assert policy.allowed_actions == ["call_capability"]
