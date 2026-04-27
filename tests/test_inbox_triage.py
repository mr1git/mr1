"""Tests for bounded inbox triage."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from mr1 import inbox_triage as inbox_triage_module
from mr1 import workflow_cli
from mr1.inbox_triage import InboxTriagePolicy, InboxTriageRunner
from mr1.kazi_runner import MockRunner
from mr1.messages import MessageStore
from mr1.mr1 import MR1, StateManager
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_compiler import WorkflowCompilerClient
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


def _build_mr1(tmp_path, workflow_store, agent_store, message_store, *, workflow_compiler=None):
    mr1 = MR1(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        workflow_runner=MockRunner(),
        workflow_auto_tick=False,
        workflow_compiler=workflow_compiler or (lambda *_: "{}"),
    )
    mr1._state = StateManager(state_path=tmp_path / "mr1_state.json")
    return mr1


def _action(action_type: str, *, reason: str = "test reason", **extra) -> dict[str, object]:
    payload = {"type": action_type, "reason": reason}
    payload.update(extra)
    return payload


def _triage(actions: list[dict[str, object]], *, summary: str = "triage summary") -> str:
    return json.dumps({"actions": actions, "summary": summary})


class FakeReasoner:
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, prompt: str) -> str:
        self.calls.append((system_prompt, prompt))
        if not self._responses:
            raise AssertionError("no reasoner responses configured")
        return self._responses.pop(0)


class FakeCompiler:
    def __init__(self, response: str):
        self._response = response

    def __call__(self, system_prompt: str, prompt: str) -> str:
        return self._response


class PendingStateSink:
    def __init__(self):
        self.pending_workflow = None

    def set_pending_workflow(self, draft):
        self.pending_workflow = draft


@dataclass
class StubRunResult:
    run_id: str = "run-1"
    workflows_created: int = 0


class StubRunRunner:
    def __init__(self, result: StubRunResult):
        self.result = result
        self.calls: list[tuple[str, object, str | None]] = []

    def run(self, agent_id: str, policy, caller_agent_id: str | None = None):
        self.calls.append((agent_id, policy, caller_agent_id))
        return self.result


@dataclass
class StubStepResult:
    iteration: int = 1
    created_workflow_id: str | None = None


class StubStepRunner:
    def __init__(self, result: StubStepResult):
        self.result = result
        self.calls: list[tuple[str, str | None]] = []

    def step(self, agent_id: str, caller_agent_id: str | None = None):
        self.calls.append((agent_id, caller_agent_id))
        return self.result


def _compiler_envelope(*, needs_confirmation: bool = False) -> str:
    return json.dumps({
        "preview": "Read notes, then summarize them.",
        "spec": {
            "title": "Read and summarize",
            "tasks": [
                {
                    "label": "read_notes",
                    "title": "Read notes",
                    "task_kind": "tool",
                    "tool_type": "read_file",
                    "tool_config": {"path": "notes.txt"},
                }
            ],
        },
        "assumptions": ["notes.txt exists"],
        "risks": ["summary may be incomplete"],
        "needs_confirmation": needs_confirmation,
        "confidence": "medium",
    })


def _root_and_child(agent_store: PersistentAgentStore, title: str = "research"):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, title)
    return root, child


def _root_message(message_store: MessageStore, agent_store: PersistentAgentStore, *, subject: str = "Update", body: str = "Body"):
    root, child = _root_and_child(agent_store)
    message = message_store.create_message(
        from_agent_id=child.agent_id,
        to_agent_id=root.agent_id,
        kind="question",
        subject=subject,
        body=body,
    )
    return root, child, message


def test_no_messages_is_noop(workflow_store, agent_store, message_store):
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([])),
    )

    result = runner.run(InboxTriagePolicy())

    assert result.processed_messages == []
    assert result.actions_executed == []
    assert result.summary == "No unread inbox messages."
    assert result.counts["messages_read"] == 0


def test_mark_read_action(workflow_store, agent_store, message_store):
    root, child, message = _root_message(message_store, agent_store)
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([_action("mark_read", message_id=message.message_id)])),
    )

    result = runner.run(InboxTriagePolicy(auto_mark_read=False), caller_agent_id=root.agent_id)

    assert result.actions_executed[0]["type"] == "mark_read"
    assert result.counts["messages_read"] == 1
    assert message_store.get_message(message.message_id).status == "read"
    assert message_store.list_outbox(root.agent_id) == []
    assert child.agent_id == message.from_agent_id


def test_archive_action(workflow_store, agent_store, message_store):
    root, _, message = _root_message(message_store, agent_store)
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([_action("archive", message_id=message.message_id)])),
    )

    result = runner.run(InboxTriagePolicy(auto_mark_read=False), caller_agent_id=root.agent_id)

    assert result.counts["messages_archived"] == 1
    assert message_store.get_message(message.message_id).status == "archived"


def test_reply_message_creates_message(workflow_store, agent_store, message_store):
    root, child, message = _root_message(message_store, agent_store, subject="Need input")
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([
            _action(
                "reply_message",
                message_id=message.message_id,
                reply_body="Please continue.",
            )
        ])),
    )

    result = runner.run(InboxTriagePolicy(auto_mark_read=False), caller_agent_id=root.agent_id)
    sent_id = result.actions_executed[0]["created_message_id"]
    sent = message_store.get_message(sent_id)

    assert result.counts["messages_sent"] == 1
    assert sent is not None
    assert sent.subject == "Re: Need input"
    assert sent.kind == message.kind
    assert sent.to_agent_id == child.agent_id


def test_agent_run_invoked(workflow_store, agent_store, message_store):
    root, child, message = _root_message(message_store, agent_store)
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    run_runner = StubRunRunner(StubRunResult(run_id="run-123", workflows_created=1))
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([_action("agent_run", agent_id=child.agent_id)])),
        mrn_run_runner=run_runner,
    )

    result = runner.run(InboxTriagePolicy(auto_mark_read=False), caller_agent_id=root.agent_id)

    assert run_runner.calls[0][0] == child.agent_id
    assert run_runner.calls[0][1].max_steps == 2
    assert run_runner.calls[0][1].max_workflows_created == 1
    assert result.counts["agents_run"] == 1
    assert result.counts["workflows_created"] == 1
    assert result.actions_executed[0]["run_id"] == "run-123"
    assert message.message_id in result.processed_messages


def test_agent_step_invoked(workflow_store, agent_store, message_store):
    root, child, _ = _root_message(message_store, agent_store)
    agent_store.assign_mission(root.agent_id, child.agent_id, "Investigate")
    step_runner = StubStepRunner(StubStepResult(iteration=4, created_workflow_id="wf-1"))
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([_action("agent_step", agent_id=child.agent_id)])),
        mrn_step_runner=step_runner,
    )

    result = runner.run(InboxTriagePolicy(auto_mark_read=False), caller_agent_id=root.agent_id)

    assert step_runner.calls == [(child.agent_id, root.agent_id)]
    assert result.counts["agents_run"] == 1
    assert result.counts["workflows_created"] == 1
    assert result.actions_executed[0]["step_iteration"] == 4


def test_assign_mission_updates_agent(workflow_store, agent_store, message_store):
    root, child, _ = _root_message(message_store, agent_store)
    child.run_status = "waiting"
    child.current_iteration = 3
    child.last_step_at = "2026-01-01T00:00:00+00:00"
    child.last_action = {"action": "ask_parent"}
    child.parent_request = "Need clarification"
    agent_store.save_agent(child)
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([
            _action("assign_mission", agent_id=child.agent_id, mission="New mission")
        ])),
    )

    runner.run(InboxTriagePolicy(auto_mark_read=False), caller_agent_id=root.agent_id)
    updated = agent_store.require_agent(child.agent_id)

    assert updated.mission == "New mission"
    assert updated.run_status == "idle"
    assert updated.current_iteration == 0
    assert updated.last_action is None
    assert updated.parent_request is None


def test_create_workflow_submits(workflow_store, agent_store, message_store):
    root, _, _ = _root_message(message_store, agent_store)
    compiler_client = WorkflowCompilerClient(
        compiler=FakeCompiler(_compiler_envelope(needs_confirmation=False)),
        scoped_agent_store=agent_store,
    )
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([
            _action("create_workflow", workflow_request="Read notes and summarize them")
        ])),
        workflow_compiler_client=compiler_client,
    )

    result = runner.run(InboxTriagePolicy(auto_mark_read=False), caller_agent_id=root.agent_id)
    workflow_id = result.actions_executed[0]["created_workflow_id"]

    assert workflow_id is not None
    assert workflow_store.load_workflow(workflow_id) is not None
    assert result.counts["workflows_created"] == 1


def test_create_workflow_confirmation_persists_pending_draft(workflow_store, agent_store, message_store):
    root, _, _ = _root_message(message_store, agent_store)
    state_sink = PendingStateSink()
    compiler_client = WorkflowCompilerClient(
        compiler=FakeCompiler(_compiler_envelope(needs_confirmation=True)),
        scoped_agent_store=agent_store,
    )
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([
            _action("create_workflow", workflow_request="Read notes and summarize them")
        ])),
        workflow_compiler_client=compiler_client,
        pending_workflow_state=state_sink,
    )

    result = runner.run(InboxTriagePolicy(auto_mark_read=False), caller_agent_id=root.agent_id)

    assert result.actions_executed[0]["status"] == "confirmation_required"
    assert result.counts["workflows_created"] == 0
    assert state_sink.pending_workflow is not None
    assert state_sink.pending_workflow["spec"]["title"] == "Read and summarize"


def test_ask_user_has_no_side_effects(workflow_store, agent_store, message_store):
    root, _, message = _root_message(message_store, agent_store)
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([
            _action("ask_user", reason="Need clarification from the user")
        ])),
    )

    result = runner.run(
        InboxTriagePolicy(auto_mark_read=False, auto_archive=False),
        caller_agent_id=root.agent_id,
    )

    assert result.actions_executed[0]["status"] == "ask_user"
    assert result.actions_executed[0]["user_message"] == "Need clarification from the user"
    assert result.counts == {
        "messages_read": 0,
        "messages_archived": 0,
        "messages_sent": 0,
        "agents_run": 0,
        "workflows_created": 0,
    }
    assert message_store.get_message(message.message_id).status == "unread"


def test_action_limit_enforced_with_correction(workflow_store, agent_store, message_store):
    root, _, _ = _root_message(message_store, agent_store)
    reasoner = FakeReasoner(
        _triage([
            _action("idle"),
            _action("idle", reason="second"),
        ]),
        _triage([_action("idle")], summary="corrected"),
    )
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=reasoner,
    )

    result = runner.run(InboxTriagePolicy(max_actions=1, auto_mark_read=False), caller_agent_id=root.agent_id)

    assert len(reasoner.calls) == 2
    assert len(result.actions_executed) == 1
    assert result.summary == "corrected"


def test_message_limit_enforced(workflow_store, agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    first = message_store.create_message(
        from_agent_id=child.agent_id,
        to_agent_id=root.agent_id,
        kind="question",
        subject="First",
        body="one",
    )
    second = message_store.create_message(
        from_agent_id=child.agent_id,
        to_agent_id=root.agent_id,
        kind="question",
        subject="Second",
        body="two",
    )
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([_action("idle")])),
    )

    result = runner.run(InboxTriagePolicy(max_messages=1, auto_mark_read=False), caller_agent_id=root.agent_id)

    assert result.processed_messages == [second.message_id]
    assert first.message_id not in result.processed_messages


def test_disallowed_action_stops_after_prior_actions(workflow_store, agent_store, message_store):
    root, child, message = _root_message(message_store, agent_store)
    agent_store.assign_mission(root.agent_id, child.agent_id, "Old mission")
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner(_triage([
            _action("mark_read", message_id=message.message_id),
            _action("assign_mission", agent_id=child.agent_id, mission="New mission"),
        ])),
    )

    result = runner.run(
        InboxTriagePolicy(
            allowed_actions=["mark_read"],
            auto_mark_read=False,
            max_actions=2,
        ),
        caller_agent_id=root.agent_id,
    )

    assert result.actions_executed[0]["status"] == "executed"
    assert result.actions_executed[1]["status"] == "stopped"
    assert "Stopped on disallowed action" in result.summary
    assert agent_store.require_agent(child.agent_id).mission == "Old mission"


def test_invalid_action_correction_then_fail(workflow_store, agent_store, message_store):
    root, _, _ = _root_message(message_store, agent_store)
    runner = InboxTriageRunner(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        reasoner=FakeReasoner("not json", "still not json"),
    )

    with pytest.raises(ValueError, match="inbox triage failed:"):
        runner.run(InboxTriagePolicy(auto_mark_read=False), caller_agent_id=root.agent_id)


def test_cli_inbox_triage_formats_output_and_persists_pending_draft(
    workflow_store,
    agent_store,
    message_store,
    tmp_path,
    capsys,
    monkeypatch,
):
    root, _, _ = _root_message(message_store, agent_store)
    fake_reasoner = FakeReasoner(_triage([
        _action("create_workflow", workflow_request="Read notes and summarize them")
    ]))
    monkeypatch.setattr(inbox_triage_module, "run_inbox_triage_reasoner", fake_reasoner)

    from mr1 import mr1 as mr1_module

    class TestStateManager(StateManager):
        def __init__(self):
            super().__init__(state_path=tmp_path / "cli_state.json")

    monkeypatch.setattr(mr1_module, "StateManager", TestStateManager)

    rc = workflow_cli.main(
        ["inbox-triage", "--max-messages", "1", "--max-actions", "1"],
        store=workflow_store,
        caller_agent_id=root.agent_id,
        scoped_agent_store=agent_store,
        message_store=message_store,
        workflow_compiler=FakeCompiler(_compiler_envelope(needs_confirmation=True)),
    )

    assert rc == 0
    out = capsys.readouterr().out
    state = TestStateManager()
    assert "summary: triage summary" in out
    assert "confirmation_required" in out
    assert state.pending_workflow is not None


def test_builtin_inbox_triage_formats_output_and_persists_pending_draft(
    workflow_store,
    agent_store,
    message_store,
    tmp_path,
    monkeypatch,
):
    fake_reasoner = FakeReasoner(_triage([
        _action("create_workflow", workflow_request="Read notes and summarize them")
    ]))
    monkeypatch.setattr(inbox_triage_module, "run_inbox_triage_reasoner", fake_reasoner)
    mr1 = _build_mr1(
        tmp_path,
        workflow_store,
        agent_store,
        message_store,
        workflow_compiler=FakeCompiler(_compiler_envelope(needs_confirmation=True)),
    )
    _root_message(message_store, agent_store)

    output = mr1._handle_builtin("/inbox triage --max-messages 1 --max-actions 1")

    assert "summary: triage summary" in output
    assert "confirmation_required" in output
    assert mr1._state.pending_workflow is not None
