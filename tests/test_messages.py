"""Tests for agent messaging and inbox commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from mr1 import workflow_cli
from mr1.worker_runner import MockRunner
from mr1.messages import MessageStore
from mr1.mr1 import MR1, StateManager
from mr1.scoped_agents import AgentScopeError, AgentStore
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def workflow_store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return AgentStore(root=tmp_path / "agents")


@pytest.fixture
def message_store(tmp_path, agent_store):
    return MessageStore(root=tmp_path / "messages", scoped_agent_store=agent_store)


def _build_mr1(tmp_path, workflow_store, agent_store, message_store):
    mr1 = MR1(
        workflow_store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
        workflow_runner=MockRunner(),
        workflow_auto_tick=False,
        workflow_compiler=lambda *_: "{}",
    )
    mr1._state = StateManager(state_path=tmp_path / "mr1_state.json")
    return mr1


def test_message_creation_persists_and_lists_inbox_outbox(agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")

    created = message_store.create_message(
        from_agent_id=root.agent_id,
        to_agent_id=child.agent_id,
        kind="request",
        subject="Investigate",
        body="Please inspect the repo.",
    )
    reloaded = message_store.get_message(created.message_id)

    assert reloaded is not None
    assert reloaded.subject == "Investigate"
    assert message_store.list_inbox(child.agent_id)[0].message_id == created.message_id
    assert message_store.list_outbox(root.agent_id)[0].message_id == created.message_id


def test_message_creation_normalizes_supported_alias_kind(agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")

    created = message_store.create_message(
        from_agent_id=root.agent_id,
        to_agent_id=child.agent_id,
        kind="proposal",
        subject="Plan",
        body="Proceed with the implementation.",
    )

    assert created.kind == "request"
    assert message_store.get_message(created.message_id).kind == "request"


def test_mark_read_and_archive(agent_store, message_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    created = message_store.create_message(
        from_agent_id=root.agent_id,
        to_agent_id=child.agent_id,
        kind="request",
        subject="Investigate",
        body="Please inspect the repo.",
    )

    read_message = message_store.mark_read(created.message_id)
    archived_message = message_store.archive_message(created.message_id)

    assert read_message is not None
    assert read_message.status == "read"
    assert read_message.read_at is not None
    assert archived_message is not None
    assert archived_message.status == "archived"
    assert archived_message.archived_at is not None
    assert message_store.list_inbox(child.agent_id) == []
    assert message_store.list_inbox(child.agent_id, include_archived=True)[0].message_id == created.message_id


def test_root_access_all_and_mrn_access_is_self_only(agent_store, message_store):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    message = message_store.create_message(
        from_agent_id=left.agent_id,
        to_agent_id=root.agent_id,
        kind="status",
        subject="Update",
        body="Done.",
    )

    assert message_store.can_agent_access_message(root.agent_id, message) is True
    assert message_store.can_agent_access_message(left.agent_id, message) is True
    assert message_store.can_agent_access_message(right.agent_id, message) is False


def test_mrn_send_permissions_follow_scope(agent_store, message_store):
    root = agent_store.ensure_root_agent()
    parent = agent_store.create_child_agent(root.agent_id, "parent")
    child = agent_store.create_child_agent(parent.agent_id, "child")
    sibling = agent_store.create_child_agent(root.agent_id, "sibling")

    assert message_store.can_agent_send_message(child.agent_id, parent.agent_id) is True
    assert message_store.can_agent_send_message(parent.agent_id, child.agent_id) is True
    assert message_store.can_agent_send_message(child.agent_id, sibling.agent_id) is False
    assert message_store.can_agent_send_message(child.agent_id, "worker") is False
    assert message_store.can_agent_send_message("worker", child.agent_id) is False


def test_cli_inbox_message_read_archive_and_send(workflow_store, agent_store, message_store, capsys, tmp_path):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    created = message_store.create_message(
        from_agent_id=child.agent_id,
        to_agent_id=root.agent_id,
        kind="question",
        subject="Need input",
        body="Clarify scope.",
    )
    body_path = tmp_path / "body.txt"
    body_path.write_text("Please continue.", encoding="utf-8")

    rc = workflow_cli.main(
        ["inbox"],
        store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    assert created.message_id in capsys.readouterr().out

    rc = workflow_cli.main(
        ["message", created.message_id],
        store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    assert "Clarify scope." in capsys.readouterr().out

    rc = workflow_cli.main(
        ["message-read", created.message_id],
        store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == created.message_id
    assert message_store.get_message(created.message_id).status == "read"

    rc = workflow_cli.main(
        ["message-archive", created.message_id],
        store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    assert capsys.readouterr().out.strip() == created.message_id
    assert message_store.get_message(created.message_id).status == "archived"

    rc = workflow_cli.main(
        ["message-send", child.agent_id, "Follow up", str(body_path)],
        store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    sent_id = capsys.readouterr().out.strip()
    assert message_store.get_message(sent_id).subject == "Follow up"


def test_cli_message_scope_and_body_file_errors(workflow_store, agent_store, message_store, capsys, tmp_path):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    message = message_store.create_message(
        from_agent_id=right.agent_id,
        to_agent_id=root.agent_id,
        kind="status",
        subject="Update",
        body="Done.",
    )

    rc = workflow_cli.main(
        ["message", message.message_id],
        store=workflow_store,
        caller_agent_id=left.agent_id,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 2
    assert capsys.readouterr().err.strip() == "error: access denied: message not in agent scope"

    rc = workflow_cli.main(
        ["message-send", right.agent_id, "Nope", str(tmp_path / "missing.txt")],
        store=workflow_store,
        caller_agent_id=left.agent_id,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 2
    assert capsys.readouterr().err.strip() == f"error: message body file not found: {tmp_path / 'missing.txt'}"

    body_path = tmp_path / "body.txt"
    body_path.write_text("Hi", encoding="utf-8")
    rc = workflow_cli.main(
        ["message-send", right.agent_id, "Nope", str(body_path)],
        store=workflow_store,
        caller_agent_id=left.agent_id,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 2
    assert capsys.readouterr().err.strip() == "error: access denied: recipient not in agent scope"


def test_mr1_builtins_inbox_and_message(workflow_store, agent_store, message_store, tmp_path):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    message = message_store.create_message(
        from_agent_id=child.agent_id,
        to_agent_id=root.agent_id,
        kind="report",
        subject="Summary",
        body="All done.",
    )
    body_path = tmp_path / "body.txt"
    body_path.write_text("Please continue.", encoding="utf-8")
    mr1 = _build_mr1(tmp_path, workflow_store, agent_store, message_store)

    inbox_output = mr1._handle_builtin("/inbox")
    detail_output = mr1._handle_builtin(f"/message {message.message_id}")
    read_output = mr1._handle_builtin(f"/message read {message.message_id}")
    send_output = mr1._handle_builtin(f"/message send {child.agent_id} Follow-up {body_path}")

    assert message.message_id in inbox_output
    assert "All done." in detail_output
    assert read_output == message.message_id
    assert message_store.get_message(message.message_id).status == "read"
    assert send_output.startswith("msg-")


def test_cli_agent_detail_stays_compact_while_message_detail_is_full(workflow_store, agent_store, message_store, capsys):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    long_mission = "Investigate " + ("repo " * 80)
    child.mission = long_mission
    agent_store.save_agent(child)
    long_body = "Detailed body " + ("x" * 5000)
    message = message_store.create_message(
        from_agent_id=child.agent_id,
        to_agent_id=root.agent_id,
        kind="report",
        subject="Long report",
        body=long_body,
    )

    rc = workflow_cli.main(
        ["agent", child.agent_id],
        store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    agent_output = capsys.readouterr().out
    assert "mission:      " in agent_output
    assert long_mission not in agent_output
    assert "..." in agent_output

    rc = workflow_cli.main(
        ["message", message.message_id],
        store=workflow_store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    message_output = capsys.readouterr().out
    assert long_body in message_output


def test_require_message_denies_sibling_access(agent_store, message_store):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    message = message_store.create_message(
        from_agent_id=right.agent_id,
        to_agent_id=root.agent_id,
        kind="status",
        subject="Update",
        body="Done.",
    )

    with pytest.raises(AgentScopeError, match="access denied: message not in agent scope"):
        if not message_store.can_agent_access_message(left.agent_id, message):
            raise AgentScopeError("access denied: message not in agent scope")
