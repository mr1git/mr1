from __future__ import annotations

import json
from pathlib import Path

import pytest

from mr1 import workflow_cli
from mr1.capability_runner import CapabilityRunner
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def message_store(tmp_path, agent_store):
    return MessageStore(root=tmp_path / "messages", scoped_agent_store=agent_store)


def test_timeline_recent_and_show(store, agent_store, message_store, capsys):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child")
    message = message_store.create_message(
        from_agent_id=root.agent_id,
        to_agent_id=child.agent_id,
        kind="status",
        subject="hello",
        body="body",
    )

    rc = workflow_cli.main(
        ["timeline", "recent", "--limit", "5"],
        store=store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "message_sent" in out

    rc = workflow_cli.main(
        ["timeline", "show", "evt-does-not-exist"],
        store=store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 2
    capsys.readouterr()

    from mr1.event_log import EventLog

    event = EventLog(store.root.parent / "events").filter_events(message_id=message.message_id)[0]
    rc = workflow_cli.main(
        ["timeline", "show", event.event_id],
        store=store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert event.event_id in out
    assert "message_sent" in out


def test_timeline_trace_blocked_approvals_and_agent_views(tmp_path, store, agent_store, message_store, capsys):
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
    result = runner.run_capability(
        "read_file",
        {"path": str(target)},
        child.agent_id,
        step_id=f"{child.agent_id}:1",
    )
    assert result.approval_request_id is not None

    from mr1.event_log import EventLog

    event_log = EventLog(store.root.parent / "events")
    blocked_event = event_log.filter_events(
        approval_request_id=result.approval_request_id,
        event_type="capability_blocked",
    )[0]

    rc = workflow_cli.main(
        ["timeline", "trace", blocked_event.correlation_id],
        store=store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "capability_requested" in out
    assert "capability_blocked" in out

    rc = workflow_cli.main(
        ["timeline", "blocked"],
        store=store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "capability_blocked" in out or "approval_requested" in out

    rc = workflow_cli.main(
        ["timeline", "approvals", "--json"],
        store=store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(item["event_type"] == "approval_requested" for item in payload)

    rc = workflow_cli.main(
        ["timeline", "agent", child.agent_id],
        store=store,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert child.agent_id in out


def test_timeline_workflow_scope_filtering(store, agent_store, message_store, capsys):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    spec = {
        "title": "Scoped workflow",
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
    workflow_id = workflow_cli.submit_spec_to_disk(
        spec,
        created_by=workflow_cli.Provenance(type="user", id="cli"),
        store=store,
        owner_agent_id=right.agent_id,
        caller_agent_id=root.agent_id,
        scoped_agent_store=agent_store,
    )

    rc = workflow_cli.main(
        ["timeline", "workflow", workflow_id],
        store=store,
        caller_agent_id=left.agent_id,
        scoped_agent_store=agent_store,
        message_store=message_store,
    )
    assert rc == 2
    assert "access denied" in capsys.readouterr().err
