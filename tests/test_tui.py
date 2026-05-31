from __future__ import annotations

from pathlib import Path

import pytest

from mr1.capability_policy import CapabilityApprovalRequest, CapabilityApprovalStore
from mr1.event_log import EventLog
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgent, PersistentAgentStore
from mr1.tui.app import main as tui_main
from mr1.tui.colors import color_for_agent, depth_color
from mr1.tui.data import (
    RuntimeDataSource,
    RuntimeSnapshot,
    build_agent_tree,
    live_focus_agent_id,
    visible_agent_ids,
)
from mr1.tui.layout import format_agent_detail, format_event_detail
from mr1.tui.navigation import (
    coerce_selected_agent_id,
    first_child_agent_id,
    next_sibling_agent_id,
    parent_agent_id,
    previous_sibling_agent_id,
)
from mr1.workflow_store import WorkflowStore


def _agent(
    agent_id: str,
    *,
    title: str,
    tree_level: int,
    parent_agent_id: str | None,
    created_at: str,
    status: str = "active",
    run_status: str = "idle",
) -> PersistentAgent:
    return PersistentAgent(
        agent_id=agent_id,
        agent_type="mr1" if tree_level == 1 else "mrn",
        title=title,
        tree_level=tree_level,
        parent_agent_id=parent_agent_id,
        created_at=created_at,
        status=status,
        run_status=run_status,
    )


def test_build_agent_tree_keeps_root_and_orders_siblings():
    root = _agent("ag-root", title="MR1", tree_level=1, parent_agent_id=None, created_at="2026-05-01T00:00:00+00:00")
    b = _agent("ag-b", title="B", tree_level=2, parent_agent_id="ag-root", created_at="2026-05-01T00:00:02+00:00")
    a = _agent("ag-a", title="A", tree_level=2, parent_agent_id="ag-root", created_at="2026-05-01T00:00:01+00:00")
    a2 = _agent("ag-a2", title="A2", tree_level=2, parent_agent_id="ag-root", created_at="2026-05-01T00:00:01+00:00")

    tree = build_agent_tree([b, root, a2, a], root_agent_id="ag-root")

    assert tree.root_agent_id == "ag-root"
    assert tree.children_by_parent["ag-root"] == ("ag-a", "ag-a2", "ag-b")


def test_navigation_left_right_up_down_uses_visible_tree():
    root = _agent("ag-root", title="MR1", tree_level=1, parent_agent_id=None, created_at="2026-05-01T00:00:00+00:00")
    left = _agent("ag-left", title="Left", tree_level=2, parent_agent_id="ag-root", created_at="2026-05-01T00:00:01+00:00")
    middle = _agent("ag-middle", title="Middle", tree_level=2, parent_agent_id="ag-root", created_at="2026-05-01T00:00:02+00:00")
    right = _agent("ag-right", title="Right", tree_level=2, parent_agent_id="ag-root", created_at="2026-05-01T00:00:03+00:00")
    child = _agent("ag-child", title="Child", tree_level=3, parent_agent_id="ag-middle", created_at="2026-05-01T00:00:04+00:00")
    tree = build_agent_tree([root, left, middle, right, child], root_agent_id="ag-root")

    assert previous_sibling_agent_id(tree, "ag-middle", show_dead=True) == "ag-left"
    assert next_sibling_agent_id(tree, "ag-middle", show_dead=True) == "ag-right"
    assert parent_agent_id(tree, "ag-child", show_dead=True) == "ag-middle"
    assert first_child_agent_id(tree, "ag-middle", show_dead=True) == "ag-child"


def test_show_dead_hides_dead_leaf_but_preserves_dead_ancestor_for_live_descendant():
    root = _agent("ag-root", title="MR1", tree_level=1, parent_agent_id=None, created_at="2026-05-01T00:00:00+00:00")
    terminated_branch = _agent(
        "ag-dead-parent",
        title="Dead Parent",
        tree_level=2,
        parent_agent_id="ag-root",
        created_at="2026-05-01T00:00:01+00:00",
        status="terminated",
    )
    live_child = _agent(
        "ag-live-child",
        title="Live Child",
        tree_level=3,
        parent_agent_id="ag-dead-parent",
        created_at="2026-05-01T00:00:02+00:00",
        run_status="running",
    )
    dead_leaf = _agent(
        "ag-dead-leaf",
        title="Dead Leaf",
        tree_level=2,
        parent_agent_id="ag-root",
        created_at="2026-05-01T00:00:03+00:00",
        status="terminated",
    )
    tree = build_agent_tree([root, terminated_branch, live_child, dead_leaf], root_agent_id="ag-root")

    visible = visible_agent_ids(tree, show_dead=False)

    assert "ag-root" in visible
    assert "ag-dead-parent" in visible
    assert "ag-live-child" in visible
    assert "ag-dead-leaf" not in visible
    assert coerce_selected_agent_id(tree, "ag-dead-leaf", show_dead=False) == "ag-root"


def test_color_mapping_uses_depth_palette_and_dims_terminal_status():
    assert depth_color(1) != depth_color(2)
    active = color_for_agent(depth=2, status="active", run_status="running")
    dead = color_for_agent(depth=2, status="terminated", run_status="idle")

    assert active.dim is False
    assert active.bold is True
    assert dead.dim is True
    assert dead.color == depth_color(2)


def test_live_focus_prefers_deepest_newest_visible_live_agent():
    root = _agent("ag-root", title="MR1", tree_level=1, parent_agent_id=None, created_at="2026-05-01T00:00:00+00:00")
    older = _agent("ag-older", title="Older", tree_level=3, parent_agent_id="ag-root", created_at="2026-05-01T00:00:01+00:00", run_status="running")
    newer = _agent("ag-newer", title="Newer", tree_level=3, parent_agent_id="ag-root", created_at="2026-05-01T00:00:02+00:00", run_status="waiting")
    done = _agent("ag-done", title="Done", tree_level=4, parent_agent_id="ag-newer", created_at="2026-05-01T00:00:03+00:00", status="terminated")
    tree = build_agent_tree([root, older, newer, done], root_agent_id="ag-root")

    assert live_focus_agent_id(tree, show_dead=True) == "ag-newer"


def test_live_count_and_visibility_exclude_legacy_active_terminated_records():
    root = _agent("ag-root", title="MR1", tree_level=1, parent_agent_id=None, created_at="2026-05-01T00:00:00+00:00")
    live = _agent("ag-live", title="Live", tree_level=2, parent_agent_id="ag-root", created_at="2026-05-01T00:00:01+00:00", run_status="waiting")
    legacy = _agent("ag-legacy", title="Legacy", tree_level=2, parent_agent_id="ag-root", created_at="2026-05-01T00:00:02+00:00", status="active", run_status="terminated")
    tree = build_agent_tree([root, live, legacy], root_agent_id="ag-root")
    snapshot = RuntimeSnapshot(tree=tree, events=(), refreshed_at="2026-05-01T00:00:03+00:00")

    assert snapshot.live_count == 2
    assert snapshot.done_count == 1
    assert visible_agent_ids(tree, show_dead=False) == {"ag-root", "ag-live"}


def test_runtime_data_source_returns_recent_bounded_events(tmp_path):
    store = WorkflowStore(root=tmp_path / "workflows")
    agent_store = PersistentAgentStore(root=tmp_path / "agents")
    message_store = MessageStore(root=tmp_path / "messages", scoped_agent_store=agent_store)
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child")
    _ = message_store
    event_log = EventLog(tmp_path / "events")
    event_log.emit(
        event_type="agent_created",
        actor_id=root.agent_id,
        actor_type=root.agent_type,
        target_id=child.agent_id,
        target_type="agent",
        status="created",
        summary="child created",
        timestamp="2026-05-01T00:00:01+00:00",
    )
    event_log.emit(
        event_type="message_sent",
        actor_id=root.agent_id,
        actor_type=root.agent_type,
        target_id=child.agent_id,
        target_type="agent",
        status="sent",
        summary="message sent",
        timestamp="2026-05-01T00:00:02+00:00",
    )
    event_log.emit(
        event_type="message_read",
        actor_id=child.agent_id,
        actor_type=child.agent_type,
        target_id=root.agent_id,
        target_type="agent",
        status="read",
        summary="message read",
        timestamp="2026-05-01T00:00:03+00:00",
    )

    data_source = RuntimeDataSource(store_root=store.root)
    snapshot = data_source.load_snapshot(event_limit=2)

    assert len(snapshot.events) == 2
    assert snapshot.events[0].event_type == "message_read"
    assert snapshot.events[1].event_type == "message_sent"


def test_detail_formatting_for_agent_and_event():
    agent_lines = format_agent_detail(
        {
            "title": "Research",
            "agent_id": "ag-123",
            "status": "active",
            "run_status": "waiting",
            "lifecycle_status": "active",
            "status_conflict": False,
            "tree_level": 2,
            "parent_agent_id": "ag-root",
            "security_clearance": 0.8,
            "owned_workflow_ids": ["wf-1"],
            "unread_inbox_count": 2,
            "latest_run_id": "run-1",
            "mission": "Investigate the runtime graph and summarize what changed.",
            "latest_inbox_messages": [{"message_id": "msg-1"}],
            "latest_outbox_messages": [{"message_id": "msg-2"}],
            "last_action": {"action": "ask_parent"},
        }
    )
    event_lines = format_event_detail(
        {
            "event": {
                "event_type": "message_sent",
                "event_id": "evt-1",
                "timestamp": "2026-05-01T00:00:01+00:00",
                "severity": "INFO",
                "status": "sent",
                "actor_id": "ag-root",
                "target_id": "ag-123",
                "workflow_id": "wf-1",
                "message_id": "msg-1",
                "approval_request_id": "cap_approval_1",
                "summary": "message sent to child",
            },
            "workflow": {"workflow_id": "wf-1", "title": "Demo", "status": "running"},
            "message": {"message_id": "msg-1", "subject": "Hello", "status": "unread"},
            "approval": {"approval_request_id": "cap_approval_1", "capability_name": "read_file", "status": "pending"},
            "payload_text": "{\n  \"kind\": \"status\"\n}",
        }
    )

    assert any("latest_message_ids: msg-1, msg-2" in line for line in agent_lines)
    assert any("lifecycle: active" in line for line in agent_lines)
    assert any("status_conflict: no" in line for line in agent_lines)
    assert any("workflow: wf-1 Demo [running]" in line for line in event_lines)
    assert any("approval: cap_approval_1 read_file [pending]" in line for line in event_lines)


def test_tui_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        tui_main(["--help"])
    assert exc.value.code == 0
