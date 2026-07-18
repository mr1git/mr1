"""Tests for persistent scoped-agent lifecycle and hierarchy rules."""

from __future__ import annotations

import json

import pytest

from mr1.worker_runner import MockRunner
from mr1.scoped_agents import (
    AgentScopeError,
    AgentRecord,
    AgentStore,
    agent_display_status,
    build_assignment_packet,
    is_agent_live,
    is_agent_terminal,
    new_agent_id,
    render_assignment_mission,
)
from mr1.scheduler import Scheduler, WorkflowSpecError, submit_spec_to_disk
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


SPEC = {
    "title": "Owned workflow",
    "tasks": [
        {
            "label": "a",
            "title": "Task A",
            "task_kind": "agent",
            "agent_type": "worker",
            "prompt": "do it",
        }
    ],
}


@pytest.fixture
def agent_store(tmp_path):
    return AgentStore(root=tmp_path / "agents")


@pytest.fixture
def workflow_store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


class TestRootBootstrap:
    def test_root_bootstrap_persists(self, agent_store):
        root = agent_store.ensure_root_agent()

        assert root.role == "orchestrator"
        assert root.mr_level == 1
        assert root.lifecycle == "standing"
        assert root.mission is None
        assert root.mode == "manual"
        assert root.run_status == "idle"
        assert root.current_iteration == 0
        assert root.security_clearance == 0.99
        assert agent_store.root_agent_id_path.exists()
        assert agent_store.agent_path(root.agent_id).exists()
        assert agent_store.memory_path(root.agent_id).exists()
        assert agent_store.logs_dir(root.agent_id).exists()
        assert agent_store.report_dir(root.agent_id).exists()

        reloaded = AgentStore(root=agent_store.root)
        same_root = reloaded.ensure_root_agent()

        assert same_root.agent_id == root.agent_id
        assert same_root.title == "MR1"


class TestHierarchy:
    def test_child_creation_persists_and_sets_tree_level(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "research")

        assert child.parent_agent_id == root.agent_id
        assert child.mr_level == root.mr_level + 1
        assert child.security_clearance == root.security_clearance
        assert agent_store.agent_path(child.agent_id).exists()
        assert agent_store.memory_path(child.agent_id).exists()
        assert agent_store.logs_dir(child.agent_id).exists()
        assert agent_store.report_dir(child.agent_id).exists()

        reloaded = agent_store.load_agent(child.agent_id)
        assert reloaded is not None
        assert reloaded.to_dict() == child.to_dict()

    def test_child_clearance_cannot_exceed_parent(self, agent_store):
        root = agent_store.ensure_root_agent()
        parent = agent_store.create_child_agent(root.agent_id, "parent", security_clearance=0.6)

        with pytest.raises(ValueError, match="security_clearance must be between 0.0 and 1.0"):
            agent_store.create_child_agent(root.agent_id, "research", security_clearance=1.1)

        with pytest.raises(ValueError, match="child security_clearance cannot exceed parent.security_clearance"):
            agent_store.create_child_agent(parent.agent_id, "research", security_clearance=0.7)

        child = agent_store.create_child_agent(parent.agent_id, "research", security_clearance=0.5)

        assert child.security_clearance == 0.5

    def test_child_title_rejects_reserved_all(self, agent_store):
        root = agent_store.ensure_root_agent()

        with pytest.raises(ValueError, match="agent title 'all' is reserved"):
            agent_store.create_child_agent(root.agent_id, "ALL")

    @pytest.mark.parametrize("duplicate_title", ["Alpha", "alpha"])
    def test_child_title_rejects_duplicate_case_insensitively(self, agent_store, duplicate_title):
        root = agent_store.ensure_root_agent()
        agent_store.create_child_agent(root.agent_id, "Alpha")

        with pytest.raises(ValueError, match="agent title already exists: Alpha"):
            agent_store.create_child_agent(root.agent_id, duplicate_title)

    def test_save_agent_rejects_new_duplicate_title_case_insensitively(self, agent_store):
        root = agent_store.ensure_root_agent()
        agent_store.create_child_agent(root.agent_id, "Alpha")
        duplicate = AgentRecord(
            agent_id=new_agent_id(),
            role="orchestrator",
            title="alpha",
            mr_level=root.mr_level + 1,
            parent_agent_id=root.agent_id,
        )

        with pytest.raises(ValueError, match="agent title already exists: Alpha"):
            agent_store.save_agent(duplicate)

    def test_child_title_allows_distinct_titles(self, agent_store):
        root = agent_store.ensure_root_agent()
        first = agent_store.create_child_agent(root.agent_id, "RepoResearcher")
        second = agent_store.create_child_agent(root.agent_id, "PaperResearcher")

        assert first.title == "RepoResearcher"
        assert second.title == "PaperResearcher"

    @pytest.mark.parametrize("title", ["Title\x00Null", "Title\nNull", "Title\tNull", "Title\x7fNull"])
    def test_child_title_rejects_control_characters(self, agent_store, title):
        root = agent_store.ensure_root_agent()

        with pytest.raises(ValueError, match="agent title must not contain control characters"):
            agent_store.create_child_agent(root.agent_id, title)

    def test_structured_assignment_packet_has_required_fields_and_nested_recursion(self, agent_store):
        root = agent_store.ensure_root_agent()
        parent = agent_store.create_child_agent(root.agent_id, "MR2", security_clearance=0.6)
        parent_packet = build_assignment_packet(
            root,
            parent.title,
            "Own repository inspection and report issues.",
            {
                "assigned_clearance": parent.security_clearance,
                "agents": [root.agent_id, {"agent_id": "ag-ignore", "name": "ignored"}],
                "messages": [{"message_id": "msg-1", "body": "ignored"}],
                "workflows": [{"workflow_id": "wf-1", "title": "ignored"}],
            },
        )
        parent = agent_store.assign_mission(
            root.agent_id,
            parent.agent_id,
            render_assignment_mission(parent_packet) or "",
            assignment_packet=parent_packet,
        )
        child = agent_store.create_child_agent(parent.agent_id, "MR3", security_clearance=0.5)
        child_request = "Investigate failing workflows and synthesize findings." + (" detail" * 200)
        child_packet = build_assignment_packet(
            parent,
            child.title,
            child_request,
            {
                "assigned_clearance": child.security_clearance,
                "agents": [
                    parent.agent_id,
                    {"agent_id": child.agent_id, "full": {"ignored": True}},
                ],
                "messages": ["msg-2", {"message_id": "msg-3", "body": {"ignored": True}}],
                "workflows": ["wf-2", {"workflow_id": "wf-3", "tasks": [{"ignored": True}]}],
            },
        )
        child = agent_store.assign_mission(
            parent.agent_id,
            child.agent_id,
            render_assignment_mission(child_packet) or "",
            assignment_packet=child_packet,
        )
        reloaded = agent_store.require_agent(child.agent_id)

        assert reloaded.assignment_packet is not None
        assert set(reloaded.assignment_packet) == {
            "parent_agent_id",
            "parent_level",
            "child_level",
            "assigned_clearance",
            "child_title",
            "full_parent_request",
            "delegated_subtask",
            "reason_for_delegation",
            "mission",
            "responsibility",
            "scope",
            "constraints",
            "success_criteria",
            "expected_first_action",
            "relevant_context",
            "escalation_rules",
            "notes",
        }
        assert reloaded.assignment_packet["parent_agent_id"] == parent.agent_id
        assert reloaded.assignment_packet["parent_level"] == parent.mr_level
        assert reloaded.assignment_packet["child_level"] == child.mr_level
        assert reloaded.assignment_packet["assigned_clearance"] == child.security_clearance
        assert reloaded.assignment_packet["full_parent_request"] == child_request
        assert reloaded.assignment_packet["relevant_context"]["agents"] == [parent.agent_id, child.agent_id]
        assert reloaded.assignment_packet["relevant_context"]["messages"] == ["msg-2", "msg-3"]
        assert reloaded.assignment_packet["relevant_context"]["workflows"] == ["wf-2", "wf-3"]
        assert reloaded.parent_request == reloaded.assignment_packet["full_parent_request"]
        assert reloaded.mission is not None
        assert "MR3, an orchestrator agent owning a scoped assignment" in reloaded.mission

    def test_visibility_is_self_and_descendants_only(self, agent_store):
        root = agent_store.ensure_root_agent()
        left = agent_store.create_child_agent(root.agent_id, "left")
        left_child = agent_store.create_child_agent(left.agent_id, "left-child")
        right = agent_store.create_child_agent(root.agent_id, "right")

        visible_left = {agent.agent_id for agent in agent_store.list_visible_agents(left.agent_id)}
        visible_root = {agent.agent_id for agent in agent_store.list_visible_agents(root.agent_id)}

        assert visible_left == {left.agent_id, left_child.agent_id}
        assert visible_root == {root.agent_id, left.agent_id, left_child.agent_id, right.agent_id}

        with pytest.raises(AgentScopeError, match="access denied: agent not in scope"):
            agent_store.get_visible_agent(left.agent_id, right.agent_id)

    def test_kill_rules_allow_self_and_descendants_only(self, agent_store):
        root = agent_store.ensure_root_agent()
        left = agent_store.create_child_agent(root.agent_id, "left")
        left_child = agent_store.create_child_agent(left.agent_id, "left-child")
        right = agent_store.create_child_agent(root.agent_id, "right")

        killed_child = agent_store.terminate_agent(left.agent_id, left_child.agent_id)
        killed_self = agent_store.terminate_agent(left.agent_id, left.agent_id)

        assert killed_child.status == "terminated"
        assert killed_child.run_status == "terminated"
        assert killed_self.status == "terminated"
        assert killed_self.run_status == "terminated"

        with pytest.raises(AgentScopeError, match="access denied: agent not in scope"):
            agent_store.terminate_agent(left.agent_id, right.agent_id)
        with pytest.raises(AgentScopeError, match="access denied: agent not in scope"):
            agent_store.terminate_agent(left.agent_id, root.agent_id)

    def test_root_cannot_be_terminated(self, agent_store):
        root = agent_store.ensure_root_agent()

        with pytest.raises(ValueError, match="cannot terminate root agent"):
            agent_store.terminate_agent(root.agent_id, root.agent_id)


class TestTerminationEffects:
    def test_kill_prevents_new_workflows_but_existing_workflows_remain_accessible(
        self,
        agent_store,
        workflow_store,
    ):
        root = agent_store.ensure_root_agent()
        parent = agent_store.create_child_agent(root.agent_id, "parent")
        child = agent_store.create_child_agent(parent.agent_id, "child")

        workflow_id = submit_spec_to_disk(
            SPEC,
            Provenance(type="user", id="cli"),
            workflow_store,
            owner_agent_id=child.agent_id,
            scoped_agent_store=agent_store,
        )
        workflow = workflow_store.load_workflow(workflow_id)
        workflow = agent_store.normalize_workflow_ownership(workflow)

        agent_store.terminate_agent(parent.agent_id, child.agent_id)

        scheduler = Scheduler(
            workflow_store,
            MockRunner(),
            auto_tick=False,
            scoped_agent_store=agent_store,
        )
        with pytest.raises(WorkflowSpecError, match=f"agent is terminated: {child.agent_id}"):
            scheduler.submit_workflow(
                SPEC,
                Provenance(type="user", id="cli"),
                owner_agent_id=child.agent_id,
            )

        assert agent_store.can_agent_access_workflow(parent.agent_id, workflow) is True
        assert agent_store.can_agent_access_workflow(root.agent_id, workflow) is True


class TestLifecycleHelpers:
    def test_live_and_terminal_helpers_handle_legacy_conflicts(self):
        active_idle = AgentRecord(
            agent_id="ag-idle",
            role="orchestrator",
            title="Idle",
            mr_level=2,
            parent_agent_id="ag-root",
            status="active",
            run_status="idle",
        )
        active_waiting = AgentRecord(
            agent_id="ag-live",
            role="orchestrator",
            title="Live",
            mr_level=2,
            parent_agent_id="ag-root",
            status="active",
            run_status="waiting",
        )
        active_working = AgentRecord(
            agent_id="ag-working",
            role="orchestrator",
            title="Working",
            mr_level=2,
            parent_agent_id="ag-root",
            status="active",
            run_status="working",
        )
        legacy_terminated = AgentRecord(
            agent_id="ag-legacy-term",
            role="orchestrator",
            title="Legacy",
            mr_level=2,
            parent_agent_id="ag-root",
            status="active",
            run_status="terminated",
        )
        legacy_completed = AgentRecord(
            agent_id="ag-legacy-complete",
            role="orchestrator",
            title="LegacyComplete",
            mr_level=2,
            parent_agent_id="ag-root",
            status="active",
            run_status="completed",
        )
        terminated = AgentRecord(
            agent_id="ag-dead",
            role="orchestrator",
            title="Dead",
            mr_level=2,
            parent_agent_id="ag-root",
            status="terminated",
            run_status="idle",
        )

        assert is_agent_live(active_idle) is True
        assert is_agent_live(active_waiting) is True
        assert is_agent_live(active_working) is True
        assert is_agent_terminal(active_waiting) is False
        assert is_agent_live(terminated) is False
        assert is_agent_terminal(terminated) is True
        assert is_agent_live(legacy_terminated) is False
        assert is_agent_terminal(legacy_terminated) is True
        assert is_agent_live(legacy_completed) is False
        assert is_agent_terminal(legacy_completed) is True
        assert agent_display_status(legacy_terminated) == "terminated"

    def test_root_bootstrap_revives_legacy_terminal_root(self, agent_store):
        root = agent_store.ensure_root_agent()
        root.status = "active"
        root.run_status = "terminated"
        agent_store.save_agent(root)

        revived = agent_store.ensure_root_agent()

        assert revived.status == "active"
        assert revived.run_status == "idle"
        assert is_agent_live(revived) is True


class TestTitleIndex:
    """N-4: agent-title uniqueness should use an O(1) index, not an O(n) full scan."""

    def test_index_file_created_on_first_save(self, agent_store):
        agent_store.ensure_root_agent()
        assert agent_store._title_index_path.exists()

    def test_index_contains_all_active_titles(self, agent_store):
        root = agent_store.ensure_root_agent()
        agent_store.create_child_agent(root.agent_id, "Alpha")
        agent_store.create_child_agent(root.agent_id, "Beta")

        index_path = agent_store._title_index_path
        import json
        index = json.loads(index_path.read_text())
        assert "mr1" in index            # root agent title
        assert "alpha" in index
        assert "beta" in index

    def test_conflict_check_does_not_scan_all_files(self, agent_store, monkeypatch):
        """After the index is built, creating an agent should not glob/read all ag-*.json."""
        root = agent_store.ensure_root_agent()
        for i in range(20):
            agent_store.create_child_agent(root.agent_id, f"agent-{i}")

        read_paths: list = []
        original_read = agent_store._read_agent_file
        def tracking_read(path):
            if path.name.startswith("ag-") and path.suffix == ".json":
                read_paths.append(path)
            return original_read(path)
        monkeypatch.setattr(agent_store, "_read_agent_file", tracking_read)

        read_paths.clear()
        agent_store.create_child_agent(root.agent_id, "UniqueNewTitle")

        # Index path read is not tracked (only ag-*.json are). At most one
        # agent file should be read (conflict verification), never all 21+.
        assert len(read_paths) <= 1, (
            f"Full scan detected: {len(read_paths)} ag-*.json reads for 1 create"
        )

    def test_index_persists_across_store_instances(self, tmp_path):
        store_a = AgentStore(root=tmp_path / "agents")
        root = store_a.ensure_root_agent()
        store_a.create_child_agent(root.agent_id, "Persistent")

        # Fresh store instance — should load existing index.
        store_b = AgentStore(root=tmp_path / "agents")
        root_b = store_b.ensure_root_agent()
        with pytest.raises(ValueError, match="agent title already exists"):
            store_b.create_child_agent(root_b.agent_id, "persistent")

    def test_terminated_agent_permanently_reserves_title(self, agent_store):
        """Terminated agents remain in the index; their title cannot be reused."""
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "Doomed")
        agent_store.terminate_agent(root.agent_id, child.agent_id)

        with pytest.raises(ValueError, match="agent title already exists"):
            agent_store.create_child_agent(root.agent_id, "Doomed")

    def test_index_rebuilds_from_agent_files_if_missing(self, agent_store):
        """If the index file is deleted, the next operation rebuilds it from ag-*.json."""
        root = agent_store.ensure_root_agent()
        agent_store.create_child_agent(root.agent_id, "Rebuild")

        agent_store._title_index_path.unlink()

        # Conflict check should trigger rebuild and still block duplicate.
        with pytest.raises(ValueError, match="agent title already exists"):
            agent_store.create_child_agent(root.agent_id, "rebuild")

        # Index should exist again after rebuild.
        assert agent_store._title_index_path.exists()

    def test_large_agent_count_stays_bounded(self, agent_store):
        """Creating N agents should only ever read O(1) ag-*.json files per creation."""
        n = 50
        root = agent_store.ensure_root_agent()
        for i in range(n):
            agent_store.create_child_agent(root.agent_id, f"worker-{i}")

        read_counts: list[int] = []
        original_read = agent_store._read_agent_file
        for i in range(5):
            reads: list = []
            original_read_local = agent_store._read_agent_file
            def tracking(path, _reads=reads, _orig=original_read_local):
                if path.name.startswith("ag-") and path.suffix == ".json":
                    _reads.append(1)
                return _orig(path)
            agent_store._read_agent_file = tracking
            try:
                agent_store.create_child_agent(root.agent_id, f"extra-{i}")
            finally:
                agent_store._read_agent_file = original_read
            read_counts.append(len(reads))

        # Each creation should read at most 1 agent file (conflict verification), not n.
        assert max(read_counts) <= 1, (
            f"O(n) scan detected: max reads = {max(read_counts)}, n = {n}"
        )
