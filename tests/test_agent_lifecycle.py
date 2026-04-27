"""Tests for persistent scoped-agent lifecycle and hierarchy rules."""

from __future__ import annotations

import json

import pytest

from mr1.kazi_runner import MockRunner
from mr1.scoped_agents import AgentScopeError, PersistentAgentStore
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
            "agent_type": "kazi",
            "prompt": "do it",
        }
    ],
}


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def workflow_store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


class TestRootBootstrap:
    def test_root_bootstrap_persists(self, agent_store):
        root = agent_store.ensure_root_agent()

        assert root.agent_type == "mr1"
        assert root.tree_level == 1
        assert agent_store.root_agent_id_path.exists()
        assert agent_store.agent_path(root.agent_id).exists()

        reloaded = PersistentAgentStore(root=agent_store.root)
        same_root = reloaded.ensure_root_agent()

        assert same_root.agent_id == root.agent_id
        assert same_root.title == "MR1"


class TestHierarchy:
    def test_child_creation_persists_and_sets_tree_level(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "research")

        assert child.parent_agent_id == root.agent_id
        assert child.tree_level == root.tree_level + 1
        assert agent_store.agent_path(child.agent_id).exists()

        reloaded = agent_store.load_agent(child.agent_id)
        assert reloaded is not None
        assert reloaded.to_dict() == child.to_dict()

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
        assert killed_self.status == "terminated"

        with pytest.raises(AgentScopeError, match="access denied: agent not in scope"):
            agent_store.terminate_agent(left.agent_id, right.agent_id)
        with pytest.raises(AgentScopeError, match="access denied: agent not in scope"):
            agent_store.terminate_agent(left.agent_id, root.agent_id)


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
