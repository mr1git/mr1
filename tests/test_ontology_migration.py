"""Tests for the agent ontology (role / mr_level / lifecycle) refactor:
- mr_level derivation and validation
- role independence from mr_level
- lifecycle values
- naming (no MR-branded fallback titles)
- legacy (agent_type/tree_level) migration, including idempotency
"""

from __future__ import annotations

import json

import pytest

from mr1.doctor import run_doctor
from mr1.scoped_agents import (
    ALLOWED_LIFECYCLES,
    ALLOWED_ROLES,
    AgentRecord,
    AgentStore,
    actor_category,
    migrate_agent_store_ontology,
    migrate_legacy_agent_dict,
    new_agent_id,
)


@pytest.fixture
def agent_store(tmp_path):
    return AgentStore(root=tmp_path / "agents")


# --------------------------------------------------------------------------
# mr_level
# --------------------------------------------------------------------------

class TestMrLevel:
    def test_root_is_level_one(self, agent_store):
        root = agent_store.ensure_root_agent()
        assert root.mr_level == 1
        assert root.parent_agent_id is None

    def test_direct_child_is_level_two(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "child")
        assert child.mr_level == 2

    def test_grandchild_is_level_three(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "child")
        grandchild = agent_store.create_child_agent(child.agent_id, "grandchild")
        assert grandchild.mr_level == 3

    def test_siblings_share_the_same_level(self, agent_store):
        root = agent_store.ensure_root_agent()
        a = agent_store.create_child_agent(root.agent_id, "a")
        b = agent_store.create_child_agent(root.agent_id, "b")
        assert a.mr_level == b.mr_level == 2

    def test_restart_preserves_levels(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "child")
        grandchild = agent_store.create_child_agent(child.agent_id, "grandchild")

        reopened = AgentStore(root=agent_store.root)
        assert reopened.require_agent(root.agent_id).mr_level == 1
        assert reopened.require_agent(child.agent_id).mr_level == 2
        assert reopened.require_agent(grandchild.agent_id).mr_level == 3

    def test_root_level_must_be_one(self):
        with pytest.raises(ValueError, match="mr_level=1 .root. must not have a parent_agent_id"):
            AgentRecord(
                agent_id="ag-bad",
                role="orchestrator",
                title="bad root",
                mr_level=1,
                parent_agent_id="ag-someone",
            )

    def test_mr_level_below_one_rejected(self):
        with pytest.raises(ValueError, match="mr_level must be >= 1"):
            AgentRecord(
                agent_id="ag-bad",
                role="orchestrator",
                title="bad",
                mr_level=0,
                parent_agent_id=None,
            )

    def test_invalid_level_parent_relation_fails_loudly_via_doctor(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "child")
        # Corrupt the child's mr_level on disk directly (simulating hand-edited state).
        path = agent_store.agent_path(child.agent_id)
        data = json.loads(path.read_text())
        data["mr_level"] = 5
        path.write_text(json.dumps(data))

        report = run_doctor(agent_store.root.parent)
        agents_check = next(c for c in report.checks if c.check_id == "agents.registry")
        assert any("mr_level" in err for err in agents_check.details["errors"])

    def test_cycle_fails_via_doctor(self, agent_store):
        root = agent_store.ensure_root_agent()
        cyclic = AgentRecord(
            agent_id="ag-cycle-a",
            role="orchestrator",
            title="cycle-a",
            mr_level=2,
            parent_agent_id="ag-cycle-b",
        )
        cyclic_b = AgentRecord(
            agent_id="ag-cycle-b",
            role="orchestrator",
            title="cycle-b",
            mr_level=2,
            parent_agent_id="ag-cycle-a",
        )
        (agent_store.root / "ag-cycle-a.json").write_text(json.dumps(cyclic.to_dict()))
        (agent_store.root / "ag-cycle-b.json").write_text(json.dumps(cyclic_b.to_dict()))

        report = run_doctor(agent_store.root.parent)
        agents_check = next(c for c in report.checks if c.check_id == "agents.registry")
        assert any("cycle detected" in err for err in agents_check.details["errors"])


# --------------------------------------------------------------------------
# role
# --------------------------------------------------------------------------

class TestRole:
    def test_root_role_is_orchestrator(self, agent_store):
        root = agent_store.ensure_root_agent()
        assert root.role == "orchestrator"

    def test_child_role_is_orchestrator(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "child")
        assert child.role == "orchestrator"

    def test_role_not_inferred_from_mr_level_alone(self):
        # Two agents at the same mr_level can (in principle) hold different
        # roles — role is a stored field, not derived from level.
        orchestrator = AgentRecord(
            agent_id="ag-o", role="orchestrator", title="o", mr_level=3, parent_agent_id="ag-p",
        )
        worker = AgentRecord(
            agent_id="ag-w", role="worker", title="w", mr_level=3, parent_agent_id="ag-p",
            lifecycle="ephemeral",
        )
        assert orchestrator.mr_level == worker.mr_level == 3
        assert orchestrator.role != worker.role

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError, match="invalid role"):
            AgentRecord(
                agent_id="ag-bad", role="manager", title="bad", mr_level=1, parent_agent_id=None,
            )

    def test_actor_category_distinguishes_root_from_other_orchestrators(self):
        assert actor_category("orchestrator", 1) == "root_orchestrator"
        assert actor_category("orchestrator", 2) == "orchestrator"
        assert actor_category("orchestrator", 7) == "orchestrator"
        assert actor_category("worker", 1) == "worker"


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------

class TestLifecycle:
    def test_root_lifecycle_is_standing(self, agent_store):
        root = agent_store.ensure_root_agent()
        assert root.lifecycle == "standing"

    def test_child_default_lifecycle_is_project_scoped(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "child")
        assert child.lifecycle == "project_scoped"

    def test_child_lifecycle_can_be_set_explicitly(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "watchtower", lifecycle="standing")
        assert child.lifecycle == "standing"

    def test_invalid_lifecycle_rejected_at_construction(self):
        with pytest.raises(ValueError, match="invalid lifecycle"):
            AgentRecord(
                agent_id="ag-bad", role="worker", title="bad", mr_level=2,
                parent_agent_id="ag-root", lifecycle="forever",
            )

    def test_invalid_lifecycle_rejected_at_creation(self, agent_store):
        root = agent_store.ensure_root_agent()
        with pytest.raises(ValueError, match="invalid lifecycle"):
            agent_store.create_child_agent(root.agent_id, "child", lifecycle="forever")

    def test_lifecycle_persists_across_restart(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "child", lifecycle="standing")
        reopened = AgentStore(root=agent_store.root)
        assert reopened.require_agent(child.agent_id).lifecycle == "standing"

    def test_all_allowed_lifecycles_are_constructible(self):
        for lifecycle in ALLOWED_LIFECYCLES:
            agent = AgentRecord(
                agent_id=new_agent_id(), role="worker", title="t", mr_level=2,
                parent_agent_id="ag-root", lifecycle=lifecycle,
            )
            assert agent.lifecycle == lifecycle

    def test_terminate_agent_still_works_regardless_of_lifecycle(self, agent_store):
        root = agent_store.ensure_root_agent()
        standing_child = agent_store.create_child_agent(root.agent_id, "standing-one", lifecycle="standing")
        terminated = agent_store.terminate_agent(root.agent_id, standing_child.agent_id)
        assert terminated.status == "terminated"

    def test_root_cannot_be_terminated(self, agent_store):
        root = agent_store.ensure_root_agent()
        with pytest.raises(ValueError, match="cannot terminate root agent"):
            agent_store.terminate_agent(root.agent_id, root.agent_id)


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------

class TestNaming:
    def test_no_fallback_title_is_mr_branded(self):
        from mr1.scoped_agents import build_assignment_packet

        parent = AgentRecord(
            agent_id="ag-parent", role="orchestrator", title="Parent", mr_level=1, parent_agent_id=None,
        )
        packet = build_assignment_packet(parent, "", "do something", {})
        assert packet["child_title"] == "Unnamed agent"
        assert packet["child_title"] not in {"MRn", "MR2", "MR3"}

    def test_explicit_title_wins(self, agent_store):
        root = agent_store.ensure_root_agent()
        child = agent_store.create_child_agent(root.agent_id, "Repository Inspector")
        assert child.title == "Repository Inspector"

    def test_duplicate_title_protection_intact(self, agent_store):
        root = agent_store.ensure_root_agent()
        agent_store.create_child_agent(root.agent_id, "Alpha")
        with pytest.raises(ValueError, match="agent title already exists"):
            agent_store.create_child_agent(root.agent_id, "alpha")


# --------------------------------------------------------------------------
# migration
# --------------------------------------------------------------------------

_LEGACY_ROOT = {
    "agent_id": "ag-legacy-root",
    "agent_type": "mr1",
    "title": "MR1",
    "tree_level": 1,
    "parent_agent_id": None,
    "status": "active",
    "created_at": "2020-01-01T00:00:00+00:00",
    "owned_workflow_ids": [],
    "mission": None,
    "mode": "manual",
    "run_status": "idle",
    "current_iteration": 0,
    "last_step_at": None,
    "last_action": None,
    "parent_request": None,
    "last_run": None,
    "step_context": {},
    "security_clearance": 0.99,
    "scope_roots": [],
    "scope_grants": [],
    "assignment_packet": None,
}


def _legacy_child(agent_id: str, parent_id: str, *, agent_type: str = "mrn", **overrides) -> dict:
    payload = dict(_LEGACY_ROOT)
    payload.update(
        agent_id=agent_id,
        agent_type=agent_type,
        title=overrides.pop("title", agent_id),
        tree_level=2,
        parent_agent_id=parent_id,
        security_clearance=1.0,
    )
    payload.update(overrides)
    return payload


class TestLegacyMigration:
    def test_legacy_mr1_migrates_to_root_orchestrator_standing(self):
        migrated = migrate_legacy_agent_dict(dict(_LEGACY_ROOT))
        agent = AgentRecord.from_dict(migrated)
        assert agent.role == "orchestrator"
        assert agent.mr_level == 1
        assert agent.lifecycle == "standing"

    def test_legacy_mrn_migrates_to_orchestrator_project_scoped(self):
        legacy = _legacy_child("ag-legacy-child", "ag-legacy-root")
        agent = AgentRecord.from_dict(legacy)
        assert agent.role == "orchestrator"
        assert agent.mr_level == 2
        assert agent.lifecycle == "project_scoped"

    def test_legacy_kazi_migrates_to_worker(self):
        legacy = _legacy_child("ag-legacy-worker", "ag-legacy-root", agent_type="kazi")
        agent = AgentRecord.from_dict(legacy)
        assert agent.role == "worker"

    def test_unknown_legacy_agent_type_fails_loudly(self):
        legacy = _legacy_child("ag-legacy-mystery", "ag-legacy-root", agent_type="kami")
        with pytest.raises(ValueError, match="cannot migrate unknown legacy agent_type"):
            AgentRecord.from_dict(legacy)

    def test_record_with_neither_role_nor_agent_type_fails_loudly(self):
        broken = {k: v for k, v in _LEGACY_ROOT.items() if k != "agent_type"}
        with pytest.raises(ValueError, match="neither 'role' nor"):
            AgentRecord.from_dict(broken)

    def test_migration_is_idempotent_at_the_dict_level(self):
        once = migrate_legacy_agent_dict(dict(_LEGACY_ROOT))
        twice = migrate_legacy_agent_dict(dict(once))
        assert once == twice

    def test_no_data_loss_across_migration(self):
        legacy = _legacy_child(
            "ag-legacy-mission",
            "ag-legacy-root",
            mission="Own the docs.",
            owned_workflow_ids=["wf-1", "wf-2"],
            scope_roots=["/tmp/scope"],
        )
        agent = AgentRecord.from_dict(legacy)
        assert agent.mission == "Own the docs."
        assert agent.owned_workflow_ids == ["wf-1", "wf-2"]
        assert agent.scope_roots == ["/tmp/scope"]


class TestMigrateAgentStoreOntology:
    def test_migrates_legacy_files_on_disk(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "ag-legacy-root.json").write_text(json.dumps(_LEGACY_ROOT))
        child = _legacy_child("ag-legacy-child", "ag-legacy-root")
        (agents_dir / "ag-legacy-child.json").write_text(json.dumps(child))

        report = migrate_agent_store_ontology(agents_dir)

        assert set(report["migrated"]) == {"ag-legacy-root", "ag-legacy-child"}
        assert report["already_current"] == []
        assert report["failed"] == []

        root_on_disk = json.loads((agents_dir / "ag-legacy-root.json").read_text())
        assert root_on_disk["role"] == "orchestrator"
        assert root_on_disk["mr_level"] == 1
        assert root_on_disk["lifecycle"] == "standing"
        assert "agent_type" not in root_on_disk
        assert "tree_level" not in root_on_disk

    def test_migration_is_idempotent_on_disk(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "ag-legacy-root.json").write_text(json.dumps(_LEGACY_ROOT))

        first = migrate_agent_store_ontology(agents_dir)
        assert first["migrated"] == ["ag-legacy-root"]
        contents_after_first = (agents_dir / "ag-legacy-root.json").read_text()

        second = migrate_agent_store_ontology(agents_dir)
        assert second["migrated"] == []
        assert second["already_current"] == ["ag-legacy-root"]
        assert (agents_dir / "ag-legacy-root.json").read_text() == contents_after_first

    def test_already_current_files_are_left_untouched(self, agent_store):
        root = agent_store.ensure_root_agent()
        agent_store.create_child_agent(root.agent_id, "child")

        report = migrate_agent_store_ontology(agent_store.root)

        assert report["migrated"] == []
        assert len(report["already_current"]) == 2
        assert report["failed"] == []

    def test_unmigratable_file_is_reported_not_dropped(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        broken = _legacy_child("ag-broken", "ag-legacy-root", agent_type="kami")
        (agents_dir / "ag-broken.json").write_text(json.dumps(broken))

        report = migrate_agent_store_ontology(agents_dir)

        assert report["migrated"] == []
        assert len(report["failed"]) == 1
        assert report["failed"][0]["agent_id"] == "ag-broken"
        # The file must survive untouched — migration never discards data.
        still_on_disk = json.loads((agents_dir / "ag-broken.json").read_text())
        assert still_on_disk["agent_type"] == "kami"

    def test_no_agents_directory_is_a_no_op(self, tmp_path):
        report = migrate_agent_store_ontology(tmp_path / "agents")
        assert report == {
            "root": str(tmp_path / "agents"),
            "migrated": [],
            "already_current": [],
            "failed": [],
        }


def test_allowed_roles_and_lifecycles_match_ontology_spec():
    assert ALLOWED_ROLES == frozenset({"worker", "orchestrator"})
    assert ALLOWED_LIFECYCLES == frozenset(
        {"ephemeral", "task_scoped", "project_scoped", "standing"}
    )
