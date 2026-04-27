"""Tests for workflow ownership metadata, visibility, and reporting."""

from __future__ import annotations

import json

from mr1.kazi_runner import MockRunner, RunStatus
from mr1.scoped_agents import PersistentAgentStore
from mr1.scheduler import (
    Scheduler,
    append_workflow_on_disk,
    replace_workflow_on_disk,
    submit_spec_to_disk,
)
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


def _replace_spec() -> dict:
    return {
        "tasks": [
            {
                "label": "a",
                "title": "Task A replaced",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "replacement",
            }
        ]
    }


def _append_spec() -> dict:
    return {
        "tasks": [
            {
                "label": "b",
                "title": "Task B",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "follow up",
                "depends_on": ["a"],
            }
        ]
    }


def test_mr1_submit_defaults_to_root_ownership(tmp_path):
    workflow_store = WorkflowStore(root=tmp_path / "workflows")
    agent_store = PersistentAgentStore(root=tmp_path / "agents")
    root = agent_store.ensure_root_agent()

    workflow_id = submit_spec_to_disk(
        SPEC,
        Provenance(type="user", id="cli"),
        workflow_store,
        scoped_agent_store=agent_store,
    )
    workflow = agent_store.normalize_workflow_ownership(workflow_store.load_workflow(workflow_id))
    owner = agent_store.require_agent(root.agent_id)

    assert workflow.owner_agent_id == root.agent_id
    assert workflow.owner_agent_title == root.title
    assert workflow.parent_agent_id is None
    assert workflow_id in owner.owned_workflow_ids


def test_mrn_submit_stamps_ownership_and_updates_agent_record(tmp_path):
    workflow_store = WorkflowStore(root=tmp_path / "workflows")
    agent_store = PersistentAgentStore(root=tmp_path / "agents")
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")

    workflow_id = submit_spec_to_disk(
        SPEC,
        Provenance(type="user", id="cli"),
        workflow_store,
        owner_agent_id=child.agent_id,
        scoped_agent_store=agent_store,
    )
    workflow = agent_store.normalize_workflow_ownership(workflow_store.load_workflow(workflow_id))
    owner = agent_store.require_agent(child.agent_id)

    assert workflow.owner_agent_id == child.agent_id
    assert workflow.owner_agent_title == "research"
    assert workflow.parent_agent_id == root.agent_id
    assert workflow_id in owner.owned_workflow_ids


def test_workflow_mutations_preserve_ownership(tmp_path):
    workflow_store = WorkflowStore(root=tmp_path / "workflows")
    agent_store = PersistentAgentStore(root=tmp_path / "agents")
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")

    workflow_id = submit_spec_to_disk(
        SPEC,
        Provenance(type="user", id="cli"),
        workflow_store,
        owner_agent_id=child.agent_id,
        scoped_agent_store=agent_store,
    )
    append_workflow_on_disk(workflow_store, workflow_id, _append_spec(), agent_id="cli")
    replace_workflow_on_disk(workflow_store, workflow_id, "a", _replace_spec(), agent_id="cli")

    workflow = agent_store.normalize_workflow_ownership(workflow_store.load_workflow(workflow_id))
    assert workflow.owner_agent_id == child.agent_id
    assert workflow.owner_agent_title == "research"
    assert workflow.parent_agent_id == root.agent_id


def test_descendant_access_allowed_and_sibling_access_denied(tmp_path):
    workflow_store = WorkflowStore(root=tmp_path / "workflows")
    agent_store = PersistentAgentStore(root=tmp_path / "agents")
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    left_child = agent_store.create_child_agent(left.agent_id, "left-child")
    right = agent_store.create_child_agent(root.agent_id, "right")

    workflow_id = submit_spec_to_disk(
        SPEC,
        Provenance(type="user", id="cli"),
        workflow_store,
        owner_agent_id=left_child.agent_id,
        scoped_agent_store=agent_store,
    )
    workflow = agent_store.normalize_workflow_ownership(workflow_store.load_workflow(workflow_id))

    assert agent_store.can_agent_access_workflow(left.agent_id, workflow) is True
    assert agent_store.can_agent_access_workflow(root.agent_id, workflow) is True
    assert agent_store.can_agent_access_workflow(right.agent_id, workflow) is False


def test_terminal_mrn_workflow_writes_report(tmp_path):
    workflow_store = WorkflowStore(root=tmp_path / "workflows")
    agent_store = PersistentAgentStore(root=tmp_path / "agents")
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research")
    runner = MockRunner()
    scheduler = Scheduler(
        workflow_store,
        runner,
        auto_tick=False,
        scoped_agent_store=agent_store,
    )

    workflow_id = scheduler.submit_workflow(
        SPEC,
        Provenance(type="user", id="cli"),
        owner_agent_id=child.agent_id,
    )
    workflow = workflow_store.load_workflow(workflow_id)
    task_id = workflow.label_to_task_id["a"]

    scheduler.tick()
    runner.complete(
        task_id,
        RunStatus.SUCCEEDED,
        summary="done",
        result_payload={
            "status": "succeeded",
            "summary": "done",
            "text": "workflow output",
        },
    )
    scheduler.tick()

    reports = agent_store.list_reports(child.agent_id)
    assert len(reports) == 1
    content = reports[0].read_text(encoding="utf-8")
    assert workflow_id in content
    assert "status: succeeded" in content
    assert "workflow output" in content
