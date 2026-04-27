"""Tests for scoped agent CLI visibility and workflow access control."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mr1 import workflow_cli
from mr1.dataflow import ResolvedTaskInput, TaskOutput
from mr1.scoped_agents import PersistentAgentStore
from mr1.scheduler import submit_spec_to_disk
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


WORKFLOW_SPEC = {
    "title": "Scoped workflow",
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

WATCHER_SPEC = {
    "title": "Wait workflow",
    "tasks": [
        {
            "label": "wait",
            "title": "Wait",
            "task_kind": "watcher",
            "watcher_type": "manual_event",
            "watch_config": {"event": "approved"},
        }
    ],
}


@pytest.fixture
def workflow_store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


def _submit(
    workflow_store: WorkflowStore,
    agent_store: PersistentAgentStore,
    owner_agent_id: str,
    spec: dict | None = None,
) -> str:
    return submit_spec_to_disk(
        spec or WORKFLOW_SPEC,
        Provenance(type="user", id="cli"),
        workflow_store,
        owner_agent_id=owner_agent_id,
        scoped_agent_store=agent_store,
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_agents_and_workflows_are_filtered_by_scope(workflow_store, agent_store, capsys, tmp_path):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    left_child = agent_store.create_child_agent(left.agent_id, "left-child")
    right = agent_store.create_child_agent(root.agent_id, "right")

    left_wf = _submit(workflow_store, agent_store, left.agent_id)
    child_wf = _submit(workflow_store, agent_store, left_child.agent_id)
    right_wf = _submit(workflow_store, agent_store, right.agent_id)

    rc = workflow_cli.main(
        ["agents", "--json"],
        store=workflow_store,
        caller_agent_id=left.agent_id,
        scoped_agent_store=agent_store,
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    ids = {item["agent_id"] for item in payload}
    assert ids == {left.agent_id, left_child.agent_id}

    rc = workflow_cli.main(
        ["workflows"],
        store=workflow_store,
        caller_agent_id=left.agent_id,
        scoped_agent_store=agent_store,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert left_wf in out
    assert child_wf in out
    assert right_wf not in out


def test_agent_inspection_blocks_out_of_scope_agents(workflow_store, agent_store, capsys):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")

    rc = workflow_cli.main(
        ["agent", right.agent_id],
        store=workflow_store,
        caller_agent_id=left.agent_id,
        scoped_agent_store=agent_store,
    )

    assert rc == 2
    assert capsys.readouterr().err.strip() == "error: access denied: agent not in scope"


def test_workflow_commands_deny_out_of_scope_workflows(workflow_store, agent_store, capsys, tmp_path):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    right_wf = _submit(workflow_store, agent_store, right.agent_id)
    watcher_wf = _submit(workflow_store, agent_store, right.agent_id, WATCHER_SPEC)

    append_path = _write_json(tmp_path / "append.json", {
        "tasks": [{
            "label": "b",
            "title": "Task B",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "next",
            "depends_on": ["a"],
        }]
    })
    insert_path = _write_json(tmp_path / "insert.json", {
        "tasks": [{
            "label": "between",
            "title": "Between",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "between",
        }]
    })
    replace_path = _write_json(tmp_path / "replace.json", {
        "tasks": [{
            "label": "a",
            "title": "Task A replaced",
            "task_kind": "agent",
            "agent_type": "kazi",
            "prompt": "replacement",
        }]
    })

    commands = [
        (["workflow", right_wf], "error: access denied: workflow not in agent scope"),
        (["events", right_wf], "error: access denied: workflow not in agent scope"),
        (["artifacts", right_wf], "error: access denied: workflow not in agent scope"),
        (["rerun", right_wf, "a"], "error: access denied: workflow not in agent scope"),
        (["cancel-workflow", right_wf], "error: access denied: workflow not in agent scope"),
        (["append-workflow", right_wf, str(append_path)], "error: access denied: workflow not in agent scope"),
        (["insert-workflow", right_wf, "a", str(insert_path)], "error: access denied: workflow not in agent scope"),
        (["replace-workflow", right_wf, "a", str(replace_path)], "error: access denied: workflow not in agent scope"),
        (["trigger", watcher_wf, "wait", "approved"], "error: access denied: workflow not in agent scope"),
    ]

    for argv, expected in commands:
        rc = workflow_cli.main(
            argv,
            store=workflow_store,
            caller_agent_id=left.agent_id,
            scoped_agent_store=agent_store,
        )
        assert rc == 2
        assert capsys.readouterr().err.strip() == expected


def test_task_lookups_do_not_leak_out_of_scope_ids(workflow_store, agent_store, capsys):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    right = agent_store.create_child_agent(root.agent_id, "right")
    workflow_id = _submit(workflow_store, agent_store, right.agent_id)
    workflow = workflow_store.load_workflow(workflow_id)
    task_id = workflow.label_to_task_id["a"]

    commands = [
        ["task", task_id],
        ["result", task_id],
        ["inputs", task_id],
        ["cancel-task", task_id],
    ]
    for argv in commands:
        rc = workflow_cli.main(
            argv,
            store=workflow_store,
            caller_agent_id=left.agent_id,
            scoped_agent_store=agent_store,
        )
        assert rc == 2
        assert capsys.readouterr().err.strip() == f"error: task not found: {task_id}"


def test_result_and_inputs_work_for_visible_tasks(workflow_store, agent_store, capsys):
    root = agent_store.ensure_root_agent()
    left = agent_store.create_child_agent(root.agent_id, "left")
    workflow_id = _submit(workflow_store, agent_store, left.agent_id)
    workflow = workflow_store.load_workflow(workflow_id)
    task = workflow.task_by_label("a")

    workflow_store.write_task_output(
        workflow_id,
        task.task_id,
        TaskOutput(
            task_id=task.task_id,
            workflow_id=workflow_id,
            status="succeeded",
            summary="done",
            text="hello",
        ),
    )
    workflow_store.write_task_inputs(
        workflow_id,
        task.task_id,
        [
            ResolvedTaskInput(
                name="x",
                source="a.result.text",
                resolved_task_id=task.task_id,
                resolved_type="text",
                value="hello",
            )
        ],
    )

    rc = workflow_cli.main(
        ["result", task.task_id],
        store=workflow_store,
        caller_agent_id=left.agent_id,
        scoped_agent_store=agent_store,
    )
    assert rc == 0
    assert "hello" in capsys.readouterr().out

    rc = workflow_cli.main(
        ["inputs", task.task_id],
        store=workflow_store,
        caller_agent_id=left.agent_id,
        scoped_agent_store=agent_store,
    )
    assert rc == 0
    assert "a.result.text" in capsys.readouterr().out
