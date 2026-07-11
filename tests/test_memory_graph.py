from __future__ import annotations

import json
from pathlib import Path

import pytest

from mr1 import workflow_cli
from mr1.capability_policy import CapabilityApprovalDecision
from mr1.capability_runner import CapabilityRunner
from mr1.event_log import EventLog
from mr1.kazi_runner import MockRunner
from mr1.memory_graph import (
    MemoryGraphStore,
    agent_node_id,
    approval_node_id,
    capability_node_id,
    capability_stats,
    compute_workflow_template_id,
    edge_id,
    failure_modes,
    file_node_id,
    graph_stats,
    project_node_id,
    top_workflow_templates,
    update_graph_from_events,
    workflow_node_id,
)
from mr1.scheduler import Scheduler, submit_spec_to_disk
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def graph_store(tmp_path):
    return MemoryGraphStore(tmp_path / "graph")


@pytest.fixture
def event_log(tmp_path):
    return EventLog(tmp_path / "events")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


def _make_tool_workflow(
    tmp_path: Path,
    store: WorkflowStore,
    agent_store: PersistentAgentStore,
) -> tuple[str, str, str]:
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research", security_clearance=0.99)
    child.scope_roots = [str(tmp_path)]
    agent_store.save_agent(child)

    target = tmp_path / "input.txt"
    target.write_text("hello", encoding="utf-8")
    spec = {
        "title": "Read file workflow",
        "tasks": [{
            "label": "read",
            "title": "Read",
            "task_kind": "tool",
            "tool_type": "read_file",
            "tool_config": {"path": str(target)},
        }],
    }
    workflow_id = submit_spec_to_disk(
        spec,
        Provenance(type="agent", id=child.agent_id),
        store,
        owner_agent_id=child.agent_id,
        caller_agent_id=root.agent_id,
        scoped_agent_store=agent_store,
    )
    scheduler = Scheduler(store, MockRunner(), auto_tick=False, scoped_agent_store=agent_store)
    try:
        scheduler.tick()
        scheduler.tick()
    finally:
        scheduler.shutdown()
    return workflow_id, child.agent_id, str(target.resolve())


def test_deterministic_ids_and_template_hash_stability(tmp_path):
    file_path = tmp_path / "a.txt"
    same_file_path = tmp_path / "." / "a.txt"
    assert file_node_id(file_path) == file_node_id(same_file_path)
    assert edge_id("agent:a", "USED_CAPABILITY", "capability:read_file") == edge_id(
        "agent:a", "USED_CAPABILITY", "capability:read_file"
    )

    spec_a = {
        "title": "First",
        "tasks": [
            {
                "label": "read",
                "title": "Read",
                "task_kind": "tool",
                "tool_type": "read_file",
                "tool_config": {"path": "/tmp/one.txt"},
            },
            {
                "label": "summarize",
                "title": "Summarize",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "summarize",
                "depends_on": ["read"],
                "inputs": [{"name": "notes", "from": "read.result.text"}],
            },
        ],
    }
    spec_b = {
        "title": "Second",
        "tasks": [
            {
                "label": "read",
                "title": "Read changed",
                "task_kind": "tool",
                "tool_type": "read_file",
                "tool_config": {"path": "/tmp/two.txt"},
            },
            {
                "label": "summarize",
                "title": "Summarize changed",
                "task_kind": "agent",
                "agent_type": "kazi",
                "prompt": "totally different prompt",
                "depends_on": ["read"],
                "inputs": [{"name": "renamed_input", "from": "read.result.text"}],
            },
        ],
    }
    spec_c = {
        "title": "Different shape",
        "tasks": [
            {
                "label": "read",
                "task_kind": "tool",
                "tool_type": "read_file",
                "tool_config": {"path": "/tmp/one.txt"},
            },
            {
                "label": "other",
                "task_kind": "tool",
                "tool_type": "read_file",
                "tool_config": {"path": "/tmp/three.txt"},
            },
        ],
    }

    assert compute_workflow_template_id(spec_a) == compute_workflow_template_id(spec_b)
    assert compute_workflow_template_id(spec_a) != compute_workflow_template_id(spec_c)


def test_store_detects_corrupt_graph_and_cursor(tmp_path):
    store = MemoryGraphStore(tmp_path / "graph")
    store.root.mkdir(parents=True, exist_ok=True)
    store.graph_path.write_text("{bad", encoding="utf-8")
    store.cursor_path.write_text("{bad", encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt graph file"):
        store.load_graph()
    with pytest.raises(ValueError, match="corrupt cursor file"):
        store.load_cursor()


def test_ingestion_soft_skips_missing_event_metadata(event_log: EventLog, graph_store: MemoryGraphStore):
    event_log.emit(
        event_type="capability_requested",
        actor_id="ag-1",
        actor_type="mrn",
        target_id=None,
        target_type="capability",
        status="requested",
        summary="missing capability target",
    )

    result = update_graph_from_events(event_log, graph_store)
    assert result.processed_events == 1
    assert result.last_processed_event_index == 1

    graph = graph_store.load_graph()
    assert agent_node_id("ag-1") in graph.nodes
    assert capability_stats(graph) == []


def test_update_graph_from_real_workflow_events(tmp_path, store, graph_store, agent_store):
    workflow_id, child_agent_id, target_path = _make_tool_workflow(tmp_path, store, agent_store)

    result = update_graph_from_events(EventLog(tmp_path / "events"), graph_store)
    assert result.processed_events > 0

    graph = graph_store.load_graph()
    cursor = graph_store.load_cursor()
    assert cursor == result.last_processed_event_index

    workflow_id_value = workflow_node_id(workflow_id)
    capability_id_value = capability_node_id("read_file")
    file_id_value = file_node_id(target_path)
    project_id_value = project_node_id(tmp_path.name.lower())

    assert workflow_id_value in graph.nodes
    assert capability_id_value in graph.nodes
    assert agent_node_id(child_agent_id) in graph.nodes
    assert file_id_value in graph.nodes
    assert project_id_value in graph.nodes

    task_nodes = [node for node in graph.nodes.values() if node.node_type == "Task"]
    assert len(task_nodes) == 1
    assert task_nodes[0].stats["success_count"] == 1
    assert graph.nodes[workflow_id_value].stats["success_count"] == 1
    assert graph.nodes[capability_id_value].stats["request_count"] == 1
    assert graph.nodes[capability_id_value].stats["execution_count"] == 1
    assert graph.nodes[file_id_value].stats["touch_count"] >= 1

    templates = top_workflow_templates(graph)
    assert len(templates) == 1
    assert templates[0]["stats"]["success_count"] == 1
    assert templates[0]["stats"]["credibility_score"] > 0.0

    capabilities = capability_stats(graph)
    assert capabilities[0]["name"] == "read_file"
    assert capabilities[0]["stats"]["reliability_score"] > 0.0

    rerun = update_graph_from_events(EventLog(tmp_path / "events"), graph_store)
    assert rerun.processed_events == 0


def test_approval_and_failure_ingestion(tmp_path, graph_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research", security_clearance=0.1)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)

    result = runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-1")
    assert result.approval_request_id is not None
    runner._approval_store.apply_decision(
        result.approval_request_id,
        decision=CapabilityApprovalDecision(
            approval_request_id=result.approval_request_id,
            decision="denied",
            decided_by=root.agent_id,
            reason="no",
            timestamp=1.0,
            approval_scope="single_use",
        ),
        scoped_agent_store=agent_store,
    )

    update_graph_from_events(EventLog(tmp_path / "events"), graph_store)
    graph = graph_store.load_graph()

    approval_id_value = approval_node_id(result.approval_request_id)
    assert graph.nodes[approval_id_value].metadata["status"] == "denied"
    assert graph.nodes[agent_node_id(child.agent_id)].stats["approval_requested_count"] == 1
    assert graph.nodes[agent_node_id(root.agent_id)].stats["approval_denied_count"] == 1
    assert any(edge.edge_type == "REQUESTED_APPROVAL" for edge in graph.edges.values())
    assert any(edge.edge_type == "DENIED_BY" for edge in graph.edges.values())


def test_query_helpers_include_failure_modes(tmp_path, graph_store, agent_store):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "research", security_clearance=0.1)
    target = tmp_path / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    runner = CapabilityRunner(scoped_agent_store=agent_store, workspace_root=tmp_path)

    runner.run_capability("read_file", {"path": str(target)}, child.agent_id, step_id="step-2")
    update_graph_from_events(EventLog(tmp_path / "events"), graph_store)
    graph = graph_store.load_graph()

    failures = failure_modes(graph)
    assert failures == []
    assert graph_stats(graph, last_processed_event_index=graph_store.load_cursor())["node_count"] >= 3


def test_memory_cli_commands(tmp_path, store, graph_store, agent_store, capsys):
    workflow_id, child_agent_id, target_path = _make_tool_workflow(tmp_path, store, agent_store)
    child_project_id = project_node_id(tmp_path.name.lower())
    child_file_id = file_node_id(target_path)

    rc = workflow_cli.main(["memory", "update", "--json"], store=store, scoped_agent_store=agent_store)
    assert rc == 0
    update_payload = json.loads(capsys.readouterr().out)
    assert update_payload["processed_events"] > 0

    rc = workflow_cli.main(["memory", "stats", "--json"], store=store, scoped_agent_store=agent_store)
    assert rc == 0
    stats_payload = json.loads(capsys.readouterr().out)
    assert stats_payload["node_count"] > 0
    assert stats_payload["last_processed_event_index"] >= 1

    rc = workflow_cli.main(["memory", "graph", "top-workflows", "--json"], store=store, scoped_agent_store=agent_store)
    assert rc == 0
    top_payload = json.loads(capsys.readouterr().out)
    assert top_payload[0]["stats"]["success_count"] == 1

    rc = workflow_cli.main(["memory", "graph", "capabilities", "--json"], store=store, scoped_agent_store=agent_store)
    assert rc == 0
    caps_payload = json.loads(capsys.readouterr().out)
    assert caps_payload[0]["name"] == "read_file"

    rc = workflow_cli.main(["memory", "graph", "failures", "--json"], store=store, scoped_agent_store=agent_store)
    assert rc == 0
    failures_payload = json.loads(capsys.readouterr().out)
    assert failures_payload == []

    rc = workflow_cli.main(
        ["memory", "graph", "agent", child_agent_id, "--json"],
        store=store,
        scoped_agent_store=agent_store,
    )
    assert rc == 0
    agent_payload = json.loads(capsys.readouterr().out)
    assert agent_payload["node"]["node_id"] == agent_node_id(child_agent_id)

    rc = workflow_cli.main(
        ["memory", "graph", "project", child_project_id, "--json"],
        store=store,
        scoped_agent_store=agent_store,
    )
    assert rc == 0
    project_payload = json.loads(capsys.readouterr().out)
    assert project_payload["node"]["node_id"] == child_project_id

    rc = workflow_cli.main(
        ["memory", "graph", "file", child_file_id, "--json"],
        store=store,
        scoped_agent_store=agent_store,
    )
    assert rc == 0
    file_payload = json.loads(capsys.readouterr().out)
    assert file_payload["node"]["node_id"] == child_file_id

    rc = workflow_cli.main(
        ["memory", "graph", "show", workflow_node_id(workflow_id), "--json"],
        store=store,
        scoped_agent_store=agent_store,
    )
    assert rc == 0
    show_payload = json.loads(capsys.readouterr().out)
    assert show_payload["node"]["node_id"] == workflow_node_id(workflow_id)
    assert show_payload["incident_edges"]
