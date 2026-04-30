from __future__ import annotations

import json

import pytest

from mr1.capabilities import default_capability_registry
from mr1.capability_runner import CapabilityRunner
from mr1.event_log import EventLog
from mr1.memory_curator import EvidenceRef, InsightStore, MemoryInsight
from mr1.memory_graph import (
    MemoryGraph,
    MemoryGraphStore,
    MemoryNode,
    agent_node_id,
    capability_node_id,
    failure_node_id,
    workflow_template_node_id,
)
from mr1.scoped_agents import PersistentAgentStore


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def runner(agent_store):
    return CapabilityRunner(scoped_agent_store=agent_store)


@pytest.fixture
def root_agent_id(agent_store):
    return agent_store.ensure_root_agent().agent_id


def _insight(
    insight_id: str,
    *,
    insight_type: str,
    title: str,
    summary: str,
    confidence: float,
    updated_at: str,
    status: str = "active",
) -> MemoryInsight:
    return MemoryInsight(
        insight_id=insight_id,
        insight_type=insight_type,
        title=title,
        summary=summary,
        confidence=confidence,
        severity="WARNING",
        recommended_action="consider narrowing scope",
        evidence=[EvidenceRef(source_type="query", source_id="query:test", reason="test fixture")],
        related_nodes=["capability:read_file"],
        created_at="2026-04-20T00:00:00+00:00",
        updated_at=updated_at,
        status=status,
        metadata={"topic": "file access approval friction"},
    )


def _seed_insights(tmp_path):
    store = InsightStore(tmp_path / "insights")
    store.save_insights({
        "insight:exact": _insight(
            "insight:exact",
            insight_type="capability_friction",
            title="approval friction",
            summary="Exact match title for approval friction",
            confidence=0.90,
            updated_at="2026-04-25T00:00:00+00:00",
        ),
        "insight:substring": _insight(
            "insight:substring",
            insight_type="capability_friction",
            title="file access friction",
            summary="This summary mentions approval friction as a substring.",
            confidence=0.70,
            updated_at="2026-04-24T00:00:00+00:00",
        ),
        "insight:dismissed": _insight(
            "insight:dismissed",
            insight_type="scope_recommendation",
            title="dismissed item",
            summary="Should be hidden by default",
            confidence=0.99,
            updated_at="2026-04-26T00:00:00+00:00",
            status="dismissed",
        ),
    })
    return store


def _seed_graph(tmp_path):
    store = MemoryGraphStore(tmp_path / "graph")
    graph = MemoryGraph(
        nodes={
            workflow_template_node_id("abc"): MemoryNode(
                node_id=workflow_template_node_id("abc"),
                node_type="WorkflowTemplate",
                name="Top template",
                stats={"credibility_score": 0.95, "success_count": 10, "failure_count": 1, "blocked_count": 0},
            ),
            capability_node_id("read_file"): MemoryNode(
                node_id=capability_node_id("read_file"),
                node_type="Capability",
                name="read_file",
                stats={"request_count": 8, "execution_count": 7, "blocked_count": 0, "failure_count": 1},
            ),
            failure_node_id("timeout"): MemoryNode(
                node_id=failure_node_id("timeout"),
                node_type="FailureMode",
                name="timeout",
                stats={"occurrence_count": 4},
                metadata={"status": "failed", "error_type": "timeout"},
            ),
            agent_node_id("ag-123"): MemoryNode(
                node_id=agent_node_id("ag-123"),
                node_type="Agent",
                name="research",
            ),
        }
    )
    store.save_graph(graph)
    store.save_cursor(12)
    return store


def test_memory_capabilities_registered_with_policy_metadata():
    registry = default_capability_registry()
    for name in (
        "memory_insights_search",
        "memory_insight_show",
        "memory_graph_top_workflows",
        "memory_graph_capabilities",
        "memory_graph_failures",
        "memory_graph_agent_summary",
    ):
        desc = registry.describe_capability(name)
        assert desc["direct_allowed"] is True
        assert desc["workflow_allowed"] is True
        assert desc["risk_score"] == 0.05
        assert desc["is_execution"] is False


def test_memory_insights_search_filters_ranks_and_limits(tmp_path, runner, root_agent_id):
    _seed_insights(tmp_path)

    result = runner.run_capability(
        "memory_insights_search",
        {
            "query": "approval friction",
            "types": ["capability_friction"],
            "status": "active",
            "limit": 1,
        },
        root_agent_id,
    )

    assert result.status == "succeeded"
    assert result.output["count"] == 1
    assert result.output["items"][0]["insight_id"] == "insight:exact"


def test_memory_insights_search_ignores_dismissed_by_default(tmp_path, runner, root_agent_id):
    _seed_insights(tmp_path)

    result = runner.run_capability(
        "memory_insights_search",
        {"query": "dismissed", "limit": 5},
        root_agent_id,
    )

    assert result.status == "succeeded"
    assert result.output["items"] == []


def test_memory_insight_show_returns_full_object_and_missing_false(tmp_path, runner, root_agent_id):
    _seed_insights(tmp_path)

    found = runner.run_capability(
        "memory_insight_show",
        {"insight_id": "insight:exact"},
        root_agent_id,
    )
    missing = runner.run_capability(
        "memory_insight_show",
        {"insight_id": "insight:missing"},
        root_agent_id,
    )

    assert found.status == "succeeded"
    assert found.output["found"] is True
    assert found.output["insight"]["insight_id"] == "insight:exact"
    assert missing.status == "succeeded"
    assert missing.output == {"found": False, "insight": None}


def test_memory_graph_queries_return_existing_helper_results(tmp_path, runner, root_agent_id):
    _seed_graph(tmp_path)

    top = runner.run_capability("memory_graph_top_workflows", {"limit": 5}, root_agent_id)
    caps = runner.run_capability("memory_graph_capabilities", {"limit": 10}, root_agent_id)
    failures = runner.run_capability("memory_graph_failures", {"limit": 10}, root_agent_id)
    agent = runner.run_capability("memory_graph_agent_summary", {"agent_id": "ag-123"}, root_agent_id)

    assert top.output["items"][0]["node_id"] == workflow_template_node_id("abc")
    assert caps.output["items"][0]["node_id"] == capability_node_id("read_file")
    assert failures.output["items"][0]["node_id"] == failure_node_id("timeout")
    assert agent.output["found"] is True
    assert agent.output["summary"]["node"]["node_id"] == agent_node_id("ag-123")


def test_memory_capabilities_return_empty_structures_when_stores_missing(runner, root_agent_id):
    top = runner.run_capability("memory_graph_top_workflows", {"limit": 5}, root_agent_id)
    caps = runner.run_capability("memory_graph_capabilities", {"limit": 10}, root_agent_id)
    failures = runner.run_capability("memory_graph_failures", {"limit": 10}, root_agent_id)
    search = runner.run_capability("memory_insights_search", {"query": "anything", "limit": 5}, root_agent_id)

    assert top.output == {"items": [], "count": 0}
    assert caps.output == {"items": [], "count": 0}
    assert failures.output == {"items": [], "count": 0}
    assert search.output["items"] == []


def test_memory_capability_invalid_input_returns_structured_validation_failure(runner, root_agent_id):
    result = runner.run_capability(
        "memory_graph_agent_summary",
        {},
        root_agent_id,
    )

    assert result.status == "failed"
    assert result.output["error_type"] == "validation_error"
    assert result.error == result.output["message"]


def test_memory_capability_direct_call_writes_audit_and_timeline(tmp_path, runner, root_agent_id):
    _seed_graph(tmp_path)

    result = runner.run_capability(
        "memory_graph_capabilities",
        {"limit": 10},
        root_agent_id,
    )

    assert result.status == "succeeded"
    audit = json.loads(open(result.audit_record_path, "r", encoding="utf-8").read())
    assert audit["capability_name"] == "memory_graph_capabilities"
    events = EventLog(tmp_path / "events").list_events()
    assert [event.event_type for event in events[-3:]] == [
        "capability_requested",
        "capability_allowed",
        "capability_executed",
    ]
