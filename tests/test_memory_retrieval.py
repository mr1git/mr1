from __future__ import annotations

import json

import pytest

from mr1.event_log import EventLog
from mr1.memory_curator import EvidenceRef, InsightStore, MemoryInsight
from mr1.memory_feedback import InsightFeedback
from mr1.memory_graph import (
    MemoryEdge,
    MemoryGraph,
    MemoryGraphStore,
    MemoryNode,
    agent_node_id,
    capability_node_id,
    edge_id,
    failure_node_id,
    workflow_node_id,
    workflow_template_node_id,
)
from mr1.memory_queries import memory_search
from mr1.memory_retrieval import LexicalMemoryRetriever, RetrievalStore, update_memory_retrieval


def _insight(insight_id: str = "insight:capability_friction:capability:read_file") -> MemoryInsight:
    return MemoryInsight(
        insight_id=insight_id,
        insight_type="capability_friction",
        title="Read file approval friction",
        summary="Read file workflows hit approval friction often.",
        confidence=0.9,
        severity="WARNING",
        recommended_action="consider narrowing scope before requesting file access",
        evidence=[
            EvidenceRef(source_type="event", source_id="evt-1", reason="workflow blocked"),
            EvidenceRef(source_type="query", source_id="query:capability_stats", reason="high request count"),
        ],
        related_nodes=[capability_node_id("read_file"), agent_node_id("ag-1")],
        created_at="2026-04-20T00:00:00+00:00",
        updated_at="2026-04-25T00:00:00+00:00",
        status="active",
        metadata={"capability_id": capability_node_id("read_file")},
    )


def _seed_insights(root):
    store = InsightStore(root / "insights")
    store.save_insights({"insight:capability_friction:capability:read_file": _insight()})
    store.append_feedback(InsightFeedback(
        feedback_id="feedback:1",
        insight_id="insight:capability_friction:capability:read_file",
        workflow_id="wf-1",
        event_id="evt-feedback-1",
        event_index=12,
        outcome="negative",
        reason="approval still blocked the workflow",
        confidence_delta=-0.05,
        evaluator_type="deterministic",
        created_at="2026-04-26T00:00:00+00:00",
        metadata={},
    ))
    store.append_feedback(InsightFeedback(
        feedback_id="feedback:2",
        insight_id="insight:capability_friction:capability:read_file",
        workflow_id="wf-2",
        event_id="evt-feedback-2",
        event_index=13,
        outcome="positive",
        reason="narrower scope avoided repeated approval churn",
        confidence_delta=0.03,
        evaluator_type="deterministic",
        created_at="2026-04-27T00:00:00+00:00",
        metadata={},
    ))
    return store


def _seed_graph(root):
    template_id = workflow_template_node_id("tmpl-1")
    workflow_id_value = workflow_node_id("wf-1")
    capability_id_value = capability_node_id("read_file")
    failure_id_value = failure_node_id("timeout")
    agent_id_value = agent_node_id("ag-1")
    graph = MemoryGraph(
        nodes={
            template_id: MemoryNode(
                node_id=template_id,
                node_type="WorkflowTemplate",
                name="template-one",
                created_at="2026-04-20T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
                stats={"credibility_score": 0.95, "success_count": 7, "failure_count": 2, "blocked_count": 1},
            ),
            workflow_id_value: MemoryNode(
                node_id=workflow_id_value,
                node_type="Workflow",
                name="workflow-one",
                created_at="2026-04-20T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
            ),
            capability_id_value: MemoryNode(
                node_id=capability_id_value,
                node_type="Capability",
                name="read_file",
                created_at="2026-04-20T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
                stats={"request_count": 8, "execution_count": 6, "blocked_count": 1, "failure_count": 1, "reliability_score": 0.666},
                metadata={"capability_name": "read_file"},
            ),
            failure_id_value: MemoryNode(
                node_id=failure_id_value,
                node_type="FailureMode",
                name="timeout",
                created_at="2026-04-21T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
                stats={"occurrence_count": 4},
                metadata={"status": "failed", "error_type": "timeout"},
            ),
            agent_id_value: MemoryNode(
                node_id=agent_id_value,
                node_type="Agent",
                name="research",
                created_at="2026-04-20T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
                stats={"blocked_count": 2, "approval_requested_count": 3},
                metadata={"agent_id": "ag-1"},
            ),
        },
        edges={
            edge_id(workflow_id_value, "HAS_TEMPLATE", template_id): MemoryEdge(
                edge_id=edge_id(workflow_id_value, "HAS_TEMPLATE", template_id),
                source_id=workflow_id_value,
                target_id=template_id,
                edge_type="HAS_TEMPLATE",
                first_seen_at="2026-04-20T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
            ),
            edge_id(workflow_id_value, "FAILED_WITH", failure_id_value): MemoryEdge(
                edge_id=edge_id(workflow_id_value, "FAILED_WITH", failure_id_value),
                source_id=workflow_id_value,
                target_id=failure_id_value,
                edge_type="FAILED_WITH",
                first_seen_at="2026-04-20T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
            ),
            edge_id(capability_id_value, "FAILED_WITH", failure_id_value): MemoryEdge(
                edge_id=edge_id(capability_id_value, "FAILED_WITH", failure_id_value),
                source_id=capability_id_value,
                target_id=failure_id_value,
                edge_type="FAILED_WITH",
                first_seen_at="2026-04-20T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
            ),
            edge_id(agent_id_value, "USED_CAPABILITY", capability_id_value): MemoryEdge(
                edge_id=edge_id(agent_id_value, "USED_CAPABILITY", capability_id_value),
                source_id=agent_id_value,
                target_id=capability_id_value,
                edge_type="USED_CAPABILITY",
                first_seen_at="2026-04-20T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
            ),
            edge_id(agent_id_value, "RAN_WORKFLOW", workflow_id_value): MemoryEdge(
                edge_id=edge_id(agent_id_value, "RAN_WORKFLOW", workflow_id_value),
                source_id=agent_id_value,
                target_id=workflow_id_value,
                edge_type="RAN_WORKFLOW",
                first_seen_at="2026-04-20T00:00:00+00:00",
                last_seen_at="2026-04-27T00:00:00+00:00",
            ),
        },
    )
    store = MemoryGraphStore(root / "graph")
    store.save_graph(graph)
    return store


def _seed_events(root):
    log = EventLog(root / "events")
    log.emit(
        event_type="workflow_created",
        actor_id="cli",
        actor_type="mr1",
        target_id="wf-1",
        target_type="workflow",
        status="pending",
        summary="workflow created",
        workflow_id="wf-1",
        correlation_id="corr-important",
        timestamp="2026-04-25T00:00:00+00:00",
    )
    log.emit(
        event_type="workflow_completed",
        actor_id="scheduler",
        actor_type="mr1",
        target_id="wf-1",
        target_type="workflow",
        status="succeeded",
        summary="workflow completed successfully",
        workflow_id="wf-1",
        correlation_id="corr-important",
        timestamp="2026-04-25T00:10:00+00:00",
    )
    log.emit(
        event_type="capability_requested",
        actor_id="ag-1",
        actor_type="mrn",
        target_id="read_file",
        target_type="capability",
        status="requested",
        summary="capability requested",
        workflow_id="wf-2",
        correlation_id="corr-unimportant",
        timestamp="2026-04-26T00:00:00+00:00",
    )
    return log


def _seed_runtime(root):
    _seed_insights(root)
    _seed_graph(root)
    _seed_events(root)


def test_update_memory_retrieval_generates_required_documents(tmp_path):
    _seed_runtime(tmp_path)

    result = update_memory_retrieval(tmp_path)
    store = RetrievalStore(tmp_path)
    documents = {item.doc_id: item for item in store.load_documents()}
    manifest = store.load_manifest()

    assert result.documents_written == len(documents)
    assert "retrieval:insight:insight:capability_friction:capability:read_file" in documents
    assert "retrieval:insight_feedback:insight:capability_friction:capability:read_file" in documents
    assert "retrieval:workflow_template:workflow_template:tmpl-1" in documents
    assert "retrieval:capability:capability:read_file" in documents
    assert "retrieval:failure_mode:failure:timeout" in documents
    assert "retrieval:timeline_trace:corr-important" in documents
    assert "retrieval:agent_summary:agent:ag-1" in documents
    assert "retrieval:timeline_trace:corr-unimportant" not in documents
    assert manifest["schema_version"] == 1
    assert manifest["document_count"] == len(documents)
    assert manifest["source_counts"]["timeline_trace"] == 1


def test_update_memory_retrieval_is_deterministic_on_rerun(tmp_path):
    _seed_runtime(tmp_path)

    first = update_memory_retrieval(tmp_path)
    docs_path = RetrievalStore(tmp_path).documents_path
    manifest_path = RetrievalStore(tmp_path).manifest_path
    first_docs_bytes = docs_path.read_bytes()
    first_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    second = update_memory_retrieval(tmp_path)
    second_docs_bytes = docs_path.read_bytes()
    second_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert first.documents_written == second.documents_written
    assert first_docs_bytes == second_docs_bytes
    assert first_manifest["schema_version"] == second_manifest["schema_version"]
    assert first_manifest["document_count"] == second_manifest["document_count"]
    assert first_manifest["source_counts"] == second_manifest["source_counts"]


def test_missing_stores_degrade_gracefully(tmp_path):
    result = update_memory_retrieval(tmp_path)
    docs = RetrievalStore(tmp_path).load_documents()

    assert result.documents_written == 0
    assert docs == []
    assert result.source_counts["insight"] == 0
    assert result.errors == []


def test_corrupt_sources_are_skipped_per_source(tmp_path):
    _seed_graph(tmp_path)
    _seed_events(tmp_path)
    insights_dir = tmp_path / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)
    (insights_dir / "insights.json").write_text("{bad json", encoding="utf-8")

    result = update_memory_retrieval(tmp_path)
    docs = RetrievalStore(tmp_path).load_documents()

    assert any("skipped insight retrieval documents" in item for item in result.errors)
    assert result.source_counts["insight"] == 0
    assert any(item.doc_type == "capability" for item in docs)


def test_lexical_retriever_title_summary_body_and_type_filters(tmp_path):
    _seed_runtime(tmp_path)
    update_memory_retrieval(tmp_path)
    store = RetrievalStore(tmp_path)
    docs = store.load_documents()
    manual = [
        item for item in docs
        if item.doc_type != "timeline_trace"
    ]
    manual.append(manual[0].__class__(
        doc_id="retrieval:capability:body-only",
        doc_type="capability",
        title="Generic capability",
        summary="No friction summary",
        body="approval friction only appears in the body",
        refs=[{"kind": "graph_node", "id": "capability:body-only"}],
        tags=["capability"],
        created_at="2026-04-20T00:00:00+00:00",
        updated_at="2026-04-20T00:00:00+00:00",
        metadata={},
    ))
    store.save(manual, source_counts=store.load_manifest().get("source_counts", {}), updated_at="2026-04-30T00:00:00+00:00")

    retriever = LexicalMemoryRetriever(tmp_path)
    results = retriever.search("approval friction", limit=5)

    assert results[0].doc_type == "insight"
    assert any(item.doc_type == "capability" for item in retriever.search("approval friction", limit=10, types=["capability"]))
    assert len(retriever.search("approval friction", limit=1)) == 1
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.search("   ")


def test_memory_search_bootstraps_when_retrieval_docs_missing(tmp_path):
    _seed_runtime(tmp_path)

    result = memory_search(tmp_path, query="approval friction", limit=5)

    assert result["retrieval_ready"] is True
    assert result["updated_now"] is True
    assert result["count"] >= 1


def test_memory_search_rebuilds_corrupt_retrieval_docs(tmp_path):
    _seed_runtime(tmp_path)
    update_memory_retrieval(tmp_path)
    store = RetrievalStore(tmp_path)
    store.documents_path.write_text("{bad json\n", encoding="utf-8")

    result = memory_search(tmp_path, query="approval friction", limit=5)

    assert result["retrieval_ready"] is True
    assert result["updated_now"] is True
    assert result["count"] >= 1
