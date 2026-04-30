from __future__ import annotations

import json

import pytest

from mr1 import workflow_cli
from mr1.capabilities import default_capability_registry
from mr1.capability_runner import CapabilityRunner
from mr1.event_log import EventLog
from mr1.memory_curator import EvidenceRef, InsightStore, MemoryInsight, default_insight_stats
from mr1.memory_feedback import (
    DeterministicInsightEvaluator,
    InsightFeedback,
    build_memory_maintenance_spec,
    evaluate_memory_feedback_due,
    feedback_id_for,
    update_insight_feedback,
)
from mr1.scoped_agents import PersistentAgentStore
from mr1.tools import default_tool_registry
from mr1.workflow_models import Provenance, Workflow
from mr1.workflow_store import WorkflowStore


PROV = Provenance(type="agent", id="MR1")


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def event_log(tmp_path):
    return EventLog(tmp_path / "events")


@pytest.fixture
def insight_store(tmp_path):
    return InsightStore(tmp_path / "insights")


@pytest.fixture
def agent_store(tmp_path):
    return PersistentAgentStore(root=tmp_path / "agents")


@pytest.fixture
def root_agent_id(agent_store):
    return agent_store.ensure_root_agent().agent_id


@pytest.fixture
def runner(agent_store):
    return CapabilityRunner(scoped_agent_store=agent_store)


def _insight(
    insight_id: str,
    *,
    confidence: float = 0.50,
    status: str = "active",
    capability_name: str = "read_file",
) -> MemoryInsight:
    return MemoryInsight(
        insight_id=insight_id,
        insight_type="capability_friction",
        title=f"{capability_name} friction",
        summary="test insight",
        confidence=confidence,
        severity="WARNING",
        recommended_action="consider narrowing scope",
        evidence=[EvidenceRef(source_type="query", source_id="query:test", reason="fixture")],
        related_nodes=[f"capability:{capability_name}"],
        created_at="2026-04-20T00:00:00+00:00",
        updated_at="2026-04-20T00:00:00+00:00",
        status=status,
        metadata={"capability_id": f"capability:{capability_name}"},
    )


def _save_workflow(store: WorkflowStore, workflow_id: str, memory_refs: list[str]) -> None:
    store.save_workflow(Workflow(
        workflow_id=workflow_id,
        title=f"Workflow {workflow_id}",
        created_by=PROV,
        metadata={
            "compiled_with_memory": True,
            "memory_refs_used": list(memory_refs),
        },
    ))


def _emit_workflow_event(
    event_log: EventLog,
    *,
    event_type: str,
    workflow_id: str,
    target_id: str | None = None,
    metadata: dict | None = None,
    timestamp: str | None = None,
):
    return event_log.emit(
        event_type=event_type,
        actor_id="scheduler",
        actor_type="scheduler",
        target_id=target_id or workflow_id,
        target_type="workflow",
        status="failed" if "failed" in event_type or "denied" in event_type or "blocked" in event_type else "succeeded",
        summary=event_type,
        workflow_id=workflow_id,
        timestamp=timestamp,
        metadata=metadata or {},
    )


def test_feedback_id_and_store_roundtrip(insight_store: InsightStore):
    feedback = InsightFeedback(
        feedback_id=feedback_id_for(
            event_id="evt-1",
            insight_id="insight:a",
            evaluator_type="deterministic",
        ),
        insight_id="insight:a",
        workflow_id="wf-1",
        event_id="evt-1",
        event_index=1,
        outcome="positive",
        reason="workflow completed after using this memory ref",
        confidence_delta=0.03,
        evaluator_type="deterministic",
        created_at="2026-04-29T00:00:00+00:00",
        metadata={"event_type": "workflow_completed"},
    )
    insight_store.append_feedback(feedback)
    insight_store.save_feedback_cursor(7)

    loaded = insight_store.load_feedback(limit=10)
    assert loaded[0].feedback_id == "feedback:evt-1:insight:a:deterministic"
    assert insight_store.load_feedback(insight_id="insight:a")[0].insight_id == "insight:a"
    assert insight_store.feedback_exists(feedback.feedback_id) is True
    assert insight_store.load_feedback_cursor() == 7


def test_legacy_insight_load_fills_default_stats(insight_store: InsightStore):
    raw = {
        "insight:test": {
            "insight_id": "insight:test",
            "insight_type": "capability_friction",
            "title": "legacy",
            "summary": "legacy",
            "confidence": 0.4,
            "severity": "WARNING",
            "recommended_action": "consider narrowing scope",
            "evidence": [{"source_type": "query", "source_id": "query:test", "reason": "fixture"}],
            "related_nodes": ["capability:read_file"],
            "created_at": "2026-04-20T00:00:00+00:00",
            "updated_at": "2026-04-20T00:00:00+00:00",
            "status": "active",
            "metadata": {"capability_id": "capability:read_file"},
        }
    }
    insight_store.insights_path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = insight_store.load_insights()["insight:test"]
    assert loaded.stats == default_insight_stats()


def test_update_feedback_positive_and_idempotent(store, event_log, insight_store):
    _save_workflow(store, "wf-1", ["insight:read"])
    insight_store.save_insights({"insight:read": _insight("insight:read")})
    _emit_workflow_event(event_log, event_type="workflow_completed", workflow_id="wf-1")

    first = update_insight_feedback(event_log, insight_store, store)
    assert first.processed_events == 1
    assert first.feedback_created == 1
    assert first.insights_updated == 1
    assert first.last_evaluated_event_index == 1

    stored = insight_store.load_insights()["insight:read"]
    assert stored.confidence == pytest.approx(0.53)
    assert stored.stats["used_count"] == 1
    assert stored.stats["positive_outcome_count"] == 1
    assert stored.stats["effectiveness_score"] == pytest.approx(0.5)

    second = update_insight_feedback(event_log, insight_store, store)
    assert second.processed_events == 0
    assert second.feedback_created == 0
    assert len(insight_store.load_feedback()) == 1


def test_failure_does_not_advance_cursor_or_append_feedback(store, event_log, insight_store):
    class ExplodingEvaluator:
        evaluator_type = "deterministic"

        def evaluate(self, event, workflow, insight, context):
            raise RuntimeError("boom")

    _save_workflow(store, "wf-1", ["insight:read"])
    insight_store.save_insights({"insight:read": _insight("insight:read")})
    _emit_workflow_event(event_log, event_type="workflow_failed", workflow_id="wf-1")

    with pytest.raises(RuntimeError, match="boom"):
        update_insight_feedback(event_log, insight_store, store, evaluator=ExplodingEvaluator())

    assert insight_store.load_feedback_cursor() == 0
    assert insight_store.load_feedback() == []
    assert insight_store.load_insights()["insight:read"].stats == default_insight_stats()


def test_negative_feedback_marks_stale_and_preserves_dismissed_status(store, event_log, insight_store):
    insight_store.save_insights({
        "insight:active": _insight("insight:active", confidence=0.2, status="active"),
        "insight:dismissed": _insight("insight:dismissed", confidence=0.2, status="dismissed"),
    })
    _save_workflow(store, "wf-1", ["insight:active", "insight:dismissed"])
    for idx in range(3):
        _emit_workflow_event(
            event_log,
            event_type="workflow_failed",
            workflow_id="wf-1",
            timestamp=f"2026-04-29T00:00:0{idx}+00:00",
        )

    result = update_insight_feedback(event_log, insight_store, store)

    assert result.feedback_created == 6
    active = insight_store.load_insights()["insight:active"]
    dismissed = insight_store.load_insights()["insight:dismissed"]
    assert active.status == "stale"
    assert active.stats["negative_outcome_count"] == 3
    assert active.stats["effectiveness_score"] == pytest.approx(0.0)
    assert dismissed.status == "dismissed"
    assert dismissed.stats["negative_outcome_count"] == 3


def test_capability_blocked_related_unrelated_and_unknown_refs(store, event_log, insight_store):
    insight_store.save_insights({
        "insight:related": _insight("insight:related", capability_name="read_file"),
        "insight:unrelated": _insight("insight:unrelated", capability_name="write_file"),
    })
    _save_workflow(store, "wf-1", ["insight:related", "insight:unrelated", "insight:missing"])
    _emit_workflow_event(
        event_log,
        event_type="capability_blocked",
        workflow_id="wf-1",
        target_id="read_file",
    )

    result = update_insight_feedback(event_log, insight_store, store)

    assert result.feedback_created == 2
    assert "unknown memory ref: insight:missing" in result.errors
    feedback = {item.insight_id: item for item in insight_store.load_feedback()}
    assert feedback["insight:related"].outcome == "negative"
    assert feedback["insight:unrelated"].outcome == "neutral"


def test_feedback_due_helper_and_direct_capability(store, event_log, insight_store, runner, root_agent_id):
    _save_workflow(store, "wf-1", ["insight:read"])
    _emit_workflow_event(event_log, event_type="approval_denied", workflow_id="wf-1")

    due = evaluate_memory_feedback_due(event_log, insight_store, store)
    assert due.due is True
    assert due.relevant_event_count == 1
    assert due.relevant_event_types == ["approval_denied"]

    result = runner.run_capability("memory_feedback_due", {}, root_agent_id)
    assert result.status == "succeeded"
    assert result.output["due"] is True


def test_new_tool_and_capability_registrations():
    tools = default_tool_registry()
    capabilities = default_capability_registry()

    assert tools.is_registered("memory_graph_update") is True
    assert tools.is_registered("memory_curate") is True
    assert tools.is_registered("memory_feedback_update") is True
    assert tools.is_registered("memory_retrieval_update") is True

    feedback_due = capabilities.describe_capability("memory_feedback_due")
    assert feedback_due["direct_allowed"] is True
    assert feedback_due["workflow_allowed"] is True
    assert feedback_due["risk_score"] == 0.05

    feedback_update = capabilities.describe_capability("memory_feedback_update")
    assert feedback_update["direct_allowed"] is False
    assert feedback_update["workflow_allowed"] is True
    assert feedback_update["risk_score"] == 0.25

    retrieval_update = capabilities.describe_capability("memory_retrieval_update")
    assert retrieval_update["direct_allowed"] is False
    assert retrieval_update["workflow_allowed"] is True
    assert retrieval_update["risk_score"] == 0.20


def test_memory_maintenance_spec_shape_is_valid():
    spec = build_memory_maintenance_spec()
    labels = [task["label"] for task in spec["tasks"]]
    assert labels == ["curation_due", "graph_update", "curate", "feedback_due", "feedback_update", "retrieval_update"]
    graph_update = next(task for task in spec["tasks"] if task["label"] == "graph_update")
    feedback_update = next(task for task in spec["tasks"] if task["label"] == "feedback_update")
    retrieval_update = next(task for task in spec["tasks"] if task["label"] == "retrieval_update")
    assert graph_update["depends_on"] == ["curation_due"]
    assert feedback_update["depends_on"] == ["feedback_due"]
    assert retrieval_update["depends_on"] == ["curate", "feedback_update"]
    assert retrieval_update["dependency_policy"] == "any_succeeded"
    assert graph_update["run_if"]["ref"] == "curation_due.result.data.due"
    assert feedback_update["run_if"]["ref"] == "feedback_due.result.data.due"


def test_cli_feedback_and_maintenance_commands(store, event_log, insight_store, capsys):
    insight_store.save_insights({"insight:read": _insight("insight:read")})
    _save_workflow(store, "wf-1", ["insight:read"])
    _emit_workflow_event(event_log, event_type="workflow_completed", workflow_id="wf-1")

    rc = workflow_cli.main(["memory", "feedback", "due", "--json"], store=store)
    assert rc == 0
    due_payload = json.loads(capsys.readouterr().out)
    assert due_payload["due"] is True

    rc = workflow_cli.main(["memory", "feedback", "update", "--json"], store=store)
    assert rc == 0
    update_payload = json.loads(capsys.readouterr().out)
    assert update_payload["feedback_created"] == 1

    rc = workflow_cli.main(["memory", "feedback", "list", "--json"], store=store)
    assert rc == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["insight_id"] == "insight:read"

    rc = workflow_cli.main(["memory", "feedback", "insight", "insight:read", "--json"], store=store)
    assert rc == 0
    by_insight = json.loads(capsys.readouterr().out)
    assert by_insight[0]["feedback_id"].startswith("feedback:")

    rc = workflow_cli.main(["memory", "insights", "effectiveness", "--json"], store=store)
    assert rc == 0
    effectiveness = json.loads(capsys.readouterr().out)
    assert effectiveness[0]["stats"]["positive_outcome_count"] == 1

    rc = workflow_cli.main(["memory", "maintenance", "spec"], store=store)
    assert rc == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["tasks"][0]["label"] == "curation_due"

    rc = workflow_cli.main(["memory", "maintenance", "run", "--json"], store=store)
    assert rc == 0
    run_payload = json.loads(capsys.readouterr().out)
    workflow_id = run_payload["workflow_id"]
    assert workflow_id.startswith("wf-")

    rc = workflow_cli.main(["memory", "maintenance", "status", "--json"], store=store)
    assert rc == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["found"] is True
    assert status_payload["workflow"]["workflow_id"] == workflow_id
