from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from mr1 import workflow_cli
from mr1.event_log import EventLog
from mr1.memory_curator import (
    InsightStore,
    EvidenceRef,
    MemoryCurationBundle,
    MemoryCuratorClient,
    MemoryCuratorFailure,
    MemoryInsight,
    build_memory_curation_bundle,
    evaluate_memory_curation_due,
    run_memory_curation,
)
from mr1.memory_graph import MemoryGraphStore, capability_node_id
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def store(tmp_path):
    return WorkflowStore(root=tmp_path / "workflows")


@pytest.fixture
def event_log(tmp_path):
    return EventLog(tmp_path / "events")


@pytest.fixture
def graph_store(tmp_path):
    return MemoryGraphStore(tmp_path / "graph")


@pytest.fixture
def insight_store(tmp_path):
    return InsightStore(tmp_path / "insights")


class FakeCurator:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, prompt: str) -> str:
        self.prompts.append((system_prompt, prompt))
        if not self._responses:
            raise AssertionError("no curator responses configured")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _important_capability_failure(event_log: EventLog) -> str:
    requested = event_log.emit(
        event_type="capability_requested",
        actor_id="ag-1",
        actor_type="orchestrator",
        target_id="read_file",
        target_type="capability",
        status="requested",
        summary="capability requested: read_file",
    )
    failed = event_log.emit(
        event_type="capability_failed",
        actor_id="ag-1",
        actor_type="orchestrator",
        target_id="read_file",
        target_type="capability",
        status="failed",
        summary="capability failed: read_file",
        parent_event_id=requested.event_id,
        metadata={"error_type": "missing_file", "reason": "path missing"},
    )
    return failed.event_id


def _valid_curator_output(event_id: str) -> str:
    return json.dumps({
        "insights": [
            {
                "insight_type": "capability_friction",
                "title": "Read-file capability fails on missing inputs",
                "summary": "Repeated read failures suggest the planner should verify file presence earlier.",
                "confidence": 0.82,
                "severity": "WARNING",
                "recommended_action": "consider adding a preflight existence check before read_file calls",
                "evidence": [
                    {
                        "source_type": "event",
                        "source_id": event_id,
                        "reason": "Recent failure shows the friction directly.",
                    }
                ],
                "related_nodes": [capability_node_id("read_file")],
                "metadata": {"capability_id": "read_file"},
            }
        ],
        "stale_insight_ids": [],
    })


def test_due_check_false_without_important_events(event_log: EventLog, insight_store: InsightStore):
    event_log.emit(
        event_type="capability_requested",
        actor_id="ag-1",
        actor_type="orchestrator",
        target_id="read_file",
        target_type="capability",
        status="requested",
        summary="capability requested: read_file",
    )

    result = evaluate_memory_curation_due(event_log, insight_store)

    assert result.due is False
    assert result.important_event_count == 0
    assert result.important_event_types == []


def test_due_check_true_with_important_events_after_cursor(event_log: EventLog, insight_store: InsightStore):
    _important_capability_failure(event_log)
    insight_store.save_cursor(1)

    result = evaluate_memory_curation_due(event_log, insight_store)

    assert result.due is True
    assert result.latest_event_index == 2
    assert result.last_curated_event_index == 1
    assert result.important_event_count == 1
    assert result.important_event_types == ["capability_failed"]
    assert result.suggested_event_window == [2, 2]


def test_bundle_construction_is_compact_and_includes_related_insights(
    event_log: EventLog,
    graph_store: MemoryGraphStore,
    insight_store: InsightStore,
):
    event_id = _important_capability_failure(event_log)
    insight_store.save_insights({
        "insight:capability_friction:capability:read_file": MemoryInsight(
            insight_id="insight:capability_friction:capability:read_file",
            insight_type="capability_friction",
            title="Existing friction",
            summary="Existing summary",
            confidence=0.5,
            severity="WARNING",
            recommended_action="consider preflight checks",
            evidence=[
                EvidenceRef(
                    source_type="query",
                    source_id="query:capability_stats",
                    reason="Existing aggregate evidence",
                )
            ],
            related_nodes=[capability_node_id("read_file")],
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            status="active",
            metadata={"capability_id": "capability:read_file"},
        ),
    })
    due = evaluate_memory_curation_due(event_log, insight_store)

    bundle, input_summary = build_memory_curation_bundle(
        event_log=event_log,
        graph_store=graph_store,
        insight_store=insight_store,
        due_result=due,
    )

    dumped = json.dumps(bundle.to_dict(), sort_keys=True)
    assert bundle.important_events[0]["event_id"] == event_id
    assert "graph_stats" in bundle.to_dict()
    assert input_summary["bundle_counts"]["important_events"] == 1
    assert bundle.existing_related_insights[0]["insight_id"] == "insight:capability_friction:capability:read_file"
    assert "record_path" not in dumped
    assert "metadata" not in bundle.important_events[0]


def test_memory_curator_client_retries_invalid_top_level_shape():
    compiler = FakeCurator(
        json.dumps({"bad": True}),
        json.dumps({"insights": [], "stale_insight_ids": []}),
    )
    client = MemoryCuratorClient(compiler=compiler)

    result = client.curate(MemoryCurationBundle(
        event_start_index=1,
        event_end_index=2,
        important_events=[],
        graph_stats={},
        top_workflow_templates=[],
        capability_stats=[],
        failure_modes=[],
        approval_history=[],
        existing_related_insights=[],
    ))

    assert result.insights == []
    assert len(compiler.prompts) == 2


def test_run_not_due_does_not_call_llm(event_log: EventLog, graph_store: MemoryGraphStore, insight_store: InsightStore):
    client = MemoryCuratorClient(compiler=FakeCurator(AssertionError("should not run")))

    run = run_memory_curation(
        event_log=event_log,
        graph_store=graph_store,
        insight_store=insight_store,
        client=client,
        trigger_reason="test",
        persist_not_due=True,
    )

    assert run.status == "not_due"
    assert insight_store.load_cursor() == 0
    assert insight_store.load_runs(limit=1)[0].status == "not_due"


def test_successful_run_updates_cursor_and_persists_insight(
    event_log: EventLog,
    graph_store: MemoryGraphStore,
    insight_store: InsightStore,
):
    event_id = _important_capability_failure(event_log)
    client = MemoryCuratorClient(compiler=FakeCurator(_valid_curator_output(event_id)))

    run = run_memory_curation(
        event_log=event_log,
        graph_store=graph_store,
        insight_store=insight_store,
        client=client,
        trigger_reason="test",
    )

    assert run.status == "succeeded"
    assert insight_store.load_cursor() == 2
    assert run.output_insight_ids == ["insight:capability_friction:capability:read_file"]
    stored = insight_store.load_insights()["insight:capability_friction:capability:read_file"]
    assert stored.created_at == stored.updated_at


def test_failed_llm_run_keeps_insight_cursor_and_retains_graph_update(
    event_log: EventLog,
    graph_store: MemoryGraphStore,
    insight_store: InsightStore,
):
    _important_capability_failure(event_log)
    client = MemoryCuratorClient(compiler=FakeCurator(
        MemoryCuratorFailure("bad"),
    ))

    run = run_memory_curation(
        event_log=event_log,
        graph_store=graph_store,
        insight_store=insight_store,
        client=client,
        trigger_reason="test",
    )

    assert run.status == "failed"
    assert insight_store.load_cursor() == 0
    assert graph_store.load_cursor() == 2


def test_partial_rejection_logs_errors_but_persists_valid_insight(
    event_log: EventLog,
    graph_store: MemoryGraphStore,
    insight_store: InsightStore,
):
    event_id = _important_capability_failure(event_log)
    client = MemoryCuratorClient(compiler=FakeCurator(json.dumps({
        "insights": [
            json.loads(_valid_curator_output(event_id))["insights"][0],
            {
                "insight_type": "capability_friction",
                "title": "Invalid friction",
                "summary": "This one should fail validation.",
                "confidence": 0.6,
                "severity": "BAD",
                "recommended_action": "consider reviewing logs",
                "evidence": [
                    {"source_type": "event", "source_id": event_id, "reason": "same event"}
                ],
                "related_nodes": [capability_node_id("read_file")],
                "metadata": {"capability_id": "read_file"},
            },
        ],
        "stale_insight_ids": [],
    })))

    run = run_memory_curation(
        event_log=event_log,
        graph_store=graph_store,
        insight_store=insight_store,
        client=client,
        trigger_reason="test",
    )

    assert run.status == "succeeded"
    assert any("candidate[1]" in item for item in run.errors)
    assert "insight:capability_friction:capability:read_file" in insight_store.load_insights()


def test_upsert_preserves_created_at_and_dismissed_status(
    event_log: EventLog,
    graph_store: MemoryGraphStore,
    insight_store: InsightStore,
):
    insight_store.save_insights({
        "insight:capability_friction:capability:read_file": MemoryInsight(
            insight_id="insight:capability_friction:capability:read_file",
            insight_type="capability_friction",
            title="Old title",
            summary="Old summary",
            confidence=0.2,
            severity="INFO",
            recommended_action="consider old action",
            evidence=[
                EvidenceRef(
                    source_type="query",
                    source_id="query:capability_stats",
                    reason="Old query evidence",
                )
            ],
            related_nodes=[capability_node_id("read_file")],
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            status="dismissed",
            metadata={"capability_id": "capability:read_file"},
        ),
    })
    event_id = _important_capability_failure(event_log)
    client = MemoryCuratorClient(compiler=FakeCurator(_valid_curator_output(event_id)))

    run_memory_curation(
        event_log=event_log,
        graph_store=graph_store,
        insight_store=insight_store,
        client=client,
        trigger_reason="test",
    )

    stored = insight_store.load_insights()["insight:capability_friction:capability:read_file"]
    assert stored.created_at == "2026-01-01T00:00:00+00:00"
    assert stored.status == "dismissed"


def test_cli_memory_curator_commands(store: WorkflowStore, capsys):
    runtime_root = store.root.parent
    event_log = EventLog(runtime_root / "events")
    event_id = _important_capability_failure(event_log)

    rc = workflow_cli.main(["memory", "curation-due", "--json"], store=store)
    assert rc == 0
    due_payload = json.loads(capsys.readouterr().out)
    assert due_payload["due"] is True

    with patch("mr1.memory_curator.run_memory_curator_agent", return_value=_valid_curator_output(event_id)):
        rc = workflow_cli.main(["memory", "curate", "--json"], store=store)
    assert rc == 0
    curate_payload = json.loads(capsys.readouterr().out)
    assert curate_payload["status"] == "succeeded"

    rc = workflow_cli.main(["memory", "insights", "list", "--json"], store=store)
    assert rc == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["insight_type"] == "capability_friction"

    insight_id = listed[0]["insight_id"]
    rc = workflow_cli.main(["memory", "insights", "show", insight_id, "--json"], store=store)
    assert rc == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["insight_id"] == insight_id

    rc = workflow_cli.main(["memory", "insights", "friction", "--json"], store=store)
    assert rc == 0
    friction = json.loads(capsys.readouterr().out)
    assert len(friction) == 1

    rc = workflow_cli.main(["memory", "insights", "recommendations", "--json"], store=store)
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == []

    rc = workflow_cli.main(["memory", "insights", "failures", "--json"], store=store)
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == []

    rc = workflow_cli.main(["memory", "curation-runs", "--json"], store=store)
    assert rc == 0
    runs = json.loads(capsys.readouterr().out)
    assert runs[0]["status"] == "succeeded"
