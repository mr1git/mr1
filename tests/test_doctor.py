from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mr1 import workflow_cli
from mr1.capability_policy import (
    CapabilityApprovalRequest,
    CapabilityRequest,
    ScopeContext,
)
from mr1.doctor import (
    DoctorCheckResult,
    DoctorReport,
    create_snapshot,
    filter_doctor_report,
    inspect_snapshot,
    list_snapshots,
    repair_state_file,
    run_doctor,
)
from mr1.event_log import EventLog
from mr1.memory_curator import EvidenceRef, InsightStore, MemoryInsight
from mr1.memory_feedback import InsightFeedback
from mr1.memory_graph import (
    MemoryEdge,
    MemoryGraph,
    MemoryGraphStore,
    MemoryNode,
    edge_id,
)
from mr1.memory_retrieval import RetrievalStore, update_memory_retrieval
from mr1.messages import MessageStore
from mr1.scoped_agents import AgentRecord, AgentStore
from mr1.scheduler import submit_spec_to_disk
from mr1.workflow_models import Provenance
from mr1.workflow_store import WorkflowStore


@pytest.fixture
def runtime_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def workflow_store(runtime_root: Path) -> WorkflowStore:
    return WorkflowStore(root=runtime_root / "workflows")


@pytest.fixture
def agent_store(runtime_root: Path) -> AgentStore:
    return AgentStore(root=runtime_root / "agents")


@pytest.fixture
def message_store(runtime_root: Path, agent_store: AgentStore) -> MessageStore:
    return MessageStore(root=runtime_root / "messages", scoped_agent_store=agent_store)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, payloads: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in payloads:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    return path


def _seed_event_log(runtime_root: Path) -> EventLog:
    log = EventLog(runtime_root / "events")
    log.emit(
        event_type="workflow_created",
        actor_id="cli",
        actor_type="root_orchestrator",
        target_id="wf-1",
        target_type="workflow",
        status="pending",
        summary="workflow created",
        workflow_id="wf-1",
        timestamp="2026-04-29T00:00:00+00:00",
    )
    log.emit(
        event_type="workflow_completed",
        actor_id="scheduler",
        actor_type="scheduler",
        target_id="wf-1",
        target_type="workflow",
        status="succeeded",
        summary="workflow completed",
        workflow_id="wf-1",
        timestamp="2026-04-29T00:05:00+00:00",
    )
    return log


def _seed_graph(runtime_root: Path, *, missing_target: bool = False, cursor: int = 2) -> None:
    graph = MemoryGraph(
        nodes={
            "agent:ag-root": MemoryNode(
                node_id="agent:ag-root",
                node_type="Agent",
                name="MR1",
                created_at="2026-04-29T00:00:00+00:00",
                last_seen_at="2026-04-29T00:05:00+00:00",
            ),
        },
        edges={
            edge_id("agent:ag-root", "USED_CAPABILITY", "capability:read_file"): MemoryEdge(
                edge_id=edge_id("agent:ag-root", "USED_CAPABILITY", "capability:read_file"),
                source_id="agent:ag-root",
                target_id="capability:read_file" if missing_target else "agent:ag-root",
                edge_type="USED_CAPABILITY",
                first_seen_at="2026-04-29T00:00:00+00:00",
                last_seen_at="2026-04-29T00:05:00+00:00",
            ),
        },
    )
    store = MemoryGraphStore(runtime_root / "graph")
    store.save_graph(graph)
    store.save_cursor(cursor)


def _seed_insights(runtime_root: Path, *, confidence: float = 0.8, used_count: int = 1) -> InsightStore:
    store = InsightStore(runtime_root / "insights")
    store.save_insights({
        "insight:test": MemoryInsight(
            insight_id="insight:test",
            insight_type="capability_friction",
            title="Read file friction",
            summary="Read file is noisy.",
            confidence=confidence,
            severity="WARNING",
            recommended_action="consider narrowing scope",
            evidence=[EvidenceRef(source_type="query", source_id="query:test", reason="fixture")],
            related_nodes=["agent:ag-root"],
            created_at="2026-04-29T00:00:00+00:00",
            updated_at="2026-04-29T00:05:00+00:00",
            status="active",
            stats={
                "used_count": used_count,
                "positive_outcome_count": 1,
                "negative_outcome_count": 0,
                "neutral_outcome_count": 0,
                "effectiveness_score": 1.0,
                "last_used_at": None,
                "last_feedback_at": None,
            },
            metadata={"capability_id": "capability:read_file"},
        ),
    })
    store.save_cursor(2)
    store.append_run(
        __import__("mr1.memory_curator", fromlist=["MemoryCurationRun"]).MemoryCurationRun(
            run_id="curation:1",
            started_at="2026-04-29T00:01:00+00:00",
            ended_at="2026-04-29T00:02:00+00:00",
            event_start_index=1,
            event_end_index=2,
            trigger_reason="test",
            status="succeeded",
            input_summary={"events": 2},
            output_insight_ids=["insight:test"],
            errors=[],
        )
    )
    return store


def _seed_feedback(runtime_root: Path, *, duplicate: bool = False) -> None:
    store = InsightStore(runtime_root / "insights")
    item = InsightFeedback(
        feedback_id="feedback:1",
        insight_id="insight:test",
        workflow_id="wf-1",
        event_id="evt-1",
        event_index=2,
        outcome="positive",
        reason="worked",
        confidence_delta=0.03,
        evaluator_type="deterministic",
        created_at="2026-04-29T00:06:00+00:00",
        metadata={},
    )
    store.append_feedback(item)
    if duplicate:
        store.append_feedback(item)
    store.save_feedback_cursor(2)


def _seed_workflow(
    workflow_store: WorkflowStore,
    agent_store: AgentStore,
    owner_agent_id: str,
    *,
    title: str = "Doctor workflow",
    metadata: dict | None = None,
) -> str:
    return submit_spec_to_disk(
        {
            "title": title,
            "tasks": [
                {
                    "label": "a",
                    "title": "A",
                    "task_kind": "agent",
                    "agent_type": "worker",
                    "prompt": "x",
                }
            ],
        },
        Provenance(type="user", id="cli"),
        workflow_store,
        owner_agent_id=owner_agent_id,
        scoped_agent_store=agent_store,
        workflow_metadata=metadata,
    )


def _seed_pending_approval(runtime_root: Path, agent_store: AgentStore) -> str:
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child", security_clearance=0.1)
    approval = CapabilityApprovalRequest(
        approval_request_id="cap_approval_test",
        requesting_actor_id=child.agent_id,
        capability_name="read_file",
        invocation_mode="direct",
        args={"path": str(runtime_root / "secret.txt")},
        risk_score=0.1,
        reason="needs access",
        scope_summary={"scope_roots": []},
        original_request=CapabilityRequest(
            actor_id=child.agent_id,
            actor_type="orchestrator",
            actor_clearance=child.security_clearance,
            invocation_mode="direct",
            capability_name="read_file",
            args={"path": str(runtime_root / "secret.txt")},
            scope=ScopeContext(allowed_roots=[], workspace_root=runtime_root),
        ),
        designated_approver_id=root.agent_id,
        created_at="2026-04-29T00:00:00+00:00",
    )
    _write_json(runtime_root / "capability_approvals" / "cap_approval_test.json", approval.to_dict())
    return child.agent_id


def test_doctor_report_rollup_and_filtering():
    report = DoctorReport(
        status="error",
        generated_at="2026-04-30T00:00:00+00:00",
        checks=[
            DoctorCheckResult("a", "runtime", "ok", "A", "ok", {}, []),
            DoctorCheckResult("b", "events", "warning", "B", "warn", {}, ["do x"]),
            DoctorCheckResult("c", "memory", "error", "C", "err", {}, ["do y"]),
        ],
        summary={
            "checks_run": 3,
            "ok_count": 1,
            "warning_count": 1,
            "error_count": 1,
            "categories": ["runtime", "events", "memory"],
            "latest_event_index": 2,
        },
    )

    filtered = filter_doctor_report(report, errors_only=True)

    assert filtered.status == "error"
    assert [item.check_id for item in filtered.checks] == ["b", "c"]
    assert filtered.summary["checks_run"] == 2
    assert json.loads(json.dumps(report.to_dict()))["status"] == "error"


def test_doctor_category_filtering(runtime_root: Path):
    _seed_event_log(runtime_root)

    report = run_doctor(runtime_root, categories=["events"])

    assert report.summary["categories"] == ["events"]
    assert [item.category for item in report.checks] == ["events"]


def test_doctor_valid_event_log_is_ok(runtime_root: Path):
    _seed_event_log(runtime_root)

    report = run_doctor(runtime_root, categories=["events"])

    assert report.status == "ok"
    assert report.summary["latest_event_index"] == 2


def test_doctor_event_log_corrupt_json_is_error(runtime_root: Path):
    (runtime_root / "events").mkdir(parents=True, exist_ok=True)
    (runtime_root / "events" / "events.jsonl").write_text("{bad json\n", encoding="utf-8")

    report = run_doctor(runtime_root, categories=["events"])

    assert report.status == "error"
    assert "corruption" in report.checks[0].summary


def test_doctor_event_log_non_monotonic_duplicate_and_dangling_errors(runtime_root: Path):
    _write_jsonl(
        runtime_root / "events" / "events.jsonl",
        [
            {
                "event_id": "evt-1",
                "event_index": 2,
                "event_version": 1,
                "timestamp": "2026-04-29T00:00:00+00:00",
                "event_type": "workflow_created",
                "event_kind": "lifecycle",
                "status": "pending",
                "severity": "INFO",
                "summary": "created",
                "metadata": {},
            },
            {
                "event_id": "evt-1",
                "event_index": 1,
                "event_version": 1,
                "timestamp": "2026-04-29T00:01:00+00:00",
                "event_type": "workflow_completed",
                "event_kind": "lifecycle",
                "status": "succeeded",
                "severity": "INFO",
                "summary": "done",
                "parent_event_id": "evt-missing",
                "metadata": {},
            },
        ],
    )

    report = run_doctor(runtime_root, categories=["events"])

    assert report.status == "error"
    details = report.checks[0].details
    assert any("duplicate event_id" in item for item in details["errors"])
    assert any("non-monotonic" in item for item in details["errors"])
    assert any("dangling parent_event_id" in item for item in details["errors"])


def test_doctor_unknown_event_values_are_warning(runtime_root: Path):
    _write_jsonl(
        runtime_root / "events" / "events.jsonl",
        [
            {
                "event_id": "evt-1",
                "event_index": 1,
                "event_version": 9,
                "timestamp": "2026-04-29T00:00:00+00:00",
                "event_type": "unknown_future_event",
                "event_kind": "future_kind",
                "status": "pending",
                "severity": "TRACE",
                "summary": "future",
                "metadata": {},
            }
        ],
    )

    report = run_doctor(runtime_root, categories=["events"])

    assert report.status == "warning"
    assert report.checks[0].details["warnings"]


def test_doctor_graph_stale_cursor_is_warning(runtime_root: Path):
    _seed_event_log(runtime_root)
    _seed_graph(runtime_root, cursor=1)

    report = run_doctor(runtime_root, categories=["memory"])

    graph_check = next(item for item in report.checks if item.check_id == "memory.graph")
    assert graph_check.status == "warning"


def test_doctor_graph_missing_node_is_error(runtime_root: Path):
    _seed_event_log(runtime_root)
    _seed_graph(runtime_root, missing_target=True, cursor=2)

    report = run_doctor(runtime_root, categories=["memory"])

    graph_check = next(item for item in report.checks if item.check_id == "memory.graph")
    assert graph_check.status == "error"


def test_doctor_invalid_insight_confidence_is_error(runtime_root: Path):
    _seed_event_log(runtime_root)
    _write_json(
        runtime_root / "insights" / "insights.json",
        {
            "insight:test": {
                "insight_id": "insight:test",
                "insight_type": "capability_friction",
                "title": "bad",
                "summary": "bad",
                "confidence": 2.0,
                "severity": "WARNING",
                "recommended_action": "consider",
                "evidence": [{"source_type": "query", "source_id": "query:test", "reason": "fixture"}],
                "related_nodes": ["agent:ag-root"],
                "created_at": "2026-04-29T00:00:00+00:00",
                "updated_at": "2026-04-29T00:01:00+00:00",
                "status": "active",
                "stats": {},
                "metadata": {},
            }
        },
    )
    _write_json(runtime_root / "insights" / "cursor.json", {"last_curated_event_index": 1})

    report = run_doctor(runtime_root, categories=["memory"])

    insights_check = next(item for item in report.checks if item.check_id == "memory.insights")
    assert insights_check.status == "error"


def test_doctor_feedback_stats_mismatch_is_warning(runtime_root: Path):
    _seed_event_log(runtime_root)
    _seed_insights(runtime_root, used_count=5)
    _seed_feedback(runtime_root)

    report = run_doctor(runtime_root, categories=["memory"])

    feedback_check = next(item for item in report.checks if item.check_id == "memory.feedback")
    assert feedback_check.status == "warning"
    assert any("used_count" in item for item in feedback_check.details["warnings"])


def test_doctor_duplicate_feedback_ids_are_error(runtime_root: Path):
    _seed_event_log(runtime_root)
    _seed_insights(runtime_root)
    _seed_feedback(runtime_root, duplicate=True)

    report = run_doctor(runtime_root, categories=["memory"])

    feedback_check = next(item for item in report.checks if item.check_id == "memory.feedback")
    assert feedback_check.status == "error"


def test_doctor_retrieval_manifest_mismatch_is_warning(runtime_root: Path):
    _seed_event_log(runtime_root)
    _seed_graph(runtime_root)
    _seed_insights(runtime_root)
    update_memory_retrieval(runtime_root)
    manifest = RetrievalStore(runtime_root).load_manifest()
    manifest["document_count"] = 999
    _write_json(runtime_root / "retrieval" / "manifest.json", manifest)

    report = run_doctor(runtime_root, categories=["memory"])

    retrieval_check = next(item for item in report.checks if item.check_id == "memory.retrieval")
    assert retrieval_check.status == "warning"


def test_doctor_missing_memory_maintenance_workflow_is_warning(runtime_root: Path):
    _seed_event_log(runtime_root)

    report = run_doctor(runtime_root, categories=["memory"])

    maintenance_check = next(item for item in report.checks if item.check_id == "memory.maintenance_workflow")
    assert maintenance_check.status == "warning"


def test_doctor_valid_agent_tree_is_ok(runtime_root: Path, agent_store: AgentStore):
    root = agent_store.ensure_root_agent()
    agent_store.create_child_agent(root.agent_id, "child")

    report = run_doctor(runtime_root, categories=["agents"])

    assert report.status == "ok"


def test_doctor_agent_missing_parent_clearance_and_cycle_are_errors(runtime_root: Path, agent_store: AgentStore):
    root = agent_store.ensure_root_agent()
    child = agent_store.create_child_agent(root.agent_id, "child", security_clearance=0.5)
    payload = child.to_dict()
    payload["parent_agent_id"] = "ag-missing"
    payload["security_clearance"] = 2.0
    _write_json(runtime_root / "agents" / f"{child.agent_id}.json", payload)
    cyclic = AgentRecord(
        agent_id="ag-cycle",
        role="orchestrator",
        title="cycle",
        mr_level=2,
        parent_agent_id="ag-cycle",
        security_clearance=0.1,
    )
    _write_json(runtime_root / "agents" / "ag-cycle.json", cyclic.to_dict())

    report = run_doctor(runtime_root, categories=["agents"])

    assert report.status == "error"


def test_doctor_capabilities_detect_invalid_metadata(monkeypatch, runtime_root: Path):
    class FakeRegistry:
        def describe_all(self) -> list[dict]:
            return [{
                "name": "memory_search",
                "kind": "tool",
                "effect": "read",
                "risk_score": 2.0,
                "direct_allowed": True,
                "workflow_allowed": True,
                "requires_scope": False,
                "blocking": False,
                "is_filesystem": False,
                "is_network": False,
                "is_execution": False,
                "is_blocking": False,
                "path_arg_fields": [],
            }]

    monkeypatch.setattr("mr1.doctor.default_capability_registry", lambda: FakeRegistry())

    report = run_doctor(runtime_root, categories=["capabilities"])

    assert report.status == "error"


def test_doctor_pending_approval_and_unread_message_are_warning(
    runtime_root: Path,
    agent_store: AgentStore,
    message_store: MessageStore,
):
    requester_id = _seed_pending_approval(runtime_root, agent_store)
    root = agent_store.ensure_root_agent()
    message_store.create_message(
        from_agent_id=requester_id,
        to_agent_id=root.agent_id,
        kind="request",
        subject="Need approval",
        body="Please approve.",
    )

    report = run_doctor(runtime_root, categories=["approvals", "messages"])

    assert report.status == "warning"
    assert any(item.category == "approvals" and item.status == "warning" for item in report.checks)
    assert any(item.category == "messages" and item.status == "warning" for item in report.checks)


def test_doctor_malformed_message_file_is_error(runtime_root: Path):
    (runtime_root / "messages").mkdir(parents=True, exist_ok=True)
    (runtime_root / "messages" / "msg-bad.json").write_text("{bad json", encoding="utf-8")

    report = run_doctor(runtime_root, categories=["messages"])

    assert report.status == "error"


def test_snapshot_create_list_and_inspect(runtime_root: Path):
    _seed_event_log(runtime_root)
    _seed_graph(runtime_root)
    _seed_insights(runtime_root)
    update_memory_retrieval(runtime_root)

    manifest = create_snapshot(
        runtime_root,
        now=datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc),
    )
    listed = list_snapshots(runtime_root)
    inspected = inspect_snapshot(runtime_root, manifest["snapshot_id"])

    assert manifest["snapshot_id"] == "snapshot_20260430T120000Z"
    assert listed[0]["snapshot_id"] == manifest["snapshot_id"]
    assert inspected["included_paths"]
    assert (runtime_root / "snapshots" / manifest["snapshot_id"] / "events" / "events.jsonl").exists()


def test_snapshot_missing_optional_stores_and_duplicate_id(runtime_root: Path):
    _seed_event_log(runtime_root)

    create_snapshot(
        runtime_root,
        now=datetime(2026, 4, 30, 12, 1, 0, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError, match="snapshot already exists"):
        create_snapshot(
            runtime_root,
            now=datetime(2026, 4, 30, 12, 1, 0, tzinfo=timezone.utc),
        )


def test_snapshot_fail_on_error_blocks_creation(runtime_root: Path):
    (runtime_root / "events").mkdir(parents=True, exist_ok=True)
    (runtime_root / "events" / "events.jsonl").write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="doctor reported error status"):
        create_snapshot(
            runtime_root,
            fail_on_error=True,
            now=datetime(2026, 4, 30, 12, 2, 0, tzinfo=timezone.utc),
        )


def test_snapshot_incomplete_is_not_listed(runtime_root: Path):
    snapshots_root = runtime_root / "snapshots"
    incomplete = snapshots_root / ".incomplete_snapshot_20260430T120300Z"
    incomplete.mkdir(parents=True)
    _write_json(incomplete / "snapshot_manifest.json", {"snapshot_id": "ignored"})

    assert list_snapshots(runtime_root) == []


def test_doctor_cli_variants(runtime_root: Path, workflow_store: WorkflowStore, capsys):
    _seed_event_log(runtime_root)
    _seed_graph(runtime_root, cursor=1)

    rc = workflow_cli.main(["doctor"], store=workflow_store)
    assert rc == 0
    assert "STATUS" in capsys.readouterr().out

    rc = workflow_cli.main(["doctor", "--json"], store=workflow_store)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["latest_event_index"] == 2

    rc = workflow_cli.main(["doctor", "--category", "events"], store=workflow_store)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Event timeline" in out
    assert "Graph memory" not in out

    rc = workflow_cli.main(["doctor", "--errors-only"], store=workflow_store)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Graph memory" in out


def test_snapshot_cli_variants(runtime_root: Path, workflow_store: WorkflowStore, capsys):
    _seed_event_log(runtime_root)

    rc = workflow_cli.main(["snapshot", "create"], store=workflow_store)
    assert rc == 0
    created = capsys.readouterr().out
    assert "snapshot_id:" in created

    rc = workflow_cli.main(["snapshot", "list"], store=workflow_store)
    assert rc == 0
    listed = capsys.readouterr().out
    assert "SNAPSHOT_ID" in listed

    snapshot_id = list_snapshots(runtime_root)[0]["snapshot_id"]
    rc = workflow_cli.main(["snapshot", "inspect", snapshot_id], store=workflow_store)
    assert rc == 0
    assert snapshot_id in capsys.readouterr().out

    rc = workflow_cli.main(["snapshot", "inspect", snapshot_id, "--json"], store=workflow_store)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["snapshot_id"] == snapshot_id


def test_doctor_workflow_maintenance_presence_and_warning(runtime_root: Path, workflow_store: WorkflowStore, agent_store: AgentStore):
    root = agent_store.ensure_root_agent()
    _seed_workflow(workflow_store, agent_store, root.agent_id, metadata={"system_workflow": "memory_maintenance"})

    report = run_doctor(runtime_root, categories=["memory", "workflows"])

    maintenance_check = next(item for item in report.checks if item.check_id == "memory.maintenance_workflow")
    assert maintenance_check.status == "ok"


# ---------------------------------------------------------------------------
# N-7: State repair
# ---------------------------------------------------------------------------

class TestRepairStateFile:
    """N-7: corrupt state files must be quarantineable via an explicit operator command."""

    def test_repair_quarantines_corrupt_json(self, tmp_path):
        state = tmp_path / "mr1_state.json"
        state.write_text("{bad json!!!", encoding="utf-8")

        result = repair_state_file(state)

        assert result["action"] == "quarantined"
        assert not state.exists(), "original path must be gone after quarantine"
        quarantine = Path(result["quarantined_path"])
        assert quarantine.exists(), "quarantined file must be preserved"
        assert quarantine.read_text() == "{bad json!!!"

    def test_repair_quarantines_non_dict_json(self, tmp_path):
        state = tmp_path / "mr1_state.json"
        state.write_text("[1, 2, 3]", encoding="utf-8")

        result = repair_state_file(state)

        assert result["action"] == "quarantined"
        assert not state.exists()
        assert Path(result["quarantined_path"]).exists()

    def test_repair_raises_on_valid_state_file(self, tmp_path):
        state = tmp_path / "mr1_state.json"
        state.write_text('{"session_id": "abc"}', encoding="utf-8")

        with pytest.raises(ValueError, match="structurally valid"):
            repair_state_file(state)

        assert state.exists(), "valid file must not be modified"

    def test_repair_raises_on_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            repair_state_file(tmp_path / "nonexistent.json")

    def test_repair_quarantine_path_contains_timestamp(self, tmp_path):
        state = tmp_path / "mr1_state.json"
        state.write_text("{corrupt", encoding="utf-8")

        result = repair_state_file(state)

        quarantine_name = Path(result["quarantined_path"]).name
        assert "bad" in quarantine_name
        assert quarantine_name.startswith("mr1_state")

    def test_statemanager_starts_fresh_after_repair(self, tmp_path):
        from mr1.orchestrator.state import StateManager

        state_path = tmp_path / "active" / "mr1_state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{corrupt json", encoding="utf-8")

        repair_state_file(state_path)
        assert not state_path.exists()

        # StateManager should now initialise a fresh state rather than raising.
        sm = StateManager(state_path=state_path)
        assert sm.session_id is not None

    def test_repair_cli_command_quarantines_file(self, tmp_path, capsys):
        state = tmp_path / "mr1_state.json"
        state.write_text("{bad", encoding="utf-8")

        from mr1.workflow_store import WorkflowStore
        from mr1.scoped_agents import AgentStore
        from mr1.cli.memory import _cmd_repair_state
        import argparse

        # Build a minimal args namespace mirroring what the real parser produces.
        args = argparse.Namespace(state_path=str(state), json=False)
        store = WorkflowStore(root=tmp_path / "workflows")
        sa = AgentStore(root=tmp_path / "agents")
        rc = _cmd_repair_state(args, store, "test-actor", sa)

        assert rc == 0
        out = capsys.readouterr().out
        assert "quarantined" in out.lower()
        assert not state.exists()

    def test_repair_cli_returns_error_on_valid_file(self, tmp_path, capsys):
        state = tmp_path / "mr1_state.json"
        state.write_text('{"session_id": "ok"}', encoding="utf-8")

        from mr1.workflow_store import WorkflowStore
        from mr1.scoped_agents import AgentStore
        from mr1.cli.memory import _cmd_repair_state
        import argparse
        import sys

        args = argparse.Namespace(state_path=str(state), json=False)
        store = WorkflowStore(root=tmp_path / "workflows")
        sa = AgentStore(root=tmp_path / "agents")
        rc = _cmd_repair_state(args, store, "test-actor", sa)

        assert rc == 2
        assert state.exists()
