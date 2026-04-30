"""
Deterministic runtime diagnostics and snapshot helpers.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from mr1.capabilities import default_capability_registry
from mr1.capability_policy import CapabilityApprovalDecision, CapabilityApprovalRequest, CapabilityMetadata
from mr1.event_log import (
    EVENT_KIND_BY_TYPE,
    EVENT_KIND_VALUES,
    EVENT_VERSION,
    SEVERITY_BY_TYPE,
    SEVERITY_VALUES,
)
from mr1.memory_curator import (
    EvidenceRef,
    InsightStore,
    MemoryCurationRun,
    MemoryInsight,
)
from mr1.memory_feedback import InsightFeedback
from mr1.memory_graph import EDGE_TYPES, NODE_TYPES, MemoryGraphStore
from mr1.memory_queries import MEMORY_CAPABILITY_NAMES
from mr1.memory_retrieval import LexicalMemoryRetriever, RETRIEVAL_DOC_TYPES, RetrievalStore
from mr1.messages import PersistentMessage
from mr1.scoped_agents import PersistentAgent
from mr1.capability_policy import normalize_path
from mr1.workflow_models import TaskStatus, Workflow, WorkflowStatus
from mr1.workflow_store import WorkflowStore


DOCTOR_CATEGORIES = (
    "runtime",
    "events",
    "memory",
    "workflows",
    "agents",
    "capabilities",
    "approvals",
    "messages",
)
_EXPECTED_RUNTIME_PATHS = (
    "events",
    "graph",
    "insights",
    "retrieval",
    "agents",
    "workflows",
    "messages",
    "capability_approvals",
)
_LEGACY_RUNTIME_PATHS = frozenset({"rag", "dumps", "active"})
_SNAPSHOT_INCLUDED_PATHS = (
    "events",
    "graph",
    "insights",
    "retrieval",
    "agents",
    "workflows",
    "messages",
    "capability_approvals",
)
_SNAPSHOT_SKIP_PATTERNS = ("__pycache__", ".DS_Store", "*.tmp", "*.lock")
_REQUIRED_MAINTENANCE_CAPABILITIES = (
    "memory_curation_due",
    "memory_feedback_due",
    "memory_graph_update",
    "memory_curate",
    "memory_feedback_update",
    "memory_retrieval_update",
)
_STUCK_RUNNING_THRESHOLD = timedelta(hours=24)


@dataclass(frozen=True)
class DoctorCheckResult:
    check_id: str
    category: str
    status: str
    title: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "status": self.status,
            "title": self.title,
            "summary": self.summary,
            "details": deepcopy(self.details),
            "recommendations": list(self.recommendations),
        }


@dataclass(frozen=True)
class DoctorReport:
    status: str
    generated_at: str
    checks: list[DoctorCheckResult]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generated_at": self.generated_at,
            "checks": [item.to_dict() for item in self.checks],
            "summary": deepcopy(self.summary),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _ensure_utc(parsed)


def _status_rollup(checks: Iterable[DoctorCheckResult]) -> str:
    statuses = {item.status for item in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"


def _build_summary(
    checks: list[DoctorCheckResult],
    *,
    categories: list[str],
    latest_event_index: int,
) -> dict[str, Any]:
    return {
        "checks_run": len(checks),
        "ok_count": sum(1 for item in checks if item.status == "ok"),
        "warning_count": sum(1 for item in checks if item.status == "warning"),
        "error_count": sum(1 for item in checks if item.status == "error"),
        "categories": list(categories),
        "latest_event_index": int(latest_event_index),
    }


def _report_from_checks(
    checks: list[DoctorCheckResult],
    *,
    generated_at: Optional[str] = None,
    categories: Optional[list[str]] = None,
    latest_event_index: int = 0,
) -> DoctorReport:
    ordered_categories = list(categories or sorted({item.category for item in checks}))
    return DoctorReport(
        status=_status_rollup(checks),
        generated_at=generated_at or _now_iso(),
        checks=list(checks),
        summary=_build_summary(
            checks,
            categories=ordered_categories,
            latest_event_index=latest_event_index,
        ),
    )


def filter_doctor_report(
    report: DoctorReport,
    *,
    errors_only: bool = False,
) -> DoctorReport:
    checks = list(report.checks)
    if errors_only:
        checks = [item for item in checks if item.status in {"warning", "error"}]
    return _report_from_checks(
        checks,
        generated_at=report.generated_at,
        categories=list(report.summary.get("categories", [])),
        latest_event_index=int(report.summary.get("latest_event_index", 0)),
    )


def _check(
    *,
    check_id: str,
    category: str,
    status: str,
    title: str,
    summary: str,
    details: Optional[dict[str, Any]] = None,
    recommendations: Optional[list[str]] = None,
) -> DoctorCheckResult:
    return DoctorCheckResult(
        check_id=check_id,
        category=category,
        status=status,
        title=title,
        summary=summary,
        details=dict(details or {}),
        recommendations=list(recommendations or []),
    )


def _load_json_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl_payloads(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: payload must be a JSON object")
            items.append(payload)
    return items


def _recommend_timeline_workflow(workflow_ids: Iterable[str]) -> list[str]:
    recommendations: list[str] = []
    for workflow_id in sorted({item for item in workflow_ids if item}):
        recommendations.append(f"Run: python -m mr1.workflow_cli timeline workflow {workflow_id}")
    return recommendations


def _workflow_files(store_root: Path) -> list[Path]:
    workflow_root = store_root / "workflows"
    if not workflow_root.exists() or not workflow_root.is_dir():
        return []
    return sorted(
        path / "workflow.json"
        for path in workflow_root.iterdir()
        if path.is_dir() and (path / "workflow.json").exists()
    )


def _load_workflows_direct(runtime_root: Path) -> tuple[list[Workflow], list[str]]:
    workflows: list[Workflow] = []
    errors: list[str] = []
    for path in _workflow_files(runtime_root):
        try:
            payload = _load_json_file(path)
            if not isinstance(payload, dict):
                raise ValueError("workflow payload must be an object")
            workflows.append(Workflow.from_dict(payload))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return workflows, errors


def _load_agents_direct(runtime_root: Path) -> tuple[dict[str, PersistentAgent], list[str]]:
    root = runtime_root / "agents"
    agents: dict[str, PersistentAgent] = {}
    errors: list[str] = []
    if not root.exists():
        return agents, errors
    for path in sorted(root.glob("ag-*.json")):
        try:
            payload = _load_json_file(path)
            if not isinstance(payload, dict):
                raise ValueError("agent payload must be an object")
            agent = PersistentAgent.from_dict(payload)
            agents[agent.agent_id] = agent
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return agents, errors


def _load_messages_direct(runtime_root: Path) -> tuple[list[PersistentMessage], list[str]]:
    root = runtime_root / "messages"
    messages: list[PersistentMessage] = []
    errors: list[str] = []
    if not root.exists():
        return messages, errors
    for path in sorted(root.glob("msg-*.json")):
        try:
            payload = _load_json_file(path)
            if not isinstance(payload, dict):
                raise ValueError("message payload must be an object")
            messages.append(PersistentMessage.from_dict(payload))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return messages, errors


def _load_approvals_direct(runtime_root: Path) -> tuple[list[CapabilityApprovalRequest], list[str]]:
    root = runtime_root / "capability_approvals"
    approvals: list[CapabilityApprovalRequest] = []
    errors: list[str] = []
    if not root.exists():
        return approvals, errors
    for path in sorted(root.glob("cap_approval_*.json")):
        try:
            payload = _load_json_file(path)
            if not isinstance(payload, dict):
                raise ValueError("approval payload must be an object")
            approvals.append(CapabilityApprovalRequest.from_dict(payload))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return approvals, errors


def _check_runtime(runtime_root: Path) -> DoctorCheckResult:
    if not runtime_root.exists():
        return _check(
            check_id="runtime.structure",
            category="runtime",
            status="error",
            title="Runtime structure",
            summary="runtime root is missing",
            details={"runtime_root": str(runtime_root)},
            recommendations=["Create the runtime root or point `--store-root` at a valid MR1 runtime."],
        )
    if not runtime_root.is_dir():
        return _check(
            check_id="runtime.structure",
            category="runtime",
            status="error",
            title="Runtime structure",
            summary="runtime root is not a directory",
            details={"runtime_root": str(runtime_root)},
        )

    missing: list[str] = []
    unreadable: list[str] = []
    invalid_types: list[str] = []
    present: list[str] = []
    for name in _EXPECTED_RUNTIME_PATHS:
        path = runtime_root / name
        if not path.exists():
            missing.append(name)
            continue
        if not path.is_dir():
            invalid_types.append(name)
            continue
        present.append(name)
        try:
            next(path.iterdir(), None)
        except OSError as exc:
            unreadable.append(f"{path}: {exc}")

    status = "ok"
    summary = "runtime root and expected stores are readable"
    recommendations: list[str] = []
    if invalid_types or unreadable:
        status = "error"
        summary = "runtime contains invalid or unreadable expected store paths"
    elif missing:
        status = "warning"
        summary = "runtime is missing one or more expected store directories"
        recommendations.append("Initialize the missing stores by running the relevant MR1 workflows or commands.")

    return _check(
        check_id="runtime.structure",
        category="runtime",
        status=status,
        title="Runtime structure",
        summary=summary,
        details={
            "runtime_root": str(runtime_root),
            "present_paths": present,
            "missing_paths": missing,
            "invalid_paths": invalid_types,
            "unreadable_paths": unreadable,
            "ignored_legacy_paths": sorted(
                name for name in _LEGACY_RUNTIME_PATHS if (runtime_root / name).exists()
            ),
        },
        recommendations=recommendations,
    )


def _validate_events(runtime_root: Path) -> tuple[list[DoctorCheckResult], int]:
    path = runtime_root / "events" / "events.jsonl"
    if not path.exists():
        return [
            _check(
                check_id="events.log",
                category="events",
                status="warning",
                title="Event timeline",
                summary="event log does not exist yet",
                details={"event_log_path": str(path), "event_count": 0, "latest_event_index": 0},
            )
        ], 0

    errors: list[str] = []
    warnings: list[str] = []
    parsed_events: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                    continue
                if not isinstance(payload, dict):
                    errors.append(f"{path}:{line_no}: payload must be a JSON object")
                    continue
                parsed_events.append(payload)
    except OSError as exc:
        return [
            _check(
                check_id="events.log",
                category="events",
                status="error",
                title="Event timeline",
                summary="event log is unreadable",
                details={"event_log_path": str(path), "errors": [str(exc)]},
                recommendations=[
                    "Inspect events/events.jsonl and restore from snapshot if available."
                ],
            )
        ], 0

    if not parsed_events and not errors:
        return [
            _check(
                check_id="events.log",
                category="events",
                status="warning",
                title="Event timeline",
                summary="event log exists but contains no events",
                details={"event_log_path": str(path), "event_count": 0, "latest_event_index": 0},
            )
        ], 0

    seen_ids: set[str] = set()
    seen_parents: set[str] = set()
    latest_event_index = 0
    previous_index: Optional[int] = None
    type_counts: dict[str, int] = {}
    for idx, payload in enumerate(parsed_events, start=1):
        event_id = payload.get("event_id")
        event_index = payload.get("event_index")
        event_version = payload.get("event_version")
        event_type = payload.get("event_type")
        event_kind = payload.get("event_kind")
        severity = payload.get("severity")

        if not isinstance(event_id, str) or not event_id.strip():
            errors.append(f"{path}:{idx}: missing or invalid event_id")
            continue
        if event_id in seen_ids:
            errors.append(f"{path}:{idx}: duplicate event_id {event_id}")
        seen_ids.add(event_id)

        try:
            event_index_int = int(event_index)
        except (TypeError, ValueError):
            errors.append(f"{path}:{idx}: invalid event_index {event_index!r}")
            continue
        if previous_index is not None and event_index_int <= previous_index:
            errors.append(
                f"{path}:{idx}: non-monotonic event_index {event_index_int} after {previous_index}"
            )
        previous_index = event_index_int
        latest_event_index = max(latest_event_index, event_index_int)

        try:
            version_int = int(event_version)
        except (TypeError, ValueError):
            warnings.append(f"{path}:{idx}: unknown event_version {event_version!r}")
        else:
            if version_int != EVENT_VERSION:
                warnings.append(f"{path}:{idx}: unknown event_version {version_int}")

        if not isinstance(event_type, str) or not event_type.strip():
            errors.append(f"{path}:{idx}: missing or invalid event_type")
        else:
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            if event_type not in EVENT_KIND_BY_TYPE:
                warnings.append(f"{path}:{idx}: unknown event_type {event_type}")
        if not isinstance(event_kind, str) or not event_kind.strip():
            errors.append(f"{path}:{idx}: missing or invalid event_kind")
        elif event_kind not in EVENT_KIND_VALUES:
            warnings.append(f"{path}:{idx}: unknown event_kind {event_kind}")
        elif isinstance(event_type, str) and event_type in EVENT_KIND_BY_TYPE and event_kind != EVENT_KIND_BY_TYPE[event_type]:
            warnings.append(
                f"{path}:{idx}: event_kind {event_kind} does not match expected {EVENT_KIND_BY_TYPE[event_type]} for {event_type}"
            )

        if not isinstance(severity, str) or not severity.strip():
            errors.append(f"{path}:{idx}: missing or invalid severity")
        elif severity not in SEVERITY_VALUES:
            warnings.append(f"{path}:{idx}: unknown severity {severity}")
        elif isinstance(event_type, str) and event_type in SEVERITY_BY_TYPE and severity != SEVERITY_BY_TYPE[event_type]:
            warnings.append(
                f"{path}:{idx}: severity {severity} does not match expected {SEVERITY_BY_TYPE[event_type]} for {event_type}"
            )

        parent_event_id = payload.get("parent_event_id")
        if parent_event_id is not None:
            if not isinstance(parent_event_id, str) or not parent_event_id.strip():
                errors.append(f"{path}:{idx}: invalid parent_event_id {parent_event_id!r}")
            else:
                seen_parents.add(parent_event_id)

    missing_parents = sorted(parent_id for parent_id in seen_parents if parent_id not in seen_ids)
    for parent_id in missing_parents:
        errors.append(f"dangling parent_event_id {parent_id}")

    if errors:
        status = "error"
        summary = "event log contains corruption or causal inconsistencies"
        recommendations = [
            "Inspect events/events.jsonl and restore from snapshot if available."
        ]
    elif warnings:
        status = "warning"
        summary = "event log is readable but contains forward-compatible or inconsistent values"
        recommendations = []
    else:
        status = "ok"
        summary = "event log is valid"
        recommendations = []

    return [
        _check(
            check_id="events.log",
            category="events",
            status=status,
            title="Event timeline",
            summary=summary,
            details={
                "event_log_path": str(path),
                "event_count": len(parsed_events),
                "latest_event_index": latest_event_index,
                "latest_event_type": parsed_events[-1].get("event_type") if parsed_events else None,
                "type_counts": dict(sorted(type_counts.items())),
                "warnings": warnings,
                "errors": errors,
            },
            recommendations=recommendations,
        )
    ], latest_event_index


def _check_graph_memory(runtime_root: Path, latest_event_index: int) -> DoctorCheckResult:
    store = MemoryGraphStore(runtime_root / "graph")
    warnings: list[str] = []
    errors: list[str] = []
    recommendations: list[str] = []

    graph_exists = store.graph_path.exists()
    cursor_exists = store.cursor_path.exists()
    if not graph_exists and not cursor_exists:
        status = "warning" if latest_event_index > 0 else "ok"
        summary = "graph memory has not been initialized" if latest_event_index > 0 else "graph memory not initialized yet"
        if latest_event_index > 0:
            recommendations.append("Run: python -m mr1.workflow_cli memory maintenance run")
        return _check(
            check_id="memory.graph",
            category="memory",
            status=status,
            title="Graph memory",
            summary=summary,
            details={
                "graph_exists": graph_exists,
                "cursor_exists": cursor_exists,
                "latest_event_index": latest_event_index,
            },
            recommendations=recommendations,
        )

    try:
        graph = store.load_graph()
        cursor = store.load_cursor()
    except ValueError as exc:
        return _check(
            check_id="memory.graph",
            category="memory",
            status="error",
            title="Graph memory",
            summary="graph memory files are corrupt",
            details={"error": str(exc)},
            recommendations=[
                "Inspect graph/graph.json and graph/cursor.json, or restore from a snapshot."
            ],
        )

    missing_nodes: list[str] = []
    for edge in graph.edges.values():
        if edge.edge_type not in EDGE_TYPES:
            errors.append(f"unknown edge_type {edge.edge_type}")
        if edge.source_id not in graph.nodes:
            missing_nodes.append(edge.source_id)
        if edge.target_id not in graph.nodes:
            missing_nodes.append(edge.target_id)
    if missing_nodes:
        errors.append(f"edges reference missing nodes: {sorted(set(missing_nodes))}")
    if cursor > latest_event_index:
        errors.append(
            f"graph cursor {cursor} is ahead of latest event index {latest_event_index}"
        )
    elif cursor < latest_event_index:
        warnings.append(
            f"graph cursor {cursor} is stale relative to latest event index {latest_event_index}"
        )
        recommendations.append("Run: python -m mr1.workflow_cli memory maintenance run")

    if errors:
        status = "error"
        summary = "graph memory is inconsistent"
    elif warnings:
        status = "warning"
        summary = "graph memory is readable but stale"
    else:
        status = "ok"
        summary = "graph memory is consistent"
    return _check(
        check_id="memory.graph",
        category="memory",
        status=status,
        title="Graph memory",
        summary=summary,
        details={
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "last_processed_event_index": cursor,
            "latest_event_index": latest_event_index,
            "warnings": warnings,
            "errors": errors,
        },
        recommendations=recommendations,
    )


def _check_insights_memory(runtime_root: Path, latest_event_index: int) -> DoctorCheckResult:
    store = InsightStore(runtime_root / "insights")
    warnings: list[str] = []
    errors: list[str] = []
    recommendations: list[str] = []

    insights_exists = store.insights_path.exists()
    cursor_exists = store.cursor_path.exists()
    runs_exists = store.runs_path.exists()
    if not insights_exists and not cursor_exists and not runs_exists:
        status = "warning" if latest_event_index > 0 else "ok"
        summary = "insight store has not been initialized" if latest_event_index > 0 else "insight store not initialized yet"
        if latest_event_index > 0:
            recommendations.append("Run: python -m mr1.workflow_cli memory maintenance run")
        return _check(
            check_id="memory.insights",
            category="memory",
            status=status,
            title="Curated insights",
            summary=summary,
            details={"latest_event_index": latest_event_index},
            recommendations=recommendations,
        )

    try:
        insights = store.load_insights()
        cursor = store.load_cursor()
        runs = store.load_runs()
    except ValueError as exc:
        return _check(
            check_id="memory.insights",
            category="memory",
            status="error",
            title="Curated insights",
            summary="insight store is corrupt",
            details={"error": str(exc)},
            recommendations=[
                "Inspect insights/insights.json, insights/cursor.json, and insights/curation_runs.jsonl."
            ],
        )

    status_counts = {"active": 0, "stale": 0, "dismissed": 0, "superseded": 0}
    for insight in insights.values():
        try:
            MemoryInsight.from_dict(insight.to_dict())
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{insight.insight_id}: {exc}")
            continue
        status_counts[insight.status] = status_counts.get(insight.status, 0) + 1
        for item in insight.evidence:
            try:
                EvidenceRef.from_dict(item.to_dict())
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{insight.insight_id}: invalid evidence ref: {exc}")
    for run in runs:
        try:
            MemoryCurationRun.from_dict(run.to_dict())
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{run.run_id}: invalid curation run: {exc}")

    if cursor > latest_event_index:
        errors.append(
            f"curation cursor {cursor} is ahead of latest event index {latest_event_index}"
        )
    elif cursor < latest_event_index:
        warnings.append(
            f"curation cursor {cursor} is stale relative to latest event index {latest_event_index}"
        )
        recommendations.append("Run: python -m mr1.workflow_cli memory maintenance run")

    if errors:
        status = "error"
        summary = "insight store contains invalid data"
    elif warnings:
        status = "warning"
        summary = "insight store is readable but stale"
    else:
        status = "ok"
        summary = "insight store is valid"
    return _check(
        check_id="memory.insights",
        category="memory",
        status=status,
        title="Curated insights",
        summary=summary,
        details={
            "insight_count": len(insights),
            "status_counts": status_counts,
            "last_curated_event_index": cursor,
            "latest_event_index": latest_event_index,
            "curation_run_count": len(runs),
            "warnings": warnings,
            "errors": errors,
        },
        recommendations=recommendations,
    )


def _check_feedback_memory(runtime_root: Path, latest_event_index: int) -> DoctorCheckResult:
    store = InsightStore(runtime_root / "insights")
    warnings: list[str] = []
    errors: list[str] = []
    recommendations: list[str] = []

    if not store.feedback_path.exists() and not store.feedback_cursor_path.exists():
        return _check(
            check_id="memory.feedback",
            category="memory",
            status="ok" if latest_event_index == 0 else "warning",
            title="Insight feedback",
            summary="insight feedback has not been initialized" if latest_event_index > 0 else "insight feedback not initialized yet",
            details={"latest_event_index": latest_event_index},
            recommendations=["Run: python -m mr1.workflow_cli memory maintenance run"] if latest_event_index > 0 else [],
        )

    try:
        insights = store.load_insights()
        feedback_items = store.load_feedback()
        cursor = store.load_feedback_cursor()
    except ValueError as exc:
        return _check(
            check_id="memory.feedback",
            category="memory",
            status="error",
            title="Insight feedback",
            summary="insight feedback store is corrupt",
            details={"error": str(exc)},
            recommendations=[
                "Inspect insights/feedback.jsonl and insights/feedback_cursor.json."
            ],
        )

    seen_ids: set[str] = set()
    for item in feedback_items:
        try:
            InsightFeedback.from_dict(item.to_dict())
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{item.feedback_id}: {exc}")
            continue
        if item.feedback_id in seen_ids:
            errors.append(f"duplicate feedback_id {item.feedback_id}")
        seen_ids.add(item.feedback_id)
        if item.insight_id not in insights:
            errors.append(f"{item.feedback_id}: missing insight ref {item.insight_id}")

    for insight in insights.values():
        stats = dict(insight.stats)
        positive = int(stats.get("positive_outcome_count", 0))
        negative = int(stats.get("negative_outcome_count", 0))
        neutral = int(stats.get("neutral_outcome_count", 0))
        used_count = int(stats.get("used_count", 0))
        if used_count != positive + negative + neutral:
            warnings.append(
                f"{insight.insight_id}: used_count {used_count} != positive+negative+neutral {positive + negative + neutral}"
            )

    if cursor > latest_event_index:
        errors.append(
            f"feedback cursor {cursor} is ahead of latest event index {latest_event_index}"
        )
    elif cursor < latest_event_index:
        warnings.append(
            f"feedback cursor {cursor} is stale relative to latest event index {latest_event_index}"
        )
        recommendations.append("Run: python -m mr1.workflow_cli memory maintenance run")

    if errors:
        status = "error"
        summary = "insight feedback contains corruption or dangling refs"
    elif warnings:
        status = "warning"
        summary = "insight feedback is readable but needs attention"
    else:
        status = "ok"
        summary = "insight feedback is valid"
    return _check(
        check_id="memory.feedback",
        category="memory",
        status=status,
        title="Insight feedback",
        summary=summary,
        details={
            "feedback_count": len(feedback_items),
            "last_evaluated_event_index": cursor,
            "latest_event_index": latest_event_index,
            "warnings": warnings,
            "errors": errors,
        },
        recommendations=recommendations,
    )


def _latest_memory_source_timestamp(runtime_root: Path) -> Optional[str]:
    timestamps: list[str] = []

    graph_store = MemoryGraphStore(runtime_root / "graph")
    if graph_store.graph_path.exists():
        try:
            graph = graph_store.load_graph()
        except ValueError:
            graph = None
        if graph is not None:
            for node in graph.nodes.values():
                if node.last_seen_at:
                    timestamps.append(node.last_seen_at)

    insight_store = InsightStore(runtime_root / "insights")
    if insight_store.insights_path.exists():
        try:
            insights = insight_store.load_insights()
        except ValueError:
            insights = {}
        for insight in insights.values():
            if insight.updated_at:
                timestamps.append(insight.updated_at)
    if insight_store.feedback_path.exists():
        try:
            feedback = insight_store.load_feedback()
        except ValueError:
            feedback = []
        for item in feedback:
            if item.created_at:
                timestamps.append(item.created_at)

    return max(timestamps) if timestamps else None


def _check_retrieval_memory(runtime_root: Path) -> DoctorCheckResult:
    store = RetrievalStore(runtime_root)
    warnings: list[str] = []
    errors: list[str] = []
    recommendations: list[str] = []

    graph_exists = (runtime_root / "graph" / "graph.json").exists()
    insights_exists = (runtime_root / "insights" / "insights.json").exists()
    feedback_exists = (runtime_root / "insights" / "feedback.jsonl").exists()

    if not store.documents_path.exists() and not store.manifest_path.exists():
        status = "warning" if graph_exists or insights_exists or feedback_exists else "ok"
        summary = "retrieval store has not been initialized" if status == "warning" else "retrieval store not initialized yet"
        if status == "warning":
            recommendations.append("Run: python -m mr1.workflow_cli memory retrieval update")
        return _check(
            check_id="memory.retrieval",
            category="memory",
            status=status,
            title="Retrieval documents",
            summary=summary,
            details={
                "documents_exists": store.documents_path.exists(),
                "manifest_exists": store.manifest_path.exists(),
            },
            recommendations=recommendations,
        )

    try:
        documents = store.load_documents()
        manifest = store.load_manifest()
    except ValueError as exc:
        return _check(
            check_id="memory.retrieval",
            category="memory",
            status="error",
            title="Retrieval documents",
            summary="retrieval store is corrupt",
            details={"error": str(exc)},
            recommendations=[
                "Inspect retrieval/documents.jsonl and retrieval/manifest.json, then rebuild retrieval if needed."
            ],
        )

    seen_ids: set[str] = set()
    type_counts: dict[str, int] = {}
    for item in documents:
        if item.doc_id in seen_ids:
            errors.append(f"duplicate document id {item.doc_id}")
        seen_ids.add(item.doc_id)
        type_counts[item.doc_type] = type_counts.get(item.doc_type, 0) + 1
        if item.doc_type not in RETRIEVAL_DOC_TYPES:
            errors.append(f"unknown document type {item.doc_type}")

    manifest_count = manifest.get("document_count")
    try:
        manifest_count_int = int(manifest_count)
    except (TypeError, ValueError):
        errors.append(f"invalid manifest document_count {manifest_count!r}")
        manifest_count_int = -1
    if manifest_count_int != len(documents):
        warnings.append(
            f"manifest document_count {manifest_count_int} does not match actual {len(documents)}"
        )
    latest_source_timestamp = _latest_memory_source_timestamp(runtime_root)
    manifest_updated_at = manifest.get("updated_at")
    parsed_manifest_updated = _parse_iso(manifest_updated_at)
    parsed_latest_source = _parse_iso(latest_source_timestamp)
    if parsed_manifest_updated is None and manifest_updated_at not in {None, ""}:
        warnings.append(f"manifest updated_at is invalid: {manifest_updated_at!r}")
    if (
        parsed_manifest_updated is not None
        and parsed_latest_source is not None
        and parsed_manifest_updated < parsed_latest_source
    ):
        warnings.append(
            f"retrieval manifest updated_at {manifest_updated_at} is older than latest memory source {latest_source_timestamp}"
        )
        recommendations.append("Run: python -m mr1.workflow_cli memory retrieval update")

    try:
        retriever = LexicalMemoryRetriever(runtime_root)
        retriever.search("memory", limit=1)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"retrieval smoke test failed: {exc}")

    if errors:
        status = "error"
        summary = "retrieval store is invalid"
    elif warnings:
        status = "warning"
        summary = "retrieval store is readable but stale or inconsistent"
    else:
        status = "ok"
        summary = "retrieval store is valid"
    return _check(
        check_id="memory.retrieval",
        category="memory",
        status=status,
        title="Retrieval documents",
        summary=summary,
        details={
            "document_count": len(documents),
            "manifest_document_count": manifest_count_int,
            "schema_version": manifest.get("schema_version"),
            "updated_at": manifest.get("updated_at"),
            "type_counts": dict(sorted(type_counts.items())),
            "warnings": warnings,
            "errors": errors,
        },
        recommendations=recommendations,
    )


def _check_memory_maintenance_workflow(runtime_root: Path) -> DoctorCheckResult:
    workflows, errors = _load_workflows_direct(runtime_root)
    if errors:
        return _check(
            check_id="memory.maintenance_workflow",
            category="memory",
            status="error",
            title="Memory maintenance workflow",
            summary="workflow store contains corruption, so maintenance workflow presence is uncertain",
            details={"errors": errors},
        )

    matching = [
        workflow.workflow_id
        for workflow in workflows
        if workflow.metadata.get("system_workflow") == "memory_maintenance"
    ]
    if not matching:
        return _check(
            check_id="memory.maintenance_workflow",
            category="memory",
            status="warning",
            title="Memory maintenance workflow",
            summary="no memory maintenance workflow is present",
            details={"matching_workflow_ids": []},
            recommendations=["Run: python -m mr1.workflow_cli memory maintenance run"],
        )
    return _check(
        check_id="memory.maintenance_workflow",
        category="memory",
        status="ok",
        title="Memory maintenance workflow",
        summary="memory maintenance workflow is present",
        details={"matching_workflow_ids": matching},
    )


def _validate_workflows(runtime_root: Path) -> DoctorCheckResult:
    store = WorkflowStore(root=runtime_root / "workflows")
    loaded_workflows = store.list_workflows()
    workflows, direct_errors = _load_workflows_direct(runtime_root)
    warnings: list[str] = []
    errors = list(direct_errors)
    recommendations: list[str] = []

    running_ids: list[str] = []
    failed_ids: list[str] = []
    stuck_ids: list[str] = []
    blocked_workflow_ids: list[str] = []
    policy_block_ids: list[str] = []
    task_counts = {
        "running_tasks": 0,
        "failed_tasks": 0,
        "blocked_tasks": 0,
    }

    now = datetime.now(timezone.utc)
    for workflow in workflows:
        if workflow.status == WorkflowStatus.RUNNING:
            running_ids.append(workflow.workflow_id)
        if workflow.status == WorkflowStatus.FAILED:
            failed_ids.append(workflow.workflow_id)
        statuses = {task.status for task in workflow.tasks.values()}
        ready_or_running = any(task.status in {TaskStatus.READY, TaskStatus.RUNNING} for task in workflow.tasks.values())
        terminal_only = bool(workflow.tasks) and all(task.is_terminal() for task in workflow.tasks.values())
        blocked_visible = any(task.status in {TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED} for task in workflow.tasks.values())

        if workflow.status in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING} and terminal_only:
            warnings.append(f"{workflow.workflow_id}: non-terminal workflow has only terminal tasks")
            stuck_ids.append(workflow.workflow_id)
        created_at = _parse_iso(workflow.created_at)
        if workflow.status == WorkflowStatus.RUNNING and created_at is not None and now - created_at > _STUCK_RUNNING_THRESHOLD:
            warnings.append(
                f"{workflow.workflow_id}: running workflow is older than {_STUCK_RUNNING_THRESHOLD}"
            )
            stuck_ids.append(workflow.workflow_id)
        if workflow.status in {WorkflowStatus.PENDING, WorkflowStatus.RUNNING} and not ready_or_running and blocked_visible:
            warnings.append(f"{workflow.workflow_id}: workflow has no runnable tasks and visible blockers")
            stuck_ids.append(workflow.workflow_id)
        if any(task.status == TaskStatus.BLOCKED for task in workflow.tasks.values()):
            blocked_workflow_ids.append(workflow.workflow_id)

        for task in workflow.tasks.values():
            if task.status == TaskStatus.RUNNING:
                task_counts["running_tasks"] += 1
            if task.status in {TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED}:
                task_counts["failed_tasks"] += 1
            if task.status == TaskStatus.BLOCKED:
                task_counts["blocked_tasks"] += 1

            candidate_paths: list[Path] = []
            if task.result_path:
                candidate_paths.append(Path(task.result_path))
            candidate_paths.append(store.task_result_path(workflow.workflow_id, task.task_id))
            for attempt in task.attempts:
                if attempt.result_path:
                    candidate_paths.append(Path(attempt.result_path))
                candidate_paths.append(
                    store.task_attempt_result_path(workflow.workflow_id, task.task_id, attempt.attempt_id)
                )
            for result_path in candidate_paths:
                if not result_path.exists():
                    continue
                try:
                    payload = _load_json_file(result_path)
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"{result_path}: {exc}")
                    continue
                if not isinstance(payload, dict):
                    errors.append(f"{result_path}: result payload must be an object")
                    continue
                if (
                    payload.get("error_type") == "policy_block"
                    or payload.get("failure_type") == "policy_block"
                    or payload.get("retryable") is False
                ):
                    policy_block_ids.append(workflow.workflow_id)
                    warnings.append(f"{workflow.workflow_id}: task {task.task_id} has non-retryable policy-block style failure")
                    break

    if len(loaded_workflows) != len(workflows) and direct_errors:
        errors.append("workflow store skipped one or more corrupt workflows during normal load")

    if failed_ids or blocked_workflow_ids or stuck_ids or policy_block_ids:
        recommendations.extend(_recommend_timeline_workflow(set(failed_ids + blocked_workflow_ids + stuck_ids + policy_block_ids)))

    if errors:
        status = "error"
        summary = "workflow store contains corruption"
    elif warnings:
        status = "warning"
        summary = "workflow store is readable but has failed, blocked, or stale workflows"
    else:
        status = "ok"
        summary = "workflow store is healthy"

    return _check(
        check_id="workflows.store",
        category="workflows",
        status=status,
        title="Workflow store",
        summary=summary,
        details={
            "workflow_count": len(workflows),
            "loaded_workflow_count": len(loaded_workflows),
            "running_workflow_count": len(running_ids),
            "failed_workflow_count": len(failed_ids),
            "blocked_workflow_count": len(set(blocked_workflow_ids)),
            "task_counts": task_counts,
            "failed_workflow_ids": sorted(set(failed_ids)),
            "blocked_workflow_ids": sorted(set(blocked_workflow_ids)),
            "stuck_workflow_ids": sorted(set(stuck_ids)),
            "warnings": warnings,
            "errors": errors,
        },
        recommendations=recommendations,
    )


def _validate_agents(runtime_root: Path) -> DoctorCheckResult:
    root = runtime_root / "agents"
    agents, errors = _load_agents_direct(runtime_root)
    warnings: list[str] = []

    pointer = root / ".root_agent_id"
    pointer_target: Optional[str] = None
    if pointer.exists():
        try:
            pointer_target = pointer.read_text(encoding="utf-8").strip() or None
        except OSError as exc:
            errors.append(f"{pointer}: {exc}")
    else:
        errors.append(f"{pointer}: missing root agent pointer")

    if pointer_target and pointer_target not in agents:
        errors.append(f"{pointer}: points to missing agent {pointer_target}")

    root_agents = [
        agent.agent_id
        for agent in agents.values()
        if agent.agent_type == "mr1" and agent.parent_agent_id is None
    ]
    if len(root_agents) != 1:
        errors.append(f"expected exactly one root MR1 agent, found {len(root_agents)}")

    child_count = 0
    for agent in agents.values():
        if agent.parent_agent_id is not None:
            child_count += 1
            parent = agents.get(agent.parent_agent_id)
            if parent is None:
                errors.append(f"{agent.agent_id}: missing parent {agent.parent_agent_id}")
            elif agent.security_clearance > parent.security_clearance:
                errors.append(
                    f"{agent.agent_id}: security_clearance {agent.security_clearance} exceeds parent {parent.security_clearance}"
                )
        seen_in_chain: set[str] = set()
        current_id = agent.agent_id
        while True:
            current = agents.get(current_id)
            if current is None or current.parent_agent_id is None:
                break
            if current.parent_agent_id in seen_in_chain:
                errors.append(f"{agent.agent_id}: cycle detected in agent hierarchy")
                break
            seen_in_chain.add(current.parent_agent_id)
            current_id = current.parent_agent_id

        for scope_root in list(agent.scope_roots or []):
            normalized = normalize_path(scope_root)
            if not normalized.exists():
                warnings.append(f"{agent.agent_id}: scope root missing on disk: {normalized}")
        agent_dir = root / agent.agent_id
        expected_files = [
            root / f"{agent.agent_id}.json",
            agent_dir / "memory.md",
            agent_dir / "logs",
            agent_dir / "reports",
        ]
        for expected in expected_files:
            if not expected.exists():
                warnings.append(f"{agent.agent_id}: expected agent file missing: {expected}")

    if child_count == 0 and agents:
        warnings.append("no child agents exist")

    if errors:
        status = "error"
        summary = "agent hierarchy contains invalid structure"
    elif warnings:
        status = "warning"
        summary = "agent registry is valid but has missing scope roots or child agents"
    else:
        status = "ok"
        summary = "agent registry is valid"

    return _check(
        check_id="agents.registry",
        category="agents",
        status=status,
        title="Scoped agents",
        summary=summary,
        details={
            "agent_count": len(agents),
            "root_pointer_target": pointer_target,
            "root_agents": sorted(root_agents),
            "child_agent_count": child_count,
            "warnings": warnings,
            "errors": errors,
        },
        recommendations=[],
    )


def _validate_capabilities() -> DoctorCheckResult:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        registry = default_capability_registry()
        descriptions = registry.describe_all()
    except Exception as exc:  # pragma: no cover - defensive
        return _check(
            check_id="capabilities.registry",
            category="capabilities",
            status="error",
            title="Capability registry",
            summary="capability registry failed to initialize",
            details={"error": str(exc)},
        )

    names = [item["name"] for item in descriptions]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        errors.append(f"duplicate capability names: {duplicates}")

    for item in descriptions:
        try:
            CapabilityMetadata.from_dict(item)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{item.get('name', '<missing>')}: {exc}")

    missing_required = sorted(
        name for name in (list(MEMORY_CAPABILITY_NAMES) + list(_REQUIRED_MAINTENANCE_CAPABILITIES))
        if name not in names
    )
    if missing_required:
        errors.append(f"missing required capabilities: {missing_required}")

    status = "error" if errors else "warning" if warnings else "ok"
    summary = (
        "capability registry contains invalid metadata"
        if errors else
        "capability registry contains warnings"
        if warnings else
        "capability registry is valid"
    )
    return _check(
        check_id="capabilities.registry",
        category="capabilities",
        status=status,
        title="Capability registry",
        summary=summary,
        details={
            "capability_count": len(descriptions),
            "missing_required": missing_required,
            "warnings": warnings,
            "errors": errors,
        },
    )


def _validate_approvals(runtime_root: Path) -> DoctorCheckResult:
    approvals, errors = _load_approvals_direct(runtime_root)
    warnings: list[str] = []
    agents, agent_errors = _load_agents_direct(runtime_root)
    errors.extend(agent_errors)

    counts = {"pending": 0, "approved": 0, "denied": 0, "superseded": 0, "expired": 0}
    for approval in approvals:
        counts[approval.status] = counts.get(approval.status, 0) + 1
        if approval.requesting_actor_id not in agents:
            warnings.append(
                f"{approval.approval_request_id}: requesting actor missing: {approval.requesting_actor_id}"
            )
        if approval.designated_approver_id and approval.designated_approver_id not in agents:
            warnings.append(
                f"{approval.approval_request_id}: designated approver missing: {approval.designated_approver_id}"
            )
        if approval.status == "pending":
            warnings.append(f"{approval.approval_request_id}: approval is still pending")
        if isinstance(approval.decision, dict):
            try:
                decision = CapabilityApprovalDecision.from_dict(approval.decision)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{approval.approval_request_id}: invalid decision payload: {exc}")
            else:
                if decision.approval_scope == "single_use" and approval.used_at is not None:
                    if approval.status != "superseded":
                        errors.append(
                            f"{approval.approval_request_id}: used single-use approval must be superseded"
                        )
                    if not approval.used_by_audit_id:
                        errors.append(
                            f"{approval.approval_request_id}: used single-use approval missing used_by_audit_id"
                        )
                    elif approval.used_by_audit_id:
                        audit_found = False
                        for path in (runtime_root / "agents").glob("ag-*/logs/capability_audits/*.json"):
                            if path.stem == approval.used_by_audit_id:
                                audit_found = True
                                break
                        if not audit_found:
                            warnings.append(
                                f"{approval.approval_request_id}: used_by_audit_id {approval.used_by_audit_id} not found on disk"
                            )

    if errors:
        status = "error"
        summary = "approval records contain invalid data"
    elif warnings:
        status = "warning"
        summary = "approval records are readable but require attention"
    else:
        status = "ok"
        summary = "approval records are valid"

    recommendations: list[str] = []
    if counts.get("pending", 0) > 0:
        recommendations.append("Run: python -m mr1.workflow_cli approvals list")

    return _check(
        check_id="approvals.records",
        category="approvals",
        status=status,
        title="Capability approvals",
        summary=summary,
        details={
            "approval_count": len(approvals),
            "counts": counts,
            "warnings": warnings,
            "errors": errors,
        },
        recommendations=recommendations,
    )


def _validate_messages(runtime_root: Path) -> DoctorCheckResult:
    messages, errors = _load_messages_direct(runtime_root)
    warnings: list[str] = []
    agents, agent_errors = _load_agents_direct(runtime_root)
    errors.extend(agent_errors)

    unread_count = 0
    for message in messages:
        if message.status == "unread":
            unread_count += 1
        if message.from_agent_id not in agents:
            warnings.append(
                f"{message.message_id}: sender missing: {message.from_agent_id}"
            )
        if message.to_agent_id not in agents:
            warnings.append(
                f"{message.message_id}: recipient missing: {message.to_agent_id}"
            )

    status = "error" if errors else "warning" if warnings or unread_count else "ok"
    summary = (
        "message store contains invalid data"
        if errors else
        "message store has unread or orphaned messages"
        if warnings or unread_count else
        "message store is valid"
    )
    return _check(
        check_id="messages.store",
        category="messages",
        status=status,
        title="Messages",
        summary=summary,
        details={
            "message_count": len(messages),
            "unread_count": unread_count,
            "warnings": warnings,
            "errors": errors,
        },
    )


def run_doctor(runtime_root: Path, categories: list[str] | None = None) -> DoctorReport:
    runtime_root = Path(runtime_root)
    selected_categories = list(categories or DOCTOR_CATEGORIES)
    invalid = sorted(set(selected_categories) - set(DOCTOR_CATEGORIES))
    if invalid:
        raise ValueError(f"unsupported doctor categories: {', '.join(invalid)}")

    checks: list[DoctorCheckResult] = []
    latest_event_index = 0
    if "runtime" in selected_categories:
        checks.append(_check_runtime(runtime_root))

    event_checks, latest_event_index = _validate_events(runtime_root)
    if "events" in selected_categories:
        checks.extend(event_checks)

    if "memory" in selected_categories:
        checks.extend([
            _check_graph_memory(runtime_root, latest_event_index),
            _check_insights_memory(runtime_root, latest_event_index),
            _check_feedback_memory(runtime_root, latest_event_index),
            _check_retrieval_memory(runtime_root),
            _check_memory_maintenance_workflow(runtime_root),
        ])
    if "workflows" in selected_categories:
        checks.append(_validate_workflows(runtime_root))
    if "agents" in selected_categories:
        checks.append(_validate_agents(runtime_root))
    if "capabilities" in selected_categories:
        checks.append(_validate_capabilities())
    if "approvals" in selected_categories:
        checks.append(_validate_approvals(runtime_root))
    if "messages" in selected_categories:
        checks.append(_validate_messages(runtime_root))

    return _report_from_checks(
        checks,
        categories=selected_categories,
        latest_event_index=latest_event_index,
    )


def _should_skip_snapshot_name(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in _SNAPSHOT_SKIP_PATTERNS)


def _copy_runtime_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns(*_SNAPSHOT_SKIP_PATTERNS),
        )
    else:
        if _should_skip_snapshot_name(src.name):
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _snapshot_stats(snapshot_dir: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for path in snapshot_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "snapshot_manifest.json":
            continue
        file_count += 1
        total_bytes += path.stat().st_size
    return file_count, total_bytes


def _snapshot_manifest_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / "snapshot_manifest.json"


def create_snapshot(
    runtime_root: Path,
    *,
    fail_on_error: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = Path(runtime_root)
    timestamp = _ensure_utc(now or datetime.now(timezone.utc))
    snapshot_id = f"snapshot_{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    snapshots_root = root / "snapshots"
    final_dir = snapshots_root / snapshot_id
    temp_dir = snapshots_root / f".incomplete_{snapshot_id}"

    report = run_doctor(root)
    if report.status == "error" and fail_on_error:
        raise ValueError("doctor reported error status; refusing snapshot creation because --fail-on-error was set")

    snapshots_root.mkdir(parents=True, exist_ok=True)
    if final_dir.exists() or temp_dir.exists():
        raise ValueError(f"snapshot already exists: {snapshot_id}")

    included_paths: list[str] = []
    errors: list[str] = []
    manifest: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "created_at": timestamp.isoformat().replace("+00:00", "Z"),
        "source_root": str(root.resolve(strict=False)),
        "doctor_status_at_creation": report.status,
        "latest_event_index": int(report.summary.get("latest_event_index", 0)),
        "file_count": 0,
        "total_bytes": 0,
        "included_paths": included_paths,
        "errors": errors,
    }

    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
        for name in _SNAPSHOT_INCLUDED_PATHS:
            src = root / name
            if not src.exists():
                continue
            included_paths.append(name)
            _copy_runtime_path(src, temp_dir / name)
        file_count, total_bytes = _snapshot_stats(temp_dir)
        manifest["file_count"] = file_count
        manifest["total_bytes"] = total_bytes
        _snapshot_manifest_path(temp_dir).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_dir.replace(final_dir)
        return inspect_snapshot(root, snapshot_id)
    except Exception as exc:
        errors.append(str(exc))
        if temp_dir.exists():
            try:
                file_count, total_bytes = _snapshot_stats(temp_dir)
                manifest["file_count"] = file_count
                manifest["total_bytes"] = total_bytes
                _snapshot_manifest_path(temp_dir).write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
        raise


def list_snapshots(runtime_root: Path) -> list[dict[str, Any]]:
    root = Path(runtime_root) / "snapshots"
    if not root.exists() or not root.is_dir():
        return []
    manifests: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".incomplete_"):
            continue
        if not path.name.startswith("snapshot_"):
            continue
        manifest_path = _snapshot_manifest_path(path)
        if not manifest_path.exists():
            continue
        try:
            payload = _load_json_file(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        manifests.append(payload)
    manifests.sort(
        key=lambda item: (str(item.get("created_at", "")), str(item.get("snapshot_id", ""))),
        reverse=True,
    )
    return manifests


def inspect_snapshot(runtime_root: Path, snapshot_id: str) -> dict[str, Any]:
    manifest_path = Path(runtime_root) / "snapshots" / snapshot_id / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"snapshot not found: {snapshot_id}")
    try:
        payload = _load_json_file(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid snapshot manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid snapshot manifest: {manifest_path}")
    return payload
