"""
Deterministic unified retrieval documents derived from MR1 memory stores.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mr1.event_log import EventLog, SystemEvent
    from mr1.memory_curator import InsightStore, MemoryInsight
    from mr1.memory_graph import MemoryEdge, MemoryGraph, MemoryNode


SCHEMA_VERSION = 1
RETRIEVAL_DOC_TYPES = frozenset({
    "insight",
    "insight_feedback",
    "workflow_template",
    "capability",
    "failure_mode",
    "timeline_trace",
    "agent_summary",
})
IMPORTANT_TIMELINE_EVENT_TYPES = frozenset({
    "workflow_failed",
    "workflow_completed",
    "capability_blocked",
    "capability_failed",
    "approval_denied",
    "approval_consumed",
})
MAX_TIMELINE_TRACES = 100
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalized_lower(value: Any) -> str:
    return _normalized_text(value).lower()


def _tokenize(value: Any) -> list[str]:
    return _TOKEN_RE.findall(_normalized_lower(value))


def _iso_sort_key(value: Optional[str]) -> tuple[int, str]:
    if not isinstance(value, str) or not value.strip():
        return (0, "")
    return (1, value)


def _line(label: str, value: Any) -> str:
    return f"{label}: {_normalized_text(value) or '-'}"


def _format_refs(refs: list[dict[str, Any]]) -> str:
    if not refs:
        return "-"
    parts = []
    for item in refs:
        kind = _normalized_text(item.get("kind"))
        item_id = _normalized_text(item.get("id"))
        reason = _normalized_text(item.get("reason"))
        entry = f"{kind}:{item_id}" if kind and item_id else item_id or kind or "-"
        if reason:
            entry = f"{entry} ({reason})"
        parts.append(entry)
    return "; ".join(parts)


def _dedupe_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for item in refs:
        normalized = {
            "kind": _normalized_text(item.get("kind")),
            "id": _normalized_text(item.get("id")),
        }
        reason = _normalized_text(item.get("reason"))
        if reason:
            normalized["reason"] = reason
        if not normalized["kind"] or not normalized["id"]:
            continue
        key = _canonical_json(normalized)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    ordered.sort(key=lambda item: (item["kind"], item["id"], item.get("reason", "")))
    return ordered


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        normalized = _normalized_lower(tag)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return sorted(ordered)


def _doc_sort_key(document: "MemoryDocument") -> tuple[str]:
    return (document.doc_id,)


def _doc_id(doc_type: str, source_id: str) -> str:
    return f"retrieval:{doc_type}:{source_id}"


def _memory_root(runtime_root: Path) -> Path:
    return Path(runtime_root)


@dataclass(frozen=True)
class MemoryDocument:
    doc_id: str
    doc_type: str
    title: str
    summary: str
    body: str
    refs: list[dict[str, Any]]
    tags: list[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.doc_type not in RETRIEVAL_DOC_TYPES:
            raise ValueError(f"invalid doc_type '{self.doc_type}'")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict")
        object.__setattr__(self, "title", _normalized_text(self.title))
        object.__setattr__(self, "summary", _normalized_text(self.summary))
        object.__setattr__(self, "body", _normalized_text(self.body))
        object.__setattr__(self, "refs", _dedupe_refs(list(self.refs)))
        object.__setattr__(self, "tags", _dedupe_tags(list(self.tags)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "summary": self.summary,
            "body": self.body,
            "refs": [dict(item) for item in self.refs],
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryDocument":
        return cls(
            doc_id=str(data["doc_id"]),
            doc_type=str(data["doc_type"]),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            body=str(data.get("body", "")),
            refs=list(data.get("refs", [])),
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RetrievalUpdateResult:
    documents_written: int
    documents_created: int
    documents_updated: int
    source_counts: dict[str, int]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_written": self.documents_written,
            "documents_created": self.documents_created,
            "documents_updated": self.documents_updated,
            "source_counts": dict(self.source_counts),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class MemorySearchResult:
    doc_id: str
    doc_type: str
    title: str
    summary: str
    score: float
    refs: list[dict[str, Any]]
    tags: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "summary": self.summary,
            "score": float(self.score),
            "refs": [dict(item) for item in self.refs],
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


class RetrievalStore:
    def __init__(self, runtime_root: Path):
        self._root = _memory_root(runtime_root) / "retrieval"
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def documents_path(self) -> Path:
        return self._root / "documents.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self._root / "manifest.json"

    def exists(self) -> bool:
        return self.documents_path.exists() and self.manifest_path.exists()

    def load_documents(self) -> list[MemoryDocument]:
        if not self.documents_path.exists():
            return []
        docs: list[MemoryDocument] = []
        try:
            with open(self.documents_path, "r", encoding="utf-8") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("document payload must be an object")
                    docs.append(MemoryDocument.from_dict(payload))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt retrieval documents file: {self.documents_path}") from exc
        docs.sort(key=_doc_sort_key)
        return docs

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest payload must be an object")
            return payload
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"corrupt retrieval manifest file: {self.manifest_path}") from exc

    def save(
        self,
        documents: list[MemoryDocument],
        *,
        source_counts: dict[str, int],
        updated_at: Optional[str] = None,
    ) -> None:
        normalized_documents = sorted(documents, key=_doc_sort_key)
        timestamp = updated_at or _now_iso()
        documents_tmp = self.documents_path.with_suffix(".jsonl.tmp")
        with open(documents_tmp, "w", encoding="utf-8") as handle:
            for item in normalized_documents:
                handle.write(_canonical_json(item.to_dict()) + "\n")
        documents_tmp.replace(self.documents_path)
        manifest = {
            "updated_at": timestamp,
            "document_count": len(normalized_documents),
            "source_counts": {
                key: int(source_counts[key])
                for key in sorted(source_counts)
            },
            "schema_version": SCHEMA_VERSION,
        }
        manifest_tmp = self.manifest_path.with_suffix(".json.tmp")
        with open(manifest_tmp, "w", encoding="utf-8") as handle:
            handle.write(_json_dumps(manifest))
            handle.write("\n")
        manifest_tmp.replace(self.manifest_path)


def _load_previous_documents(runtime_root: Path) -> dict[str, MemoryDocument]:
    store = RetrievalStore(runtime_root)
    try:
        return {item.doc_id: item for item in store.load_documents()}
    except ValueError:
        return {}


def _graph_neighbors_by_node(graph: MemoryGraph) -> dict[str, list[tuple[MemoryEdge, Optional[MemoryNode]]]]:
    neighbors: dict[str, list[tuple[MemoryEdge, Optional[MemoryNode]]]] = defaultdict(list)
    for edge in graph.edges.values():
        target = graph.nodes.get(edge.target_id)
        source = graph.nodes.get(edge.source_id)
        neighbors[edge.source_id].append((edge, target))
        neighbors[edge.target_id].append((edge, source))
    return neighbors


def _node_type_slug(node_type: str) -> str:
    return _normalized_lower(node_type).replace(" ", "_")


def _insight_document(insight: MemoryInsight) -> MemoryDocument:
    refs = [{"kind": "insight", "id": insight.insight_id}]
    refs.extend(
        {
            "kind": f"evidence_{item.source_type}",
            "id": item.source_id,
            "reason": item.reason,
        }
        for item in insight.evidence
    )
    refs.extend({"kind": "graph_node", "id": node_id} for node_id in insight.related_nodes)
    body = " | ".join([
        _line("title", insight.title),
        _line("summary", insight.summary),
        _line("recommended_action", insight.recommended_action),
        _line("confidence", f"{float(insight.confidence):.3f}"),
        _line("severity", insight.severity),
        _line("status", insight.status),
        _line("used_count", insight.stats.get("used_count", 0)),
        _line("positive_outcome_count", insight.stats.get("positive_outcome_count", 0)),
        _line("negative_outcome_count", insight.stats.get("negative_outcome_count", 0)),
        _line("neutral_outcome_count", insight.stats.get("neutral_outcome_count", 0)),
        _line("effectiveness_score", f"{float(insight.stats.get('effectiveness_score', 0.0)):.3f}"),
        _line("evidence_refs", _format_refs(refs[1:1 + len(insight.evidence)])),
        _line("related_nodes", ", ".join(sorted(insight.related_nodes))),
    ])
    return MemoryDocument(
        doc_id=_doc_id("insight", insight.insight_id),
        doc_type="insight",
        title=insight.title,
        summary=insight.summary,
        body=body,
        refs=refs,
        tags=[
            "insight",
            insight.insight_type,
            insight.severity,
            insight.status,
        ],
        created_at=insight.created_at,
        updated_at=insight.updated_at,
        metadata={
            "insight_id": insight.insight_id,
            "insight_type": insight.insight_type,
            "confidence": float(insight.confidence),
            "severity": insight.severity,
            "status": insight.status,
            "used_count": int(insight.stats.get("used_count", 0)),
            "positive_outcome_count": int(insight.stats.get("positive_outcome_count", 0)),
            "negative_outcome_count": int(insight.stats.get("negative_outcome_count", 0)),
            "neutral_outcome_count": int(insight.stats.get("neutral_outcome_count", 0)),
            "effectiveness_score": float(insight.stats.get("effectiveness_score", 0.0)),
        },
    )


def _feedback_documents(
    insights: dict[str, MemoryInsight],
    feedback_items: list[Any],
) -> list[MemoryDocument]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in feedback_items:
        grouped[item.insight_id].append(item)
    documents: list[MemoryDocument] = []
    for insight_id in sorted(grouped):
        entries = sorted(grouped[insight_id], key=lambda item: (item.created_at, item.feedback_id))
        insight = insights.get(insight_id)
        counts = Counter(item.outcome for item in entries)
        positive = int(counts.get("positive", 0))
        negative = int(counts.get("negative", 0))
        neutral = int(counts.get("neutral", 0))
        effectiveness = positive / (positive + negative + neutral + 1)
        recent_entries = entries[-5:]
        recent_reasons = "; ".join(
            f"{item.outcome}:{_normalized_text(item.reason)}"
            for item in recent_entries
        )
        refs = [{"kind": "insight", "id": insight_id}]
        refs.extend({"kind": "feedback", "id": item.feedback_id} for item in recent_entries)
        refs.extend({"kind": "event", "id": item.event_id} for item in recent_entries)
        title = f"Feedback for {insight.title}" if insight is not None else f"Feedback for {insight_id}"
        summary = (
            f"positive={positive} negative={negative} neutral={neutral} "
            f"effectiveness={effectiveness:.3f}"
        )
        body = " | ".join([
            _line("insight_id", insight_id),
            _line("positive_count", positive),
            _line("negative_count", negative),
            _line("neutral_count", neutral),
            _line("effectiveness_score", f"{effectiveness:.3f}"),
            _line("recent_feedback_reasons", recent_reasons or "-"),
            _line("last_feedback_at", entries[-1].created_at),
        ])
        documents.append(MemoryDocument(
            doc_id=_doc_id("insight_feedback", insight_id),
            doc_type="insight_feedback",
            title=title,
            summary=summary,
            body=body,
            refs=refs,
            tags=[
                "insight_feedback",
                "feedback",
                insight.insight_type if insight is not None else "",
            ],
            created_at=entries[0].created_at,
            updated_at=entries[-1].created_at,
            metadata={
                "insight_id": insight_id,
                "feedback_count": len(entries),
                "positive_count": positive,
                "negative_count": negative,
                "neutral_count": neutral,
                "effectiveness_score": effectiveness,
                "last_feedback_at": entries[-1].created_at,
            },
        ))
    return documents


def _graph_documents(graph: MemoryGraph) -> list[MemoryDocument]:
    neighbors = _graph_neighbors_by_node(graph)
    documents: list[MemoryDocument] = []
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        related = neighbors.get(node_id, [])
        related_nodes = sorted({
            other.node_id
            for _, other in related
            if other is not None
        })
        refs = [{"kind": "graph_node", "id": node.node_id}]
        refs.extend(
            {"kind": "graph_node", "id": other_id}
            for other_id in related_nodes
        )
        if node.node_type == "WorkflowTemplate":
            workflow_ids = sorted({
                edge.source_id
                for edge, other in related
                if edge.edge_type in {"HAS_TEMPLATE", "SUCCEEDED_WITH"}
                and edge.source_id.startswith("workflow:")
            })
            failure_ids = sorted({
                other.node_id
                for edge, other in related
                if edge.edge_type == "FAILED_WITH" and other is not None and other.node_type == "FailureMode"
            })
            body = " | ".join([
                _line("template_id", node.node_id),
                _line("credibility_score", f"{float(node.stats.get('credibility_score', 0.0)):.3f}"),
                _line("success_count", int(node.stats.get("success_count", 0))),
                _line("failure_count", int(node.stats.get("failure_count", 0))),
                _line("blocked_count", int(node.stats.get("blocked_count", 0))),
                _line("attached_workflows", ", ".join(workflow_ids)),
                _line("common_failures", ", ".join(failure_ids)),
            ])
            documents.append(MemoryDocument(
                doc_id=_doc_id("workflow_template", node.node_id),
                doc_type="workflow_template",
                title=f"Workflow template {node.name}",
                summary=(
                    f"credibility={float(node.stats.get('credibility_score', 0.0)):.3f} "
                    f"success={int(node.stats.get('success_count', 0))} "
                    f"failure={int(node.stats.get('failure_count', 0))}"
                ),
                body=body,
                refs=refs,
                tags=["workflow_template"],
                created_at=node.created_at,
                updated_at=node.last_seen_at,
                metadata={
                    "template_id": node.node_id,
                    "credibility_score": float(node.stats.get("credibility_score", 0.0)),
                    "success_count": int(node.stats.get("success_count", 0)),
                    "failure_count": int(node.stats.get("failure_count", 0)),
                    "blocked_count": int(node.stats.get("blocked_count", 0)),
                },
            ))
        elif node.node_type == "Capability":
            related_agents = sorted({
                other.node_id
                for edge, other in related
                if other is not None and other.node_type == "Agent"
            })
            related_workflows = sorted({
                other.node_id
                for edge, other in related
                if other is not None and other.node_type == "Workflow"
            })
            body = " | ".join([
                _line("capability_name", node.name),
                _line("request_count", int(node.stats.get("request_count", 0))),
                _line("execution_count", int(node.stats.get("execution_count", 0))),
                _line("blocked_count", int(node.stats.get("blocked_count", 0))),
                _line("failure_count", int(node.stats.get("failure_count", 0))),
                _line("reliability_score", f"{float(node.stats.get('reliability_score', 0.0)):.3f}"),
                _line("related_agents", ", ".join(related_agents)),
                _line("related_workflows", ", ".join(related_workflows)),
            ])
            documents.append(MemoryDocument(
                doc_id=_doc_id("capability", node.node_id),
                doc_type="capability",
                title=f"Capability {node.name}",
                summary=(
                    f"requests={int(node.stats.get('request_count', 0))} "
                    f"reliability={float(node.stats.get('reliability_score', 0.0)):.3f}"
                ),
                body=body,
                refs=refs,
                tags=["capability", node.name],
                created_at=node.created_at,
                updated_at=node.last_seen_at,
                metadata={
                    "capability_id": node.node_id,
                    "capability_name": node.name,
                    "request_count": int(node.stats.get("request_count", 0)),
                    "execution_count": int(node.stats.get("execution_count", 0)),
                    "blocked_count": int(node.stats.get("blocked_count", 0)),
                    "failure_count": int(node.stats.get("failure_count", 0)),
                    "reliability_score": float(node.stats.get("reliability_score", 0.0)),
                },
            ))
        elif node.node_type == "FailureMode":
            body = " | ".join([
                _line("failure_mode_name", node.name),
                _line("occurrence_count", int(node.stats.get("occurrence_count", 0))),
                _line("related_nodes", ", ".join(related_nodes)),
            ])
            documents.append(MemoryDocument(
                doc_id=_doc_id("failure_mode", node.node_id),
                doc_type="failure_mode",
                title=f"Failure mode {node.name}",
                summary=f"occurrence_count={int(node.stats.get('occurrence_count', 0))}",
                body=body,
                refs=refs,
                tags=[
                    "failure_mode",
                    str(node.metadata.get("status") or ""),
                    str(node.metadata.get("error_type") or ""),
                ],
                created_at=node.created_at,
                updated_at=node.last_seen_at,
                metadata={
                    "failure_mode_id": node.node_id,
                    "occurrence_count": int(node.stats.get("occurrence_count", 0)),
                    "status": node.metadata.get("status"),
                    "error_type": node.metadata.get("error_type"),
                },
            ))
        elif node.node_type == "Agent":
            related_capabilities = sorted({
                other.node_id
                for edge, other in related
                if other is not None and other.node_type == "Capability"
            })
            related_workflows = sorted({
                other.node_id
                for edge, other in related
                if other is not None and other.node_type == "Workflow"
            })
            common_friction = sorted({
                other.node_id
                for edge, other in related
                if other is not None and other.node_type in {"Capability", "ApprovalRequest", "FailureMode"}
            })
            body = " | ".join([
                _line("agent_id", node.metadata.get("agent_id") or node.node_id.removeprefix("agent:")),
                _line("blocked_count", int(node.stats.get("blocked_count", 0))),
                _line("approval_requested_count", int(node.stats.get("approval_requested_count", 0))),
                _line("used_capabilities", ", ".join(related_capabilities)),
                _line("created_workflows", ", ".join(related_workflows)),
                _line("common_friction", ", ".join(common_friction)),
            ])
            documents.append(MemoryDocument(
                doc_id=_doc_id("agent_summary", node.node_id),
                doc_type="agent_summary",
                title=f"Agent summary {node.name}",
                summary=(
                    f"blocked={int(node.stats.get('blocked_count', 0))} "
                    f"approval_requested={int(node.stats.get('approval_requested_count', 0))}"
                ),
                body=body,
                refs=refs,
                tags=["agent_summary", node.name],
                created_at=node.created_at,
                updated_at=node.last_seen_at,
                metadata={
                    "agent_node_id": node.node_id,
                    "agent_id": node.metadata.get("agent_id") or node.node_id.removeprefix("agent:"),
                    "blocked_count": int(node.stats.get("blocked_count", 0)),
                    "approval_requested_count": int(node.stats.get("approval_requested_count", 0)),
                },
            ))
    return documents


def _timeline_documents(event_log: EventLog) -> list[MemoryDocument]:
    events = event_log.list_events()
    grouped: dict[str, list[SystemEvent]] = defaultdict(list)
    for event in events:
        if (
            event.correlation_id
            and event.event_type in IMPORTANT_TIMELINE_EVENT_TYPES
        ):
            grouped[event.correlation_id].append(event)
    qualifying = sorted(
        grouped.items(),
        key=lambda item: (
            max(event.event_index for event in item[1]),
            item[0],
        ),
        reverse=True,
    )[:MAX_TIMELINE_TRACES]
    documents: list[MemoryDocument] = []
    for correlation_id, important_events in sorted(qualifying, key=lambda item: item[0]):
        trace = event_log.trace_by_correlation(correlation_id)
        trace.sort(key=lambda item: item.event_index)
        recent_types = sorted({event.event_type for event in important_events})
        refs: list[dict[str, Any]] = [{"kind": "correlation", "id": correlation_id}]
        for event in trace:
            refs.append({"kind": "event", "id": event.event_id})
            if event.workflow_id:
                refs.append({"kind": "workflow", "id": event.workflow_id})
            if event.task_id:
                refs.append({"kind": "task", "id": event.task_id})
            if event.target_id and event.target_type == "capability":
                refs.append({"kind": "capability", "id": event.target_id})
            if event.approval_request_id:
                refs.append({"kind": "approval", "id": event.approval_request_id})
        event_summaries = "; ".join(
            f"{event.event_type}/{event.status}/{event.severity}:{_normalized_text(event.summary)}"
            for event in important_events
        )
        workflows = sorted({event.workflow_id for event in trace if event.workflow_id})
        tasks = sorted({event.task_id for event in trace if event.task_id})
        actors = sorted({event.actor_id for event in trace if event.actor_id})
        targets = sorted({event.target_id for event in trace if event.target_id})
        documents.append(MemoryDocument(
            doc_id=_doc_id("timeline_trace", correlation_id),
            doc_type="timeline_trace",
            title=f"Timeline trace {correlation_id}",
            summary=", ".join(recent_types),
            body=" | ".join([
                _line("correlation_id", correlation_id),
                _line("important_event_summaries", event_summaries),
                _line("actors", ", ".join(actors)),
                _line("targets", ", ".join(targets)),
                _line("workflow_context", ", ".join(workflows)),
                _line("task_context", ", ".join(tasks)),
            ]),
            refs=refs,
            tags=["timeline_trace"] + recent_types,
            created_at=trace[0].timestamp if trace else None,
            updated_at=trace[-1].timestamp if trace else None,
            metadata={
                "correlation_id": correlation_id,
                "important_event_types": recent_types,
                "event_count": len(trace),
            },
        ))
    return documents


def update_memory_retrieval(runtime_root: Path) -> RetrievalUpdateResult:
    from mr1.event_log import EventLog
    from mr1.memory_curator import InsightStore
    from mr1.memory_graph import MemoryGraphStore

    root = _memory_root(runtime_root)
    store = RetrievalStore(root)
    previous = _load_previous_documents(root)
    documents: list[MemoryDocument] = []
    source_counts: dict[str, int] = {doc_type: 0 for doc_type in sorted(RETRIEVAL_DOC_TYPES)}
    errors: list[str] = []

    insight_store = InsightStore(root / "insights")
    insights: dict[str, MemoryInsight] = {}
    try:
        if insight_store.insights_path.exists():
            insights = insight_store.load_insights()
            insight_docs = [_insight_document(insight) for insight in insights.values()]
            documents.extend(insight_docs)
            source_counts["insight"] = len(insight_docs)
    except ValueError as exc:
        errors.append(f"skipped insight retrieval documents: {exc}")

    try:
        if insight_store.feedback_path.exists():
            feedback_docs = _feedback_documents(insights, insight_store.load_feedback())
            documents.extend(feedback_docs)
            source_counts["insight_feedback"] = len(feedback_docs)
    except ValueError as exc:
        errors.append(f"skipped insight feedback retrieval documents: {exc}")

    try:
        graph_store = MemoryGraphStore(root / "graph")
        if graph_store.graph_path.exists():
            graph_docs = _graph_documents(graph_store.load_graph())
            documents.extend(graph_docs)
            type_counter = Counter(item.doc_type for item in graph_docs)
            source_counts["workflow_template"] = int(type_counter.get("workflow_template", 0))
            source_counts["capability"] = int(type_counter.get("capability", 0))
            source_counts["failure_mode"] = int(type_counter.get("failure_mode", 0))
            source_counts["agent_summary"] = int(type_counter.get("agent_summary", 0))
    except ValueError as exc:
        errors.append(f"skipped graph retrieval documents: {exc}")

    try:
        event_log = EventLog(root / "events")
        if event_log.path.exists():
            timeline_docs = _timeline_documents(event_log)
            documents.extend(timeline_docs)
            source_counts["timeline_trace"] = len(timeline_docs)
    except ValueError as exc:
        errors.append(f"skipped timeline trace retrieval documents: {exc}")

    documents = sorted(documents, key=_doc_sort_key)
    created = sum(1 for item in documents if item.doc_id not in previous)
    updated = sum(
        1
        for item in documents
        if item.doc_id in previous and previous[item.doc_id] != item
    )
    store.save(documents, source_counts=source_counts, updated_at=_now_iso())
    return RetrievalUpdateResult(
        documents_written=len(documents),
        documents_created=created,
        documents_updated=updated,
        source_counts=source_counts,
        errors=errors,
    )


class LexicalMemoryRetriever:
    def __init__(self, runtime_root: Path):
        self._store = RetrievalStore(runtime_root)
        self._documents = self._store.load_documents()

    def search(
        self,
        query: str,
        limit: int = 5,
        types: Optional[list[str]] = None,
    ) -> list[MemorySearchResult]:
        normalized_query = _normalized_lower(query)
        if not normalized_query:
            raise ValueError("query must be a non-empty string")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be an integer >= 0")
        validated_types: Optional[set[str]] = None
        if types is not None:
            if not isinstance(types, list) or any(not isinstance(item, str) or not item.strip() for item in types):
                raise ValueError("types must be a list of non-empty strings")
            normalized_types = {_normalized_text(item) for item in types}
            invalid = sorted(item for item in normalized_types if item not in RETRIEVAL_DOC_TYPES)
            if invalid:
                raise ValueError(f"invalid retrieval document types: {', '.join(invalid)}")
            validated_types = normalized_types
        query_tokens = sorted(set(_tokenize(normalized_query)))
        ranked: list[tuple[float, MemoryDocument]] = []
        for document in self._documents:
            if validated_types is not None and document.doc_type not in validated_types:
                continue
            title_tokens = set(_tokenize(document.title))
            summary_tokens = set(_tokenize(document.summary))
            body_tokens = set(_tokenize(document.body))
            tag_tokens = set(token for tag in document.tags for token in _tokenize(tag))
            score = 0.0
            for token in query_tokens:
                if token in title_tokens:
                    score += 5.0
                if token in summary_tokens:
                    score += 3.0
                if token in body_tokens:
                    score += 1.0
                if token in tag_tokens:
                    score += 2.0
            combined = " ".join([document.title.lower(), document.summary.lower(), document.body.lower()])
            if normalized_query in combined:
                score += 4.0
            if score <= 0.0:
                continue
            ranked.append((score, document))
        ranked.sort(
            key=lambda item: (
                -item[0],
                _iso_sort_key(item[1].updated_at),
                item[1].doc_id,
            ),
        )
        results = [
            MemorySearchResult(
                doc_id=document.doc_id,
                doc_type=document.doc_type,
                title=document.title,
                summary=document.summary,
                score=score,
                refs=document.refs,
                tags=document.tags,
                metadata=document.metadata,
            )
            for score, document in ranked[:limit]
        ]
        return results
