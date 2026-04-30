"""
Deterministic insight feedback and memory maintenance helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from mr1.event_log import EventLog, SystemEvent
from mr1.memory_curator import (
    InsightStore,
    MemoryInsight,
    default_insight_stats,
)
from mr1.memory_graph import MemoryGraphStore, capability_node_id, update_graph_from_events
from mr1.memory_curator import run_memory_curation
from mr1.scoped_agents import PersistentAgentStore
from mr1.workflow_models import Provenance, Workflow
from mr1.workflow_store import WorkflowStore


OUTCOME_RELEVANT_EVENT_TYPES = frozenset({
    "workflow_completed",
    "workflow_failed",
    "workflow_task_failed",
    "capability_blocked",
    "approval_denied",
})
_NEGATIVE_DEMOTION_THRESHOLD = 3
_STALE_EFFECTIVENESS_THRESHOLD = 0.30


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_root(runtime_root: Optional[Path]) -> Path:
    if runtime_root is None:
        return Path(__file__).resolve().parent / "memory"
    return Path(runtime_root)


def _dedupe_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        ordered.append(ref)
    return ordered


@dataclass(frozen=True)
class InsightFeedback:
    feedback_id: str
    insight_id: str
    workflow_id: Optional[str]
    event_id: str
    event_index: int
    outcome: str
    reason: str
    confidence_delta: float
    evaluator_type: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in {"positive", "negative", "neutral"}:
            raise ValueError(f"invalid outcome '{self.outcome}'")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
        if not isinstance(self.evaluator_type, str) or not self.evaluator_type.strip():
            raise ValueError("evaluator_type must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "insight_id": self.insight_id,
            "workflow_id": self.workflow_id,
            "event_id": self.event_id,
            "event_index": int(self.event_index),
            "outcome": self.outcome,
            "reason": self.reason,
            "confidence_delta": float(self.confidence_delta),
            "evaluator_type": self.evaluator_type,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InsightFeedback":
        return cls(
            feedback_id=data["feedback_id"],
            insight_id=data["insight_id"],
            workflow_id=data.get("workflow_id"),
            event_id=data["event_id"],
            event_index=int(data["event_index"]),
            outcome=data["outcome"],
            reason=data["reason"],
            confidence_delta=float(data["confidence_delta"]),
            evaluator_type=data["evaluator_type"],
            created_at=data["created_at"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class FeedbackUpdateResult:
    processed_events: int
    feedback_created: int
    insights_updated: int
    last_evaluated_event_index: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_events": self.processed_events,
            "feedback_created": self.feedback_created,
            "insights_updated": self.insights_updated,
            "last_evaluated_event_index": self.last_evaluated_event_index,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class MemoryFeedbackDueResult:
    due: bool
    latest_event_index: int
    last_evaluated_event_index: int
    relevant_event_count: int
    relevant_event_types: list[str]
    suggested_event_window: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "due": self.due,
            "latest_event_index": self.latest_event_index,
            "last_evaluated_event_index": self.last_evaluated_event_index,
            "relevant_event_count": self.relevant_event_count,
            "relevant_event_types": list(self.relevant_event_types),
            "suggested_event_window": list(self.suggested_event_window),
        }


class InsightEvaluator(Protocol):
    evaluator_type: str

    def evaluate(
        self,
        event: SystemEvent,
        workflow: Optional[Workflow],
        insight: MemoryInsight,
        context: dict[str, Any],
    ) -> Optional[InsightFeedback]: ...


class DeterministicInsightEvaluator:
    evaluator_type = "deterministic"

    def evaluate(
        self,
        event: SystemEvent,
        workflow: Optional[Workflow],
        insight: MemoryInsight,
        context: dict[str, Any],
    ) -> Optional[InsightFeedback]:
        del workflow
        outcome: Optional[str] = None
        reason: Optional[str] = None
        confidence_delta = 0.0

        if event.event_type == "workflow_completed":
            outcome = "positive"
            reason = "workflow completed after using this memory ref"
            confidence_delta = 0.03
        elif event.event_type == "workflow_failed":
            outcome = "negative"
            reason = "workflow failed after using this memory ref"
            confidence_delta = -0.05
        elif event.event_type == "workflow_task_failed":
            outcome = "negative"
            reason = "workflow task failed after using this memory ref"
            confidence_delta = -0.05
        elif event.event_type == "approval_denied":
            outcome = "negative"
            reason = "approval denied after memory-informed workflow planning"
            confidence_delta = -0.05
        elif event.event_type == "capability_blocked":
            capability_name = str(context.get("capability_name") or "").strip()
            related = _insight_related_to_capability(insight, capability_name)
            if related:
                outcome = "negative"
                reason = "capability was still blocked despite related memory insight"
                confidence_delta = -0.05
            else:
                outcome = "neutral"
                reason = "workflow encountered unrelated capability block"
                confidence_delta = 0.0

        if outcome is None or reason is None:
            return None
        return InsightFeedback(
            feedback_id=feedback_id_for(
                event_id=event.event_id,
                insight_id=insight.insight_id,
                evaluator_type=self.evaluator_type,
            ),
            insight_id=insight.insight_id,
            workflow_id=event.workflow_id,
            event_id=event.event_id,
            event_index=event.event_index,
            outcome=outcome,
            reason=reason,
            confidence_delta=confidence_delta,
            evaluator_type=self.evaluator_type,
            created_at=_now_iso(),
            metadata={
                "event_type": event.event_type,
                "memory_refs": list(context.get("memory_refs", [])),
                "capability_name": context.get("capability_name"),
            },
        )


def feedback_id_for(*, event_id: str, insight_id: str, evaluator_type: str) -> str:
    return f"feedback:{event_id}:{insight_id}:{evaluator_type}"


def extract_memory_refs_for_event(event: SystemEvent, workflow_store: WorkflowStore) -> list[str]:
    refs: list[str] = []
    metadata_refs = event.metadata.get("memory_refs_used")
    if isinstance(metadata_refs, list):
        refs.extend(item for item in metadata_refs if isinstance(item, str) and item.strip())
    if event.workflow_id:
        workflow = workflow_store.load_workflow(event.workflow_id)
        if workflow is not None:
            workflow_refs = workflow.metadata.get("memory_refs_used")
            if isinstance(workflow_refs, list):
                refs.extend(item for item in workflow_refs if isinstance(item, str) and item.strip())
    return _dedupe_refs(refs)


def evaluate_memory_feedback_due(
    event_log: EventLog,
    insight_store: InsightStore,
    workflow_store: WorkflowStore,
) -> MemoryFeedbackDueResult:
    events = event_log.list_events()
    latest_event_index = events[-1].event_index if events else 0
    last_evaluated_event_index = insight_store.load_feedback_cursor()
    relevant_events = [
        event
        for event in events
        if event.event_index > last_evaluated_event_index
        and event.event_type in OUTCOME_RELEVANT_EVENT_TYPES
        and extract_memory_refs_for_event(event, workflow_store)
    ]
    relevant_types = sorted({event.event_type for event in relevant_events})
    due = bool(relevant_events)
    window_start = last_evaluated_event_index + 1 if latest_event_index > last_evaluated_event_index else 0
    window = [window_start, latest_event_index] if window_start > 0 and latest_event_index >= window_start else []
    return MemoryFeedbackDueResult(
        due=due,
        latest_event_index=latest_event_index,
        last_evaluated_event_index=last_evaluated_event_index,
        relevant_event_count=len(relevant_events),
        relevant_event_types=relevant_types,
        suggested_event_window=window,
    )


def evaluate_memory_feedback_due_for_runtime_root(runtime_root: Optional[Path] = None) -> MemoryFeedbackDueResult:
    root = _memory_root(runtime_root)
    return evaluate_memory_feedback_due(
        EventLog(root / "events"),
        InsightStore(root / "insights"),
        WorkflowStore(root=root / "workflows"),
    )


def update_insight_feedback(
    event_log: EventLog,
    insight_store: InsightStore,
    workflow_store: WorkflowStore,
    evaluator: Optional[InsightEvaluator] = None,
) -> FeedbackUpdateResult:
    events = event_log.list_events()
    last_evaluated_event_index = insight_store.load_feedback_cursor()
    pending_events = [event for event in events if event.event_index > last_evaluated_event_index]
    relevant_events = [event for event in pending_events if event.event_type in OUTCOME_RELEVANT_EVENT_TYPES]
    insights = insight_store.load_insights()
    evaluator_impl = evaluator or DeterministicInsightEvaluator()
    existing_feedback_ids = {item.feedback_id for item in insight_store.load_feedback()}
    pending_feedback: list[InsightFeedback] = []
    updated_insight_ids: set[str] = set()
    errors: list[str] = []
    staged_insights = dict(insights)

    for event in relevant_events:
        memory_refs = extract_memory_refs_for_event(event, workflow_store)
        if not memory_refs:
            continue
        workflow = workflow_store.load_workflow(event.workflow_id) if event.workflow_id else None
        capability_name = _event_capability_name(event)
        for ref in memory_refs:
            insight = staged_insights.get(ref)
            if insight is None:
                errors.append(f"unknown memory ref: {ref}")
                continue
            context = {
                "memory_refs": list(memory_refs),
                "capability_name": capability_name,
            }
            feedback = evaluator_impl.evaluate(event, workflow, insight, context)
            if feedback is None:
                continue
            if feedback.feedback_id in existing_feedback_ids or any(
                item.feedback_id == feedback.feedback_id for item in pending_feedback
            ):
                continue
            pending_feedback.append(feedback)
            _apply_feedback_to_insight(insight, feedback)
            updated_insight_ids.add(insight.insight_id)

    for feedback in pending_feedback:
        insight_store.append_feedback(feedback)
    if updated_insight_ids:
        insight_store.save_insights(staged_insights)
    latest_processed_index = (
        relevant_events[-1].event_index
        if relevant_events else
        last_evaluated_event_index
    )
    insight_store.save_feedback_cursor(latest_processed_index)
    return FeedbackUpdateResult(
        processed_events=len(relevant_events),
        feedback_created=len(pending_feedback),
        insights_updated=len(updated_insight_ids),
        last_evaluated_event_index=latest_processed_index,
        errors=errors,
    )


def _apply_feedback_to_insight(insight: MemoryInsight, feedback: InsightFeedback) -> None:
    stats = default_insight_stats()
    stats.update(insight.stats)
    stats["used_count"] = int(stats.get("used_count", 0)) + 1
    if feedback.outcome == "positive":
        stats["positive_outcome_count"] = int(stats.get("positive_outcome_count", 0)) + 1
    elif feedback.outcome == "negative":
        stats["negative_outcome_count"] = int(stats.get("negative_outcome_count", 0)) + 1
    else:
        stats["neutral_outcome_count"] = int(stats.get("neutral_outcome_count", 0)) + 1
    positive = int(stats.get("positive_outcome_count", 0))
    negative = int(stats.get("negative_outcome_count", 0))
    neutral = int(stats.get("neutral_outcome_count", 0))
    stats["effectiveness_score"] = positive / (positive + negative + neutral + 1)
    stats["last_used_at"] = feedback.created_at
    stats["last_feedback_at"] = feedback.created_at
    insight.stats = stats
    insight.confidence = max(0.0, min(1.0, float(insight.confidence) + feedback.confidence_delta))
    insight.updated_at = feedback.created_at
    if insight.status != "dismissed":
        if (
            int(stats.get("negative_outcome_count", 0)) >= _NEGATIVE_DEMOTION_THRESHOLD
            and float(stats.get("effectiveness_score", 0.0)) < _STALE_EFFECTIVENESS_THRESHOLD
        ):
            insight.status = "stale"


def _event_capability_name(event: SystemEvent) -> str:
    target = str(event.target_id or "").strip()
    if target:
        return target
    metadata_name = event.metadata.get("capability_name")
    return str(metadata_name or "").strip()


def _insight_related_to_capability(insight: MemoryInsight, capability_name: str) -> bool:
    if not capability_name:
        return False
    normalized_node_id = capability_node_id(capability_name)
    if normalized_node_id in set(insight.related_nodes):
        return True
    metadata_capability = insight.metadata.get("capability_id")
    if isinstance(metadata_capability, str):
        if metadata_capability == capability_name or metadata_capability == normalized_node_id:
            return True
    return False


def build_memory_maintenance_spec() -> dict[str, Any]:
    from mr1.scheduler import validate_spec

    spec = {
        "title": "Memory maintenance",
        "tasks": [
            {
                "label": "curation_due",
                "title": "Check memory curation",
                "task_kind": "watcher",
                "watcher_type": "memory_curation_due",
                "watch_config": {},
            },
            {
                "label": "graph_update",
                "title": "Update graph memory",
                "task_kind": "tool",
                "tool_type": "memory_graph_update",
                "depends_on": ["curation_due"],
                "run_if": {
                    "ref": "curation_due.result.data.due",
                    "op": "eq",
                    "value": True,
                },
            },
            {
                "label": "curate",
                "title": "Curate insights",
                "task_kind": "tool",
                "tool_type": "memory_curate",
                "depends_on": ["graph_update"],
            },
            {
                "label": "feedback_due",
                "title": "Check insight feedback",
                "task_kind": "watcher",
                "watcher_type": "memory_feedback_due",
                "watch_config": {},
            },
            {
                "label": "feedback_update",
                "title": "Update insight feedback",
                "task_kind": "tool",
                "tool_type": "memory_feedback_update",
                "depends_on": ["feedback_due"],
                "run_if": {
                    "ref": "feedback_due.result.data.due",
                    "op": "eq",
                    "value": True,
                },
            },
            {
                "label": "retrieval_update",
                "title": "Update unified retrieval documents",
                "task_kind": "tool",
                "tool_type": "memory_retrieval_update",
                "depends_on": ["curate", "feedback_update"],
                "dependency_policy": "any_succeeded",
            },
        ],
    }
    validate_spec(spec)
    return spec


def submit_memory_maintenance_workflow(
    store: WorkflowStore,
    *,
    scoped_agent_store: Optional[PersistentAgentStore] = None,
) -> str:
    from mr1.scheduler import submit_spec_to_disk

    scoped_agents = scoped_agent_store or PersistentAgentStore(root=store.root.parent / "agents")
    root_agent_id = scoped_agents.root_agent_id
    return submit_spec_to_disk(
        build_memory_maintenance_spec(),
        Provenance(type="agent", id="MR1"),
        store,
        owner_agent_id=root_agent_id,
        caller_agent_id=root_agent_id,
        workflow_metadata={"system_workflow": "memory_maintenance", "phase": 21},
        scoped_agent_store=scoped_agents,
    )


def latest_memory_maintenance_workflow(store: WorkflowStore) -> Optional[Workflow]:
    matches = [
        workflow
        for workflow in store.list_workflows()
        if workflow.metadata.get("system_workflow") == "memory_maintenance"
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (item.created_at, item.workflow_id), reverse=True)
    return matches[0]


def run_memory_graph_update(*, event_log: EventLog, graph_store: MemoryGraphStore) -> dict[str, Any]:
    return update_graph_from_events(event_log, graph_store).to_dict()


def run_memory_curate(
    *,
    event_log: EventLog,
    graph_store: MemoryGraphStore,
    insight_store: InsightStore,
    trigger_reason: str = "memory_curate_capability",
    persist_not_due: bool = True,
) -> dict[str, Any]:
    return run_memory_curation(
        event_log=event_log,
        graph_store=graph_store,
        insight_store=insight_store,
        trigger_reason=trigger_reason,
        persist_not_due=persist_not_due,
    ).to_dict()


def maintenance_status_payload(store: WorkflowStore) -> dict[str, Any]:
    workflow = latest_memory_maintenance_workflow(store)
    if workflow is None:
        return {"found": False, "workflow": None}
    return {
        "found": True,
        "workflow": {
            "workflow_id": workflow.workflow_id,
            "title": workflow.title,
            "status": workflow.status.value,
            "created_at": workflow.created_at,
            "finished_at": workflow.finished_at,
            "metadata": dict(workflow.metadata),
        },
    }
