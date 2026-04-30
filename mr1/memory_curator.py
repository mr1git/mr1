"""
LLM-assisted memory curator for validated insight generation.

Phase 19 keeps curation bounded and advisory-only. The curator may
recommend actions, but it never executes them or mutates runtime
behavior outside the insight store.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from mr1.agents import (
    AgentRuntimeError,
    build_agent_command,
    load_agent_runtime_config,
    parse_agent_json_envelope,
)
from mr1.capability_policy import normalize_path
from mr1.core import Dispatcher, PermissionDenied
from mr1.event_log import EventLog, SystemEvent
from mr1.memory_graph import (
    MemoryGraph,
    MemoryGraphStore,
    agent_node_id,
    capability_node_id,
    capability_stats,
    failure_modes,
    failure_node_id,
    graph_stats,
    top_workflow_templates,
    update_graph_from_events,
    workflow_template_node_id,
)


CuratorFn = Callable[[str, str], str]

if TYPE_CHECKING:
    from mr1.memory_feedback import InsightFeedback

IMPORTANT_EVENT_TYPES = frozenset({
    "workflow_failed",
    "workflow_completed",
    "capability_blocked",
    "capability_failed",
    "approval_approved",
    "approval_denied",
    "approval_consumed",
})
_ALLOWED_EVIDENCE_SOURCE_TYPES = frozenset({"event", "node", "edge", "query"})
_ALLOWED_INSIGHT_TYPES = frozenset({
    "workflow_recommendation",
    "capability_friction",
    "approval_friction",
    "failure_pattern",
    "agent_pattern",
    "scope_recommendation",
    "system_design_lesson",
})
_ALLOWED_SEVERITIES = frozenset({"INFO", "WARNING", "ERROR", "CRITICAL"})
_ALLOWED_STATUSES = frozenset({"active", "stale", "dismissed", "superseded"})
_ALLOWED_RUN_STATUSES = frozenset({"succeeded", "failed", "not_due"})
_MUTATING_ACTION_PREFIXES = (
    "run",
    "execute",
    "invoke",
    "spawn",
    "approve",
    "grant",
    "revoke",
    "modify",
    "edit",
    "write",
    "delete",
    "apply",
    "patch",
    "select",
)
_ADVISORY_ACTION_PREFIXES = (
    "consider",
    "review",
    "investigate",
    "prefer",
    "avoid",
    "document",
    "monitor",
    "evaluate",
)
_QUERY_SOURCE_IDS = frozenset({
    "query:graph_stats",
    "query:top_workflow_templates",
    "query:capability_stats",
    "query:failure_modes",
    "query:approval_history",
    "query:existing_related_insights",
})
_WORKFLOW_TEMPLATE_LIMIT = 10
_CAPABILITY_LIMIT = 10
_FAILURE_LIMIT = 10
_APPROVAL_LIMIT = 25
_RELATED_INSIGHT_LIMIT = 20

_MEMORY_CURATOR_SYSTEM_PROMPT = """\
You are MemoryCurator for MR1.

You are a memory curator, not an actor.
You may recommend actions, but you may not execute them.
You must return JSON only with this exact top-level shape:
{
  "insights": [
    {
      "insight_type": "workflow_recommendation | capability_friction | approval_friction | failure_pattern | agent_pattern | scope_recommendation | system_design_lesson",
      "title": "string",
      "summary": "string",
      "confidence": 0.0,
      "severity": "INFO | WARNING | ERROR | CRITICAL",
      "recommended_action": "advisory-only text",
      "evidence": [
        {
          "source_type": "event | node | edge | query",
          "source_id": "string",
          "reason": "string"
        }
      ],
      "related_nodes": ["node_id"],
      "metadata": {}
    }
  ],
  "stale_insight_ids": ["insight_id"]
}

Rules:
- Return exactly one JSON object and nothing else.
- Do not use markdown fences.
- Only cite evidence refs that appear in the provided bundle.
- Do not invent evidence.
- Prefer high-value insights that improve future workflow planning, scope management, approval reduction, failure avoidance, or system design.
- Avoid obvious low-value observations.
- recommended_action must stay advisory. Do not output executable or mutating instructions.
- stale_insight_ids may only reference provided existing related insights.

Metadata requirements for stable IDs:
- workflow_recommendation: include metadata.workflow_template_id
- capability_friction: include metadata.capability_id
- approval_friction: include metadata.agent_id
- failure_pattern: include metadata.failure_mode_id
- agent_pattern: include metadata.agent_id
- scope_recommendation: include metadata.agent_id and metadata.path
- system_design_lesson: include metadata.topic when possible
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def default_insight_stats() -> dict[str, Any]:
    return {
        "used_count": 0,
        "positive_outcome_count": 0,
        "negative_outcome_count": 0,
        "neutral_outcome_count": 0,
        "effectiveness_score": 0.0,
        "last_used_at": None,
        "last_feedback_at": None,
    }


def _normalize_insight_stats(value: Any) -> dict[str, Any]:
    stats = default_insight_stats()
    if not isinstance(value, dict):
        return stats
    for key in (
        "used_count",
        "positive_outcome_count",
        "negative_outcome_count",
        "neutral_outcome_count",
    ):
        raw = value.get(key, stats[key])
        try:
            stats[key] = max(0, int(raw))
        except (TypeError, ValueError):
            stats[key] = default_insight_stats()[key]
    try:
        score = float(value.get("effectiveness_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    stats["effectiveness_score"] = max(0.0, min(1.0, score))
    for key in ("last_used_at", "last_feedback_at"):
        raw = value.get(key)
        stats[key] = raw if isinstance(raw, str) and raw.strip() else None
    return stats


def _snippet(text: str, *, limit: int = 240) -> str:
    compact = " ".join(str(text).split())
    if len(compact) > limit:
        compact = compact[:limit] + "..."
    return compact


def _extract_json_object(text: str) -> dict[str, Any]:
    payload = text.strip()
    if not payload:
        raise MemoryCuratorFailure("memory curator returned empty output")
    if payload.startswith("```"):
        parts = payload.split("```")
        for part in parts:
            part = part.strip()
            if not part or part == "json":
                continue
            payload = part.removeprefix("json").strip()
            break
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MemoryCuratorFailure(
            f"memory curator returned invalid JSON: {exc}; raw={_snippet(payload)!r}"
        ) from exc
    if not isinstance(data, dict):
        raise MemoryCuratorFailure(
            f"memory curator must return a JSON object; raw={_snippet(payload)!r}"
        )
    return data


def _normalize_insight_id_component(value: str) -> str:
    return value.strip()


def _event_summary(event: SystemEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_index": event.event_index,
        "timestamp": event.timestamp,
        "event_type": event.event_type,
        "status": event.status,
        "severity": event.severity,
        "summary": event.summary,
        "actor_id": event.actor_id,
        "target_id": event.target_id,
        "workflow_id": event.workflow_id,
        "task_id": event.task_id,
        "approval_request_id": event.approval_request_id,
    }


def _compact_approval_event(event: SystemEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_index": event.event_index,
        "timestamp": event.timestamp,
        "event_type": event.event_type,
        "status": event.status,
        "approval_request_id": event.approval_request_id,
        "actor_id": event.actor_id,
        "target_id": event.target_id,
        "summary": event.summary,
    }


@dataclass(frozen=True)
class EvidenceRef:
    source_type: str
    source_id: str
    reason: str

    def __post_init__(self) -> None:
        if self.source_type not in _ALLOWED_EVIDENCE_SOURCE_TYPES:
            raise ValueError(f"invalid source_type '{self.source_type}'")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRef":
        return cls(
            source_type=str(data["source_type"]),
            source_id=str(data["source_id"]),
            reason=str(data["reason"]),
        )


@dataclass
class MemoryInsight:
    insight_id: str
    insight_type: str
    title: str
    summary: str
    confidence: float
    severity: str
    recommended_action: str
    evidence: list[EvidenceRef]
    related_nodes: list[str]
    created_at: str
    updated_at: str
    status: str
    stats: dict[str, Any] = field(default_factory=default_insight_stats)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.insight_type not in _ALLOWED_INSIGHT_TYPES:
            raise ValueError(f"invalid insight_type '{self.insight_type}'")
        if self.severity not in _ALLOWED_SEVERITIES:
            raise ValueError(f"invalid severity '{self.severity}'")
        if self.status not in _ALLOWED_STATUSES:
            raise ValueError(f"invalid status '{self.status}'")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not isinstance(self.stats, dict):
            raise ValueError("stats must be a dict")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict")
        if not isinstance(self.evidence, list) or not self.evidence:
            raise ValueError("evidence must be a non-empty list")
        if any(not isinstance(item, EvidenceRef) for item in self.evidence):
            raise ValueError("evidence must contain EvidenceRef values")
        if any(not isinstance(item, str) or not item for item in self.related_nodes):
            raise ValueError("related_nodes must contain non-empty strings")
        self.stats = _normalize_insight_stats(self.stats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type,
            "title": self.title,
            "summary": self.summary,
            "confidence": float(self.confidence),
            "severity": self.severity,
            "recommended_action": self.recommended_action,
            "evidence": [item.to_dict() for item in self.evidence],
            "related_nodes": list(self.related_nodes),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "stats": dict(self.stats),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryInsight":
        return cls(
            insight_id=data["insight_id"],
            insight_type=data["insight_type"],
            title=data["title"],
            summary=data["summary"],
            confidence=float(data["confidence"]),
            severity=data["severity"],
            recommended_action=data["recommended_action"],
            evidence=[EvidenceRef.from_dict(item) for item in list(data.get("evidence", []))],
            related_nodes=list(data.get("related_nodes", [])),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=data["status"],
            stats=_normalize_insight_stats(data.get("stats")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MemoryCurationRun:
    run_id: str
    started_at: str
    ended_at: Optional[str]
    event_start_index: int
    event_end_index: int
    trigger_reason: str
    status: str
    input_summary: dict[str, Any]
    output_insight_ids: list[str]
    errors: list[str]

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_RUN_STATUSES:
            raise ValueError(f"invalid run status '{self.status}'")
        if not isinstance(self.input_summary, dict):
            raise ValueError("input_summary must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "event_start_index": self.event_start_index,
            "event_end_index": self.event_end_index,
            "trigger_reason": self.trigger_reason,
            "status": self.status,
            "input_summary": dict(self.input_summary),
            "output_insight_ids": list(self.output_insight_ids),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryCurationRun":
        return cls(
            run_id=data["run_id"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            event_start_index=int(data["event_start_index"]),
            event_end_index=int(data["event_end_index"]),
            trigger_reason=data["trigger_reason"],
            status=data["status"],
            input_summary=dict(data.get("input_summary", {})),
            output_insight_ids=list(data.get("output_insight_ids", [])),
            errors=list(data.get("errors", [])),
        )


@dataclass(frozen=True)
class MemoryCurationBundle:
    event_start_index: int
    event_end_index: int
    important_events: list[dict[str, Any]]
    graph_stats: dict[str, Any]
    top_workflow_templates: list[dict[str, Any]]
    capability_stats: list[dict[str, Any]]
    failure_modes: list[dict[str, Any]]
    approval_history: list[dict[str, Any]]
    existing_related_insights: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_start_index": self.event_start_index,
            "event_end_index": self.event_end_index,
            "important_events": list(self.important_events),
            "graph_stats": dict(self.graph_stats),
            "top_workflow_templates": list(self.top_workflow_templates),
            "capability_stats": list(self.capability_stats),
            "failure_modes": list(self.failure_modes),
            "approval_history": list(self.approval_history),
            "existing_related_insights": list(self.existing_related_insights),
        }


@dataclass(frozen=True)
class MemoryCurationDueResult:
    due: bool
    latest_event_index: int
    last_curated_event_index: int
    important_event_count: int
    important_event_types: list[str]
    suggested_event_window: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "due": self.due,
            "latest_event_index": self.latest_event_index,
            "last_curated_event_index": self.last_curated_event_index,
            "important_event_count": self.important_event_count,
            "important_event_types": list(self.important_event_types),
            "suggested_event_window": list(self.suggested_event_window),
        }


@dataclass(frozen=True)
class MemoryCuratorCandidateOutput:
    insights: list[dict[str, Any]]
    stale_insight_ids: list[str]
    raw_output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "insights": list(self.insights),
            "stale_insight_ids": list(self.stale_insight_ids),
        }


class MemoryCuratorFailure(ValueError):
    """Deterministic memory curator failure."""


class InsightStore:
    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def insights_path(self) -> Path:
        return self._root / "insights.json"

    @property
    def cursor_path(self) -> Path:
        return self._root / "cursor.json"

    @property
    def runs_path(self) -> Path:
        return self._root / "curation_runs.jsonl"

    @property
    def feedback_path(self) -> Path:
        return self._root / "feedback.jsonl"

    @property
    def feedback_cursor_path(self) -> Path:
        return self._root / "feedback_cursor.json"

    def load_cursor(self) -> int:
        if not self.cursor_path.exists():
            return 0
        try:
            payload = json.loads(self.cursor_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("cursor payload must be an object")
            value = int(payload["last_curated_event_index"])
            if value < 0:
                raise ValueError("cursor must be >= 0")
            return value
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt cursor file: {self.cursor_path}") from exc

    def save_cursor(self, last_curated_event_index: int) -> None:
        self._write_json_file(
            self.cursor_path,
            {"last_curated_event_index": int(last_curated_event_index)},
        )

    def load_insights(self) -> dict[str, MemoryInsight]:
        if not self.insights_path.exists():
            return {}
        try:
            payload = json.loads(self.insights_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("insights payload must be an object")
            insights = {
                insight_id: MemoryInsight.from_dict(item)
                for insight_id, item in payload.items()
            }
            return dict(sorted(insights.items()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt insights file: {self.insights_path}") from exc

    def save_insights(self, insights: dict[str, MemoryInsight]) -> None:
        payload = {
            insight_id: insights[insight_id].to_dict()
            for insight_id in sorted(insights)
        }
        self._write_json_file(self.insights_path, payload)

    def append_run(self, run: MemoryCurationRun) -> None:
        with open(self.runs_path, "a", encoding="utf-8") as handle:
            handle.write(_canonical_json(run.to_dict()) + "\n")

    def load_runs(self, *, limit: Optional[int] = None) -> list[MemoryCurationRun]:
        if not self.runs_path.exists():
            return []
        runs: list[MemoryCurationRun] = []
        try:
            with open(self.runs_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        continue
                    runs.append(MemoryCurationRun.from_dict(payload))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt curation run log: {self.runs_path}") from exc
        runs.sort(key=lambda item: (item.started_at, item.run_id), reverse=True)
        return runs if limit is None else runs[:limit]

    def load_feedback_cursor(self) -> int:
        if not self.feedback_cursor_path.exists():
            return 0
        try:
            payload = json.loads(self.feedback_cursor_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("feedback cursor payload must be an object")
            value = int(payload["last_evaluated_event_index"])
            if value < 0:
                raise ValueError("feedback cursor must be >= 0")
            return value
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt feedback cursor file: {self.feedback_cursor_path}") from exc

    def save_feedback_cursor(self, last_evaluated_event_index: int) -> None:
        self._write_json_file(
            self.feedback_cursor_path,
            {"last_evaluated_event_index": int(last_evaluated_event_index)},
        )

    def feedback_exists(self, feedback_id: str) -> bool:
        return any(item.feedback_id == feedback_id for item in self.load_feedback())

    def append_feedback(self, feedback: "InsightFeedback") -> None:
        with open(self.feedback_path, "a", encoding="utf-8") as handle:
            handle.write(_canonical_json(feedback.to_dict()) + "\n")

    def load_feedback(
        self,
        *,
        limit: Optional[int] = None,
        insight_id: Optional[str] = None,
    ) -> list["InsightFeedback"]:
        if not self.feedback_path.exists():
            return []
        from mr1.memory_feedback import InsightFeedback

        items: list[InsightFeedback] = []
        try:
            with open(self.feedback_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        continue
                    item = InsightFeedback.from_dict(payload)
                    if insight_id is not None and item.insight_id != insight_id:
                        continue
                    items.append(item)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"corrupt insight feedback log: {self.feedback_path}") from exc
        items.sort(key=lambda item: (item.created_at, item.feedback_id), reverse=True)
        return items if limit is None else items[:limit]

    def _write_json_file(self, path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(_json_dumps(payload))
            handle.write("\n")
        tmp.replace(path)


def _default_runtime_root() -> Path:
    return Path(__file__).resolve().parent / "memory"


def default_insight_store() -> InsightStore:
    return InsightStore(_default_runtime_root() / "insights")


def evaluate_memory_curation_due(
    event_log: EventLog,
    insight_store: InsightStore,
) -> MemoryCurationDueResult:
    events = event_log.list_events()
    latest_event_index = events[-1].event_index if events else 0
    last_curated_event_index = insight_store.load_cursor()
    important_events = [
        event
        for event in events
        if event.event_index > last_curated_event_index
        and event.event_type in IMPORTANT_EVENT_TYPES
    ]
    important_types = sorted({event.event_type for event in important_events})
    due = bool(important_events)
    window_start = last_curated_event_index + 1 if latest_event_index > last_curated_event_index else 0
    window = [window_start, latest_event_index] if latest_event_index >= window_start and window_start > 0 else []
    return MemoryCurationDueResult(
        due=due,
        latest_event_index=latest_event_index,
        last_curated_event_index=last_curated_event_index,
        important_event_count=len(important_events),
        important_event_types=important_types,
        suggested_event_window=window,
    )


def evaluate_memory_curation_due_for_runtime_root(runtime_root: Optional[Path] = None) -> MemoryCurationDueResult:
    root = Path(runtime_root) if runtime_root is not None else _default_runtime_root()
    event_log = EventLog(root / "events")
    insight_store = InsightStore(root / "insights")
    return evaluate_memory_curation_due(event_log, insight_store)


def run_memory_curator_agent(system_prompt: str, prompt: str) -> str:
    config = load_agent_runtime_config("memory_curator")
    cmd = build_agent_command(
        "memory_curator",
        prompt,
        config=config,
    )
    cmd.extend(["--append-system-prompt", system_prompt])
    cli_flags = [tok for tok in cmd[1:] if tok.startswith("-")]
    dispatcher = Dispatcher()
    try:
        dispatcher.validate_full_spawn(
            "memory_curator",
            cli_flags,
            list(config.get("allowed_tools", [])),
        )
    except PermissionDenied as exc:
        raise MemoryCuratorFailure(str(exc)) from exc
    timeout_s = int(config.get("timeout_s") or 300)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise MemoryCuratorFailure(
            f"memory curator timed out after {timeout_s}s"
        ) from exc
    except OSError as exc:
        raise MemoryCuratorFailure(f"failed to run memory curator: {exc}") from exc
    raw_output = proc.stdout or ""
    if proc.returncode != 0:
        detail = (proc.stderr or raw_output).strip() or f"exit {proc.returncode}"
        raise MemoryCuratorFailure(f"memory curator failed: {detail}")
    try:
        parsed = parse_agent_json_envelope(raw_output)
    except AgentRuntimeError as exc:
        raise MemoryCuratorFailure(str(exc)) from exc
    if parsed["is_error"]:
        detail = parsed["text"] or "memory curator agent returned an error"
        raise MemoryCuratorFailure(detail)
    return parsed["text"]


class MemoryCuratorClient:
    def __init__(self, *, compiler: Optional[CuratorFn] = None):
        self._compiler = compiler or run_memory_curator_agent

    def curate(self, bundle: MemoryCurationBundle) -> MemoryCuratorCandidateOutput:
        payload = {
            "bundle": bundle.to_dict(),
            "allowed_query_source_ids": sorted(_QUERY_SOURCE_IDS),
            "important_event_types": sorted(IMPORTANT_EVENT_TYPES),
        }
        raw = self._compiler(_MEMORY_CURATOR_SYSTEM_PROMPT, _json_dumps(payload))
        try:
            return self._parse_output(raw)
        except MemoryCuratorFailure as exc:
            correction_prompt = self._build_correction_prompt(
                payload=payload,
                invalid_output=raw,
                error=str(exc),
            )
            corrected_raw = self._compiler(_MEMORY_CURATOR_SYSTEM_PROMPT, correction_prompt)
            return self._parse_output(corrected_raw)

    def _parse_output(self, raw: str) -> MemoryCuratorCandidateOutput:
        data = _extract_json_object(raw)
        insights = data.get("insights")
        if not isinstance(insights, list) or any(not isinstance(item, dict) for item in insights):
            raise MemoryCuratorFailure("memory curator field 'insights' must be a list of objects")
        stale_insight_ids = data.get("stale_insight_ids")
        if (
            not isinstance(stale_insight_ids, list)
            or any(not isinstance(item, str) or not item.strip() for item in stale_insight_ids)
        ):
            raise MemoryCuratorFailure(
                "memory curator field 'stale_insight_ids' must be a list of non-empty strings"
            )
        return MemoryCuratorCandidateOutput(
            insights=list(insights),
            stale_insight_ids=list(stale_insight_ids),
            raw_output=raw,
        )

    def _build_correction_prompt(
        self,
        *,
        payload: dict[str, Any],
        invalid_output: str,
        error: str,
    ) -> str:
        return _json_dumps({
            "request": "Return corrected curator JSON.",
            "bundle_payload": payload,
            "invalid_output": invalid_output,
            "error": error,
        })


def build_memory_curation_bundle(
    *,
    event_log: EventLog,
    graph_store: MemoryGraphStore,
    insight_store: InsightStore,
    due_result: MemoryCurationDueResult,
) -> tuple[MemoryCurationBundle, dict[str, Any]]:
    update_graph_from_events(event_log, graph_store)
    graph = graph_store.load_graph()
    graph_cursor = graph_store.load_cursor()
    events = [
        event
        for event in event_log.list_events()
        if event.event_index > due_result.last_curated_event_index
    ]
    important_events = [
        _event_summary(event)
        for event in events
        if event.event_type in IMPORTANT_EVENT_TYPES
    ]
    approvals = [
        _compact_approval_event(event)
        for event in events
        if event.event_type.startswith("approval_")
    ][: _APPROVAL_LIMIT]
    graph_stats_payload = graph_stats(graph, last_processed_event_index=graph_cursor)
    templates = top_workflow_templates(graph, limit=_WORKFLOW_TEMPLATE_LIMIT)
    capabilities = capability_stats(graph, limit=_CAPABILITY_LIMIT)
    failures = failure_modes(graph, limit=_FAILURE_LIMIT)
    related = _existing_related_insights(
        graph=graph,
        insight_store=insight_store,
        top_templates=templates,
        capabilities=capabilities,
        failures=failures,
        approvals=approvals,
        important_events=important_events,
    )
    bundle = MemoryCurationBundle(
        event_start_index=due_result.last_curated_event_index + 1,
        event_end_index=due_result.latest_event_index,
        important_events=important_events,
        graph_stats=graph_stats_payload,
        top_workflow_templates=templates,
        capability_stats=capabilities,
        failure_modes=failures,
        approval_history=approvals,
        existing_related_insights=related,
    )
    input_summary = {
        "due": due_result.to_dict(),
        "bundle_counts": {
            "important_events": len(important_events),
            "top_workflow_templates": len(templates),
            "capability_stats": len(capabilities),
            "failure_modes": len(failures),
            "approval_history": len(approvals),
            "existing_related_insights": len(related),
        },
        "graph_stats": {
            "node_count": graph_stats_payload.get("node_count", 0),
            "edge_count": graph_stats_payload.get("edge_count", 0),
            "last_processed_event_index": graph_stats_payload.get("last_processed_event_index", 0),
        },
    }
    return bundle, input_summary


def run_memory_curation(
    *,
    event_log: EventLog,
    graph_store: MemoryGraphStore,
    insight_store: InsightStore,
    client: Optional[MemoryCuratorClient] = None,
    trigger_reason: str = "memory_curate_cli",
    persist_not_due: bool = True,
) -> MemoryCurationRun:
    started_at = _now_iso()
    due_result = evaluate_memory_curation_due(event_log, insight_store)
    run = MemoryCurationRun(
        run_id=f"curation_run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}",
        started_at=started_at,
        ended_at=None,
        event_start_index=due_result.last_curated_event_index + 1,
        event_end_index=due_result.latest_event_index,
        trigger_reason=trigger_reason,
        status="not_due",
        input_summary=due_result.to_dict(),
        output_insight_ids=[],
        errors=[],
    )
    if not due_result.due:
        run.ended_at = _now_iso()
        if persist_not_due:
            insight_store.append_run(run)
        return run

    active_client = client or MemoryCuratorClient()
    llm_attempted = False
    try:
        bundle, input_summary = build_memory_curation_bundle(
            event_log=event_log,
            graph_store=graph_store,
            insight_store=insight_store,
            due_result=due_result,
        )
        run.input_summary = input_summary
        llm_attempted = True
        candidate_output = active_client.curate(bundle)
        graph = graph_store.load_graph()
        accepted, validation_errors, accepted_ids = _validate_and_materialize_insights(
            graph=graph,
            bundle=bundle,
            candidates=candidate_output.insights,
        )
        store_errors = list(validation_errors)
        insights = insight_store.load_insights()
        _mark_stale_insights(
            insights,
            candidate_output.stale_insight_ids,
            related_insights=bundle.existing_related_insights,
            accepted_ids=accepted_ids,
            errors=store_errors,
        )
        for insight in accepted:
            insights[insight.insight_id] = _upsert_insight(
                existing=insights.get(insight.insight_id),
                candidate=insight,
            )
        insight_store.save_insights(insights)
        insight_store.save_cursor(due_result.latest_event_index)
        run.status = "succeeded"
        run.output_insight_ids = sorted(accepted_ids)
        run.errors = store_errors
    except Exception as exc:
        run.status = "failed"
        run.errors = [str(exc)]
        if llm_attempted:
            # Graph-memory mutation is retained. The insight cursor must remain unchanged.
            pass
    run.ended_at = _now_iso()
    insight_store.append_run(run)
    return run


def _existing_related_insights(
    *,
    graph: MemoryGraph,
    insight_store: InsightStore,
    top_templates: list[dict[str, Any]],
    capabilities: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    approvals: list[dict[str, Any]],
    important_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    relevant_node_ids = {
        item["node_id"]
        for item in top_templates + capabilities + failures
        if isinstance(item.get("node_id"), str)
    }
    relevant_source_ids = set(relevant_node_ids)
    for event in important_events:
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            relevant_source_ids.add(event_id)
        for field in ("actor_id", "target_id"):
            value = event.get(field)
            if isinstance(value, str) and value:
                relevant_source_ids.add(value)
                relevant_source_ids.add(agent_node_id(value))
    for item in approvals:
        approval_request_id = item.get("approval_request_id")
        if isinstance(approval_request_id, str) and approval_request_id:
            relevant_source_ids.add(approval_request_id)
            relevant_source_ids.add(f"approval:{approval_request_id}")
    insights = insight_store.load_insights()
    items: list[dict[str, Any]] = []
    for insight in insights.values():
        if insight.status not in {"active", "stale"}:
            continue
        if _insight_is_related(insight, relevant_node_ids, relevant_source_ids):
            items.append({
                "insight_id": insight.insight_id,
                "insight_type": insight.insight_type,
                "title": insight.title,
                "summary": insight.summary,
                "status": insight.status,
                "related_nodes": list(insight.related_nodes),
                "evidence": [item.to_dict() for item in insight.evidence],
                "metadata": dict(insight.metadata),
            })
    items.sort(key=lambda item: item["insight_id"])
    return items[:_RELATED_INSIGHT_LIMIT]


def _insight_is_related(
    insight: MemoryInsight,
    relevant_node_ids: set[str],
    relevant_source_ids: set[str],
) -> bool:
    if set(insight.related_nodes).intersection(relevant_node_ids):
        return True
    for evidence in insight.evidence:
        if evidence.source_id in relevant_source_ids:
            return True
    metadata = insight.metadata
    for key in ("workflow_template_id", "capability_id", "failure_mode_id", "agent_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value in relevant_source_ids:
            return True
    return False


def _validate_and_materialize_insights(
    *,
    graph: MemoryGraph,
    bundle: MemoryCurationBundle,
    candidates: list[dict[str, Any]],
) -> tuple[list[MemoryInsight], list[str], set[str]]:
    accepted: list[MemoryInsight] = []
    errors: list[str] = []
    accepted_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        try:
            insight = _materialize_insight(graph=graph, bundle=bundle, candidate=candidate)
        except ValueError as exc:
            errors.append(f"candidate[{index}]: {exc}")
            continue
        accepted.append(insight)
        accepted_ids.add(insight.insight_id)
    return accepted, errors, accepted_ids


def _materialize_insight(
    *,
    graph: MemoryGraph,
    bundle: MemoryCurationBundle,
    candidate: dict[str, Any],
) -> MemoryInsight:
    insight_type = str(candidate.get("insight_type") or "").strip()
    if insight_type not in _ALLOWED_INSIGHT_TYPES:
        raise ValueError("invalid insight_type")
    title = str(candidate.get("title") or "").strip()
    if not title:
        raise ValueError("missing title")
    summary = str(candidate.get("summary") or "").strip()
    if not summary:
        raise ValueError("missing summary")
    try:
        confidence = float(candidate.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0.0, 1.0]")
    severity = str(candidate.get("severity") or "").strip()
    if severity not in _ALLOWED_SEVERITIES:
        raise ValueError("invalid severity")
    recommended_action = str(candidate.get("recommended_action") or "").strip()
    if not recommended_action:
        raise ValueError("missing recommended_action")
    if _recommended_action_is_executable(recommended_action):
        raise ValueError("recommended_action must stay advisory")
    raw_evidence = candidate.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ValueError("evidence must be a non-empty list")
    evidence = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            raise ValueError("evidence items must be objects")
        ref = EvidenceRef.from_dict(item)
        _validate_evidence_ref(ref, graph=graph, bundle=bundle)
        evidence.append(ref)
    related_nodes = candidate.get("related_nodes", [])
    if not isinstance(related_nodes, list) or any(not isinstance(item, str) or not item for item in related_nodes):
        raise ValueError("related_nodes must be a list of non-empty strings")
    for node_id in related_nodes:
        if node_id not in graph.nodes:
            raise ValueError(f"related node not found: {node_id}")
    metadata = candidate.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    insight_id, normalized_metadata = _build_insight_id(
        insight_type=insight_type,
        title=title,
        metadata=metadata,
    )
    timestamp = _now_iso()
    return MemoryInsight(
        insight_id=insight_id,
        insight_type=insight_type,
        title=title,
        summary=summary,
        confidence=confidence,
        severity=severity,
        recommended_action=recommended_action,
        evidence=evidence,
        related_nodes=list(related_nodes),
        created_at=timestamp,
        updated_at=timestamp,
        status="active",
        metadata=normalized_metadata,
    )


def _build_insight_id(
    *,
    insight_type: str,
    title: str,
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    normalized_metadata = dict(metadata)
    if insight_type == "workflow_recommendation":
        template_id = str(metadata.get("workflow_template_id") or "").strip()
        if not template_id:
            raise ValueError("workflow_recommendation requires metadata.workflow_template_id")
        normalized_template = workflow_template_node_id(template_id)
        normalized_metadata["workflow_template_id"] = normalized_template
        return f"insight:{insight_type}:{_normalize_insight_id_component(normalized_template)}", normalized_metadata
    if insight_type == "capability_friction":
        capability_id = str(metadata.get("capability_id") or "").strip()
        if not capability_id:
            raise ValueError("capability_friction requires metadata.capability_id")
        normalized_capability = capability_id if capability_id.startswith("capability:") else capability_node_id(capability_id)
        normalized_metadata["capability_id"] = normalized_capability
        return f"insight:{insight_type}:{_normalize_insight_id_component(normalized_capability)}", normalized_metadata
    if insight_type in {"approval_friction", "agent_pattern"}:
        agent_id = _normalized_agent_id(metadata.get("agent_id"))
        normalized_metadata["agent_id"] = agent_id
        return f"insight:{insight_type}:{agent_id}", normalized_metadata
    if insight_type == "failure_pattern":
        failure_id = str(metadata.get("failure_mode_id") or "").strip()
        if not failure_id:
            raise ValueError("failure_pattern requires metadata.failure_mode_id")
        normalized_failure = failure_id if failure_id.startswith("failure:") else failure_node_id(failure_id)
        normalized_metadata["failure_mode_id"] = normalized_failure
        return f"insight:{insight_type}:{_normalize_insight_id_component(normalized_failure)}", normalized_metadata
    if insight_type == "scope_recommendation":
        agent_id = _normalized_agent_id(metadata.get("agent_id"))
        raw_path = str(metadata.get("path") or "").strip()
        if not raw_path:
            raise ValueError("scope_recommendation requires metadata.path")
        normalized_path = str(normalize_path(raw_path))
        normalized_metadata["agent_id"] = agent_id
        normalized_metadata["path"] = normalized_path
        return f"insight:{insight_type}:{agent_id}:{_hash_text(normalized_path)}", normalized_metadata
    topic = str(metadata.get("topic") or title).strip()
    if not topic:
        raise ValueError("system_design_lesson requires title or metadata.topic")
    normalized_metadata["topic"] = topic
    return f"insight:{insight_type}:{_hash_text(_normalized_text(topic))}", normalized_metadata


def _normalized_agent_id(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("metadata.agent_id is required")
    if raw.startswith("agent:"):
        return raw.split(":", 1)[1]
    return raw


def _validate_evidence_ref(
    ref: EvidenceRef,
    *,
    graph: MemoryGraph,
    bundle: MemoryCurationBundle,
) -> None:
    if ref.source_type == "event":
        event_ids = {
            item["event_id"]
            for item in bundle.important_events
            if isinstance(item.get("event_id"), str)
        }
        if ref.source_id not in event_ids:
            raise ValueError(f"evidence event not found: {ref.source_id}")
        return
    if ref.source_type == "node":
        if ref.source_id not in graph.nodes:
            raise ValueError(f"evidence node not found: {ref.source_id}")
        return
    if ref.source_type == "edge":
        if ref.source_id not in graph.edges:
            raise ValueError(f"evidence edge not found: {ref.source_id}")
        return
    if ref.source_id not in _QUERY_SOURCE_IDS:
        raise ValueError(f"evidence query not found: {ref.source_id}")


def _recommended_action_is_executable(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return True
    lowered = _normalized_text(normalized)
    if "```" in normalized or "$(" in normalized or "|" in normalized or ";" in normalized:
        return True
    if any(lowered.startswith(prefix + " ") for prefix in _ADVISORY_ACTION_PREFIXES):
        return False
    return any(lowered.startswith(prefix + " ") for prefix in _MUTATING_ACTION_PREFIXES)


def _mark_stale_insights(
    insights: dict[str, MemoryInsight],
    stale_ids: list[str],
    *,
    related_insights: list[dict[str, Any]],
    accepted_ids: set[str],
    errors: list[str],
) -> None:
    allowed_stale_ids = {
        str(item.get("insight_id"))
        for item in related_insights
        if isinstance(item.get("insight_id"), str)
    }
    now = _now_iso()
    for insight_id in stale_ids:
        if insight_id in accepted_ids:
            errors.append(f"stale_insight_ids includes accepted insight: {insight_id}")
            continue
        if insight_id not in allowed_stale_ids:
            errors.append(f"stale_insight_id not related: {insight_id}")
            continue
        existing = insights.get(insight_id)
        if existing is None:
            errors.append(f"stale_insight_id not found: {insight_id}")
            continue
        if existing.status == "dismissed":
            continue
        existing.status = "stale"
        existing.updated_at = now


def _upsert_insight(
    *,
    existing: Optional[MemoryInsight],
    candidate: MemoryInsight,
) -> MemoryInsight:
    if existing is None:
        return candidate
    preserved_status = existing.status if existing.status == "dismissed" else "active"
    return MemoryInsight(
        insight_id=existing.insight_id,
        insight_type=candidate.insight_type,
        title=candidate.title,
        summary=candidate.summary,
        confidence=candidate.confidence,
        severity=candidate.severity,
        recommended_action=candidate.recommended_action,
        evidence=list(candidate.evidence),
        related_nodes=list(candidate.related_nodes),
        created_at=existing.created_at,
        updated_at=candidate.updated_at,
        status=preserved_status,
        stats=dict(existing.stats),
        metadata=dict(candidate.metadata),
    )
