"""
Unified append-only system event timeline for MR1 runtime activity.

The timeline is additive. Existing workflow events, approval records,
capability audits, messages, and agent logs remain the source of truth.
This module provides a compact causal index over those stores.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


EVENT_VERSION = 1
EVENT_KIND_VALUES = frozenset({
    "lifecycle",
    "decision",
    "action",
    "communication",
})
SEVERITY_VALUES = frozenset({
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
})

EVENT_KIND_BY_TYPE = {
    "agent_created": "lifecycle",
    "agent_scope_granted": "action",
    "agent_scope_revoked": "action",
    "workflow_created": "lifecycle",
    "workflow_started": "lifecycle",
    "workflow_task_started": "lifecycle",
    "workflow_task_completed": "lifecycle",
    "workflow_task_failed": "lifecycle",
    "workflow_completed": "lifecycle",
    "workflow_failed": "lifecycle",
    "capability_requested": "action",
    "capability_allowed": "decision",
    "capability_blocked": "decision",
    "capability_executed": "action",
    "capability_failed": "action",
    "approval_requested": "decision",
    "approval_approved": "decision",
    "approval_denied": "decision",
    "approval_consumed": "action",
    "message_sent": "communication",
    "message_read": "communication",
    "mrn_step_started": "communication",
    "mrn_step_completed": "communication",
    "mrn_reported": "communication",
}

SEVERITY_BY_TYPE = {
    "agent_created": "INFO",
    "agent_scope_granted": "INFO",
    "agent_scope_revoked": "WARNING",
    "workflow_created": "INFO",
    "workflow_started": "INFO",
    "workflow_task_started": "INFO",
    "workflow_task_completed": "INFO",
    "workflow_task_failed": "ERROR",
    "workflow_completed": "INFO",
    "workflow_failed": "CRITICAL",
    "capability_requested": "INFO",
    "capability_allowed": "INFO",
    "capability_blocked": "WARNING",
    "capability_executed": "INFO",
    "capability_failed": "ERROR",
    "approval_requested": "WARNING",
    "approval_approved": "INFO",
    "approval_denied": "WARNING",
    "approval_consumed": "INFO",
    "message_sent": "INFO",
    "message_read": "INFO",
    "mrn_step_started": "INFO",
    "mrn_step_completed": "INFO",
    "mrn_reported": "INFO",
}

_EVENT_FILE_NAME = "events.jsonl"
_CONTEXT_CORRELATION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mr1_event_correlation_id",
    default=None,
)
_LOCKS: dict[str, threading.RLock] = {}
_LAST_APPEND_NS: dict[str, int] = {}


def _now_iso_from_ns(now_ns: int) -> str:
    return datetime.fromtimestamp(now_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return _now_iso_from_ns(time.time_ns())


def _normalized_timestamp_bucket(timestamp: str) -> str:
    dt = datetime.fromisoformat(timestamp)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _build_event_id(
    *,
    event_version: int,
    event_type: str,
    actor_id: Optional[str],
    target_id: Optional[str],
    correlation_id: Optional[str],
    parent_event_id: Optional[str],
    workflow_id: Optional[str],
    task_id: Optional[str],
    step_id: Optional[str],
    approval_request_id: Optional[str],
    audit_id: Optional[str],
    timestamp: str,
) -> str:
    digest = hashlib.sha256(_canonical_json({
        "event_version": event_version,
        "event_type": event_type,
        "actor_id": actor_id,
        "target_id": target_id,
        "correlation_id": correlation_id,
        "parent_event_id": parent_event_id,
        "workflow_id": workflow_id,
        "task_id": task_id,
        "step_id": step_id,
        "approval_request_id": approval_request_id,
        "audit_id": audit_id,
        "normalized_timestamp_bucket": _normalized_timestamp_bucket(timestamp),
    }).encode("utf-8")).hexdigest()[:16]
    return f"evt-{digest}"


def workflow_correlation_id(workflow_id: str) -> str:
    return f"wf:{workflow_id}"


def mrn_step_correlation_id(agent_id: str, iteration: int) -> str:
    return f"mrn:{agent_id}:step:{iteration}"


def cli_correlation_id(caller_agent_id: str, command: str, timestamp_ms: Optional[int] = None) -> str:
    ts_ms = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    return f"cli:{caller_agent_id}:{command}:{ts_ms}"


def current_correlation_id() -> Optional[str]:
    return _CONTEXT_CORRELATION_ID.get()


@contextlib.contextmanager
def bind_correlation_id(correlation_id: Optional[str]) -> Iterator[None]:
    token = _CONTEXT_CORRELATION_ID.set(correlation_id)
    try:
        yield
    finally:
        _CONTEXT_CORRELATION_ID.reset(token)


@dataclass(frozen=True)
class SystemEvent:
    event_id: str
    event_index: int
    timestamp: str
    event_type: str
    status: str
    severity: str
    summary: str
    event_version: int = EVENT_VERSION
    event_kind: str = ""
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None
    target_id: Optional[str] = None
    target_type: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    step_id: Optional[str] = None
    message_id: Optional[str] = None
    approval_request_id: Optional[str] = None
    audit_id: Optional[str] = None
    record_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_version < 1:
            raise ValueError("event_version must be >= 1")
        if self.event_kind not in EVENT_KIND_VALUES:
            raise ValueError(f"invalid event_kind '{self.event_kind}'")
        if self.severity not in SEVERITY_VALUES:
            raise ValueError(f"invalid severity '{self.severity}'")
        if not isinstance(self.event_index, int) or self.event_index < 0:
            raise ValueError("event_index must be >= 0")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_index": self.event_index,
            "event_version": self.event_version,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "event_kind": self.event_kind,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "status": self.status,
            "severity": self.severity,
            "summary": self.summary,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "message_id": self.message_id,
            "approval_request_id": self.approval_request_id,
            "audit_id": self.audit_id,
            "record_path": self.record_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemEvent":
        return cls(
            event_id=data["event_id"],
            event_index=int(data["event_index"]),
            event_version=int(data.get("event_version", EVENT_VERSION)),
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            event_kind=data["event_kind"],
            actor_id=data.get("actor_id"),
            actor_type=data.get("actor_type"),
            target_id=data.get("target_id"),
            target_type=data.get("target_type"),
            status=data["status"],
            severity=data["severity"],
            summary=data["summary"],
            correlation_id=data.get("correlation_id"),
            parent_event_id=data.get("parent_event_id"),
            workflow_id=data.get("workflow_id"),
            task_id=data.get("task_id"),
            step_id=data.get("step_id"),
            message_id=data.get("message_id"),
            approval_request_id=data.get("approval_request_id"),
            audit_id=data.get("audit_id"),
            record_path=data.get("record_path"),
            metadata=dict(data.get("metadata", {})),
        )


class EventLog:
    def __init__(self, root: Path):
        root_path = Path(root)
        if root_path.suffix == ".jsonl":
            self._path = root_path
        else:
            self._path = root_path / _EVENT_FILE_NAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self._path.resolve(strict=False))
        self._lock = _LOCKS.setdefault(key, threading.RLock())
        _LAST_APPEND_NS.setdefault(key, 0)
        self._lock_key = key

    @property
    def path(self) -> Path:
        return self._path

    def append_event(self, event: SystemEvent) -> SystemEvent:
        with self._lock:
            existing = self._get_event_locked(event.event_id)
            if existing is not None:
                return existing
            if event.parent_event_id is not None:
                parent = self._get_event_locked(event.parent_event_id)
                if parent is None:
                    raise ValueError(f"parent event not found: {event.parent_event_id}")
            next_index = self._next_event_index_locked()
            if event.event_index not in {0, next_index}:
                raise ValueError(
                    f"invalid event_index {event.event_index}; expected 0 or {next_index}"
                )
            now_ns = time.time_ns()
            if now_ns < _LAST_APPEND_NS[self._lock_key]:
                raise RuntimeError("event append order must be monotonic within one process")
            persisted = SystemEvent.from_dict({
                **event.to_dict(),
                "event_index": next_index,
            })
            line = _canonical_json(persisted.to_dict()) + "\n"
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            _LAST_APPEND_NS[self._lock_key] = now_ns
            return persisted

    def emit(
        self,
        *,
        event_type: str,
        actor_id: Optional[str],
        actor_type: Optional[str],
        target_id: Optional[str],
        target_type: Optional[str],
        status: str,
        summary: str,
        severity: Optional[str] = None,
        correlation_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        message_id: Optional[str] = None,
        approval_request_id: Optional[str] = None,
        audit_id: Optional[str] = None,
        record_path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> SystemEvent:
        if event_type not in EVENT_KIND_BY_TYPE:
            raise ValueError(f"unsupported event_type '{event_type}'")
        event_timestamp = timestamp or _now_iso()
        with self._lock:
            resolved_correlation_id = self._resolve_correlation_id_locked(
                explicit=correlation_id,
                event_type=event_type,
                workflow_id=workflow_id,
                step_id=step_id,
                approval_request_id=approval_request_id,
                audit_id=audit_id,
            )
            resolved_parent_id = self._resolve_parent_event_id_locked(
                explicit=parent_event_id,
                event_type=event_type,
                actor_id=actor_id,
                target_id=target_id,
                correlation_id=resolved_correlation_id,
                workflow_id=workflow_id,
                task_id=task_id,
                step_id=step_id,
                approval_request_id=approval_request_id,
                audit_id=audit_id,
            )
            event_id = _build_event_id(
                event_version=EVENT_VERSION,
                event_type=event_type,
                actor_id=actor_id,
                target_id=target_id,
                correlation_id=resolved_correlation_id,
                parent_event_id=resolved_parent_id,
                workflow_id=workflow_id,
                task_id=task_id,
                step_id=step_id,
                approval_request_id=approval_request_id,
                audit_id=audit_id,
                timestamp=event_timestamp,
            )
            return self.append_event(SystemEvent(
                event_id=event_id,
                event_index=0,
                event_version=EVENT_VERSION,
                timestamp=event_timestamp,
                event_type=event_type,
                event_kind=EVENT_KIND_BY_TYPE[event_type],
                actor_id=actor_id,
                actor_type=actor_type,
                target_id=target_id,
                target_type=target_type,
                status=status,
                severity=severity or SEVERITY_BY_TYPE[event_type],
                summary=summary,
                correlation_id=resolved_correlation_id,
                parent_event_id=resolved_parent_id,
                workflow_id=workflow_id,
                task_id=task_id,
                step_id=step_id,
                message_id=message_id,
                approval_request_id=approval_request_id,
                audit_id=audit_id,
                record_path=record_path,
                metadata=dict(metadata or {}),
            ))

    def list_events(self, *, limit: Optional[int] = None) -> list[SystemEvent]:
        with self._lock:
            events = self._load_events_locked()
        if limit is not None:
            return events[-limit:]
        return events

    def get_event(self, event_id: str) -> Optional[SystemEvent]:
        with self._lock:
            return self._get_event_locked(event_id)

    def filter_events(
        self,
        *,
        actor_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        message_id: Optional[str] = None,
        approval_request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        event_type: Optional[str] = None,
        target_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[SystemEvent]:
        events = self.list_events()
        filtered: list[SystemEvent] = []
        for event in events:
            if actor_id is not None and event.actor_id != actor_id:
                continue
            if workflow_id is not None and event.workflow_id != workflow_id:
                continue
            if task_id is not None and event.task_id != task_id:
                continue
            if step_id is not None and event.step_id != step_id:
                continue
            if message_id is not None and event.message_id != message_id:
                continue
            if approval_request_id is not None and event.approval_request_id != approval_request_id:
                continue
            if correlation_id is not None and event.correlation_id != correlation_id:
                continue
            if event_type is not None and event.event_type != event_type:
                continue
            if target_id is not None and event.target_id != target_id:
                continue
            if severity is not None and event.severity != severity:
                continue
            if status is not None and event.status != status:
                continue
            filtered.append(event)
        return filtered

    def trace_by_correlation(self, correlation_id: str) -> list[SystemEvent]:
        events = self.filter_events(correlation_id=correlation_id)
        return sorted(events, key=lambda item: item.event_index)

    def blocked_now(self) -> list[SystemEvent]:
        blocked_by_key: dict[str, SystemEvent] = {}
        for event in self.list_events():
            key = self._block_key(event)
            if key is None:
                continue
            if event.event_type in {"capability_blocked", "approval_requested", "workflow_task_failed"}:
                if event.event_type == "workflow_task_failed" and event.status != "blocked":
                    continue
                blocked_by_key[key] = event
                continue
            if event.event_type in {
                "capability_allowed",
                "capability_executed",
                "approval_approved",
                "approval_denied",
                "workflow_task_completed",
            }:
                blocked_by_key.pop(key, None)
        return sorted(blocked_by_key.values(), key=lambda item: item.event_index)

    def approval_history(self, approval_request_id: Optional[str] = None) -> list[SystemEvent]:
        events = [
            event
            for event in self.list_events()
            if event.event_type.startswith("approval_")
        ]
        if approval_request_id is not None:
            events = [
                event for event in events
                if event.approval_request_id == approval_request_id
            ]
        return sorted(events, key=lambda item: item.event_index)

    def agent_activity(self, agent_id: str) -> list[SystemEvent]:
        return [
            event for event in self.list_events()
            if event.actor_id == agent_id or event.target_id == agent_id
        ]

    def workflow_trace(self, workflow_id: str) -> list[SystemEvent]:
        return sorted(
            self.filter_events(workflow_id=workflow_id),
            key=lambda item: item.event_index,
        )

    def recent_activity(self, limit: int = 20) -> list[SystemEvent]:
        return sorted(self.list_events(), key=lambda item: item.event_index, reverse=True)[:limit]

    def _next_event_index_locked(self) -> int:
        events = self._load_events_locked()
        if not events:
            return 1
        return events[-1].event_index + 1

    def _get_event_locked(self, event_id: str) -> Optional[SystemEvent]:
        for event in self._load_events_locked():
            if event.event_id == event_id:
                return event
        return None

    def _load_events_locked(self) -> list[SystemEvent]:
        if not self._path.exists():
            return []
        events: list[SystemEvent] = []
        with open(self._path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    events.append(SystemEvent.from_dict(payload))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        events.sort(key=lambda item: item.event_index)
        return events

    def _resolve_correlation_id_locked(
        self,
        *,
        explicit: Optional[str],
        event_type: str,
        workflow_id: Optional[str],
        step_id: Optional[str],
        approval_request_id: Optional[str],
        audit_id: Optional[str],
    ) -> Optional[str]:
        if explicit:
            return explicit
        current = current_correlation_id()
        if current:
            return current
        if approval_request_id:
            prior = self._find_latest_locked(
                lambda event: (
                    event.approval_request_id == approval_request_id
                    and event.correlation_id is not None
                )
            )
            if prior is not None:
                return prior.correlation_id
        if audit_id:
            prior = self._find_latest_locked(
                lambda event: event.audit_id == audit_id and event.correlation_id is not None
            )
            if prior is not None:
                return prior.correlation_id
        if step_id and event_type == "mrn_step_started":
            return step_id
        if workflow_id:
            return workflow_correlation_id(workflow_id)
        return None

    def _resolve_parent_event_id_locked(
        self,
        *,
        explicit: Optional[str],
        event_type: str,
        actor_id: Optional[str],
        target_id: Optional[str],
        correlation_id: Optional[str],
        workflow_id: Optional[str],
        task_id: Optional[str],
        step_id: Optional[str],
        approval_request_id: Optional[str],
        audit_id: Optional[str],
    ) -> Optional[str]:
        if explicit:
            if self._get_event_locked(explicit) is None:
                raise ValueError(f"parent event not found: {explicit}")
            return explicit
        parent_type: Optional[str] = None
        if event_type in {"capability_allowed", "capability_blocked"}:
            parent_type = "capability_requested"
        elif event_type in {"capability_executed", "capability_failed"}:
            parent_type = "capability_allowed"
        elif event_type == "approval_requested":
            parent_type = "capability_blocked"
        elif event_type in {"approval_approved", "approval_denied"}:
            parent_type = "approval_requested"
        elif event_type == "approval_consumed":
            parent_type = "approval_approved"
        elif event_type == "mrn_step_completed":
            parent_type = "mrn_step_started"
        elif event_type == "mrn_reported":
            parent_type = "mrn_step_completed"
        elif event_type in {"workflow_completed", "workflow_failed"}:
            parent = self._find_latest_locked(
                lambda event: (
                    event.workflow_id == workflow_id
                    and event.event_type in {"workflow_task_completed", "workflow_task_failed"}
                )
            )
            return parent.event_id if parent is not None else None
        if parent_type is None:
            return None
        parent = self._find_latest_locked(
            lambda event: (
                event.event_type == parent_type
                and (correlation_id is None or event.correlation_id == correlation_id)
                and (approval_request_id is None or event.approval_request_id == approval_request_id)
                and (step_id is None or event.step_id == step_id)
                and (workflow_id is None or event.workflow_id == workflow_id)
                and (task_id is None or event.task_id == task_id or parent_type.startswith("capability_"))
                and (actor_id is None or event.actor_id == actor_id or event.event_type.startswith("approval_"))
                and (target_id is None or event.target_id == target_id or event.event_type.startswith("approval_"))
                and (audit_id is None or event.audit_id == audit_id or parent_type == "capability_requested")
            )
        )
        return parent.event_id if parent is not None else None

    def _find_latest_locked(self, predicate) -> Optional[SystemEvent]:
        for event in reversed(self._load_events_locked()):
            if predicate(event):
                return event
        return None

    @staticmethod
    def _block_key(event: SystemEvent) -> Optional[str]:
        if event.approval_request_id:
            return f"approval:{event.approval_request_id}"
        if event.event_type.startswith("capability_"):
            return f"capability:{event.correlation_id}:{event.target_id}"
        if event.workflow_id and event.task_id:
            return f"task:{event.workflow_id}:{event.task_id}"
        return None
