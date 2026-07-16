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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from mr1.event_archive import ArchiveSegment, EventArchive, EventArchiveError


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
    "approval_expired": "decision",
    "approval_consumed": "action",
    "runtime_turn_decided": "decision",
    "inbox_triage_failed": "action",
    "inbox_triage_skipped": "decision",
    "message_sent": "communication",
    "message_read": "communication",
    "mrn_step_started": "communication",
    "mrn_step_completed": "communication",
    "mrn_reported": "communication",
    # Autonomous operation (Phase A).
    "scheduler_tick_failed": "action",
    "supervisor_started": "lifecycle",
    "supervisor_stopped": "lifecycle",
    "supervisor_tick": "lifecycle",
    "supervisor_tick_failed": "action",
    "control_mode_changed": "decision",
    "objective_created": "lifecycle",
    "objective_state_changed": "lifecycle",
    "objective_planned": "decision",
    "objective_plan_failed": "action",
    "objective_workflow_submitted": "action",
    "objective_succeeded": "lifecycle",
    "objective_satisfied": "lifecycle",
    "objective_recovery": "decision",
    "objective_quarantined": "lifecycle",
    "escalation_raised": "communication",
    "consent_grant_created": "decision",
    "consent_grant_revoked": "decision",
    "consent_grant_expired": "decision",
    "consent_grant_used": "action",
    "budget_exhausted": "decision",
    # Retention and maintenance (Phase B).
    "retention_run": "action",
    "events_rotated": "action",
    "backpressure_applied": "decision",
    "backpressure_lifted": "decision",
    "notification_delivered": "communication",
    "notification_failed": "communication",
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
    "approval_expired": "WARNING",
    "approval_consumed": "INFO",
    "runtime_turn_decided": "INFO",
    "inbox_triage_failed": "ERROR",
    "inbox_triage_skipped": "INFO",
    "message_sent": "INFO",
    "message_read": "INFO",
    "mrn_step_started": "INFO",
    "mrn_step_completed": "INFO",
    "mrn_reported": "INFO",
    # Autonomous operation (Phase A).
    "scheduler_tick_failed": "ERROR",
    "supervisor_started": "INFO",
    "supervisor_stopped": "INFO",
    "supervisor_tick": "INFO",
    "supervisor_tick_failed": "ERROR",
    "control_mode_changed": "WARNING",
    "objective_created": "INFO",
    "objective_state_changed": "INFO",
    "objective_planned": "INFO",
    "objective_plan_failed": "ERROR",
    "objective_workflow_submitted": "INFO",
    "objective_succeeded": "INFO",
    "objective_satisfied": "INFO",
    "objective_recovery": "WARNING",
    "objective_quarantined": "CRITICAL",
    "escalation_raised": "WARNING",
    "consent_grant_created": "WARNING",
    "consent_grant_revoked": "WARNING",
    "consent_grant_expired": "WARNING",
    "consent_grant_used": "INFO",
    "budget_exhausted": "WARNING",
    # Retention and maintenance (Phase B).
    "retention_run": "INFO",
    "events_rotated": "INFO",
    "backpressure_applied": "WARNING",
    "backpressure_lifted": "INFO",
    "notification_delivered": "INFO",
    "notification_failed": "ERROR",
}

# Maximum number of events kept in the in-process cache. Older events remain on
# disk — in events.jsonl and, once rotated, in sealed archive segments — but are
# evicted from memory past this limit. At ~500 bytes each, 50 000 events ≈ 25 MB.
#
# The cache is a *memory* bound, and B1's job was to stop it being mistaken for a
# *history* bound. `list_events()` and everything built on it now return complete
# history: they read the cache when the cache provably holds all of it, and fall
# back to disk (archive segments + live file) when it does not. A query that
# would have silently returned a 50 000-event window now returns the whole log.
# `recent_events()` is the explicit opt-in to the cheap cached window.
_MAX_CACHE_EVENTS = 50_000
_EVENT_FILE_NAME = "events.jsonl"
_CONTEXT_CORRELATION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "mr1_event_correlation_id",
    default=None,
)
_LOCKS: dict[str, threading.RLock] = {}
_LOCK_SUFFIX = ".lock"


def _now_iso_from_ns(now_ns: int) -> str:
    return datetime.fromtimestamp(now_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return _now_iso_from_ns(time.time_ns())


def _normalized_timestamp(timestamp: str) -> str:
    dt = datetime.fromisoformat(timestamp)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="microseconds")


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
        "normalized_timestamp": _normalized_timestamp(timestamp),
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


@dataclass
class _EventCache:
    # Bounded sliding window: oldest events are popleft()'d when _MAX_CACHE_EVENTS is exceeded.
    events: deque = field(default_factory=deque)
    event_by_id: dict[str, SystemEvent] = field(default_factory=dict)
    file_offset: int = 0
    file_mtime_ns: int = 0
    initialized: bool = False
    # True once *any* event has been dropped from the window — by eviction here
    # or by rotation into the archive. It is the flag that tells a history query
    # it may not answer from memory.
    truncated: bool = False
    # The highest index already sealed into the archive. After rotation the live
    # file may be short or empty, and the next index must still continue from
    # here — an index that restarts at 1 forks the log.
    base_index: int = 0

    @property
    def last_index(self) -> int:
        if not self.events:
            return self.base_index
        return max(self.events[-1].event_index, self.base_index)


class EventLog:
    def __init__(self, root: Path, *, compress_archive: bool = True):
        root_path = Path(root)
        if root_path.suffix == ".jsonl":
            self._path = root_path
        else:
            self._path = root_path / _EVENT_FILE_NAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self._path.resolve(strict=False))
        self._lock = _LOCKS.setdefault(key, threading.RLock())
        self._lock_key = key
        self._append_lock_path = self._path.with_name(f"{self._path.name}{_LOCK_SUFFIX}")
        self._cache = _EventCache()
        self._archive = EventArchive(self._path.parent, compress=compress_archive)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def archive(self) -> EventArchive:
        return self._archive

    def append_event(self, event: SystemEvent) -> SystemEvent:
        with self._lock:
            with self._append_guard_locked():
                self._refresh_cache_locked()
                existing = self._cache.event_by_id.get(event.event_id)
                if existing is not None:
                    return existing
                if event.parent_event_id is not None:
                    # A parent that has been archived is still a real parent.
                    # Before rotation existed this could only be a caller error;
                    # now it is also just old, so check the archive before
                    # rejecting the child and breaking the causal chain.
                    parent = self._lookup_event_anywhere_locked(event.parent_event_id)
                    if parent is None:
                        raise ValueError(f"parent event not found: {event.parent_event_id}")
                next_index = self._cache.last_index + 1
                if event.event_index not in {0, next_index}:
                    raise ValueError(
                        f"invalid event_index {event.event_index}; expected 0 or {next_index}"
                    )
                persisted = SystemEvent.from_dict({
                    **event.to_dict(),
                    "event_index": next_index,
                })
                self._append_persisted_event_locked(persisted)
                self._cache_append_locked(persisted)
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
        """
        Complete history, oldest first. Never silently truncated.

        Served from the in-memory cache while the cache provably holds
        everything — which is the normal case, and costs what it always did.
        Once history has outgrown the cache or been rotated into the archive,
        this reads from disk instead of quietly returning the tail.

        For the cheap bounded window, ask for it by name: `recent_events()`.
        """
        with self._lock:
            events = self._complete_history_locked()
        if limit is not None:
            return events[-limit:]
        return events

    def recent_events(self, limit: Optional[int] = None) -> list[SystemEvent]:
        """
        The cached window — the newest events, bounded by memory, not by history.

        Explicitly *not* complete. Use it where recency is the question
        ("what just happened") and completeness is not.
        """
        with self._lock:
            events = self._load_events_locked()
        if limit is not None:
            return events[-limit:]
        return events

    @property
    def cache_is_complete(self) -> bool:
        """True when the cached window holds all of history."""
        with self._lock:
            self._refresh_cache_locked()
            return not self._cache.truncated

    def history_stats(self) -> dict[str, Any]:
        """What history costs, and where it lives. For `mr1 status` and the doctor."""
        with self._lock:
            self._refresh_cache_locked()
            live_bytes = self._live_bytes_locked()
            segments = self._archive.segments()
            return {
                "total_events": self._cache.last_index,
                "cached_events": len(self._cache.events),
                "cache_is_complete": not self._cache.truncated,
                "cache_limit": _MAX_CACHE_EVENTS,
                "live_bytes": live_bytes,
                "archive_bytes": sum(item.size_bytes for item in segments),
                "archive_segments": len(segments),
                "archived_events": sum(item.count for item in segments),
                "archived_through_index": self._archive.archived_last_index(),
            }

    def get_event(self, event_id: str) -> Optional[SystemEvent]:
        with self._lock:
            found = self._get_event_locked(event_id)
            if found is not None:
                return found
            if not self._cache.truncated:
                return None
            # Evicted or archived — look it up in the full log rather than
            # reporting an event that exists as though it never happened.
            for event in self._complete_history_locked():
                if event.event_id == event_id:
                    return event
            return None

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
                "approval_expired",
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
        # Recency, not completeness — the cached window is exactly the right
        # source, and reading all of history to throw away everything but the
        # last 20 would be the wrong trade.
        return sorted(self.recent_events(), key=lambda item: item.event_index, reverse=True)[:limit]

    # ------------------------------------------------------------------
    # Rotation (B1)
    # ------------------------------------------------------------------

    def rotate(
        self,
        *,
        keep_recent: int = 1_000,
        now_iso: Optional[str] = None,
    ) -> Optional[ArchiveSegment]:
        """
        Seal the live log into an archive segment and shrink `events.jsonl`.

        `keep_recent` events stay behind in the live file. That is not just a
        convenience: `emit()` resolves an event's `parent_event_id` and
        `correlation_id` by searching backwards through the cache. Rotating the
        whole file out from under an in-flight causal chain would orphan the
        next event in it. Keeping a tail of recent history means live chains
        stay resolvable across a rotation.

        Returns the sealed segment, or None when there was nothing to seal.
        """
        with self._lock:
            with self._append_guard_locked():
                payloads = self._read_live_payloads_locked()
                if not payloads:
                    return None
                cutoff = len(payloads) - max(0, keep_recent)
                if cutoff <= 0:
                    return None
                to_seal = payloads[:cutoff]
                to_keep = payloads[cutoff:]

                # Durable first. Nothing leaves the live file until it is
                # sealed on disk and named in the manifest.
                segment = self._archive.seal(
                    to_seal,
                    sealed_at=now_iso or _now_iso(),
                )
                self._rewrite_live_locked(to_keep)
                self._rebuild_cache_locked()
                return segment

    def _read_live_payloads_locked(self) -> list[dict[str, Any]]:
        """Raw payloads from events.jsonl, in index order, deduped."""
        if not self._path.exists():
            return []
        payloads: list[dict[str, Any]] = []
        seen: set[str] = set()
        with open(self._path, "rb") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                event_id = payload.get("event_id")
                if not event_id or event_id in seen:
                    continue
                seen.add(event_id)
                payloads.append(payload)
        payloads.sort(key=lambda item: int(item.get("event_index", 0)))
        return payloads

    def _rewrite_live_locked(self, payloads: list[dict[str, Any]]) -> None:
        tmp = self._path.with_name(self._path.name + ".rotate.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(_canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self._path)
        _fsync_dir(self._path.parent)

    def _live_bytes_locked(self) -> int:
        try:
            return int(self._path.stat().st_size)
        except OSError:
            return 0

    def _complete_history_locked(self) -> list[SystemEvent]:
        """
        Every event ever emitted, oldest first.

        The cache answers when it holds everything. Otherwise this reads the
        archive segments and the live file and merges them — the archive is
        raised as an error if a segment named in the manifest is missing,
        because a short answer to a full-history question is worse than none.
        """
        self._refresh_cache_locked()
        if not self._cache.truncated:
            return list(self._cache.events)

        merged: dict[str, SystemEvent] = {}
        for payload in self._archive.iter_raw_events():
            try:
                event = SystemEvent.from_dict(payload)
            except (KeyError, ValueError):
                continue
            merged.setdefault(event.event_id, event)
        for payload in self._read_live_payloads_locked():
            try:
                event = SystemEvent.from_dict(payload)
            except (KeyError, ValueError):
                continue
            merged.setdefault(event.event_id, event)
        return sorted(merged.values(), key=lambda item: item.event_index)

    def _next_event_index_locked(self) -> int:
        self._refresh_cache_locked()
        return self._cache.last_index + 1

    def _get_event_locked(self, event_id: str) -> Optional[SystemEvent]:
        self._refresh_cache_locked()
        return self._cache.event_by_id.get(event_id)

    def _lookup_event_anywhere_locked(self, event_id: str) -> Optional[SystemEvent]:
        """Cache first, then the full log. Only pays for disk on a cache miss."""
        found = self._get_event_locked(event_id)
        if found is not None:
            return found
        if not self._cache.truncated:
            return None
        for event in self._complete_history_locked():
            if event.event_id == event_id:
                return event
        return None

    def _load_events_locked(self) -> list[SystemEvent]:
        self._refresh_cache_locked()
        return list(self._cache.events)

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
            if self._lookup_event_anywhere_locked(explicit) is None:
                raise ValueError(f"parent event not found: {explicit}")
            return explicit
        parent_type: Optional[str] = None
        if event_type in {"capability_allowed", "capability_blocked"}:
            parent_type = "capability_requested"
        elif event_type in {"capability_executed", "capability_failed"}:
            parent_type = "capability_allowed"
        elif event_type == "approval_requested":
            parent_type = "capability_blocked"
        elif event_type in {"approval_approved", "approval_denied", "approval_expired"}:
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
        self._refresh_cache_locked()
        for event in reversed(self._cache.events):
            if predicate(event):
                return event
        return None

    @contextlib.contextmanager
    def _append_guard_locked(self) -> Iterator[None]:
        self._append_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._append_lock_path, "a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _refresh_cache_locked(self) -> None:
        if not self._cache.initialized:
            self._rebuild_cache_locked()
            return
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            self._cache = _EventCache(initialized=True)
            return
        if stat.st_size < self._cache.file_offset:
            # The live file shrank: another process rotated it. Rebuild rather
            # than read on from an offset that now points into different data.
            self._rebuild_cache_locked()
            return
        if (
            stat.st_size == self._cache.file_offset
            and stat.st_mtime_ns == self._cache.file_mtime_ns
        ):
            return
        with open(self._path, "rb") as handle:
            handle.seek(self._cache.file_offset)
            while True:
                line = handle.readline()
                if not line:
                    break
                self._parse_and_cache_line_locked(line)
            self._cache.file_offset = handle.tell()
        # Evict oldest events when the incremental read pushes cache over the limit.
        while len(self._cache.events) > _MAX_CACHE_EVENTS:
            evicted = self._cache.events.popleft()
            self._cache.event_by_id.pop(evicted.event_id, None)
            self._cache.truncated = True
        try:
            stat = self._path.stat()
            self._cache.file_mtime_ns = stat.st_mtime_ns
        except FileNotFoundError:
            self._cache.file_mtime_ns = 0

    def _rebuild_cache_locked(self) -> None:
        # Build with a temp list so we can sort then trim before creating the deque.
        temp_events: list[SystemEvent] = []
        temp_by_id: dict[str, SystemEvent] = {}
        file_offset = 0
        mtime_ns = 0
        truncated = False
        if self._path.exists():
            with open(self._path, "rb") as handle:
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                        event = SystemEvent.from_dict(payload)
                    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError):
                        continue
                    if event.event_id not in temp_by_id:
                        temp_events.append(event)
                        temp_by_id[event.event_id] = event
                file_offset = handle.tell()
            try:
                mtime_ns = self._path.stat().st_mtime_ns
            except FileNotFoundError:
                mtime_ns = 0
            temp_events.sort(key=lambda item: item.event_index)
            if len(temp_events) > _MAX_CACHE_EVENTS:
                temp_events = temp_events[-_MAX_CACHE_EVENTS:]
                truncated = True

        # Anything sealed into the archive is history the cache does not hold —
        # and the index it stopped at is where the live log must carry on from.
        archived_last_index = self._archive.archived_last_index()
        if archived_last_index > 0:
            truncated = True

        cache = _EventCache(initialized=True)
        cache.events = deque(temp_events)
        cache.event_by_id = {e.event_id: e for e in temp_events}
        cache.file_offset = file_offset
        cache.file_mtime_ns = mtime_ns
        cache.truncated = truncated
        cache.base_index = archived_last_index
        self._cache = cache

    def _parse_and_cache_line_locked(self, line: bytes) -> None:
        raw = line.strip()
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
            event = SystemEvent.from_dict(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError):
            return
        if self._cache.event_by_id.get(event.event_id) is not None:
            return
        self._cache.events.append(event)
        self._cache.event_by_id[event.event_id] = event

    def _append_persisted_event_locked(self, event: SystemEvent) -> None:
        line = (_canonical_json(event.to_dict()) + "\n").encode("utf-8")
        with open(self._path, "ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            self._cache.file_offset = handle.tell()
        try:
            self._cache.file_mtime_ns = self._path.stat().st_mtime_ns
        except FileNotFoundError:
            self._cache.file_mtime_ns = 0

    def _cache_append_locked(self, event: SystemEvent) -> None:
        self._cache.events.append(event)
        self._cache.event_by_id[event.event_id] = event
        if len(self._cache.events) > _MAX_CACHE_EVENTS:
            evicted = self._cache.events.popleft()
            self._cache.event_by_id.pop(evicted.event_id, None)
            # The window no longer holds all of history. Every full-history
            # query from here on must go to disk.
            self._cache.truncated = True
        self._cache.initialized = True

    @staticmethod
    def _block_key(event: SystemEvent) -> Optional[str]:
        if event.approval_request_id:
            return f"approval:{event.approval_request_id}"
        if event.event_type.startswith("capability_"):
            return f"capability:{event.correlation_id}:{event.target_id}"
        if event.workflow_id and event.task_id:
            return f"task:{event.workflow_id}:{event.task_id}"
        return None


def _fsync_dir(path: Path) -> None:
    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
