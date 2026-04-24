"""
Canonical event helpers for the workflow scheduler.

Scheduler code emits events through these helpers rather than
constructing `WorkflowEvent` instances directly, so every callsite uses
the same event names and shapes. This also gives us one place to
version or extend the event vocabulary later.
"""

from __future__ import annotations

from typing import Any, Optional

from mr1.workflow_models import WorkflowEvent
from mr1.workflow_store import WorkflowStore


# Canonical event types emitted in Phase 1.
WORKFLOW_SUBMITTED = "workflow_submitted"
WORKFLOW_SUCCEEDED = "workflow_succeeded"
WORKFLOW_FAILED = "workflow_failed"
WORKFLOW_CANCELLED = "workflow_cancelled"

TASK_CREATED = "task_created"
TASK_READY = "task_ready"
TASK_STARTED = "task_started"
TASK_SUCCEEDED = "task_succeeded"
TASK_FAILED = "task_failed"
TASK_TIMED_OUT = "task_timed_out"
TASK_CANCELLED = "task_cancelled"
TASK_BLOCKED = "task_blocked"
TASK_UNBLOCKED = "task_unblocked"


class WorkflowEventLog:
    """
    Thin helper that stamps events with the conventional agent_id and
    persists them through the store.

    Callers should already hold the store lock (via `store.locked()`)
    when emitting events that must be atomic with a state change.
    """

    def __init__(self, store: WorkflowStore, default_agent_id: str = "scheduler"):
        self._store = store
        self._default_agent_id = default_agent_id

    def emit(
        self,
        event_type: str,
        workflow_id: str,
        *,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        message: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> WorkflowEvent:
        event = WorkflowEvent.new(
            event_type=event_type,
            workflow_id=workflow_id,
            task_id=task_id,
            agent_id=agent_id or self._default_agent_id,
            message=message,
            metadata=metadata,
        )
        self._store.append_event(event)
        return event

    # ------------------------------------------------------------------
    # Convenience wrappers (optional but keep callsites legible)
    # ------------------------------------------------------------------

    def workflow_submitted(self, workflow_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(WORKFLOW_SUBMITTED, workflow_id, **kw)

    def workflow_succeeded(self, workflow_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(WORKFLOW_SUCCEEDED, workflow_id, **kw)

    def workflow_failed(self, workflow_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(WORKFLOW_FAILED, workflow_id, **kw)

    def task_created(self, workflow_id: str, task_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(TASK_CREATED, workflow_id, task_id=task_id, **kw)

    def task_ready(self, workflow_id: str, task_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(TASK_READY, workflow_id, task_id=task_id, **kw)

    def task_started(self, workflow_id: str, task_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(TASK_STARTED, workflow_id, task_id=task_id, **kw)

    def task_succeeded(self, workflow_id: str, task_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(TASK_SUCCEEDED, workflow_id, task_id=task_id, **kw)

    def task_failed(self, workflow_id: str, task_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(TASK_FAILED, workflow_id, task_id=task_id, **kw)

    def task_timed_out(self, workflow_id: str, task_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(TASK_TIMED_OUT, workflow_id, task_id=task_id, **kw)

    def task_cancelled(self, workflow_id: str, task_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(TASK_CANCELLED, workflow_id, task_id=task_id, **kw)

    def task_blocked(self, workflow_id: str, task_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(TASK_BLOCKED, workflow_id, task_id=task_id, **kw)

    def task_unblocked(self, workflow_id: str, task_id: str, **kw: Any) -> WorkflowEvent:
        return self.emit(TASK_UNBLOCKED, workflow_id, task_id=task_id, **kw)
