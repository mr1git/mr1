"""
Workflow data models — pure data, no I/O.

A workflow is a DAG of tasks. Each task has a deterministic lifecycle
managed by the scheduler. Every field on these models is JSON-safe so
the store can round-trip them through `workflow.json`.

Status meaning:
  created     — exists in the workflow plan, nothing launched yet
  waiting     — has unmet dependencies
  ready       — all dependencies satisfied, scheduler may launch
  running     — runner has an active process/handle for this task
  succeeded   — terminal, clean exit
  skipped     — terminal, intentionally not executed
  failed      — terminal, non-zero exit for a reason other than timeout
  timed_out   — terminal, killed after exceeding timeout
  cancelled   — terminal, cancelled by user or shutdown
  blocked     — terminal, a prerequisite failed/timed_out/cancelled
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from mr1.dataflow import Artifact, TaskInputSpec


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    CREATED = "created"
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_TASK_STATUSES = frozenset({
    TaskStatus.SUCCEEDED,
    TaskStatus.SKIPPED,
    TaskStatus.FAILED,
    TaskStatus.TIMED_OUT,
    TaskStatus.CANCELLED,
    TaskStatus.BLOCKED,
})

FAILED_TASK_STATUSES = frozenset({
    TaskStatus.FAILED,
    TaskStatus.TIMED_OUT,
    TaskStatus.CANCELLED,
    TaskStatus.BLOCKED,
})

TERMINAL_WORKFLOW_STATUSES = frozenset({
    WorkflowStatus.SUCCEEDED,
    WorkflowStatus.FAILED,
    WorkflowStatus.CANCELLED,
})


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def new_workflow_id() -> str:
    """Deterministic format, non-guessable suffix."""
    return f"wf-{_ts_compact()}-{uuid.uuid4().hex[:6]}"


def new_task_id() -> str:
    return f"tk-{_ts_compact()}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass
class Provenance:
    """Who created this workflow or task."""
    type: str  # "agent" | "user" | "scheduler"
    id: str    # "MR1", "cli", "scheduler", etc.

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Provenance":
        return cls(type=data["type"], id=data["id"])


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class TaskAttempt:
    attempt_id: int
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    status: TaskStatus = TaskStatus.CREATED
    exit_code: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    result_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskAttempt":
        return cls(
            attempt_id=int(data["attempt_id"]),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            status=TaskStatus(data.get("status", "created")),
            exit_code=data.get("exit_code"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            stdout_path=data.get("stdout_path"),
            stderr_path=data.get("stderr_path"),
            result_path=data.get("result_path"),
        )


@dataclass
class Task:
    """
    One unit of work within a workflow.

    `depends_on` holds resolved task IDs (not labels) once the scheduler
    has ingested a submitted spec.
    """
    task_id: str
    workflow_id: str
    label: str
    title: str
    task_kind: str                  # "agent" | "watcher" | "tool"
    agent_type: Optional[str]       # "kazi" | "mrn" | None
    prompt: str
    watcher_type: Optional[str] = None
    watch_config: dict[str, Any] = field(default_factory=dict)
    watch_started_at: Optional[str] = None
    watch_satisfied_at: Optional[str] = None
    tool_type: Optional[str] = None
    tool_config: dict[str, Any] = field(default_factory=dict)
    tool_started_at: Optional[str] = None
    tool_finished_at: Optional[str] = None
    tool_error: Optional[str] = None
    last_checked_at: Optional[str] = None
    last_check_result: Optional[dict[str, Any]] = None
    condition: Optional[dict[str, Any]] = None
    run_if: Optional[dict[str, Any]] = None
    dependency_policy: str = "all_succeeded"
    depends_on: list[str] = field(default_factory=list)
    inputs: list[TaskInputSpec] = field(default_factory=list)
    attempt_count: int = 0
    current_attempt: int = 0
    attempts: list[TaskAttempt] = field(default_factory=list)
    status: TaskStatus = TaskStatus.CREATED
    created_by: Optional[Provenance] = None
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    last_error: Optional[str] = None
    last_error_type: Optional[str] = None
    result_summary: Optional[str] = None
    log_stdout_path: Optional[str] = None
    log_stderr_path: Optional[str] = None
    result_path: Optional[str] = None
    output_path: Optional[str] = None
    inputs_path: Optional[str] = None
    materialized_prompt_path: Optional[str] = None
    artifacts: list[Artifact] = field(default_factory=list)
    dataflow_error: Optional[str] = None
    blocked_reason: Optional[str] = None
    blocked_by: list[str] = field(default_factory=list)
    blocked_at: Optional[str] = None
    skip_reason: Optional[str] = None
    condition_result: Optional[dict[str, Any]] = None
    timeout_s: Optional[int] = None

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_TASK_STATUSES

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["created_by"] = self.created_by.to_dict() if self.created_by else None
        d["inputs"] = [item.to_dict() for item in self.inputs]
        d["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        d["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        raw_created_by = data.get("created_by")
        return cls(
            task_id=data["task_id"],
            workflow_id=data["workflow_id"],
            label=data["label"],
            title=data["title"],
            task_kind=data["task_kind"],
            agent_type=data.get("agent_type"),
            prompt=data.get("prompt", ""),
            watcher_type=data.get("watcher_type"),
            watch_config=dict(data.get("watch_config", {})),
            watch_started_at=data.get("watch_started_at"),
            watch_satisfied_at=data.get("watch_satisfied_at"),
            tool_type=data.get("tool_type"),
            tool_config=dict(data.get("tool_config", {})),
            tool_started_at=data.get("tool_started_at"),
            tool_finished_at=data.get("tool_finished_at"),
            tool_error=data.get("tool_error"),
            last_checked_at=data.get("last_checked_at"),
            last_check_result=dict(data["last_check_result"])
            if data.get("last_check_result") is not None else None,
            condition=dict(data["condition"])
            if data.get("condition") is not None else None,
            run_if=dict(data["run_if"])
            if data.get("run_if") is not None else None,
            dependency_policy=data.get("dependency_policy", "all_succeeded"),
            depends_on=list(data.get("depends_on", [])),
            inputs=[
                TaskInputSpec.from_dict(item)
                for item in data.get("inputs", [])
            ],
            attempt_count=data.get("attempt_count", 0),
            current_attempt=data.get("current_attempt", 0),
            attempts=[
                TaskAttempt.from_dict(item)
                for item in data.get("attempts", [])
            ],
            status=TaskStatus(data.get("status", "created")),
            created_by=(
                Provenance.from_dict(raw_created_by)
                if raw_created_by else None
            ),
            created_at=data.get("created_at", _now_iso()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            pid=data.get("pid"),
            exit_code=data.get("exit_code"),
            last_error=data.get("last_error"),
            last_error_type=data.get("last_error_type"),
            result_summary=data.get("result_summary"),
            log_stdout_path=data.get("log_stdout_path"),
            log_stderr_path=data.get("log_stderr_path"),
            result_path=data.get("result_path"),
            output_path=data.get("output_path"),
            inputs_path=data.get("inputs_path"),
            materialized_prompt_path=data.get("materialized_prompt_path"),
            artifacts=[
                Artifact.from_dict(item)
                for item in data.get("artifacts", [])
            ],
            dataflow_error=data.get("dataflow_error"),
            blocked_reason=data.get("blocked_reason"),
            blocked_by=list(data.get("blocked_by", [])),
            blocked_at=data.get("blocked_at"),
            skip_reason=data.get("skip_reason"),
            condition_result=dict(data["condition_result"])
            if data.get("condition_result") is not None else None,
            timeout_s=data.get("timeout_s"),
        )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@dataclass
class Workflow:
    workflow_id: str
    title: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_by: Optional[Provenance] = None
    created_at: str = field(default_factory=_now_iso)
    finished_at: Optional[str] = None
    tasks: dict[str, Task] = field(default_factory=dict)
    label_to_task_id: dict[str, str] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_WORKFLOW_STATUSES

    def task_by_label(self, label: str) -> Optional[Task]:
        task_id = self.label_to_task_id.get(label)
        return self.tasks.get(task_id) if task_id else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "title": self.title,
            "status": self.status.value,
            "created_by": self.created_by.to_dict() if self.created_by else None,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
            "label_to_task_id": dict(self.label_to_task_id),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Workflow":
        raw_created_by = data.get("created_by")
        tasks = {
            tid: Task.from_dict(tdata)
            for tid, tdata in data.get("tasks", {}).items()
        }
        return cls(
            workflow_id=data["workflow_id"],
            title=data["title"],
            status=WorkflowStatus(data.get("status", "pending")),
            created_by=(
                Provenance.from_dict(raw_created_by)
                if raw_created_by else None
            ),
            created_at=data.get("created_at", _now_iso()),
            finished_at=data.get("finished_at"),
            tasks=tasks,
            label_to_task_id=dict(data.get("label_to_task_id", {})),
        )


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


@dataclass
class WorkflowEvent:
    """
    One entry in a workflow's append-only event log.

    `event_type` is a short kebab-like string, e.g. "task_started".
    `agent_id` is the acting party (scheduler, MR1, kazi, etc.).
    `metadata` holds anything extra the emitter wants to preserve.
    """
    timestamp: str
    event_type: str
    workflow_id: str
    task_id: Optional[str] = None
    attempt_id: Optional[int] = None
    agent_id: Optional[str] = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        event_type: str,
        workflow_id: str,
        task_id: Optional[str] = None,
        attempt_id: Optional[int] = None,
        agent_id: Optional[str] = None,
        message: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> "WorkflowEvent":
        return cls(
            timestamp=_now_iso(),
            event_type=event_type,
            workflow_id=workflow_id,
            task_id=task_id,
            attempt_id=attempt_id,
            agent_id=agent_id,
            message=message,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowEvent":
        return cls(
            timestamp=data["timestamp"],
            event_type=data["event_type"],
            workflow_id=data["workflow_id"],
            task_id=data.get("task_id"),
            attempt_id=data.get("attempt_id"),
            agent_id=data.get("agent_id"),
            message=data.get("message", ""),
            metadata=dict(data.get("metadata", {})),
        )
