from __future__ import annotations

from typing import Any, Optional

from mr1.event_log import EventLog
from mr1.workflow_events import WorkflowEventLog
from mr1.workflow_models import Task, TaskStatus, Workflow


class SchedulerEventAdapter:
    def __init__(
        self,
        *,
        workflow_events: WorkflowEventLog,
        timeline: EventLog,
        agent_id: str,
    ) -> None:
        self._workflow_events = workflow_events
        self._timeline = timeline
        self._agent_id = agent_id

    def emit_extra_events(
        self,
        workflow_id: str,
        task_id: str,
        attempt_id: Optional[int],
        extra_events: Optional[list[tuple[str, str, dict[str, Any]]]],
    ) -> None:
        for event_type, event_message, event_metadata in extra_events or []:
            self._workflow_events.emit(
                event_type,
                workflow_id,
                task_id=task_id,
                attempt_id=attempt_id,
                agent_id=self._agent_id,
                message=event_message,
                metadata=dict(event_metadata),
            )

    def emit_attempt_started(
        self,
        workflow_id: str,
        task_id: str,
        *,
        attempt_id: int,
    ) -> None:
        self._workflow_events.task_attempt_started(
            workflow_id,
            task_id,
            agent_id=self._agent_id,
            attempt_id=attempt_id,
            message="task attempt started",
            metadata={"status": TaskStatus.RUNNING.value},
        )

    def emit_task_started(
        self,
        workflow_id: str,
        task_id: str,
        *,
        attempt_id: int,
        message: str,
        pid: Optional[int],
    ) -> None:
        metadata = {"status": TaskStatus.RUNNING.value, "pid": pid} if pid is not None else {
            "status": TaskStatus.RUNNING.value
        }
        self._workflow_events.task_started(
            workflow_id,
            task_id,
            agent_id=self._agent_id,
            attempt_id=attempt_id,
            message=message,
            metadata=metadata,
        )

    def emit_timeline_task_started(
        self,
        workflow: Workflow,
        task: Task,
        *,
        attempt_id: int,
        message: str,
        pid: Optional[int],
        record_path: str,
    ) -> None:
        self._timeline.emit(
            event_type="workflow_task_started",
            actor_id=self._agent_id,
            actor_type="scheduler",
            target_id=task.task_id,
            target_type="task",
            status=TaskStatus.RUNNING.value,
            summary=message,
            workflow_id=workflow.workflow_id,
            task_id=task.task_id,
            record_path=record_path,
            metadata={
                "attempt_id": attempt_id,
                "task_kind": task.task_kind,
                "pid": pid,
            },
        )

    def emit_attempt_finished(
        self,
        workflow_id: str,
        task_id: str,
        *,
        attempt_id: int,
        status: TaskStatus,
        error_type: Optional[str],
    ) -> None:
        self._workflow_events.task_attempt_finished(
            workflow_id,
            task_id,
            agent_id=self._agent_id,
            attempt_id=attempt_id,
            message="task attempt finished",
            metadata={
                "status": status.value,
                "error_type": error_type,
            },
        )

    def emit_task_finished(
        self,
        workflow_id: str,
        task_id: str,
        *,
        attempt_id: Optional[int],
        event: str,
        message: str,
        status: TaskStatus,
    ) -> None:
        self._workflow_events.emit(
            event,
            workflow_id,
            task_id=task_id,
            attempt_id=attempt_id,
            agent_id=self._agent_id,
            message=message,
            metadata={"status": status.value},
        )

    def emit_timeline_task_finished(
        self,
        workflow: Workflow,
        task: Task,
        *,
        attempt_id: Optional[int],
        status: TaskStatus,
        message: str,
        error_type: Optional[str],
        record_path: str,
    ) -> None:
        self._timeline.emit(
            event_type=(
                "workflow_task_completed"
                if status in {TaskStatus.SUCCEEDED, TaskStatus.SKIPPED} else
                "workflow_task_failed"
            ),
            actor_id=self._agent_id,
            actor_type="scheduler",
            target_id=task.task_id,
            target_type="task",
            status=status.value,
            summary=message,
            workflow_id=workflow.workflow_id,
            task_id=task.task_id,
            record_path=record_path,
            metadata={
                "attempt_id": attempt_id,
                "task_kind": task.task_kind,
                "error_type": error_type,
            },
        )

    def emit_watcher_check(
        self,
        workflow_id: str,
        task_id: str,
        *,
        attempt_id: Optional[int],
        message: str,
        metadata: dict[str, Any],
    ) -> None:
        self._workflow_events.emit(
            "watcher_checked",
            workflow_id,
            task_id=task_id,
            attempt_id=attempt_id,
            agent_id=self._agent_id,
            message=message,
            metadata=dict(metadata),
        )
