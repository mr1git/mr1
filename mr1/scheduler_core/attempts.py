from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.kazi_runner import RunHandle, RunResult, RunStatus
from mr1.scheduler_core.events import SchedulerEventAdapter
from mr1.workflow_models import Task, TaskAttempt, TaskStatus, Workflow
from mr1.workflow_store import WorkflowStore, sync_task_view, sync_workflow_view


UNSET = object()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def attempt_result_payload(
    task: Task,
    *,
    status: TaskStatus,
    exit_code: Optional[int],
    error: Optional[str],
    error_type: Optional[str],
    result_summary: Optional[str],
) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "workflow_id": task.workflow_id,
        "attempt_id": task.current_attempt or None,
        "status": status.value,
        "exit_code": exit_code,
        "error": error,
        "error_type": error_type,
        "summary": result_summary,
    }


def set_task_terminal_attempt(
    store: WorkflowStore,
    workflow: Workflow,
    task: Task,
    *,
    status: TaskStatus,
    exit_code: Optional[int],
    error: Optional[str],
    error_type: Optional[str],
    result_summary: Optional[str],
    result_payload: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    if task.current_attempt <= 0 or task.current_attempt > len(task.attempts):
        return None
    attempt = task.attempts[task.current_attempt - 1]
    result_path = str(store.write_attempt_result(
        workflow.workflow_id,
        task.task_id,
        attempt.attempt_id,
        result_payload or attempt_result_payload(
            task,
            status=status,
            exit_code=exit_code,
            error=error,
            error_type=error_type,
            result_summary=result_summary,
        ),
    ))
    now_iso = _now_iso()
    attempt.finished_at = now_iso
    attempt.status = status
    attempt.exit_code = exit_code
    attempt.error = error
    attempt.error_type = error_type
    attempt.result_path = result_path
    task.status = status
    task.finished_at = now_iso
    task.exit_code = exit_code
    task.pid = None
    task.last_error = error
    task.last_error_type = error_type
    task.result_summary = result_summary
    task.result_path = result_path
    return result_path


def run_result_from_payload(
    task: Task,
    payload: dict[str, Any],
) -> Optional[RunResult]:
    status_value = payload.get("status")
    try:
        status = RunStatus(status_value)
    except ValueError:
        return None
    if status not in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.TIMED_OUT}:
        return None
    stdout_path = Path(task.log_stdout_path) if task.log_stdout_path else None
    stderr_path = Path(task.log_stderr_path) if task.log_stderr_path else None
    return RunResult(
        status=status,
        exit_code=payload.get("exit_code"),
        summary=payload.get("summary"),
        error=payload.get("error"),
        error_type=payload.get("error_type"),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        result_payload=dict(payload),
    )


class AttemptManager:
    def __init__(
        self,
        *,
        store: WorkflowStore,
        events: SchedulerEventAdapter,
        handle_registry: dict[str, RunHandle],
        now_fn: Any = _now_iso,
    ) -> None:
        self._store = store
        self._events = events
        self._handles = handle_registry
        self._now = now_fn

    def load_persisted_result(
        self,
        task: Task,
    ) -> tuple[Optional[RunResult], Optional[str]]:
        candidates: list[tuple[str, Optional[Path]]] = [
            ("task_result", Path(task.result_path) if task.result_path else None),
        ]
        if task.current_attempt > 0:
            candidates.append((
                "attempt_result",
                self._store.task_attempt_result_path(
                    task.workflow_id,
                    task.task_id,
                    task.current_attempt,
                ),
            ))
        for source, path in candidates:
            if path is None:
                continue
            payload = self._store._read_json_file(path)
            if not isinstance(payload, dict):
                continue
            result = run_result_from_payload(task, payload)
            if result is not None:
                return result, source
        return None, None

    def begin_attempt(
        self,
        workflow: Workflow,
        task: Task,
        *,
        message: str,
        pid: Optional[int] = None,
        watch_started: bool = False,
        tool_started: bool = False,
        policy_audit_path: Optional[str] = None,
        extra_events: Optional[list[tuple[str, str, dict[str, Any]]]] = None,
        run_handle: Optional[RunHandle] = None,
    ) -> int:
        attempt_id = task.current_attempt or (task.attempt_count + 1)
        original_workflow = workflow
        original_task = task
        with self._store.locked():
            task_id = task.task_id
            workflow = self._store.load_workflow(workflow.workflow_id)
            if workflow is None:
                raise RuntimeError(f"workflow not found during attempt start: {task.workflow_id}")
            task = workflow.tasks.get(task_id)
            if task is None:
                raise RuntimeError(f"task not found during attempt start: {task_id}")
            stdout_path, stderr_path = self._store.task_attempt_log_paths(
                workflow.workflow_id,
                task.task_id,
                attempt_id,
            )
            now_iso = self._now()
            task.attempt_count = attempt_id
            task.current_attempt = attempt_id
            task.status = TaskStatus.RUNNING
            task.started_at = now_iso
            task.finished_at = None
            task.pid = pid
            task.exit_code = None
            task.last_error = None
            task.last_error_type = None
            task.result_summary = None
            task.log_stdout_path = str(stdout_path)
            task.log_stderr_path = str(stderr_path)
            task.result_path = None
            task.dataflow_error = None
            task.blocked_by = []
            task.blocked_reason = None
            task.blocked_at = None
            task.skip_reason = None
            task.condition_result = None
            task.watch_satisfied_at = None
            task.last_checked_at = None
            task.last_check_result = None
            task.tool_finished_at = None
            task.tool_error = None
            if watch_started:
                task.watch_started_at = now_iso
            if tool_started:
                task.tool_started_at = now_iso
            if len(task.attempts) != attempt_id - 1:
                raise RuntimeError(
                    f"task attempts out of sequence for {task.task_id}: "
                    f"count={task.attempt_count} len={len(task.attempts)}"
                )
            task.attempts.append(TaskAttempt(
                attempt_id=attempt_id,
                started_at=now_iso,
                status=TaskStatus.RUNNING,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                policy_audit_path=policy_audit_path,
            ))
            self._store.save_workflow(workflow)
            if run_handle is not None:
                self._handles[task.task_id] = run_handle
            self._events.emit_attempt_started(
                workflow.workflow_id,
                task.task_id,
                attempt_id=attempt_id,
            )
            self._events.emit_task_started(
                workflow.workflow_id,
                task.task_id,
                attempt_id=attempt_id,
                message=message,
                pid=pid,
            )
            self._events.emit_timeline_task_started(
                workflow,
                task,
                attempt_id=attempt_id,
                message=message,
                pid=pid,
                record_path=str(self._store.workflow_json_path(workflow.workflow_id)),
            )
            self._events.emit_extra_events(
                workflow.workflow_id,
                task.task_id,
                attempt_id,
                extra_events,
            )
            sync_workflow_view(original_workflow, workflow)
            sync_task_view(original_task, task)
        return attempt_id

    def finish_attempt(
        self,
        workflow: Workflow,
        task: Task,
        new_status: TaskStatus,
        *,
        event: str,
        message: str,
        exit_code: Optional[int] = None,
        result_summary: Optional[str] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        log_stdout_path: Optional[str] = None,
        log_stderr_path: Optional[str] = None,
        result_path: Optional[str] = None,
        output_path: Optional[str] = None,
        inputs_path: Optional[str] = None,
        materialized_prompt_path: Optional[str] = None,
        artifacts: Optional[list[Any]] = None,
        dataflow_error: Any = UNSET,
        blocked_by: Optional[list[str]] = None,
        blocked_reason: Any = UNSET,
        blocked_at: Any = UNSET,
        watch_satisfied_at: Any = UNSET,
        last_checked_at: Any = UNSET,
        last_check_result: Any = UNSET,
        condition: Any = UNSET,
        tool_finished_at: Any = UNSET,
        tool_error: Any = UNSET,
        extra_events: Optional[list[tuple[str, str, dict[str, Any]]]] = None,
    ) -> None:
        attempt_id = task.current_attempt or None
        original_workflow = workflow
        original_task = task
        with self._store.locked():
            task_id = task.task_id
            workflow = self._store.load_workflow(workflow.workflow_id)
            if workflow is None:
                raise RuntimeError(f"workflow not found during attempt finish: {task.workflow_id}")
            task = workflow.tasks.get(task_id)
            if task is None:
                raise RuntimeError(f"task not found during attempt finish: {task_id}")
            now_iso = self._now()
            task.status = new_status
            task.finished_at = now_iso
            task.pid = None
            task.exit_code = exit_code
            task.last_error = error
            task.last_error_type = error_type
            if result_summary is not None:
                task.result_summary = result_summary
            if log_stdout_path is not None:
                task.log_stdout_path = log_stdout_path
            if log_stderr_path is not None:
                task.log_stderr_path = log_stderr_path
            if result_path is not None:
                task.result_path = result_path
            if output_path is not None:
                task.output_path = output_path
            if inputs_path is not None:
                task.inputs_path = inputs_path
            if materialized_prompt_path is not None:
                task.materialized_prompt_path = materialized_prompt_path
            if artifacts is not None:
                task.artifacts = list(artifacts)
            if dataflow_error is not UNSET:
                task.dataflow_error = dataflow_error
            if blocked_by is not None:
                task.blocked_by = list(blocked_by)
            if blocked_reason is not UNSET:
                task.blocked_reason = blocked_reason
            if blocked_at is not UNSET:
                task.blocked_at = blocked_at
            if watch_satisfied_at is not UNSET:
                task.watch_satisfied_at = watch_satisfied_at
            if last_checked_at is not UNSET:
                task.last_checked_at = last_checked_at
            if last_check_result is not UNSET:
                task.last_check_result = (
                    dict(last_check_result)
                    if last_check_result is not None else None
                )
            if condition is not UNSET:
                task.condition = dict(condition) if condition is not None else None
            if tool_finished_at is not UNSET:
                task.tool_finished_at = tool_finished_at
            if tool_error is not UNSET:
                task.tool_error = tool_error
            if attempt_id is not None and 0 < attempt_id <= len(task.attempts):
                attempt = task.attempts[attempt_id - 1]
                attempt.finished_at = now_iso
                attempt.status = new_status
                attempt.exit_code = exit_code
                attempt.error = error
                attempt.error_type = error_type
                if log_stdout_path is not None:
                    attempt.stdout_path = log_stdout_path
                if log_stderr_path is not None:
                    attempt.stderr_path = log_stderr_path
                if result_path is not None:
                    attempt.result_path = result_path
            self._store.save_workflow(workflow)
            if attempt_id is not None:
                self._events.emit_attempt_finished(
                    workflow.workflow_id,
                    task.task_id,
                    attempt_id=attempt_id,
                    status=new_status,
                    error_type=error_type,
                )
            self._events.emit_task_finished(
                workflow.workflow_id,
                task.task_id,
                attempt_id=attempt_id,
                event=event,
                message=message,
                status=new_status,
            )
            self._events.emit_timeline_task_finished(
                workflow,
                task,
                attempt_id=attempt_id,
                status=new_status,
                message=message,
                error_type=error_type,
                record_path=result_path or str(self._store.workflow_json_path(workflow.workflow_id)),
            )
            self._events.emit_extra_events(
                workflow.workflow_id,
                task.task_id,
                attempt_id,
                extra_events,
            )
            sync_workflow_view(original_workflow, workflow)
            sync_task_view(original_task, task)
