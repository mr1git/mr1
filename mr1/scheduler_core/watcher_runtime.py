"""Watcher-task polling service.

Encapsulates the body of `Scheduler._poll_running_watcher` plus its four
private helpers (`_should_evaluate_watcher`, `_watcher_timeout_message`,
`_watcher_result_payload`, `_record_watcher_check`). The helpers were
only called from `_poll_running_watcher` so they move with it.

Extracted from `mr1.scheduler`. `Scheduler` keeps a thin
`_poll_running_watcher` wrapper that delegates here, per the Stage 6
plan.

Design notes
------------
Like `ToolTaskRunner`, the service is constructed once per `Scheduler`
with explicit collaborators and callbacks rather than a full
`Scheduler` reference. Four scheduler-owned procedures
(`current_attempt_policy_audit_path`, `finalize_policy_audit`,
`append_policy_audit_index_from_path`, `finish_attempt`) are injected
as callables.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from mr1 import workflow_events as ev
from mr1.capability_policy import CapabilityApprovalStore, CapabilityAuditRecord
from mr1.clock import Clock, default_clock
from mr1.dataflow import build_watcher_task_output
from mr1.scheduler_core.attempts import UNSET as _UNSET
from mr1.scheduler_core.events import SchedulerEventAdapter
from mr1.watchers import WatchEvaluation, WatcherRegistry
from mr1.workflow_models import Task, TaskStatus, Workflow
from mr1.workflow_store import WorkflowStore, sync_task_view, sync_workflow_view


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WatcherPollService:
    """Polls a single running watcher task once per scheduler tick."""

    def __init__(
        self,
        *,
        store: WorkflowStore,
        watchers: WatcherRegistry,
        approval_store: CapabilityApprovalStore,
        event_adapter: SchedulerEventAdapter,
        current_attempt_policy_audit_path: Callable[[Task], Optional[Path]],
        finalize_policy_audit: Callable[..., None],
        append_policy_audit_index_from_path: Callable[..., None],
        finish_attempt: Callable[..., None],
        clock: Optional[Clock] = None,
    ) -> None:
        self._clock = clock or default_clock()
        self._store = store
        self._watchers = watchers
        self._approval_store = approval_store
        self._event_adapter = event_adapter
        self._current_attempt_policy_audit_path = current_attempt_policy_audit_path
        self._finalize_policy_audit = finalize_policy_audit
        self._append_policy_audit_index_from_path = append_policy_audit_index_from_path
        self._finish_attempt = finish_attempt

    # ------------------------------------------------------------------
    # Helpers (private; previously methods on Scheduler).
    # ------------------------------------------------------------------

    def _should_evaluate_watcher(self, task: Task) -> bool:
        if not task.last_checked_at:
            return True
        interval_s = task.watch_config.get("poll_interval_s", 1)
        if not isinstance(interval_s, (int, float)) or interval_s < 0:
            interval_s = 1
        last_checked = datetime.fromisoformat(task.last_checked_at)
        return (self._clock.now() - last_checked).total_seconds() >= interval_s

    def _watcher_timeout_message(self, task: Task) -> Optional[str]:
        max_wait_s = task.watch_config.get("max_wait_s")
        if not isinstance(max_wait_s, (int, float)) or max_wait_s <= 0:
            return None
        started_at = task.watch_started_at or task.started_at
        if not started_at:
            return None
        started_dt = datetime.fromisoformat(started_at)
        elapsed_s = (self._clock.now() - started_dt).total_seconds()
        if elapsed_s < max_wait_s:
            return None
        return f"watcher exceeded max_wait_s={int(max_wait_s)}"

    def _watcher_result_payload(
        self,
        task: Task,
        evaluation: WatchEvaluation,
    ) -> dict[str, Any]:
        payload = {
            "state": evaluation.state,
            "message": evaluation.message,
            "watcher_type": task.watcher_type,
        }
        payload.update(dict(evaluation.metadata))
        return payload

    def _record_watcher_check(
        self,
        wf: Workflow,
        task: Task,
        *,
        checked_at: str,
        check_payload: dict[str, Any],
    ) -> None:
        original_wf = wf
        original_task = task
        with self._store.locked():
            live_wf = self._store.load_workflow(wf.workflow_id)
            if live_wf is None:
                raise RuntimeError(f"workflow not found during watcher check: {wf.workflow_id}")
            live_task = live_wf.tasks.get(task.task_id)
            if live_task is None:
                raise RuntimeError(f"task not found during watcher check: {task.task_id}")
            live_task.last_checked_at = checked_at
            live_task.last_check_result = dict(check_payload)
            self._store.save_workflow(live_wf)
            self._event_adapter.emit_watcher_check(
                live_wf.workflow_id,
                live_task.task_id,
                attempt_id=live_task.current_attempt or None,
                message=check_payload.get("message", ""),
                metadata=dict(check_payload),
            )
            sync_workflow_view(original_wf, live_wf)
            sync_task_view(original_task, live_task)

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    def poll(self, wf: Workflow, task: Task) -> bool:
        audit_path = self._current_attempt_policy_audit_path(task)
        timeout_message = self._watcher_timeout_message(task)
        if timeout_message is not None:
            checked_at = self._clock.now_iso()
            payload = {
                "state": "timed_out",
                "message": timeout_message,
                "watcher_type": task.watcher_type,
            }
            result_path = str(self._store.write_attempt_result(
                wf.workflow_id,
                task.task_id,
                task.current_attempt,
                payload,
            ))
            if audit_path is not None:
                self._finalize_policy_audit(
                    audit_path,
                    execution_result=dict(payload),
                    error=timeout_message,
                )
                self._append_policy_audit_index_from_path(
                    audit_path,
                    execution_status=TaskStatus.TIMED_OUT.value,
                    error=timeout_message,
                )
            self._finish_attempt(
                wf,
                task,
                TaskStatus.TIMED_OUT,
                event=ev.TASK_TIMED_OUT,
                message=timeout_message,
                error=timeout_message,
                error_type="timeout",
                result_path=result_path,
                watch_satisfied_at=_UNSET,
                last_checked_at=checked_at,
                last_check_result=payload,
                extra_events=[(
                    ev.WATCHER_TIMED_OUT,
                    timeout_message,
                    payload,
                )],
            )
            return True

        if not self._should_evaluate_watcher(task):
            return False

        now = self._clock.now()
        try:
            evaluation = self._watchers.evaluate(task, now)
        except Exception as exc:
            evaluation = WatchEvaluation(
                state="failed",
                message=f"watcher evaluation error: {exc}",
                metadata={"error": str(exc), "watcher_type": task.watcher_type},
            )

        checked_at = now.isoformat()
        check_payload = self._watcher_result_payload(task, evaluation)
        if evaluation.state == "not_satisfied":
            self._record_watcher_check(
                wf,
                task,
                checked_at=checked_at,
                check_payload=check_payload,
            )
            return True

        if evaluation.state == "satisfied":
            task.status = TaskStatus.SUCCEEDED
            task.last_checked_at = checked_at
            task.last_check_result = check_payload
            task.watch_satisfied_at = self._clock.now_iso()
            output = build_watcher_task_output(task)
            output_path = str(self._store.write_task_output(
                wf.workflow_id,
                task.task_id,
                output,
            ))
            result_path = str(self._store.write_attempt_result(
                wf.workflow_id,
                task.task_id,
                task.current_attempt,
                check_payload,
            ))
            if audit_path is not None:
                self._finalize_policy_audit(
                    audit_path,
                    execution_result=dict(check_payload),
                )
                approval_request_id = None
                try:
                    with open(audit_path, "r", encoding="utf-8") as handle:
                        audit_record = CapabilityAuditRecord.from_dict(json.load(handle))
                    approval_request_id = audit_record.decision.get("metadata", {}).get("approval_request_id")
                except (OSError, json.JSONDecodeError, KeyError, ValueError):
                    approval_request_id = None
                if isinstance(approval_request_id, str) and approval_request_id:
                    self._approval_store.mark_used(
                        approval_request_id,
                        audit_id=audit_path.stem,
                    )
                self._append_policy_audit_index_from_path(
                    audit_path,
                    execution_status=TaskStatus.SUCCEEDED.value,
                    approval_request_id=approval_request_id if isinstance(approval_request_id, str) else None,
                )
            self._finish_attempt(
                wf,
                task,
                TaskStatus.SUCCEEDED,
                event=ev.TASK_SUCCEEDED,
                message=evaluation.message,
                watch_satisfied_at=self._clock.now_iso(),
                last_checked_at=checked_at,
                last_check_result=check_payload,
                extra_events=[(
                    ev.WATCHER_SATISFIED,
                    evaluation.message,
                    check_payload,
                ), (
                    ev.OUTPUT_WRITTEN,
                    "normalized output written",
                    {"path": output_path},
                )],
                output_path=output_path,
                result_path=result_path,
            )
            return True

        if evaluation.state == "timed_out":
            result_path = str(self._store.write_attempt_result(
                wf.workflow_id,
                task.task_id,
                task.current_attempt,
                check_payload,
            ))
            if audit_path is not None:
                self._finalize_policy_audit(
                    audit_path,
                    execution_result=dict(check_payload),
                    error=evaluation.message,
                )
                self._append_policy_audit_index_from_path(
                    audit_path,
                    execution_status=TaskStatus.TIMED_OUT.value,
                    error=evaluation.message,
                )
            self._finish_attempt(
                wf,
                task,
                TaskStatus.TIMED_OUT,
                event=ev.TASK_TIMED_OUT,
                message=evaluation.message,
                error=evaluation.message,
                error_type="timeout",
                result_path=result_path,
                last_checked_at=checked_at,
                last_check_result=check_payload,
                extra_events=[(
                    ev.WATCHER_TIMED_OUT,
                    evaluation.message,
                    check_payload,
                )],
            )
            return True

        result_path = str(self._store.write_attempt_result(
            wf.workflow_id,
            task.task_id,
            task.current_attempt,
            check_payload,
        ))
        if audit_path is not None:
            self._finalize_policy_audit(
                audit_path,
                execution_result=dict(check_payload),
                error=evaluation.message,
            )
            self._append_policy_audit_index_from_path(
                audit_path,
                execution_status=TaskStatus.FAILED.value,
                error=evaluation.message,
            )
        self._finish_attempt(
            wf,
            task,
            TaskStatus.FAILED,
            event=ev.TASK_FAILED,
            message=evaluation.message,
            error=evaluation.message,
            error_type="unknown",
            result_path=result_path,
            last_checked_at=checked_at,
            last_check_result=check_payload,
            extra_events=[(
                ev.WATCHER_FAILED,
                evaluation.message,
                check_payload,
            )],
        )
        return True
