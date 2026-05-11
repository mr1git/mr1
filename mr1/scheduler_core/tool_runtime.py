"""Tool-task execution service.

Encapsulates the body of `Scheduler._run_tool_task`: runs a tool against
a task, normalizes the result, registers artifacts, writes the
normalized output, writes the result-payload file, finalizes the policy
audit, marks any used approval, emits timeline events, and finishes the
attempt.

Extracted from `mr1.scheduler` so `scheduler.py` no longer owns this
~200-line procedure directly. Scheduler still exposes a thin
`_run_tool_task` wrapper that delegates here, preserving any callers
and any tests that monkey-patch the method name.

Design notes
------------
The runner takes explicit collaborators rather than a full `Scheduler`
reference. Several collaborators are scheduler-owned procedures
(`apply_semantic_validation`, `write_policy_audit`,
`append_policy_audit_index`, `policy_audit_id`, `finish_attempt`); they
are injected as callables so this service stays decoupled from the
`Scheduler` class itself. The boundary is not perfectly clean — five
callbacks signal that further refactoring of audit/finalize is possible
in a later pass — but it is explicit, and the alternative (passing the
whole `Scheduler`) was rejected in the Stage 6 plan.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from mr1 import workflow_events as ev
from mr1.capability_policy import (
    CapabilityApprovalStore,
    CapabilityMetadata,
    CapabilityRequest,
)
from mr1.dataflow import (
    DataflowError,
    build_tool_task_output,
    register_artifacts,
)
from mr1.event_log import EventLog
from mr1.tools import ToolRegistry, ToolResult
from mr1.workflow_models import Task, TaskStatus, Workflow
from mr1.workflow_store import WorkflowStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolTaskRunner:
    """Stateless tool-task execution service.

    Constructed once per `Scheduler` instance with the collaborators it
    needs. `run()` does not mutate any state on the runner itself; all
    state lives on the workflow/task objects, the store, and the
    scheduler's audit/finalize callbacks.
    """

    def __init__(
        self,
        *,
        tools: ToolRegistry,
        store: WorkflowStore,
        approval_store: CapabilityApprovalStore,
        timeline: EventLog,
        apply_semantic_validation: Callable[..., tuple],
        write_policy_audit: Callable[..., None],
        append_policy_audit_index: Callable[..., None],
        policy_audit_id: Callable[[Path], str],
        finish_attempt: Callable[..., None],
    ) -> None:
        self._tools = tools
        self._store = store
        self._approval_store = approval_store
        self._timeline = timeline
        self._apply_semantic_validation = apply_semantic_validation
        self._write_policy_audit = write_policy_audit
        self._append_policy_audit_index = append_policy_audit_index
        self._policy_audit_id = policy_audit_id
        self._finish_attempt = finish_attempt

    def run(
        self,
        wf: Workflow,
        task: Task,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        decision: dict[str, Any],
        audit_path: Path,
    ) -> None:
        try:
            tool_result = self._tools.run(task, self._store, wf)
        except Exception as exc:
            tool_result = ToolResult(
                state="failed",
                summary=f"tool failed: {task.tool_type}",
                text="",
                error=str(exc),
                metadata={"tool_type": task.tool_type},
            )

        state_map = {
            "succeeded": (TaskStatus.SUCCEEDED, ev.TASK_SUCCEEDED, ev.TOOL_SUCCEEDED),
            "failed": (TaskStatus.FAILED, ev.TASK_FAILED, ev.TOOL_FAILED),
            "timed_out": (TaskStatus.TIMED_OUT, ev.TASK_TIMED_OUT, ev.TOOL_TIMED_OUT),
        }
        target_status, task_event, tool_event = state_map.get(
            tool_result.state,
            (TaskStatus.FAILED, ev.TASK_FAILED, ev.TOOL_FAILED),
        )
        error_type: Optional[str] = "timeout" if target_status is TaskStatus.TIMED_OUT else None

        extra_events: list[tuple[str, str, dict[str, Any]]] = []
        output_path: Optional[str] = None
        dataflow_error: Optional[str] = None
        artifacts: list[Any] = []
        output = None
        try:
            artifacts = register_artifacts(task, self._store, tool_result.artifacts)
            for artifact in artifacts:
                extra_events.append((
                    ev.ARTIFACT_REGISTERED,
                    f"artifact registered: {artifact.name}",
                    {"name": artifact.name, "kind": artifact.kind, "path": artifact.path},
                ))
            if target_status is TaskStatus.SUCCEEDED:
                output = build_tool_task_output(
                    replace(
                        task,
                        status=target_status,
                        tool_error=tool_result.error,
                    ),
                    tool_result,
                )
        except DataflowError as exc:
            target_status = TaskStatus.FAILED
            task_event = ev.TASK_FAILED
            tool_event = ev.TOOL_FAILED
            dataflow_error = str(exc)
            error_type = "unknown"
            tool_result = ToolResult(
                state="failed",
                summary=tool_result.summary,
                text=tool_result.text,
                data=tool_result.data,
                metrics=tool_result.metrics,
                artifacts=[],
                metadata=tool_result.metadata,
                error=str(exc),
            )

        result_payload = {
            "task_id": task.task_id,
            "workflow_id": wf.workflow_id,
            "attempt_id": task.current_attempt,
            "status": target_status.value,
            "summary": tool_result.summary,
            "text": tool_result.text,
            "data": tool_result.data,
            "metrics": tool_result.metrics,
            "error": dataflow_error or tool_result.error,
            "error_type": error_type,
            "failure_type": None,
            "retryable": None,
            "audit_record_path": str(audit_path),
        }
        target_status, task_event, tool_result_summary, semantic_error, semantic_error_type, result_payload = self._apply_semantic_validation(
            wf,
            task,
            target_status=target_status,
            event_type=task_event,
            summary=output.summary if output is not None else tool_result.summary,
            text=output.text if output is not None else tool_result.text,
            data=output.data if output is not None else tool_result.data,
            result_payload=result_payload,
        )
        if semantic_error_type is not None:
            error_type = semantic_error_type
            tool_event = ev.TOOL_FAILED
        tool_result_summary = tool_result_summary or tool_result.summary
        if target_status is TaskStatus.SUCCEEDED and output is not None:
            output_path = str(self._store.write_task_output(
                wf.workflow_id,
                task.task_id,
                output,
            ))
            extra_events.append((
                ev.OUTPUT_WRITTEN,
                "normalized output written",
                {"path": output_path},
            ))
        if target_status is TaskStatus.FAILED and error_type is None:
            error_type = "unknown"
        result_payload["error_type"] = error_type
        result_path = str(self._store.write_attempt_result(
            wf.workflow_id,
            task.task_id,
            task.current_attempt,
            result_payload,
        ))
        self._write_policy_audit(
            audit_path,
            capability_name=request.capability_name,
            request=request,
            metadata=metadata,
            decision=decision,
            execution_result={
                "status": target_status.value,
                "summary": result_payload.get("summary"),
                "text": result_payload.get("text"),
                "data": dict(result_payload.get("data") or {}),
                "metrics": dict(result_payload.get("metrics") or {}),
            },
            error=result_payload.get("error"),
        )
        approval_request_id = decision.get("metadata", {}).get("approval_request_id")
        if (
            target_status is TaskStatus.SUCCEEDED
            and isinstance(approval_request_id, str)
            and approval_request_id
        ):
            self._approval_store.mark_used(
                approval_request_id,
                audit_id=audit_path.stem,
            )
        self._append_policy_audit_index(
            actor_id=request.actor_id,
            audit_id=self._policy_audit_id(audit_path),
            audit_path=audit_path,
            request=request,
            metadata=metadata,
            decision=decision,
            execution_status=target_status.value,
            error=result_payload.get("error"),
            approval_request_id=approval_request_id if isinstance(approval_request_id, str) else None,
        )
        self._timeline.emit(
            event_type="capability_executed" if target_status is TaskStatus.SUCCEEDED else "capability_failed",
            actor_id=request.actor_id,
            actor_type=request.actor_type,
            target_id=request.capability_name,
            target_type="capability",
            status=target_status.value,
            summary=(
                f"capability executed: {request.capability_name}"
                if target_status is TaskStatus.SUCCEEDED else
                f"capability failed: {request.capability_name}"
            ),
            workflow_id=wf.workflow_id,
            task_id=task.task_id,
            approval_request_id=approval_request_id if isinstance(approval_request_id, str) else None,
            audit_id=self._policy_audit_id(audit_path),
            record_path=str(audit_path),
            metadata={"error": result_payload.get("error")},
        )
        message = semantic_error or dataflow_error or tool_result.error or tool_result.summary or ""
        event_metadata = {
            "tool_type": task.tool_type,
            "state": target_status.value,
        }
        if result_payload.get("error"):
            event_metadata["error"] = result_payload.get("error")
        if result_payload.get("data"):
            event_metadata["data"] = dict(result_payload["data"])
        extra_events.insert(0, (tool_event, message, event_metadata))
        self._finish_attempt(
            wf,
            task,
            target_status,
            event=task_event,
            message=message,
            result_summary=tool_result_summary,
            error=semantic_error or dataflow_error or tool_result.error,
            error_type=error_type,
            result_path=result_path,
            output_path=output_path,
            artifacts=artifacts,
            dataflow_error=dataflow_error,
            tool_finished_at=_now_iso(),
            tool_error=dataflow_error or tool_result.error,
            extra_events=extra_events,
        )
