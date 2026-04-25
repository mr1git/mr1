"""
Workflow scheduler — deterministic DAG driver.

The scheduler owns the lifecycle of every task in every workflow it
discovers under its `WorkflowStore` root. It is intentionally dumb:

  * no LLM reasoning,
  * no conditional branching (Phase 1),
  * no retry/backoff policies (Phase 1),
  * dependency rule is "all parents succeeded",
  * if a parent reaches a non-succeeded terminal status, all descendants
    transition to `blocked`.

The scheduler runs inside the MR1 process only (Phase 1). The CLI writes
a new workflow directory to disk and exits; the MR1-owned scheduler
discovers it on the next `tick()` and drives it to completion.

Concurrency cap is configurable via `concurrency=` (default 4).

Every state change follows the same atomic shape:
    store.locked():
        mutate task/workflow
        store.save_workflow(wf)
        event_log.emit(...)
"""

from __future__ import annotations

from dataclasses import replace
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.agents import AgentRegistry, default_agent_registry
from mr1 import workflow_events as ev
from mr1.dataflow import (
    DataflowError,
    TaskInputSpec,
    build_agent_task_output,
    build_materialized_prompt,
    build_tool_task_output,
    build_watcher_task_output,
    materialize_task_inputs,
    parse_input_reference,
    register_artifacts,
)
from mr1.kazi_runner import (
    MockRunner,
    RunHandle,
    RunResult,
    RunStatus,
    Runner,
)
from mr1.tools import (
    ToolConfigError,
    ToolRegistry,
    ToolResult,
    default_tool_registry,
)
from mr1.watchers import (
    WatcherConfigError,
    WatchEvaluation,
    WatcherRegistry,
    default_watcher_registry,
)
from mr1.workflow_events import WorkflowEventLog
from mr1.workflow_models import (
    FAILED_TASK_STATUSES,
    Provenance,
    Task,
    TaskAttempt,
    TaskStatus,
    Workflow,
    WorkflowStatus,
    new_task_id,
    new_workflow_id,
)
from mr1.workflow_store import WorkflowStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_UNSET = object()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class WorkflowSpecError(ValueError):
    """Raised for any malformed or unsupported workflow spec."""


class WatcherTriggerError(ValueError):
    """Raised when a manual watcher trigger request is invalid."""


def validate_spec(
    spec: dict[str, Any],
    watcher_registry: Optional[WatcherRegistry] = None,
    tool_registry: Optional[ToolRegistry] = None,
    agent_registry: Optional[AgentRegistry] = None,
) -> None:
    """
    Validate a user-submitted workflow spec without mutating it.

    Rules:
      * `tasks` is a non-empty list.
      * each task has a non-empty `label`.
      * labels are unique across the workflow.
      * each agent task has `agent_type == "kazi"`.
      * each watcher task has a registered `watcher_type` and valid config.
      * `depends_on` entries must reference labels defined in the same spec.
      * no cycles (topological sort must succeed).
    """
    registry = watcher_registry or default_watcher_registry()
    tools = tool_registry or default_tool_registry()
    agents = agent_registry or default_agent_registry()
    if not isinstance(spec, dict):
        raise WorkflowSpecError("workflow spec must be a JSON object")

    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise WorkflowSpecError("workflow spec must contain a non-empty 'tasks' list")

    labels: list[str] = []
    label_set: set[str] = set()
    for idx, raw in enumerate(tasks):
        if not isinstance(raw, dict):
            raise WorkflowSpecError(f"task[{idx}] must be a JSON object")
        label = raw.get("label")
        if not isinstance(label, str) or not label:
            raise WorkflowSpecError(f"task[{idx}] missing non-empty 'label'")
        if label in label_set:
            raise WorkflowSpecError(f"duplicate label '{label}' in workflow spec")
        labels.append(label)
        label_set.add(label)

        task_kind = raw.get("task_kind", "agent")
        if task_kind == "agent":
            agent_type = raw.get("agent_type", "kazi")
            if not agents.is_registered(agent_type):
                supported_agents = ", ".join(repr(name) for name in agents.list_agents()) or "'kazi'"
                raise WorkflowSpecError(
                    f"task '{label}': agent_type '{agent_type}' not supported (only {supported_agents})"
                )
        elif task_kind == "watcher":
            try:
                registry.validate_spec(
                    raw.get("watcher_type"),
                    raw.get("watch_config", {}),
                )
            except WatcherConfigError as exc:
                raise WorkflowSpecError(f"task '{label}': {exc}") from exc
        elif task_kind == "tool":
            try:
                tools.validate_spec(
                    raw.get("tool_type"),
                    raw.get("tool_config", {}),
                )
            except ToolConfigError as exc:
                raise WorkflowSpecError(f"task '{label}': {exc}") from exc
        else:
            raise WorkflowSpecError(
                f"task '{label}': task_kind '{task_kind}' not supported"
            )

    # depends_on reference check.
    depends_on_by_label: dict[str, list[str]] = {}
    for raw in tasks:
        deps = list(raw.get("depends_on", []) or [])
        depends_on_by_label[raw["label"]] = deps
        for dep in deps:
            if dep not in label_set:
                raise WorkflowSpecError(
                    f"task '{raw['label']}' depends_on unknown label '{dep}'"
                )

    # Cycle detection via Kahn's algorithm.
    indeg = {label: 0 for label in labels}
    graph: dict[str, list[str]] = {label: [] for label in labels}
    for raw in tasks:
        label = raw["label"]
        for dep in raw.get("depends_on", []) or []:
            graph[dep].append(label)
            indeg[label] += 1
    ready = [label for label, d in indeg.items() if d == 0]
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for nxt in graph[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if visited != len(labels):
        raise WorkflowSpecError("workflow spec contains a dependency cycle")

    ancestors = _compute_ancestor_labels(depends_on_by_label)
    for raw in tasks:
        raw_inputs = raw.get("inputs", [])
        if not isinstance(raw_inputs, list):
            raise WorkflowSpecError(f"task '{raw['label']}': inputs must be a list")
        for idx, item in enumerate(raw_inputs):
            if not isinstance(item, dict):
                raise WorkflowSpecError(
                    f"task '{raw['label']}': inputs[{idx}] must be a JSON object"
                )
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise WorkflowSpecError(
                    f"task '{raw['label']}': inputs[{idx}] missing non-empty 'name'"
                )
            from_ref = item.get("from")
            if not isinstance(from_ref, str) or not from_ref:
                raise WorkflowSpecError(
                    f"task '{raw['label']}': inputs[{idx}] missing non-empty 'from'"
                )
            try:
                parsed = parse_input_reference(from_ref)
            except DataflowError as exc:
                raise WorkflowSpecError(f"task '{raw['label']}': {exc}") from exc
            if parsed.label not in label_set:
                raise WorkflowSpecError(
                    f"task '{raw['label']}': input source label '{parsed.label}' is unknown"
                )
            if parsed.label not in ancestors.get(raw["label"], set()):
                raise WorkflowSpecError(
                    f"task '{raw['label']}': input source '{parsed.label}' must be an upstream dependency"
                )


# ---------------------------------------------------------------------------
# Public: write a workflow to disk without running a scheduler
# ---------------------------------------------------------------------------


def build_workflow_from_spec(
    spec: dict[str, Any],
    created_by: Provenance,
    watcher_registry: Optional[WatcherRegistry] = None,
    tool_registry: Optional[ToolRegistry] = None,
    agent_registry: Optional[AgentRegistry] = None,
) -> Workflow:
    """
    Convert a validated spec into a `Workflow` with generated IDs and
    resolved dependencies. Does not touch disk.
    """
    validate_spec(
        spec,
        watcher_registry=watcher_registry,
        tool_registry=tool_registry,
        agent_registry=agent_registry,
    )

    workflow_id = new_workflow_id()
    title = spec.get("title") or f"workflow {workflow_id}"
    wf = Workflow(
        workflow_id=workflow_id,
        title=title,
        status=WorkflowStatus.PENDING,
        created_by=created_by,
    )

    # First pass: create tasks and the label→id map.
    raw_tasks = spec["tasks"]
    label_to_task_id: dict[str, str] = {}
    task_objs: list[tuple[dict[str, Any], Task]] = []
    for raw in raw_tasks:
        tid = new_task_id()
        label_to_task_id[raw["label"]] = tid
        task = Task(
            task_id=tid,
            workflow_id=workflow_id,
            label=raw["label"],
            title=raw.get("title", raw["label"]),
            task_kind=raw.get("task_kind", "agent"),
            agent_type=raw.get("agent_type", "kazi")
            if raw.get("task_kind", "agent") == "agent" else None,
            prompt=raw.get("prompt", "")
            if raw.get("task_kind", "agent") == "agent" else "",
            watcher_type=raw.get("watcher_type"),
            watch_config=dict(raw.get("watch_config", {})),
            tool_type=raw.get("tool_type"),
            tool_config=dict(raw.get("tool_config", {})),
            condition=dict(raw["condition"]) if raw.get("condition") is not None else None,
            status=TaskStatus.CREATED,
            created_by=created_by,
            timeout_s=raw.get("timeout_s"),
            inputs=[
                TaskInputSpec.from_dict(item)
                for item in (raw.get("inputs") or [])
            ],
        )
        task_objs.append((raw, task))

    # Second pass: resolve depends_on labels to task IDs.
    for raw, task in task_objs:
        task.depends_on = [
            label_to_task_id[label]
            for label in (raw.get("depends_on") or [])
        ]
        wf.tasks[task.task_id] = task

    wf.label_to_task_id = label_to_task_id
    return wf


def submit_spec_to_disk(
    spec: dict[str, Any],
    created_by: Provenance,
    store: WorkflowStore,
    watcher_registry: Optional[WatcherRegistry] = None,
    tool_registry: Optional[ToolRegistry] = None,
    agent_registry: Optional[AgentRegistry] = None,
) -> str:
    """
    Persist a submitted workflow so the MR1-owned scheduler will pick
    it up on its next tick. Returns the new workflow ID.

    Called by both `Scheduler.submit_workflow` (in-process) and the
    standalone `workflow_cli submit` command.
    """
    wf = build_workflow_from_spec(
        spec,
        created_by,
        watcher_registry=watcher_registry,
        tool_registry=tool_registry,
        agent_registry=agent_registry,
    )
    log = WorkflowEventLog(store, default_agent_id=created_by.id)
    with store.locked():
        store.save_workflow(wf)
        log.workflow_submitted(
            wf.workflow_id,
            agent_id=created_by.id,
            message=f"submitted '{wf.title}' with {len(wf.tasks)} task(s)",
            metadata={"task_count": len(wf.tasks)},
        )
    return wf.workflow_id


def trigger_watcher_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    label_or_task_id: str,
    *,
    event_name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    agent_id: str = "scheduler",
) -> str:
    """
    Atomically trigger a persisted manual_event watcher without needing
    a live Scheduler instance.
    """
    event_log = WorkflowEventLog(store, default_agent_id=agent_id)
    with store.locked():
        wf = store.load_workflow(workflow_id)
        if wf is None:
            raise WatcherTriggerError(f"workflow not found: {workflow_id}")

        task = wf.tasks.get(label_or_task_id) or wf.task_by_label(label_or_task_id)
        if task is None:
            raise WatcherTriggerError(
                f"watcher not found in workflow {workflow_id}: {label_or_task_id}"
            )
        if task.task_kind != "watcher":
            raise WatcherTriggerError(f"task is not a watcher: {label_or_task_id}")
        if task.watcher_type != "manual_event":
            raise WatcherTriggerError(
                f"watcher is not manual_event: {task.label or task.task_id}"
            )
        if task.is_terminal():
            raise WatcherTriggerError(
                f"watcher already terminal: {task.label or task.task_id}"
            )

        expected_event = task.watch_config.get("event")
        if event_name and expected_event and event_name != expected_event:
            raise WatcherTriggerError(
                f"manual event mismatch: expected '{expected_event}', got '{event_name}'"
            )

        now_iso = _now_iso()
        trigger_metadata = dict(metadata or {})
        task.condition = {
            "triggered": True,
            "event": expected_event,
            "triggered_at": now_iso,
            "metadata": trigger_metadata,
        }
        task.status = TaskStatus.SUCCEEDED
        task.finished_at = now_iso
        task.watch_satisfied_at = now_iso
        task.last_checked_at = now_iso
        task.last_check_result = {
            "state": "satisfied",
            "message": f"manual event triggered: {expected_event}",
            "watcher_type": task.watcher_type,
            "event": expected_event,
            "trigger_metadata": trigger_metadata,
        }
        output = build_watcher_task_output(task)
        output_path = store.write_task_output(
            wf.workflow_id,
            task.task_id,
            output,
        )
        task.output_path = str(output_path)
        store.save_workflow(wf)
        event_log.emit(
            ev.WATCHER_SATISFIED,
            wf.workflow_id,
            task_id=task.task_id,
            agent_id=agent_id,
            message=task.last_check_result["message"],
            metadata=dict(task.last_check_result),
        )
        event_log.emit(
            ev.TASK_SUCCEEDED,
            wf.workflow_id,
            task_id=task.task_id,
            agent_id=agent_id,
            message=task.last_check_result["message"],
            metadata={"status": TaskStatus.SUCCEEDED.value, "watcher_type": task.watcher_type},
        )
        event_log.emit(
            ev.OUTPUT_WRITTEN,
            wf.workflow_id,
            task_id=task.task_id,
            agent_id=agent_id,
            message="normalized output written",
            metadata={"path": str(output_path)},
        )
        return task.task_id


def _label_for_task_id(workflow: Workflow, task_id: str) -> Optional[str]:
    for label, candidate in workflow.label_to_task_id.items():
        if candidate == task_id:
            return label
    return None


def _task_for_label_or_id(workflow: Workflow, label_or_task_id: str) -> Optional[Task]:
    return workflow.tasks.get(label_or_task_id) or workflow.task_by_label(label_or_task_id)


def _task_spec_for_workflow(workflow: Workflow, task: Task) -> dict[str, Any]:
    dep_labels = [
        _label_for_task_id(workflow, parent_id) or parent_id
        for parent_id in task.depends_on
    ]
    task_spec: dict[str, Any] = {
        "label": task.label,
        "title": task.title,
        "task_kind": task.task_kind,
    }
    if dep_labels:
        task_spec["depends_on"] = dep_labels
    if task.inputs:
        task_spec["inputs"] = [item.to_dict() for item in task.inputs]
    if task.timeout_s is not None:
        task_spec["timeout_s"] = task.timeout_s
    if task.task_kind == "agent":
        task_spec["agent_type"] = task.agent_type or "kazi"
        task_spec["prompt"] = task.prompt
    elif task.task_kind == "tool":
        task_spec["tool_type"] = task.tool_type
        task_spec["tool_config"] = dict(task.tool_config)
    elif task.task_kind == "watcher":
        task_spec["watcher_type"] = task.watcher_type
        task_spec["watch_config"] = dict(task.watch_config)
        if task.condition is not None:
            task_spec["condition"] = dict(task.condition)
    return task_spec


def _workflow_to_spec(workflow: Workflow) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for label, task_id in workflow.label_to_task_id.items():
        task = workflow.tasks.get(task_id)
        if task is None:
            continue
        tasks.append(_task_spec_for_workflow(workflow, task))
    return {
        "title": workflow.title,
        "tasks": tasks,
    }


def _reopen_workflow(workflow: Workflow) -> None:
    workflow.finished_at = None
    if any(task.status is TaskStatus.RUNNING for task in workflow.tasks.values()):
        workflow.status = WorkflowStatus.RUNNING
    else:
        workflow.status = WorkflowStatus.PENDING


def _attempt_result_payload(
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


def _set_task_terminal_attempt(
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
        result_payload or _attempt_result_payload(
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


def rerun_task_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    label_or_task_id: str,
    *,
    agent_id: str = "scheduler",
) -> str:
    event_log = WorkflowEventLog(store, default_agent_id=agent_id)
    with store.locked():
        workflow = store.load_workflow(workflow_id)
        if workflow is None:
            raise WorkflowSpecError(f"workflow not found: {workflow_id}")
        task = _task_for_label_or_id(workflow, label_or_task_id)
        if task is None:
            raise WorkflowSpecError(
                f"task not found in workflow {workflow_id}: {label_or_task_id}"
            )
        if task.status not in {
            TaskStatus.FAILED,
            TaskStatus.TIMED_OUT,
            TaskStatus.CANCELLED,
            TaskStatus.SUCCEEDED,
        }:
            raise WorkflowSpecError(
                f"task cannot be rerun from status '{task.status.value}': {task.label}"
            )
        parents_ok = all(
            (parent := workflow.tasks.get(parent_id)) is not None
            and parent.status is TaskStatus.SUCCEEDED
            for parent_id in task.depends_on
        )
        task.status = TaskStatus.READY if parents_ok else TaskStatus.WAITING
        task.started_at = None
        task.finished_at = None
        task.pid = None
        task.exit_code = None
        task.last_error = None
        task.last_error_type = None
        task.result_summary = None
        task.log_stdout_path = None
        task.log_stderr_path = None
        task.result_path = None
        task.dataflow_error = None
        task.blocked_by = []
        task.blocked_reason = None
        task.blocked_at = None
        task.watch_started_at = None
        task.watch_satisfied_at = None
        task.last_checked_at = None
        task.last_check_result = None
        task.tool_started_at = None
        task.tool_finished_at = None
        task.tool_error = None
        _reopen_workflow(workflow)
        store.save_workflow(workflow)
        event_log.task_rerun(
            workflow.workflow_id,
            task.task_id,
            agent_id=agent_id,
            message=f"task rerun requested: {task.label}",
            metadata={"status": task.status.value},
        )
    return task.task_id


def cancel_task_on_disk(
    store: WorkflowStore,
    task_id: str,
    *,
    agent_id: str = "scheduler",
) -> str:
    event_log = WorkflowEventLog(store, default_agent_id=agent_id)
    with store.locked():
        workflow: Optional[Workflow] = None
        task: Optional[Task] = None
        for candidate in store.list_workflows():
            maybe = candidate.tasks.get(task_id)
            if maybe is not None:
                workflow = candidate
                task = maybe
                break
        if workflow is None or task is None:
            raise WorkflowSpecError(f"task not found: {task_id}")
        if task.is_terminal() and task.status is not TaskStatus.BLOCKED:
            raise WorkflowSpecError(
                f"task already terminal: {task.label or task.task_id}"
            )
        original_status = task.status
        task.status = TaskStatus.CANCELLED
        task.last_error = "task cancelled"
        task.last_error_type = "cancelled"
        task.blocked_by = []
        task.blocked_reason = None
        task.blocked_at = None
        if original_status is TaskStatus.RUNNING and task.current_attempt > 0:
            _set_task_terminal_attempt(
                store,
                workflow,
                task,
                status=TaskStatus.CANCELLED,
                exit_code=None,
                error="task cancelled",
                error_type="cancelled",
                result_summary="task cancelled",
            )
        else:
            task.started_at = None
            task.finished_at = _now_iso()
            task.pid = None
            task.exit_code = None
            task.log_stdout_path = None
            task.log_stderr_path = None
            task.result_path = None
        _reopen_workflow(workflow)
        store.save_workflow(workflow)
        event_log.task_cancelled(
            workflow.workflow_id,
            task.task_id,
            agent_id=agent_id,
            attempt_id=task.current_attempt or None,
            message="task cancelled",
            metadata={"status": TaskStatus.CANCELLED.value},
        )
        if task.current_attempt > 0:
            event_log.task_attempt_finished(
                workflow.workflow_id,
                task.task_id,
                agent_id=agent_id,
                attempt_id=task.current_attempt,
                message="task attempt cancelled",
                metadata={
                    "status": TaskStatus.CANCELLED.value,
                    "error_type": "cancelled",
                },
            )
    return task.task_id


def cancel_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    *,
    agent_id: str = "scheduler",
) -> str:
    event_log = WorkflowEventLog(store, default_agent_id=agent_id)
    with store.locked():
        workflow = store.load_workflow(workflow_id)
        if workflow is None:
            raise WorkflowSpecError(f"workflow not found: {workflow_id}")
        for task in workflow.tasks.values():
            if task.is_terminal():
                continue
            original_status = task.status
            task.status = TaskStatus.CANCELLED
            task.last_error = "workflow cancelled"
            task.last_error_type = "cancelled"
            task.blocked_by = []
            task.blocked_reason = None
            task.blocked_at = None
            if original_status is TaskStatus.RUNNING and task.current_attempt > 0:
                _set_task_terminal_attempt(
                    store,
                    workflow,
                    task,
                    status=TaskStatus.CANCELLED,
                    exit_code=None,
                    error="workflow cancelled",
                    error_type="cancelled",
                    result_summary="workflow cancelled",
                )
            else:
                task.started_at = None
                task.finished_at = _now_iso()
                task.pid = None
                task.exit_code = None
                task.log_stdout_path = None
                task.log_stderr_path = None
                task.result_path = None
            event_log.task_cancelled(
                workflow.workflow_id,
                task.task_id,
                agent_id=agent_id,
                attempt_id=task.current_attempt or None,
                message="workflow cancelled",
                metadata={"status": TaskStatus.CANCELLED.value},
            )
            if task.current_attempt > 0:
                event_log.task_attempt_finished(
                    workflow.workflow_id,
                    task.task_id,
                    agent_id=agent_id,
                    attempt_id=task.current_attempt,
                    message="task attempt cancelled",
                    metadata={
                        "status": TaskStatus.CANCELLED.value,
                        "error_type": "cancelled",
                    },
                )
        workflow.status = WorkflowStatus.CANCELLED
        workflow.finished_at = _now_iso()
        store.save_workflow(workflow)
        event_log.workflow_cancelled(
            workflow.workflow_id,
            agent_id=agent_id,
            message="workflow cancelled",
        )
    return workflow.workflow_id


def _new_task_from_spec(
    workflow: Workflow,
    raw: dict[str, Any],
    *,
    created_by: Provenance,
) -> Task:
    return Task(
        task_id=new_task_id(),
        workflow_id=workflow.workflow_id,
        label=raw["label"],
        title=raw.get("title", raw["label"]),
        task_kind=raw.get("task_kind", "agent"),
        agent_type=raw.get("agent_type", "kazi")
        if raw.get("task_kind", "agent") == "agent" else None,
        prompt=raw.get("prompt", "")
        if raw.get("task_kind", "agent") == "agent" else "",
        watcher_type=raw.get("watcher_type"),
        watch_config=dict(raw.get("watch_config", {})),
        tool_type=raw.get("tool_type"),
        tool_config=dict(raw.get("tool_config", {})),
        condition=dict(raw["condition"]) if raw.get("condition") is not None else None,
        created_by=created_by,
        timeout_s=raw.get("timeout_s"),
        inputs=[
            TaskInputSpec.from_dict(item)
            for item in (raw.get("inputs") or [])
        ],
        status=TaskStatus.CREATED,
    )


def _require_fragment_tasks(spec_fragment: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = spec_fragment.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise WorkflowSpecError("workflow fragment must contain a non-empty 'tasks' list")
    return list(tasks)


def append_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    spec_fragment: dict[str, Any],
    *,
    agent_id: str = "scheduler",
) -> str:
    event_log = WorkflowEventLog(store, default_agent_id=agent_id)
    created_by = Provenance(type="user", id=agent_id)
    with store.locked():
        workflow = store.load_workflow(workflow_id)
        if workflow is None:
            raise WorkflowSpecError(f"workflow not found: {workflow_id}")
        new_tasks = _require_fragment_tasks(spec_fragment)
        merged = _workflow_to_spec(workflow)
        merged["tasks"].extend(new_tasks)
        validate_spec(merged)
        for raw in new_tasks:
            if raw["label"] in workflow.label_to_task_id:
                raise WorkflowSpecError(f"duplicate label '{raw['label']}' in workflow")
        added: list[str] = []
        for raw in new_tasks:
            task = _new_task_from_spec(workflow, raw, created_by=created_by)
            task.depends_on = [
                workflow.label_to_task_id[label]
                for label in (raw.get("depends_on") or [])
            ]
            workflow.tasks[task.task_id] = task
            workflow.label_to_task_id[task.label] = task.task_id
            added.append(task.task_id)
        _reopen_workflow(workflow)
        store.save_workflow(workflow)
        event_log.workflow_updated(
            workflow.workflow_id,
            agent_id=agent_id,
            message=f"workflow appended with {len(added)} task(s)",
            metadata={"operation": "append", "task_ids": added},
        )
    return workflow.workflow_id


def insert_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    after_task: str,
    spec_fragment: dict[str, Any],
    *,
    agent_id: str = "scheduler",
) -> str:
    event_log = WorkflowEventLog(store, default_agent_id=agent_id)
    created_by = Provenance(type="user", id=agent_id)
    with store.locked():
        workflow = store.load_workflow(workflow_id)
        if workflow is None:
            raise WorkflowSpecError(f"workflow not found: {workflow_id}")
        anchor = _task_for_label_or_id(workflow, after_task)
        if anchor is None:
            raise WorkflowSpecError(f"task not found in workflow {workflow_id}: {after_task}")
        fragment_tasks = _require_fragment_tasks(spec_fragment)
        if len(fragment_tasks) != 1:
            raise WorkflowSpecError("insert-workflow requires exactly one task in 'tasks'")
        raw = dict(fragment_tasks[0])
        if raw.get("depends_on"):
            raise WorkflowSpecError("insert-workflow task must not declare depends_on")
        direct_children = [
            task for task in workflow.tasks.values()
            if anchor.task_id in task.depends_on
        ]
        for child in direct_children:
            if child.status is TaskStatus.RUNNING:
                raise WorkflowSpecError(f"cannot mutate running task: {child.label}")
            if child.status is TaskStatus.SUCCEEDED:
                raise WorkflowSpecError(f"cannot mutate succeeded task: {child.label}")
        if raw["label"] in workflow.label_to_task_id:
            raise WorkflowSpecError(f"duplicate label '{raw['label']}' in workflow")
        merged = _workflow_to_spec(workflow)
        merged["tasks"].append({
            **raw,
            "depends_on": [_label_for_task_id(workflow, anchor.task_id) or anchor.label],
        })
        validate_spec(merged)
        inserted = _new_task_from_spec(workflow, raw, created_by=created_by)
        inserted.depends_on = [anchor.task_id]
        workflow.tasks[inserted.task_id] = inserted
        workflow.label_to_task_id[inserted.label] = inserted.task_id
        for child in direct_children:
            child.depends_on = [
                inserted.task_id if dep == anchor.task_id else dep
                for dep in child.depends_on
            ]
            if child.status not in {
                TaskStatus.FAILED,
                TaskStatus.TIMED_OUT,
                TaskStatus.CANCELLED,
            }:
                child.status = TaskStatus.WAITING
                child.finished_at = None
                child.blocked_by = []
                child.blocked_reason = None
                child.blocked_at = None
        _reopen_workflow(workflow)
        store.save_workflow(workflow)
        event_log.workflow_updated(
            workflow.workflow_id,
            agent_id=agent_id,
            message=f"workflow inserted task '{inserted.label}'",
            metadata={
                "operation": "insert",
                "task_ids": [inserted.task_id],
                "after_task_id": anchor.task_id,
            },
        )
    return workflow.workflow_id


def replace_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    target_task: str,
    spec_fragment: dict[str, Any],
    *,
    agent_id: str = "scheduler",
) -> str:
    event_log = WorkflowEventLog(store, default_agent_id=agent_id)
    with store.locked():
        workflow = store.load_workflow(workflow_id)
        if workflow is None:
            raise WorkflowSpecError(f"workflow not found: {workflow_id}")
        task = _task_for_label_or_id(workflow, target_task)
        if task is None:
            raise WorkflowSpecError(f"task not found in workflow {workflow_id}: {target_task}")
        if task.status is TaskStatus.RUNNING:
            raise WorkflowSpecError(f"cannot replace running task: {task.label}")
        if task.status is TaskStatus.SUCCEEDED:
            raise WorkflowSpecError(f"cannot replace succeeded task: {task.label}")
        if task.attempt_count > 0 and task.status not in {
            TaskStatus.FAILED,
            TaskStatus.TIMED_OUT,
            TaskStatus.CANCELLED,
        }:
            raise WorkflowSpecError(
                f"replace-workflow allowed only for unstarted or failed/cancelled tasks: {task.label}"
            )
        fragment_tasks = _require_fragment_tasks(spec_fragment)
        if len(fragment_tasks) != 1:
            raise WorkflowSpecError("replace-workflow requires exactly one task in 'tasks'")
        raw = dict(fragment_tasks[0])
        if raw.get("label") != task.label:
            raise WorkflowSpecError(
                f"replace-workflow task label must match target label '{task.label}'"
            )
        merged = _workflow_to_spec(workflow)
        for idx, item in enumerate(merged["tasks"]):
            if item["label"] == task.label:
                merged["tasks"][idx] = raw
                break
        validate_spec(merged)
        task.title = raw.get("title", raw["label"])
        task.task_kind = raw.get("task_kind", "agent")
        task.agent_type = raw.get("agent_type", "kazi") if task.task_kind == "agent" else None
        task.prompt = raw.get("prompt", "") if task.task_kind == "agent" else ""
        task.watcher_type = raw.get("watcher_type")
        task.watch_config = dict(raw.get("watch_config", {}))
        task.tool_type = raw.get("tool_type")
        task.tool_config = dict(raw.get("tool_config", {}))
        task.condition = dict(raw["condition"]) if raw.get("condition") is not None else None
        task.timeout_s = raw.get("timeout_s")
        task.inputs = [
            TaskInputSpec.from_dict(item)
            for item in (raw.get("inputs") or [])
        ]
        task.depends_on = [
            workflow.label_to_task_id[label]
            for label in (raw.get("depends_on") or [])
        ]
        task.status = TaskStatus.CREATED
        task.started_at = None
        task.finished_at = None
        task.pid = None
        task.exit_code = None
        task.last_error = None
        task.last_error_type = None
        task.result_summary = None
        task.log_stdout_path = None
        task.log_stderr_path = None
        task.result_path = None
        task.dataflow_error = None
        task.blocked_by = []
        task.blocked_reason = None
        task.blocked_at = None
        task.watch_started_at = None
        task.watch_satisfied_at = None
        task.last_checked_at = None
        task.last_check_result = None
        task.tool_started_at = None
        task.tool_finished_at = None
        task.tool_error = None
        _reopen_workflow(workflow)
        store.save_workflow(workflow)
        event_log.workflow_updated(
            workflow.workflow_id,
            agent_id=agent_id,
            message=f"workflow replaced task '{task.label}'",
            metadata={
                "operation": "replace",
                "task_ids": [task.task_id],
            },
        )
    return workflow.workflow_id


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """
    DAG driver. One scheduler instance per MR1 process.

    The scheduler is event-driven in spirit (every state transition
    emits a `WorkflowEvent`), but it physically advances inside `tick()`.
    A daemon thread calls `tick()` on a configurable interval when
    `auto_tick=True`.
    """

    def __init__(
        self,
        store: WorkflowStore,
        runner: Optional[Runner] = None,
        *,
        event_log: Optional[WorkflowEventLog] = None,
        watcher_registry: Optional[WatcherRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        concurrency: int = 4,
        auto_tick: bool = True,
        tick_interval_s: float = 1.0,
        agent_id: str = "scheduler",
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._store = store
        self._runner = runner or MockRunner()
        self._events = event_log or WorkflowEventLog(store, default_agent_id=agent_id)
        self._watchers = watcher_registry or default_watcher_registry()
        self._tools = tool_registry or default_tool_registry()
        self._concurrency = concurrency
        self._tick_interval_s = tick_interval_s
        self._agent_id = agent_id

        self._handles: dict[str, RunHandle] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if auto_tick:
            self._thread = threading.Thread(
                target=self._run_loop,
                name="workflow-scheduler",
                daemon=True,
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, cancel_running: bool = False) -> None:
        """Stop the background thread and optionally cancel live runs."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._tick_interval_s * 3)
        if cancel_running:
            for handle in list(self._handles.values()):
                try:
                    self._runner.cancel(handle)
                except Exception:
                    pass
            self._handles.clear()

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                # A crashing tick must not kill the daemon thread; the
                # scheduler is a fail-soft control surface in Phase 1.
                pass
            self._stop.wait(self._tick_interval_s)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_workflow(
        self,
        spec: dict[str, Any],
        created_by: Provenance,
    ) -> str:
        """
        Validate a spec, persist the resulting workflow, and emit
        `workflow_submitted`. Returns the new workflow ID.
        """
        return submit_spec_to_disk(
            spec,
            created_by,
            self._store,
            watcher_registry=self._watchers,
            tool_registry=self._tools,
        )

    # ------------------------------------------------------------------
    # Read-through accessors
    # ------------------------------------------------------------------

    def list_workflows(self) -> list[Workflow]:
        return self._store.list_workflows()

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        return self._store.load_workflow(workflow_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        for wf in self._store.list_workflows():
            if task_id in wf.tasks:
                return wf.tasks[task_id]
        return None

    def list_watchers(self) -> list[tuple[Workflow, Task]]:
        watchers: list[tuple[Workflow, Task]] = []
        for wf in self._store.list_workflows():
            for task in wf.tasks.values():
                if task.task_kind != "watcher" or task.is_terminal():
                    continue
                watchers.append((wf, task))
        return watchers

    def trigger_watcher(
        self,
        workflow_id: str,
        label_or_task_id: str,
        event_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return trigger_watcher_on_disk(
            self._store,
            workflow_id,
            label_or_task_id,
            event_name=event_name,
            metadata=metadata,
            agent_id=self._agent_id,
        )

    def rerun_task(self, workflow_id: str, label_or_task_id: str) -> str:
        return rerun_task_on_disk(
            self._store,
            workflow_id,
            label_or_task_id,
            agent_id=self._agent_id,
        )

    def cancel_task(self, task_id: str) -> str:
        return cancel_task_on_disk(
            self._store,
            task_id,
            agent_id=self._agent_id,
        )

    def append_workflow(self, workflow_id: str, spec_fragment: dict[str, Any]) -> str:
        return append_workflow_on_disk(
            self._store,
            workflow_id,
            spec_fragment,
            agent_id=self._agent_id,
        )

    def insert_workflow(
        self,
        workflow_id: str,
        after_task: str,
        spec_fragment: dict[str, Any],
    ) -> str:
        return insert_workflow_on_disk(
            self._store,
            workflow_id,
            after_task,
            spec_fragment,
            agent_id=self._agent_id,
        )

    def replace_workflow(
        self,
        workflow_id: str,
        target_task: str,
        spec_fragment: dict[str, Any],
    ) -> str:
        return replace_workflow_on_disk(
            self._store,
            workflow_id,
            target_task,
            spec_fragment,
            agent_id=self._agent_id,
        )

    # ------------------------------------------------------------------
    # Tick — the deterministic advance
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """
        One deterministic pass over every on-disk workflow.

        Order per workflow:
          1. Poll running tasks and persist terminal transitions.
          2. Cascade failures → blocked.
          3. Promote waiting → ready when all parents succeeded.
          4. Promote created → waiting/ready on first sight.
          5. Launch ready tasks up to the concurrency cap.
          6. Finalise the workflow when every task is terminal.
        """
        for wf in self._store.list_workflows():
            self._reconcile_external_control(wf)
            if wf.is_terminal() and not self._workflow_has_live_handles(wf):
                continue
            self._tick_workflow(wf)

    def _workflow_has_live_handles(self, wf: Workflow) -> bool:
        return any(task.task_id in self._handles for task in wf.tasks.values())

    def _reconcile_external_control(self, wf: Workflow) -> None:
        for task in wf.tasks.values():
            handle = self._handles.get(task.task_id)
            if handle is None:
                continue
            if task.status is TaskStatus.RUNNING:
                continue
            try:
                self._runner.cancel(handle)
            except Exception:
                pass
            self._handles.pop(task.task_id, None)

    def _tick_workflow(self, wf: Workflow) -> None:
        changed = False

        # 0. Promote created → waiting/ready on first discovery.
        changed |= self._initialize_created_tasks(wf)

        # 1. Poll running tasks.
        changed |= self._poll_running_tasks(wf)

        # 2. Cascade failures to dependents.
        changed |= self._cascade_blocked(wf)

        # 3. Recover previously blocked tasks when dependencies recover.
        changed |= self._recover_blocked(wf)

        # 4. Promote waiting → ready.
        changed |= self._promote_ready(wf)

        # 5. Workflow-level transitions.
        changed |= self._transition_workflow_status(wf)

        # 6. Launch ready tasks (bounded by concurrency).
        self._launch_ready(wf)

        # Nothing persists here — every mutation above went through
        # `_commit()` which holds the store lock.

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------

    def _initialize_created_tasks(self, wf: Workflow) -> bool:
        """Move newly-seen `created` tasks to `waiting` or `ready`."""
        changed = False
        for task in wf.tasks.values():
            if task.status is not TaskStatus.CREATED:
                continue
            if not task.depends_on:
                self._commit(wf, task, TaskStatus.READY, event=ev.TASK_READY,
                             message="no dependencies; ready to run")
            else:
                self._commit(wf, task, TaskStatus.WAITING, event=ev.TASK_CREATED,
                             message=f"waiting on {len(task.depends_on)} dependency(ies)")
            changed = True
        return changed

    def _poll_running_tasks(self, wf: Workflow) -> bool:
        changed = False
        for task in list(wf.tasks.values()):
            if task.status is not TaskStatus.RUNNING:
                continue
            if task.task_kind == "watcher":
                changed |= self._poll_running_watcher(wf, task)
                continue
            handle = self._handles.get(task.task_id)
            if handle is None:
                # Lost the handle (e.g. scheduler restart mid-run). Mark
                # the task failed so downstream work doesn't stall.
                result_path = str(self._store.write_attempt_result(
                    wf.workflow_id,
                    task.task_id,
                    task.current_attempt,
                    _attempt_result_payload(
                        task,
                        status=TaskStatus.FAILED,
                        exit_code=None,
                        error="run handle lost (scheduler restart?)",
                        error_type="unknown",
                        result_summary="run handle lost",
                    ),
                ))
                self._finish_attempt(
                    wf, task, TaskStatus.FAILED, event=ev.TASK_FAILED,
                    message="run handle lost (scheduler restart?)",
                    exit_code=None,
                    error="run handle lost (scheduler restart?)",
                    error_type="unknown",
                    result_path=result_path,
                    result_summary="run handle lost",
                )
                changed = True
                continue
            result = self._runner.poll(handle)
            if result is None:
                continue
            changed |= self._finalize_task(wf, task, handle, result)
        return changed

    def _poll_running_watcher(self, wf: Workflow, task: Task) -> bool:
        timeout_message = self._watcher_timeout_message(task)
        if timeout_message is not None:
            checked_at = _now_iso()
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

        now = datetime.now(timezone.utc)
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
            task.watch_satisfied_at = _now_iso()
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
            self._finish_attempt(
                wf,
                task,
                TaskStatus.SUCCEEDED,
                event=ev.TASK_SUCCEEDED,
                message=evaluation.message,
                watch_satisfied_at=_now_iso(),
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

    def _should_evaluate_watcher(self, task: Task) -> bool:
        if not task.last_checked_at:
            return True
        interval_s = task.watch_config.get("poll_interval_s", 1)
        if not isinstance(interval_s, (int, float)) or interval_s < 0:
            interval_s = 1
        last_checked = datetime.fromisoformat(task.last_checked_at)
        return (datetime.now(timezone.utc) - last_checked).total_seconds() >= interval_s

    def _watcher_timeout_message(self, task: Task) -> Optional[str]:
        max_wait_s = task.watch_config.get("max_wait_s")
        if not isinstance(max_wait_s, (int, float)) or max_wait_s <= 0:
            return None
        started_at = task.watch_started_at or task.started_at
        if not started_at:
            return None
        started_dt = datetime.fromisoformat(started_at)
        elapsed_s = (datetime.now(timezone.utc) - started_dt).total_seconds()
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

    def _normalize_error_type(
        self,
        result: RunResult,
        target_status: TaskStatus,
    ) -> Optional[str]:
        if target_status is TaskStatus.SUCCEEDED:
            return None
        if result.status is RunStatus.TIMED_OUT:
            return "timeout"
        if result.error_type in {
            "auth_error",
            "cli_error",
            "timeout",
            "parse_error",
            "cancelled",
            "unknown",
        }:
            return result.error_type
        return "unknown"

    def _record_watcher_check(
        self,
        wf: Workflow,
        task: Task,
        *,
        checked_at: str,
        check_payload: dict[str, Any],
    ) -> None:
        with self._store.locked():
            task.last_checked_at = checked_at
            task.last_check_result = dict(check_payload)
            self._store.save_workflow(wf)
            self._events.emit(
                ev.WATCHER_CHECKED,
                wf.workflow_id,
                task_id=task.task_id,
                attempt_id=task.current_attempt or None,
                agent_id=self._agent_id,
                message=check_payload.get("message", ""),
                metadata=dict(check_payload),
            )

    def _finalize_task(
        self,
        wf: Workflow,
        task: Task,
        handle: RunHandle,
        result: RunResult,
    ) -> bool:
        self._handles.pop(task.task_id, None)

        status_map = {
            RunStatus.SUCCEEDED: (TaskStatus.SUCCEEDED, ev.TASK_SUCCEEDED),
            RunStatus.FAILED: (TaskStatus.FAILED, ev.TASK_FAILED),
            RunStatus.TIMED_OUT: (TaskStatus.TIMED_OUT, ev.TASK_TIMED_OUT),
        }
        target_status, event_type = status_map.get(
            result.status, (TaskStatus.FAILED, ev.TASK_FAILED),
        )
        error_type = self._normalize_error_type(result, target_status)

        result_path = str(self._store.write_attempt_result(
            wf.workflow_id,
            task.task_id,
            task.current_attempt,
            result.result_payload or {},
        ))

        extra_events: list[tuple[str, str, dict[str, Any]]] = []
        output_path: Optional[str] = None
        dataflow_error: Optional[str] = None
        try:
            task.artifacts = register_artifacts(
                task,
                self._store,
                (result.result_payload or {}).get("artifacts"),
            )
            for artifact in task.artifacts:
                extra_events.append((
                    ev.ARTIFACT_REGISTERED,
                    f"artifact registered: {artifact.name}",
                    {"name": artifact.name, "kind": artifact.kind, "path": artifact.path},
                ))
            if target_status is TaskStatus.SUCCEEDED:
                output = build_agent_task_output(
                    replace(
                        task,
                        status=target_status,
                        result_summary=(result.summary or "")[:500] if result.summary else None,
                        exit_code=result.exit_code,
                    ),
                    result.result_payload or {},
                )
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
        except DataflowError as exc:
            target_status = TaskStatus.FAILED
            event_type = ev.TASK_FAILED
            dataflow_error = str(exc)
            result = RunResult(
                status=RunStatus.FAILED,
                exit_code=result.exit_code if result.exit_code is not None else 1,
                summary=result.summary,
                error=str(exc),
                error_type=error_type,
                stdout_path=result.stdout_path,
                stderr_path=result.stderr_path,
                result_payload=result.result_payload,
            )
            error_type = "unknown"

        self._finish_attempt(
            wf,
            task,
            target_status,
            event=event_type,
            message=dataflow_error or result.error or result.summary or "",
            exit_code=result.exit_code,
            result_summary=(result.summary or "")[:500] if result.summary else None,
            error=dataflow_error or result.error,
            error_type=error_type,
            log_stdout_path=str(result.stdout_path) if result.stdout_path else None,
            log_stderr_path=str(result.stderr_path) if result.stderr_path else None,
            result_path=result_path,
            output_path=output_path,
            dataflow_error=dataflow_error,
            artifacts=task.artifacts,
            extra_events=extra_events,
        )
        return True

    def _cascade_blocked(self, wf: Workflow) -> bool:
        changed = False
        for task in list(wf.tasks.values()):
            if task.is_terminal() or task.status is TaskStatus.RUNNING:
                continue
            failed_parents = [
                parent_id for parent_id in task.depends_on
                if (p := wf.tasks.get(parent_id)) is not None
                and p.status in FAILED_TASK_STATUSES
            ]
            if not failed_parents:
                continue
            parent_statuses = [
                f"{wf.tasks[pid].label}={wf.tasks[pid].status.value}"
                for pid in failed_parents
            ]
            self._commit(
                wf, task, TaskStatus.BLOCKED, event=ev.TASK_BLOCKED,
                message=f"blocked by {', '.join(parent_statuses)}",
                finished=True,
                blocked_by=failed_parents,
                blocked_reason=", ".join(parent_statuses),
                blocked_at=_now_iso(),
            )
            changed = True
        return changed

    def _promote_ready(self, wf: Workflow) -> bool:
        changed = False
        for task in wf.tasks.values():
            if task.status is not TaskStatus.WAITING:
                continue
            parents_ok = all(
                (p := wf.tasks.get(pid)) is not None
                and p.status is TaskStatus.SUCCEEDED
                for pid in task.depends_on
            )
            if parents_ok:
                self._commit(
                    wf, task, TaskStatus.READY, event=ev.TASK_READY,
                    message="all dependencies succeeded",
                )
                changed = True
        return changed

    def _recover_blocked(self, wf: Workflow) -> bool:
        changed = False
        for task in wf.tasks.values():
            if task.status is not TaskStatus.BLOCKED:
                continue
            if task.last_error_type == "cancelled":
                continue
            failed_parents = [
                parent_id for parent_id in task.depends_on
                if (parent := wf.tasks.get(parent_id)) is not None
                and parent.status in FAILED_TASK_STATUSES
            ]
            if failed_parents:
                continue
            parents_ok = all(
                (parent := wf.tasks.get(parent_id)) is not None
                and parent.status is TaskStatus.SUCCEEDED
                for parent_id in task.depends_on
            )
            with self._store.locked():
                task.status = TaskStatus.READY if parents_ok else TaskStatus.WAITING
                task.finished_at = None
                task.blocked_by = []
                task.blocked_reason = None
                task.blocked_at = None
                self._store.save_workflow(wf)
                self._events.task_unblocked(
                    wf.workflow_id,
                    task.task_id,
                    agent_id=self._agent_id,
                    message="dependencies recovered",
                    metadata={"status": task.status.value},
                )
            changed = True
        return changed

    def _transition_workflow_status(self, wf: Workflow) -> bool:
        """Flip `pending → running → succeeded/failed` based on tasks."""
        all_terminal = all(t.is_terminal() for t in wf.tasks.values())
        any_running_or_live = any(
            t.status in (TaskStatus.RUNNING, TaskStatus.READY, TaskStatus.WAITING)
            for t in wf.tasks.values()
        )

        target: Optional[WorkflowStatus] = None
        event_type: Optional[str] = None
        message = ""

        if wf.status is WorkflowStatus.PENDING and any_running_or_live:
            target = WorkflowStatus.RUNNING

        if all_terminal:
            any_failed = any(
                t.status in FAILED_TASK_STATUSES
                for t in wf.tasks.values()
            )
            if any_failed:
                target = WorkflowStatus.FAILED
                event_type = ev.WORKFLOW_FAILED
                message = "one or more tasks did not succeed"
            else:
                target = WorkflowStatus.SUCCEEDED
                event_type = ev.WORKFLOW_SUCCEEDED
                message = "all tasks succeeded"

        if target is None or target is wf.status:
            return False

        with self._store.locked():
            wf.status = target
            if target in (WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED):
                wf.finished_at = _now_iso()
            self._store.save_workflow(wf)
            if event_type is not None:
                self._events.emit(
                    event_type, wf.workflow_id,
                    agent_id=self._agent_id, message=message,
                )
        return True

    def _begin_attempt(
        self,
        wf: Workflow,
        task: Task,
        *,
        message: str,
        pid: Optional[int] = None,
        watch_started: bool = False,
        tool_started: bool = False,
        extra_events: Optional[list[tuple[str, str, dict[str, Any]]]] = None,
    ) -> int:
        attempt_id = task.current_attempt or (task.attempt_count + 1)
        with self._store.locked():
            stdout_path, stderr_path = self._store.task_attempt_log_paths(
                wf.workflow_id,
                task.task_id,
                attempt_id,
            )
            now_iso = _now_iso()
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
            ))
            self._store.save_workflow(wf)
            self._events.task_attempt_started(
                wf.workflow_id,
                task.task_id,
                agent_id=self._agent_id,
                attempt_id=attempt_id,
                message="task attempt started",
                metadata={"status": TaskStatus.RUNNING.value},
            )
            self._events.task_started(
                wf.workflow_id,
                task.task_id,
                agent_id=self._agent_id,
                attempt_id=attempt_id,
                message=message,
                metadata={"status": TaskStatus.RUNNING.value, "pid": pid} if pid is not None else {"status": TaskStatus.RUNNING.value},
            )
            for event_type, event_message, event_metadata in extra_events or []:
                self._events.emit(
                    event_type,
                    wf.workflow_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    agent_id=self._agent_id,
                    message=event_message,
                    metadata=dict(event_metadata),
                )
        return attempt_id

    def _finish_attempt(
        self,
        wf: Workflow,
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
        dataflow_error: Any = _UNSET,
        blocked_by: Optional[list[str]] = None,
        blocked_reason: Optional[str] = None,
        blocked_at: Optional[str] = None,
        watch_satisfied_at: Any = _UNSET,
        last_checked_at: Any = _UNSET,
        last_check_result: Any = _UNSET,
        condition: Any = _UNSET,
        tool_finished_at: Any = _UNSET,
        tool_error: Any = _UNSET,
        extra_events: Optional[list[tuple[str, str, dict[str, Any]]]] = None,
    ) -> None:
        attempt_id = task.current_attempt or None
        with self._store.locked():
            now_iso = _now_iso()
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
            if dataflow_error is not _UNSET:
                task.dataflow_error = dataflow_error
            if blocked_by is not None:
                task.blocked_by = list(blocked_by)
            if blocked_reason is not None:
                task.blocked_reason = blocked_reason
            if blocked_at is not None:
                task.blocked_at = blocked_at
            if watch_satisfied_at is not _UNSET:
                task.watch_satisfied_at = watch_satisfied_at
            if last_checked_at is not _UNSET:
                task.last_checked_at = last_checked_at
            if last_check_result is not _UNSET:
                task.last_check_result = (
                    dict(last_check_result)
                    if last_check_result is not None else None
                )
            if condition is not _UNSET:
                task.condition = dict(condition) if condition is not None else None
            if tool_finished_at is not _UNSET:
                task.tool_finished_at = tool_finished_at
            if tool_error is not _UNSET:
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
            self._store.save_workflow(wf)
            if attempt_id is not None:
                self._events.task_attempt_finished(
                    wf.workflow_id,
                    task.task_id,
                    agent_id=self._agent_id,
                    attempt_id=attempt_id,
                    message="task attempt finished",
                    metadata={
                        "status": new_status.value,
                        "error_type": error_type,
                    },
                )
            self._events.emit(
                event,
                wf.workflow_id,
                task_id=task.task_id,
                attempt_id=attempt_id,
                agent_id=self._agent_id,
                message=message,
                metadata={"status": new_status.value},
            )
            for event_type, event_message, event_metadata in extra_events or []:
                self._events.emit(
                    event_type,
                    wf.workflow_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    agent_id=self._agent_id,
                    message=event_message,
                    metadata=dict(event_metadata),
                )

    def _launch_ready(self, wf: Workflow) -> None:
        running_now = sum(
            1
            for t in wf.tasks.values()
            if t.status is TaskStatus.RUNNING and t.task_kind == "agent"
        )
        slots = self._concurrency - running_now
        for task in wf.tasks.values():
            if task.status is not TaskStatus.READY:
                continue
            if task.task_kind == "watcher":
                task.current_attempt = task.attempt_count + 1
                self._begin_attempt(
                    wf,
                    task,
                    message="watcher started",
                    watch_started=True,
                    extra_events=[(
                        ev.WATCHER_STARTED,
                        "watcher started",
                        {
                            "watcher_type": task.watcher_type,
                            "status": TaskStatus.RUNNING.value,
                        },
                    )],
                )
                continue
            if task.task_kind == "tool":
                task.current_attempt = task.attempt_count + 1
                self._begin_attempt(
                    wf,
                    task,
                    message="tool started",
                    tool_started=True,
                    extra_events=[(
                        ev.TOOL_STARTED,
                        "tool started",
                        {
                            "tool_type": task.tool_type,
                            "status": TaskStatus.RUNNING.value,
                        },
                    )],
                )
                self._run_tool_task(wf, task)
                continue
            if slots <= 0:
                break
            if task.inputs:
                if self._materialize_inputs_for_task(wf, task):
                    continue
            previous_attempt = task.current_attempt
            task.current_attempt = task.attempt_count + 1
            try:
                launch_task = self._build_launch_task(task)
                handle = self._runner.start(launch_task)
            except Exception as exc:
                task.current_attempt = previous_attempt
                self._commit(
                    wf, task, TaskStatus.FAILED, event=ev.TASK_FAILED,
                    message=f"runner.start failed: {exc}",
                    finished=True,
                    exit_code=None,
                    last_error=str(exc),
                    last_error_type="unknown",
                    result_summary=str(exc),
                )
                continue
            self._handles[task.task_id] = handle
            self._begin_attempt(
                wf,
                task,
                message="task started",
                pid=handle.pid,
            )
            slots -= 1

    def _build_launch_task(self, task: Task) -> Task:
        prompt = task.prompt
        if task.materialized_prompt_path:
            materialized_path = Path(task.materialized_prompt_path)
            if materialized_path.exists():
                prompt = materialized_path.read_text(encoding="utf-8")
        return replace(task, prompt=prompt)

    def _materialize_inputs_for_task(self, wf: Workflow, task: Task) -> bool:
        resolved_inputs = materialize_task_inputs(wf, task, self._store)
        inputs_path = self._store.write_task_inputs(
            wf.workflow_id,
            task.task_id,
            resolved_inputs,
        )
        materialized_prompt = build_materialized_prompt(task.prompt, resolved_inputs)
        prompt_path = self._store.write_materialized_prompt(
            wf.workflow_id,
            task.task_id,
            materialized_prompt,
        )
        missing = [item for item in resolved_inputs if item.resolved_type == "missing"]
        extra_events = [(
            ev.INPUT_MATERIALIZED,
            f"materialized {len(resolved_inputs)} workflow input(s)",
            {
                "count": len(resolved_inputs),
                "inputs_path": str(inputs_path),
                "materialized_prompt_path": str(prompt_path),
            },
        )]
        if missing:
            missing_sources = [item.source for item in missing]
            self._commit(
                wf,
                task,
                TaskStatus.FAILED,
                event=ev.TASK_FAILED,
                message=f"failed to resolve workflow input(s): {', '.join(missing_sources)}",
                finished=True,
                last_error=f"failed to resolve workflow input(s): {', '.join(missing_sources)}",
                last_error_type="unknown",
                inputs_path=str(inputs_path),
                materialized_prompt_path=str(prompt_path),
                dataflow_error=f"failed to resolve workflow input(s): {', '.join(missing_sources)}",
                extra_events=extra_events + [(
                    ev.INPUT_RESOLUTION_FAILED,
                    f"failed to resolve workflow input(s): {', '.join(missing_sources)}",
                    {"sources": missing_sources},
                )],
            )
            return True
        with self._store.locked():
            task.inputs_path = str(inputs_path)
            task.materialized_prompt_path = str(prompt_path)
            task.dataflow_error = None
            self._store.save_workflow(wf)
            for event_type, event_message, event_metadata in extra_events:
                self._events.emit(
                    event_type,
                    wf.workflow_id,
                    task_id=task.task_id,
                    agent_id=self._agent_id,
                    message=event_message,
                    metadata=dict(event_metadata),
                )
        return False

    def _run_tool_task(self, wf: Workflow, task: Task) -> None:
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

        if target_status is TaskStatus.FAILED and error_type is None:
            error_type = "unknown"
        result_path = str(self._store.write_attempt_result(
            wf.workflow_id,
            task.task_id,
            task.current_attempt,
            {
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
            },
        ))
        message = dataflow_error or tool_result.error or tool_result.summary or ""
        event_metadata = {
            "tool_type": task.tool_type,
            "state": target_status.value,
        }
        if tool_result.error:
            event_metadata["error"] = tool_result.error
        if tool_result.data:
            event_metadata["data"] = dict(tool_result.data)
        extra_events.insert(0, (tool_event, message, event_metadata))
        self._finish_attempt(
            wf,
            task,
            target_status,
            event=task_event,
            message=message,
            result_summary=tool_result.summary,
            error=dataflow_error or tool_result.error,
            error_type=error_type,
            result_path=result_path,
            output_path=output_path,
            artifacts=artifacts,
            dataflow_error=dataflow_error,
            tool_finished_at=_now_iso(),
            tool_error=dataflow_error or tool_result.error,
            extra_events=extra_events,
        )

    # ------------------------------------------------------------------
    # Atomic commit helper
    # ------------------------------------------------------------------

    def _commit(
        self,
        wf: Workflow,
        task: Task,
        new_status: TaskStatus,
        *,
        event: str,
        message: str = "",
        started: bool = False,
        finished: bool = False,
        attempt_id: Optional[int] = None,
        pid: Any = _UNSET,
        exit_code: Any = _UNSET,
        last_error: Any = _UNSET,
        last_error_type: Any = _UNSET,
        result_summary: Optional[str] = None,
        log_stdout_path: Optional[str] = None,
        log_stderr_path: Optional[str] = None,
        result_path: Optional[str] = None,
        output_path: Optional[str] = None,
        inputs_path: Optional[str] = None,
        materialized_prompt_path: Optional[str] = None,
        artifacts: Optional[list[Any]] = None,
        dataflow_error: Any = _UNSET,
        blocked_by: Optional[list[str]] = None,
        blocked_reason: Optional[str] = None,
        blocked_at: Optional[str] = None,
        watch_started: bool = False,
        watch_satisfied_at: Any = _UNSET,
        last_checked_at: Any = _UNSET,
        last_check_result: Any = _UNSET,
        condition: Any = _UNSET,
        tool_started: bool = False,
        tool_finished_at: Any = _UNSET,
        tool_error: Any = _UNSET,
        extra_events: Optional[list[tuple[str, str, dict[str, Any]]]] = None,
    ) -> None:
        """
        Mutate the task, persist the workflow, and append the event —
        all under the store lock so the trio is atomic.
        """
        with self._store.locked():
            active_attempt_id = attempt_id
            if active_attempt_id is None and task.current_attempt > 0:
                active_attempt_id = task.current_attempt
            task.status = new_status
            if started:
                task.started_at = _now_iso()
            if finished:
                task.finished_at = _now_iso()
            if pid is not _UNSET:
                task.pid = pid
            if exit_code is not _UNSET:
                task.exit_code = exit_code
            if last_error is not _UNSET:
                task.last_error = last_error
            if last_error_type is not _UNSET:
                task.last_error_type = last_error_type
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
            if dataflow_error is not _UNSET:
                task.dataflow_error = dataflow_error
            if blocked_by is not None:
                task.blocked_by = list(blocked_by)
            if blocked_reason is not None:
                task.blocked_reason = blocked_reason
            if blocked_at is not None:
                task.blocked_at = blocked_at
            if watch_started and task.watch_started_at is None:
                task.watch_started_at = _now_iso()
            if watch_satisfied_at is not _UNSET:
                task.watch_satisfied_at = watch_satisfied_at
            if last_checked_at is not _UNSET:
                task.last_checked_at = last_checked_at
            if last_check_result is not _UNSET:
                task.last_check_result = (
                    dict(last_check_result)
                    if last_check_result is not None else None
                )
            if condition is not _UNSET:
                task.condition = dict(condition) if condition is not None else None
            if tool_started and task.tool_started_at is None:
                task.tool_started_at = _now_iso()
            if tool_finished_at is not _UNSET:
                task.tool_finished_at = tool_finished_at
            if tool_error is not _UNSET:
                task.tool_error = tool_error

            self._store.save_workflow(wf)
            metadata: dict[str, Any] = {"status": new_status.value}
            if task.pid is not None:
                metadata["pid"] = task.pid
            if active_attempt_id is not None:
                metadata["attempt_id"] = active_attempt_id
            if blocked_by:
                metadata["blocked_by"] = blocked_by
            if task.watcher_type:
                metadata["watcher_type"] = task.watcher_type
            if task.tool_type:
                metadata["tool_type"] = task.tool_type
            if task.dataflow_error:
                metadata["dataflow_error"] = task.dataflow_error
            if task.tool_error:
                metadata["tool_error"] = task.tool_error
            if task.last_error:
                metadata["error"] = task.last_error
            if task.last_error_type:
                metadata["error_type"] = task.last_error_type
            self._events.emit(
                event, wf.workflow_id,
                task_id=task.task_id,
                attempt_id=active_attempt_id,
                agent_id=self._agent_id,
                message=message,
                metadata=metadata,
            )
            for event_type, event_message, event_metadata in extra_events or []:
                self._events.emit(
                    event_type,
                    wf.workflow_id,
                    task_id=task.task_id,
                    attempt_id=active_attempt_id,
                    agent_id=self._agent_id,
                    message=event_message,
                    metadata=dict(event_metadata),
                )

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_workflow(self, workflow_id: str) -> bool:
        wf = self._store.load_workflow(workflow_id)
        if wf is None or wf.is_terminal():
            return False
        cancel_workflow_on_disk(
            self._store,
            workflow_id,
            agent_id=self._agent_id,
        )
        for task in wf.tasks.values():
            handle = self._handles.pop(task.task_id, None)
            if handle is None:
                continue
            try:
                self._runner.cancel(handle)
            except Exception:
                pass
        return True


def _compute_ancestor_labels(depends_on_by_label: dict[str, list[str]]) -> dict[str, set[str]]:
    memo: dict[str, set[str]] = {}

    def visit(label: str) -> set[str]:
        if label in memo:
            return memo[label]
        ancestors: set[str] = set()
        for dep in depends_on_by_label.get(label, []):
            ancestors.add(dep)
            ancestors.update(visit(dep))
        memo[label] = ancestors
        return ancestors

    return {label: visit(label) for label in depends_on_by_label}
