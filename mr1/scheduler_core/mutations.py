from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from mr1.dataflow import TaskInputSpec
from mr1.workflow_events import WorkflowEventLog
from mr1.workflow_models import Provenance, Task, TaskStatus, Workflow, WorkflowStatus, new_task_id
from mr1.workflow_store import WorkflowStore

from .dependencies import status_for_reset
from .state_machine import reopen_workflow, reset_task_runtime_state


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def label_for_task_id(workflow: Workflow, task_id: str) -> Optional[str]:
    for label, candidate in workflow.label_to_task_id.items():
        if candidate == task_id:
            return label
    return None


def task_for_label_or_id(workflow: Workflow, label_or_task_id: str) -> Optional[Task]:
    return workflow.tasks.get(label_or_task_id) or workflow.task_by_label(label_or_task_id)


def task_spec_for_workflow(workflow: Workflow, task: Task) -> dict[str, Any]:
    dep_labels = [
        label_for_task_id(workflow, parent_id) or parent_id
        for parent_id in task.depends_on
    ]
    task_spec: dict[str, Any] = {
        "label": task.label,
        "title": task.title,
        "task_kind": task.task_kind,
    }
    if dep_labels:
        task_spec["depends_on"] = dep_labels
    if task.dependency_policy != "all_succeeded":
        task_spec["dependency_policy"] = task.dependency_policy
    if task.run_if is not None:
        task_spec["run_if"] = dict(task.run_if)
    if task.inputs:
        task_spec["inputs"] = [item.to_dict() for item in task.inputs]
    if task.timeout_s is not None:
        task_spec["timeout_s"] = task.timeout_s
    if task.task_kind == "agent":
        task_spec["agent_type"] = task.agent_type or "worker"
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


def workflow_to_spec(workflow: Workflow) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for label, task_id in workflow.label_to_task_id.items():
        task = workflow.tasks.get(task_id)
        if task is None:
            continue
        tasks.append(task_spec_for_workflow(workflow, task))
    return {
        "title": workflow.title,
        "tasks": tasks,
    }


def new_task_from_spec(
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
        agent_type=raw.get("agent_type", "worker")
        if raw.get("task_kind", "agent") == "agent" else None,
        prompt=raw.get("prompt", "")
        if raw.get("task_kind", "agent") == "agent" else "",
        watcher_type=raw.get("watcher_type"),
        watch_config=dict(raw.get("watch_config", {})),
        tool_type=raw.get("tool_type"),
        tool_config=dict(raw.get("tool_config", {})),
        condition=dict(raw["condition"]) if raw.get("condition") is not None else None,
        run_if=dict(raw["run_if"]) if raw.get("run_if") is not None else None,
        dependency_policy=raw.get("dependency_policy", "all_succeeded"),
        created_by=created_by,
        timeout_s=raw.get("timeout_s"),
        inputs=[
            TaskInputSpec.from_dict(item)
            for item in (raw.get("inputs") or [])
        ],
        status=TaskStatus.CREATED,
    )


def require_fragment_tasks(spec_fragment: dict[str, Any], *, error_cls: type[Exception]) -> list[dict[str, Any]]:
    tasks = spec_fragment.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise error_cls("workflow fragment must contain a non-empty 'tasks' list")
    return list(tasks)


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


class WorkflowMutationEngine:
    def __init__(
        self,
        *,
        store: WorkflowStore,
        agent_id: str,
        event_log: WorkflowEventLog,
        validate_spec: Callable[[dict[str, Any]], None],
        error_cls: type[Exception],
    ) -> None:
        self._store = store
        self._agent_id = agent_id
        self._events = event_log
        self._validate_spec = validate_spec
        self._error_cls = error_cls

    def _ensure_workflow_mutable(self, workflow: Workflow) -> None:
        if workflow.status is WorkflowStatus.CANCELLED:
            raise self._error_cls(
                f"workflow cancelled and cannot be mutated: {workflow.workflow_id}"
            )

    def rerun_task(self, workflow_id: str, label_or_task_id: str) -> str:
        with self._store.locked():
            workflow = self._store.load_workflow(workflow_id)
            if workflow is None:
                raise self._error_cls(f"workflow not found: {workflow_id}")
            self._ensure_workflow_mutable(workflow)
            task = task_for_label_or_id(workflow, label_or_task_id)
            if task is None:
                raise self._error_cls(
                    f"task not found in workflow {workflow_id}: {label_or_task_id}"
                )
            if task.status not in {
                TaskStatus.FAILED,
                TaskStatus.TIMED_OUT,
                TaskStatus.CANCELLED,
                TaskStatus.SUCCEEDED,
                TaskStatus.SKIPPED,
            }:
                raise self._error_cls(
                    f"task cannot be rerun from status '{task.status.value}': {task.label}"
                )
            reset_task_runtime_state(task, status=status_for_reset(workflow, task))
            reopen_workflow(workflow)
            self._store.save_workflow(workflow)
            self._events.task_rerun(
                workflow.workflow_id,
                task.task_id,
                agent_id=self._agent_id,
                message=f"task rerun requested: {task.label}",
                metadata={"status": task.status.value},
            )
        return task.task_id

    def cancel_task(self, task_id: str) -> str:
        with self._store.locked():
            workflow: Optional[Workflow] = None
            task: Optional[Task] = None
            for candidate in self._store.list_workflows():
                maybe = candidate.tasks.get(task_id)
                if maybe is not None:
                    workflow = candidate
                    task = maybe
                    break
            if workflow is None or task is None:
                raise self._error_cls(f"task not found: {task_id}")
            if task.is_terminal() and task.status is not TaskStatus.BLOCKED:
                raise self._error_cls(
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
                set_task_terminal_attempt(
                    self._store,
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
            reopen_workflow(workflow)
            self._store.save_workflow(workflow)
            self._events.task_cancelled(
                workflow.workflow_id,
                task.task_id,
                agent_id=self._agent_id,
                attempt_id=task.current_attempt or None,
                message="task cancelled",
                metadata={"status": TaskStatus.CANCELLED.value},
            )
            if task.current_attempt > 0:
                self._events.task_attempt_finished(
                    workflow.workflow_id,
                    task.task_id,
                    agent_id=self._agent_id,
                    attempt_id=task.current_attempt,
                    message="task attempt cancelled",
                    metadata={
                        "status": TaskStatus.CANCELLED.value,
                        "error_type": "cancelled",
                    },
                )
        return task.task_id

    def cancel_workflow(self, workflow_id: str) -> str:
        with self._store.locked():
            workflow = self._store.load_workflow(workflow_id)
            if workflow is None:
                raise self._error_cls(f"workflow not found: {workflow_id}")
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
                    set_task_terminal_attempt(
                        self._store,
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
                self._events.task_cancelled(
                    workflow.workflow_id,
                    task.task_id,
                    agent_id=self._agent_id,
                    attempt_id=task.current_attempt or None,
                    message="workflow cancelled",
                    metadata={"status": TaskStatus.CANCELLED.value},
                )
                if task.current_attempt > 0:
                    self._events.task_attempt_finished(
                        workflow.workflow_id,
                        task.task_id,
                        agent_id=self._agent_id,
                        attempt_id=task.current_attempt,
                        message="task attempt cancelled",
                        metadata={
                            "status": TaskStatus.CANCELLED.value,
                            "error_type": "cancelled",
                        },
                    )
            workflow.status = WorkflowStatus.CANCELLED
            workflow.finished_at = _now_iso()
            self._store.save_workflow(workflow)
            self._events.workflow_cancelled(
                workflow.workflow_id,
                agent_id=self._agent_id,
                message="workflow cancelled",
            )
        return workflow.workflow_id

    def append_workflow(self, workflow_id: str, spec_fragment: dict[str, Any]) -> str:
        created_by = Provenance(type="user", id=self._agent_id)
        with self._store.locked():
            workflow = self._store.load_workflow(workflow_id)
            if workflow is None:
                raise self._error_cls(f"workflow not found: {workflow_id}")
            self._ensure_workflow_mutable(workflow)
            new_tasks = require_fragment_tasks(spec_fragment, error_cls=self._error_cls)
            merged = workflow_to_spec(workflow)
            merged["tasks"].extend(new_tasks)
            self._validate_spec(merged)
            for raw in new_tasks:
                if raw["label"] in workflow.label_to_task_id:
                    raise self._error_cls(f"duplicate label '{raw['label']}' in workflow")
            added: list[str] = []
            for raw in new_tasks:
                task = new_task_from_spec(workflow, raw, created_by=created_by)
                task.depends_on = [
                    workflow.label_to_task_id[label]
                    for label in (raw.get("depends_on") or [])
                ]
                workflow.tasks[task.task_id] = task
                workflow.label_to_task_id[task.label] = task.task_id
                added.append(task.task_id)
            reopen_workflow(workflow)
            self._store.save_workflow(workflow)
            self._events.workflow_updated(
                workflow.workflow_id,
                agent_id=self._agent_id,
                message=f"workflow appended with {len(added)} task(s)",
                metadata={"operation": "append", "task_ids": added},
            )
        return workflow.workflow_id

    def insert_workflow(
        self,
        workflow_id: str,
        after_task: str,
        spec_fragment: dict[str, Any],
    ) -> str:
        created_by = Provenance(type="user", id=self._agent_id)
        with self._store.locked():
            workflow = self._store.load_workflow(workflow_id)
            if workflow is None:
                raise self._error_cls(f"workflow not found: {workflow_id}")
            self._ensure_workflow_mutable(workflow)
            anchor = task_for_label_or_id(workflow, after_task)
            if anchor is None:
                raise self._error_cls(f"task not found in workflow {workflow_id}: {after_task}")
            fragment_tasks = require_fragment_tasks(spec_fragment, error_cls=self._error_cls)
            if len(fragment_tasks) != 1:
                raise self._error_cls("insert-workflow requires exactly one task in 'tasks'")
            raw = dict(fragment_tasks[0])
            if raw.get("depends_on"):
                raise self._error_cls("insert-workflow task must not declare depends_on")
            direct_children = [
                task for task in workflow.tasks.values()
                if anchor.task_id in task.depends_on
            ]
            for child in direct_children:
                if child.status is TaskStatus.RUNNING:
                    raise self._error_cls(f"cannot mutate running task: {child.label}")
                if child.status is TaskStatus.SUCCEEDED:
                    raise self._error_cls(f"cannot mutate succeeded task: {child.label}")
            if raw["label"] in workflow.label_to_task_id:
                raise self._error_cls(f"duplicate label '{raw['label']}' in workflow")
            merged = workflow_to_spec(workflow)
            merged["tasks"].append({
                **raw,
                "depends_on": [label_for_task_id(workflow, anchor.task_id) or anchor.label],
            })
            self._validate_spec(merged)
            inserted = new_task_from_spec(workflow, raw, created_by=created_by)
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
            reopen_workflow(workflow)
            self._store.save_workflow(workflow)
            self._events.workflow_updated(
                workflow.workflow_id,
                agent_id=self._agent_id,
                message=f"workflow inserted task '{inserted.label}'",
                metadata={
                    "operation": "insert",
                    "task_ids": [inserted.task_id],
                    "after_task_id": anchor.task_id,
                },
            )
        return workflow.workflow_id

    def replace_workflow(
        self,
        workflow_id: str,
        target_task: str,
        spec_fragment: dict[str, Any],
    ) -> str:
        with self._store.locked():
            workflow = self._store.load_workflow(workflow_id)
            if workflow is None:
                raise self._error_cls(f"workflow not found: {workflow_id}")
            self._ensure_workflow_mutable(workflow)
            task = task_for_label_or_id(workflow, target_task)
            if task is None:
                raise self._error_cls(f"task not found in workflow {workflow_id}: {target_task}")
            if task.status is TaskStatus.RUNNING:
                raise self._error_cls(f"cannot replace running task: {task.label}")
            if task.status is TaskStatus.SUCCEEDED:
                raise self._error_cls(f"cannot replace succeeded task: {task.label}")
            if task.attempt_count > 0 and task.status not in {
                TaskStatus.FAILED,
                TaskStatus.TIMED_OUT,
                TaskStatus.CANCELLED,
                TaskStatus.SKIPPED,
            }:
                raise self._error_cls(
                    f"replace-workflow allowed only for unstarted or failed/cancelled tasks: {task.label}"
                )
            fragment_tasks = require_fragment_tasks(spec_fragment, error_cls=self._error_cls)
            if len(fragment_tasks) != 1:
                raise self._error_cls("replace-workflow requires exactly one task in 'tasks'")
            raw = dict(fragment_tasks[0])
            if raw.get("label") != task.label:
                raise self._error_cls(
                    f"replace-workflow task label must match target label '{task.label}'"
                )
            merged = workflow_to_spec(workflow)
            for idx, item in enumerate(merged["tasks"]):
                if item["label"] == task.label:
                    merged["tasks"][idx] = raw
                    break
            self._validate_spec(merged)
            task.title = raw.get("title", raw["label"])
            task.task_kind = raw.get("task_kind", "agent")
            task.agent_type = raw.get("agent_type", "worker") if task.task_kind == "agent" else None
            task.prompt = raw.get("prompt", "") if task.task_kind == "agent" else ""
            task.watcher_type = raw.get("watcher_type")
            task.watch_config = dict(raw.get("watch_config", {}))
            task.tool_type = raw.get("tool_type")
            task.tool_config = dict(raw.get("tool_config", {}))
            task.condition = dict(raw["condition"]) if raw.get("condition") is not None else None
            task.run_if = dict(raw["run_if"]) if raw.get("run_if") is not None else None
            task.dependency_policy = raw.get("dependency_policy", "all_succeeded")
            task.timeout_s = raw.get("timeout_s")
            task.inputs = [
                TaskInputSpec.from_dict(item)
                for item in (raw.get("inputs") or [])
            ]
            task.depends_on = [
                workflow.label_to_task_id[label]
                for label in (raw.get("depends_on") or [])
            ]
            reset_task_runtime_state(task, status=status_for_reset(workflow, task))
            reopen_workflow(workflow)
            self._store.save_workflow(workflow)
            self._events.workflow_updated(
                workflow.workflow_id,
                agent_id=self._agent_id,
                message=f"workflow replaced task '{task.label}'",
                metadata={
                    "operation": "replace",
                    "task_ids": [task.task_id],
                    "status": task.status.value,
                },
            )
        return workflow.workflow_id


def rerun_task_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    label_or_task_id: str,
    *,
    agent_id: str,
    validate_spec: Callable[[dict[str, Any]], None],
    event_log: WorkflowEventLog,
    error_cls: type[Exception],
) -> str:
    engine = WorkflowMutationEngine(
        store=store,
        agent_id=agent_id,
        event_log=event_log,
        validate_spec=validate_spec,
        error_cls=error_cls,
    )
    return engine.rerun_task(workflow_id, label_or_task_id)


def cancel_task_on_disk(
    store: WorkflowStore,
    task_id: str,
    *,
    agent_id: str,
    validate_spec: Callable[[dict[str, Any]], None],
    event_log: WorkflowEventLog,
    error_cls: type[Exception],
) -> str:
    del validate_spec
    engine = WorkflowMutationEngine(
        store=store,
        agent_id=agent_id,
        event_log=event_log,
        validate_spec=lambda spec: None,
        error_cls=error_cls,
    )
    return engine.cancel_task(task_id)


def cancel_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    *,
    agent_id: str,
    validate_spec: Callable[[dict[str, Any]], None],
    event_log: WorkflowEventLog,
    error_cls: type[Exception],
) -> str:
    del validate_spec
    engine = WorkflowMutationEngine(
        store=store,
        agent_id=agent_id,
        event_log=event_log,
        validate_spec=lambda spec: None,
        error_cls=error_cls,
    )
    return engine.cancel_workflow(workflow_id)


def append_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    spec_fragment: dict[str, Any],
    *,
    agent_id: str,
    validate_spec: Callable[[dict[str, Any]], None],
    event_log: WorkflowEventLog,
    error_cls: type[Exception],
) -> str:
    engine = WorkflowMutationEngine(
        store=store,
        agent_id=agent_id,
        event_log=event_log,
        validate_spec=validate_spec,
        error_cls=error_cls,
    )
    return engine.append_workflow(workflow_id, spec_fragment)


def insert_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    after_task: str,
    spec_fragment: dict[str, Any],
    *,
    agent_id: str,
    validate_spec: Callable[[dict[str, Any]], None],
    event_log: WorkflowEventLog,
    error_cls: type[Exception],
) -> str:
    engine = WorkflowMutationEngine(
        store=store,
        agent_id=agent_id,
        event_log=event_log,
        validate_spec=validate_spec,
        error_cls=error_cls,
    )
    return engine.insert_workflow(workflow_id, after_task, spec_fragment)


def replace_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    target_task: str,
    spec_fragment: dict[str, Any],
    *,
    agent_id: str,
    validate_spec: Callable[[dict[str, Any]], None],
    event_log: WorkflowEventLog,
    error_cls: type[Exception],
) -> str:
    engine = WorkflowMutationEngine(
        store=store,
        agent_id=agent_id,
        event_log=event_log,
        validate_spec=validate_spec,
        error_cls=error_cls,
    )
    return engine.replace_workflow(workflow_id, target_task, spec_fragment)
