"""
Workflow scheduler — deterministic DAG driver.

The scheduler owns the lifecycle of every task in every workflow it
discovers under its `WorkflowStore` root. It is intentionally dumb:

  * no LLM reasoning,
  * deterministic conditional branching,
  * no retry/backoff policies (Phase 1),
  * dependency gates are explicit and deterministic,
  * false conditions and non-winning branch paths transition to `skipped`.

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
import json
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.agents import AgentRegistry, default_agent_registry
from mr1.autonomy.consent import ConsentGrantStore
from mr1.capabilities import CapabilityRegistry, default_capability_registry
from mr1.clock import Clock, default_clock
from mr1.capability_policy import (
    CapabilityApprovalStore,
    CapabilityAuditRecord,
    CapabilityAuditWriter,
    CapabilityMetadata,
    CapabilityRequest,
    PolicyEngine,
)
from mr1.core import Logger
from mr1.conditions import (
    ConditionEvaluation,
    evaluate_condition,
    validate_condition,
    validate_dependency_policy,
)
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
from mr1.event_log import EventLog
from mr1.messages import MessageStore
from mr1.scoped_agents import PersistentAgentStore, is_agent_terminal
from mr1.scheduler_core.attempts import (
    AttemptManager,
    UNSET as ATTEMPT_UNSET,
    attempt_result_payload as _attempt_result_payload_impl,
    run_result_from_payload as _run_result_from_payload_impl,
    set_task_terminal_attempt as _set_task_terminal_attempt_impl,
)
from mr1.scheduler_core.capability_gate import CapabilityGate
from mr1.scheduler_core.dependencies import (
    DependencyGateDecision,
    compute_ancestor_labels,
    evaluate_dependency_gate,
    status_for_reset,
)
from mr1.scheduler_core.diagnostics import (
    available_agent_slots,
    build_scheduler_log_metadata,
    probe_path,
)
from mr1.scheduler_core.discovery import WorkflowQueryService, workflow_has_live_handles
from mr1.scheduler_core.events import SchedulerEventAdapter
from mr1.scheduler_core.mutations import (
    append_workflow_on_disk as _mutation_append_workflow_on_disk,
    cancel_task_on_disk as _mutation_cancel_task_on_disk,
    cancel_workflow_on_disk as _mutation_cancel_workflow_on_disk,
    insert_workflow_on_disk as _mutation_insert_workflow_on_disk,
    label_for_task_id as _label_for_task_id_impl,
    new_task_from_spec as _new_task_from_spec_impl,
    replace_workflow_on_disk as _mutation_replace_workflow_on_disk,
    require_fragment_tasks as _require_fragment_tasks_impl,
    rerun_task_on_disk as _mutation_rerun_task_on_disk,
    task_for_label_or_id as _task_for_label_or_id_impl,
    task_spec_for_workflow as _task_spec_for_workflow_impl,
    workflow_to_spec as _workflow_to_spec_impl,
)
from mr1.scheduler_core.reporting import WorkflowReporter
from mr1.scheduler_core.semantic_validation import (
    SemanticValidator,
    extract_semantic_expectation,
)
from mr1.scheduler_core.state_machine import reopen_workflow
from mr1.scheduler_core.tool_runtime import ToolTaskRunner
from mr1.scheduler_core.watcher_runtime import WatcherPollService
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
    TaskStatus,
    Workflow,
    WorkflowStatus,
    new_task_id,
    new_workflow_id,
)
from mr1.workflow_store import WorkflowStore, sync_task_view, sync_workflow_view


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_UNSET = ATTEMPT_UNSET
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
            if not agents.workflow_task_allowed(agent_type):
                raise WorkflowSpecError(
                    f"task '{label}': agent_type '{agent_type}' is not allowed in workflow tasks"
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
        try:
            validate_dependency_policy(
                raw.get("dependency_policy", "all_succeeded"),
                task_label=label,
            )
            validate_condition(raw.get("run_if"), spec, raw)
        except ValueError as exc:
            raise WorkflowSpecError(str(exc)) from exc

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

    ancestors = compute_ancestor_labels(depends_on_by_label)
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
    owner_agent_id: Optional[str] = None,
    workflow_metadata: Optional[dict[str, Any]] = None,
    scoped_agent_store: Optional[PersistentAgentStore] = None,
    watcher_registry: Optional[WatcherRegistry] = None,
    tool_registry: Optional[ToolRegistry] = None,
    agent_registry: Optional[AgentRegistry] = None,
    clock: Optional[Clock] = None,
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

    scoped_agents = scoped_agent_store or PersistentAgentStore()
    resolved_owner_agent_id = owner_agent_id or scoped_agents.root_agent_id
    owner_agent = scoped_agents.require_agent(resolved_owner_agent_id)
    if is_agent_terminal(owner_agent):
        raise WorkflowSpecError(f"agent is terminated: {resolved_owner_agent_id}")

    workflow_id = new_workflow_id()
    title = spec.get("title") or f"workflow {workflow_id}"
    wf = Workflow(
        workflow_id=workflow_id,
        title=title,
        status=WorkflowStatus.PENDING,
        created_by=created_by,
        owner_agent_id=owner_agent.agent_id,
        owner_agent_title=owner_agent.title,
        parent_agent_id=owner_agent.parent_agent_id,
        metadata=dict(workflow_metadata or {}),
        # `created_at` is what the stuck-workflow sweep measures against, so it
        # has to come from the same clock the supervisor reads.
        created_at=(clock or default_clock()).now_iso(),
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
            run_if=dict(raw["run_if"]) if raw.get("run_if") is not None else None,
            dependency_policy=raw.get("dependency_policy", "all_succeeded"),
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

    semantic_expectations: dict[str, dict[str, Any]] = {}
    for raw, task in task_objs:
        expectation = extract_semantic_expectation(raw, workflow_title=wf.title)
        if expectation is not None:
            semantic_expectations[task.task_id] = expectation
    if semantic_expectations:
        wf.metadata["semantic_expectations"] = semantic_expectations

    wf.label_to_task_id = label_to_task_id
    return wf


def submit_spec_to_disk(
    spec: dict[str, Any],
    created_by: Provenance,
    store: WorkflowStore,
    owner_agent_id: Optional[str] = None,
    caller_agent_id: Optional[str] = None,
    workflow_metadata: Optional[dict[str, Any]] = None,
    scoped_agent_store: Optional[PersistentAgentStore] = None,
    watcher_registry: Optional[WatcherRegistry] = None,
    tool_registry: Optional[ToolRegistry] = None,
    agent_registry: Optional[AgentRegistry] = None,
    clock: Optional[Clock] = None,
) -> str:
    """
    Persist a submitted workflow so the MR1-owned scheduler will pick
    it up on its next tick. Returns the new workflow ID.

    Called by both `Scheduler.submit_workflow` (in-process) and the
    standalone `workflow_cli submit` command.
    """
    scoped_agents = scoped_agent_store or PersistentAgentStore()
    if caller_agent_id is not None:
        scoped_agents.require_agent(caller_agent_id)
    resolved_owner_agent_id = (
        owner_agent_id
        or caller_agent_id
        or scoped_agents.root_agent_id
    )
    if (
        caller_agent_id is not None
        and not scoped_agents.can_manage_agent(caller_agent_id, resolved_owner_agent_id)
    ):
        raise WorkflowSpecError("access denied: owner agent not in caller scope")
    wf = build_workflow_from_spec(
        spec,
        created_by,
        resolved_owner_agent_id,
        workflow_metadata=workflow_metadata,
        scoped_agent_store=scoped_agents,
        watcher_registry=watcher_registry,
        tool_registry=tool_registry,
        agent_registry=agent_registry,
        clock=clock,
    )
    log = WorkflowEventLog(store, default_agent_id=created_by.id)
    timeline = EventLog(store.root.parent / "events")
    with store.locked():
        store.save_workflow(wf)
        scoped_agents.append_owned_workflow(resolved_owner_agent_id, wf.workflow_id)
        log.workflow_submitted(
            wf.workflow_id,
            agent_id=created_by.id,
            message=f"submitted '{wf.title}' with {len(wf.tasks)} task(s)",
            metadata={"task_count": len(wf.tasks)},
        )
        timeline.emit(
            event_type="workflow_created",
            actor_id=created_by.id,
            actor_type=created_by.type,
            target_id=wf.workflow_id,
            target_type="workflow",
            status=wf.status.value,
            summary=f"workflow created: {wf.title}",
            workflow_id=wf.workflow_id,
            record_path=str(store.workflow_json_path(wf.workflow_id)),
            metadata={
                "title": wf.title,
                "owner_agent_id": wf.owner_agent_id,
                "task_count": len(wf.tasks),
            },
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
    return _label_for_task_id_impl(workflow, task_id)


def _task_for_label_or_id(workflow: Workflow, label_or_task_id: str) -> Optional[Task]:
    return _task_for_label_or_id_impl(workflow, label_or_task_id)


def _task_spec_for_workflow(workflow: Workflow, task: Task) -> dict[str, Any]:
    return _task_spec_for_workflow_impl(workflow, task)


def _workflow_to_spec(workflow: Workflow) -> dict[str, Any]:
    return _workflow_to_spec_impl(workflow)


def _reopen_workflow(workflow: Workflow) -> None:
    reopen_workflow(workflow)


def _evaluate_dependency_gate(workflow: Workflow, task: Task) -> DependencyGateDecision:
    return evaluate_dependency_gate(workflow, task)


def _status_for_reset(workflow: Workflow, task: Task) -> TaskStatus:
    return status_for_reset(workflow, task)


def _condition_message(evaluation: ConditionEvaluation) -> str:
    if evaluation.expected is None:
        return f"{evaluation.ref} {evaluation.op} evaluated {'true' if evaluation.passed else 'false'}"
    return (
        f"{evaluation.ref} {evaluation.op} {evaluation.expected} "
        f"evaluated {'true' if evaluation.passed else 'false'}"
    )


def _condition_event_payload(
    task: Task,
    evaluation: ConditionEvaluation,
) -> tuple[str, str, dict[str, Any]]:
    return (
        ev.CONDITION_EVALUATED,
        _condition_message(evaluation),
        {
            "task_id": task.task_id,
            "ref": evaluation.ref,
            "op": evaluation.op,
            "expected": evaluation.expected,
            "actual": evaluation.actual,
            "passed": evaluation.passed,
            "reason": evaluation.reason,
            **dict(evaluation.metadata),
        },
    )


def _attempt_result_payload(
    task: Task,
    *,
    status: TaskStatus,
    exit_code: Optional[int],
    error: Optional[str],
    error_type: Optional[str],
    result_summary: Optional[str],
) -> dict[str, Any]:
    return _attempt_result_payload_impl(
        task,
        status=status,
        exit_code=exit_code,
        error=error,
        error_type=error_type,
        result_summary=result_summary,
    )


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
    return _set_task_terminal_attempt_impl(
        store,
        workflow,
        task,
        status=status,
        exit_code=exit_code,
        error=error,
        error_type=error_type,
        result_summary=result_summary,
        result_payload=result_payload,
    )


def rerun_task_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    label_or_task_id: str,
    *,
    agent_id: str = "scheduler",
) -> str:
    return _mutation_rerun_task_on_disk(
        store,
        workflow_id,
        label_or_task_id,
        agent_id=agent_id,
        validate_spec=validate_spec,
        event_log=WorkflowEventLog(store, default_agent_id=agent_id),
        error_cls=WorkflowSpecError,
    )


def cancel_task_on_disk(
    store: WorkflowStore,
    task_id: str,
    *,
    agent_id: str = "scheduler",
) -> str:
    return _mutation_cancel_task_on_disk(
        store,
        task_id,
        agent_id=agent_id,
        validate_spec=validate_spec,
        event_log=WorkflowEventLog(store, default_agent_id=agent_id),
        error_cls=WorkflowSpecError,
    )


def cancel_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    *,
    agent_id: str = "scheduler",
) -> str:
    return _mutation_cancel_workflow_on_disk(
        store,
        workflow_id,
        agent_id=agent_id,
        validate_spec=validate_spec,
        event_log=WorkflowEventLog(store, default_agent_id=agent_id),
        error_cls=WorkflowSpecError,
    )


def _new_task_from_spec(
    workflow: Workflow,
    raw: dict[str, Any],
    *,
    created_by: Provenance,
) -> Task:
    return _new_task_from_spec_impl(workflow, raw, created_by=created_by)


def _require_fragment_tasks(spec_fragment: dict[str, Any]) -> list[dict[str, Any]]:
    return _require_fragment_tasks_impl(spec_fragment, error_cls=WorkflowSpecError)


def append_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    spec_fragment: dict[str, Any],
    *,
    agent_id: str = "scheduler",
) -> str:
    return _mutation_append_workflow_on_disk(
        store,
        workflow_id,
        spec_fragment,
        agent_id=agent_id,
        validate_spec=validate_spec,
        event_log=WorkflowEventLog(store, default_agent_id=agent_id),
        error_cls=WorkflowSpecError,
    )


def insert_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    after_task: str,
    spec_fragment: dict[str, Any],
    *,
    agent_id: str = "scheduler",
) -> str:
    return _mutation_insert_workflow_on_disk(
        store,
        workflow_id,
        after_task,
        spec_fragment,
        agent_id=agent_id,
        validate_spec=validate_spec,
        event_log=WorkflowEventLog(store, default_agent_id=agent_id),
        error_cls=WorkflowSpecError,
    )


def replace_workflow_on_disk(
    store: WorkflowStore,
    workflow_id: str,
    target_task: str,
    spec_fragment: dict[str, Any],
    *,
    agent_id: str = "scheduler",
) -> str:
    return _mutation_replace_workflow_on_disk(
        store,
        workflow_id,
        target_task,
        spec_fragment,
        agent_id=agent_id,
        validate_spec=validate_spec,
        event_log=WorkflowEventLog(store, default_agent_id=agent_id),
        error_cls=WorkflowSpecError,
    )


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
        capability_registry: Optional[CapabilityRegistry] = None,
        concurrency: int = 4,
        auto_tick: bool = True,
        tick_interval_s: float = 1.0,
        agent_id: str = "scheduler",
        scoped_agent_store: Optional[PersistentAgentStore] = None,
        message_store: Optional[MessageStore] = None,
        workspace_root: Optional[Path] = None,
        logger: Optional[Logger] = None,
        clock: Optional[Clock] = None,
        runtime_error_sink: Optional[Any] = None,
        consent_store: Optional[Any] = None,
    ):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._clock = clock or default_clock()
        self._runtime_error_sink = runtime_error_sink
        self._tick_errors = 0
        self._consecutive_tick_errors = 0
        self._tick_count = 0
        self._last_tick_duration_ms = 0.0
        self._last_tick_at: Optional[str] = None
        self._last_tick_error: Optional[str] = None
        self._last_tick_error_at: Optional[str] = None
        self._store = store
        self._runner = runner or MockRunner()
        self._events = event_log or WorkflowEventLog(store, default_agent_id=agent_id)
        self._watchers = watcher_registry or default_watcher_registry()
        self._tools = tool_registry or default_tool_registry()
        self._capabilities = capability_registry or default_capability_registry()
        self._scoped_agents = scoped_agent_store or PersistentAgentStore(
            root=self._store.root.parent / "agents"
        )
        self._message_store = message_store or MessageStore(
            root=self._store.root.parent / "messages",
            scoped_agent_store=self._scoped_agents,
        )
        self._workspace_root = Path(workspace_root) if workspace_root else self._store.root.parent
        self._concurrency = concurrency
        self._tick_interval_s = tick_interval_s
        self._agent_id = agent_id
        self._policy_engine = PolicyEngine()
        self._approval_store = CapabilityApprovalStore(
            self._store.root.parent / "capability_approvals",
            clock=self._clock,
        )
        self._audit_writer = CapabilityAuditWriter()
        self._timeline = EventLog(self._store.root.parent / "events")
        self._logger = logger or Logger()
        self._event_adapter = SchedulerEventAdapter(
            workflow_events=self._events,
            timeline=self._timeline,
            agent_id=self._agent_id,
        )
        self._semantic_validator = SemanticValidator(
            workspace_root=self._workspace_root,
            approval_store=self._approval_store,
            scoped_agents=self._scoped_agents,
        )
        self._workflow_queries = WorkflowQueryService(
            store=self._store,
            scoped_agents=self._scoped_agents,
        )

        self._handles: dict[str, RunHandle] = {}
        self._attempts = AttemptManager(
            store=self._store,
            events=self._event_adapter,
            handle_registry=self._handles,
            now_fn=self._clock.now_iso,
        )
        self._consent_store = consent_store or ConsentGrantStore(
            self._store.root.parent,
            clock=self._clock,
            scoped_agent_store=self._scoped_agents,
        )
        self._capability_gate = CapabilityGate(
            store=self._store,
            capabilities=self._capabilities,
            scoped_agents=self._scoped_agents,
            message_store=self._message_store,
            policy_engine=self._policy_engine,
            approval_store=self._approval_store,
            audit_writer=self._audit_writer,
            timeline=self._timeline,
            workspace_root=self._workspace_root,
            now_fn=self._clock.now_iso,
            consent_store=self._consent_store,
            clock=self._clock,
        )
        self._reporter = WorkflowReporter(
            scoped_agents=self._scoped_agents,
            message_store=self._message_store,
        )
        self._tool_task_runner = ToolTaskRunner(
            tools=self._tools,
            store=self._store,
            approval_store=self._approval_store,
            timeline=self._timeline,
            apply_semantic_validation=self._apply_semantic_validation,
            write_policy_audit=self._write_policy_audit,
            append_policy_audit_index=self._append_policy_audit_index,
            policy_audit_id=self._policy_audit_id,
            finish_attempt=self._finish_attempt,
        )
        self._watcher_poll_service = WatcherPollService(
            store=self._store,
            watchers=self._watchers,
            approval_store=self._approval_store,
            event_adapter=self._event_adapter,
            current_attempt_policy_audit_path=self._current_attempt_policy_audit_path,
            finalize_policy_audit=self._finalize_policy_audit,
            append_policy_audit_index_from_path=self._append_policy_audit_index_from_path,
            finish_attempt=self._finish_attempt,
            clock=self._clock,
        )
        self._tick_lock = threading.Lock()
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
            except Exception as exc:
                # A crashing tick must not kill the daemon thread, but it must
                # never be invisible either: a wedged scheduler and an idle one
                # have to be distinguishable from outside the process.
                self._record_tick_failure(exc)
            self._stop.wait(self._tick_interval_s)

    def _record_tick_failure(self, exc: BaseException) -> None:
        """Persist, publish, and count a failed tick. Must never raise."""
        detail = f"{type(exc).__name__}: {exc}"
        self._tick_errors += 1
        self._consecutive_tick_errors += 1
        self._last_tick_error = detail
        self._last_tick_error_at = self._safe_now_iso()
        trace = traceback.format_exc()
        if self._runtime_error_sink is not None:
            try:
                self._runtime_error_sink(
                    "scheduler_tick",
                    detail,
                    details={
                        "tick_errors": self._tick_errors,
                        "consecutive_tick_errors": self._consecutive_tick_errors,
                        "traceback": trace[-2000:],
                    },
                )
            except Exception:
                pass
        try:
            self._timeline.emit(
                event_type="scheduler_tick_failed",
                actor_id=self._agent_id,
                actor_type="mr1",
                target_id="scheduler",
                target_type="scheduler",
                status="error",
                summary=f"scheduler tick failed: {detail}"[:300],
                metadata={
                    "error": detail,
                    "error_type": type(exc).__name__,
                    "tick_errors": self._tick_errors,
                    "consecutive_tick_errors": self._consecutive_tick_errors,
                },
            )
        except Exception:
            pass
        try:
            self._logger.log(
                "scheduler",
                "scheduler",
                "tick_failed",
                "error",
                metadata={"error": detail, "tick_errors": self._tick_errors},
            )
        except Exception:
            pass

    def _safe_now_iso(self) -> str:
        try:
            return self._clock.now_iso()
        except Exception:
            return ""

    @property
    def consent_store(self) -> Any:
        return self._consent_store

    def metrics(self) -> dict[str, Any]:
        """Loop health, for the supervisor's gauge set and `mr1 status`."""
        return {
            "tick_count": self._tick_count,
            "tick_errors": self._tick_errors,
            "consecutive_tick_errors": self._consecutive_tick_errors,
            "last_tick_at": self._last_tick_at,
            "last_tick_duration_ms": round(self._last_tick_duration_ms, 3),
            "last_tick_error": self._last_tick_error,
            "last_tick_error_at": self._last_tick_error_at,
            "live_handles": len(self._handles),
        }

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_workflow(
        self,
        spec: dict[str, Any],
        created_by: Provenance,
        owner_agent_id: Optional[str] = None,
        caller_agent_id: Optional[str] = None,
        workflow_metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Validate a spec, persist the resulting workflow, and emit
        `workflow_submitted`. Returns the new workflow ID.
        """
        return submit_spec_to_disk(
            spec,
            created_by,
            self._store,
            owner_agent_id=owner_agent_id or caller_agent_id or self._scoped_agents.root_agent_id,
            caller_agent_id=caller_agent_id,
            workflow_metadata=workflow_metadata,
            scoped_agent_store=self._scoped_agents,
            watcher_registry=self._watchers,
            tool_registry=self._tools,
            clock=self._clock,
        )

    # ------------------------------------------------------------------
    # Read-through accessors
    # ------------------------------------------------------------------

    def list_workflows(self) -> list[Workflow]:
        return self._workflow_queries.list_workflows()

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflow_queries.get_workflow(workflow_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._workflow_queries.get_task(task_id)

    def list_watchers(self) -> list[tuple[Workflow, Task]]:
        return self._workflow_queries.list_watchers()

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
          1. Promote created tasks into queued states.
          2. Poll running tasks and persist terminal transitions.
          3. Reconcile non-running tasks against dependency policies and conditions.
          4. Finalise the workflow when every task is terminal.
          5. Launch ready tasks up to the concurrency cap.
        """
        started = self._clock.monotonic()
        with self._tick_lock:
            try:
                for wf in self._workflow_queries.active_workflows(set(self._handles)):
                    self._reconcile_external_control(wf)
                    self._tick_workflow(wf)
            finally:
                self._tick_count += 1
                self._last_tick_duration_ms = (self._clock.monotonic() - started) * 1000.0
                self._last_tick_at = self._safe_now_iso()
        self._consecutive_tick_errors = 0

    def _workflow_has_live_handles(self, wf: Workflow) -> bool:
        return workflow_has_live_handles(wf, set(self._handles))

    @staticmethod
    def _probe_path(path: Optional[str | Path]) -> dict[str, Any]:
        return probe_path(path)

    def _scheduler_log_metadata(
        self,
        task: Task,
        *,
        handle: Optional[RunHandle] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return build_scheduler_log_metadata(
            task,
            store=self._store,
            handle_present=task.task_id in self._handles,
            handle=handle,
            extra=extra,
        )

    def _log_scheduler_task(
        self,
        task: Task,
        action: str,
        *,
        result: str = "ok",
        handle: Optional[RunHandle] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._logger.log(
            task.task_id,
            "scheduler",
            action,
            result,
            metadata=self._scheduler_log_metadata(task, handle=handle, extra=metadata),
        )

    def _drop_handle(self, task: Task, *, reason: str) -> Optional[RunHandle]:
        handle = self._handles.pop(task.task_id, None)
        if handle is not None:
            self._log_scheduler_task(
                task,
                "handle_removed",
                handle=handle,
                metadata={"reason": reason},
            )
        return handle

    @staticmethod
    def _run_result_from_payload(
        task: Task,
        payload: dict[str, Any],
    ) -> Optional[RunResult]:
        return _run_result_from_payload_impl(task, payload)

    def _load_persisted_result(
        self,
        task: Task,
    ) -> tuple[Optional[RunResult], Optional[str]]:
        return self._attempts.load_persisted_result(task)

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
            self._drop_handle(task, reason="external_control")

    def _tick_workflow(self, wf: Workflow) -> None:
        changed = False

        # 0. Promote created tasks into queued states on first discovery.
        changed |= self._initialize_created_tasks(wf)

        # 1. Poll running tasks.
        changed |= self._poll_running_tasks(wf)

        # 2. Reconcile dependency gates and task-level conditions.
        changed |= self._reconcile_queued_tasks(wf)

        # 3. Workflow-level transitions.
        changed |= self._transition_workflow_status(wf)

        # 4. Launch ready tasks (bounded by concurrency).
        self._launch_ready(wf)

        # Nothing persists here — every mutation above went through
        # `_commit()` which holds the store lock.

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------

    def _initialize_created_tasks(self, wf: Workflow) -> bool:
        """Move newly-seen `created` tasks into their initial queued state."""
        changed = False
        for task in wf.tasks.values():
            if task.status is not TaskStatus.CREATED:
                continue
            gate = _evaluate_dependency_gate(wf, task)
            if gate.state == "wait":
                self._commit(
                    wf,
                    task,
                    TaskStatus.WAITING,
                    event=ev.TASK_CREATED,
                    message=gate.reason,
                    pid=None,
                    finished_at=None,
                    blocked_by=[],
                    blocked_reason=None,
                    blocked_at=None,
                    skip_reason=None,
                    condition_result=None,
                )
            elif gate.state == "block":
                self._commit_blocked(wf, task, gate)
            elif gate.state == "skip":
                self._commit_skipped(wf, task, gate.reason)
            else:
                self._promote_task_ready(wf, task, gate.reason)
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
                recovered: Optional[RunResult] = None
                recovery_source: Optional[str] = None
                self._log_scheduler_task(
                    task,
                    "handle_missing",
                    result="error",
                    metadata={"reason": "running task has no live scheduler handle"},
                )
                recovered, recovery_source = self._load_persisted_result(task)
                if recovered is None:
                    try:
                        recovered = self._runner.recover_result(task)
                    except Exception:
                        recovered = None
                    if recovered is not None:
                        recovery_source = "stdout_log"
                if recovered is not None:
                    self._log_scheduler_task(
                        task,
                        "handle_recovered",
                        metadata={"source": recovery_source},
                    )
                    stub_handle = RunHandle(
                        task_id=task.task_id,
                        workflow_id=wf.workflow_id,
                    )
                    changed |= self._handle_terminal_result(wf, task, stub_handle, recovered)
                    continue
                result_path = str(self._store.write_attempt_result(
                    wf.workflow_id,
                    task.task_id,
                    task.current_attempt,
                    _attempt_result_payload(
                        task,
                        status=TaskStatus.FAILED,
                        exit_code=None,
                        error="run handle lost: no recoverable persisted result",
                        error_type="infrastructure_failure",
                        result_summary="run handle lost",
                    ),
                ))
                self._finish_attempt(
                    wf, task, TaskStatus.FAILED, event=ev.TASK_FAILED,
                    message="run handle lost: no recoverable persisted result",
                    exit_code=None,
                    error="run handle lost: no recoverable persisted result",
                    error_type="infrastructure_failure",
                    result_path=result_path,
                    result_summary="run handle lost",
                )
                changed = True
                continue
            result = self._runner.poll(handle)
            if result is None:
                continue
            changed |= self._handle_terminal_result(wf, task, handle, result)
        return changed

    def _handle_terminal_result(
        self,
        wf: Workflow,
        task: Task,
        handle: RunHandle,
        result: RunResult,
    ) -> bool:
        try:
            return self._finalize_task(wf, task, handle, result)
        except Exception as exc:
            detail = f"task finalization failed: {type(exc).__name__}: {exc}"
            trace = traceback.format_exc()
            stderr_path = result.stderr_path or handle.payload.get("stderr_path")
            if isinstance(stderr_path, Path):
                try:
                    with open(stderr_path, "a", encoding="utf-8") as handle_out:
                        if handle_out.tell() > 0:
                            handle_out.write("\n")
                        handle_out.write(trace)
                except OSError:
                    pass
            result_path = str(self._store.write_attempt_result(
                wf.workflow_id,
                task.task_id,
                task.current_attempt,
                _attempt_result_payload(
                    task,
                    status=TaskStatus.FAILED,
                    exit_code=result.exit_code,
                    error=detail,
                    error_type="internal_error",
                    result_summary="task finalization failed",
                ),
            ))
            self._finish_attempt(
                wf,
                task,
                TaskStatus.FAILED,
                event=ev.TASK_FAILED,
                message=detail,
                exit_code=result.exit_code,
                error=detail,
                error_type="internal_error",
                log_stdout_path=str(result.stdout_path) if result.stdout_path else None,
                log_stderr_path=str(stderr_path) if isinstance(stderr_path, Path) else None,
                result_path=result_path,
                result_summary="task finalization failed",
            )
            return True

    def _poll_running_watcher(self, wf: Workflow, task: Task) -> bool:
        return self._watcher_poll_service.poll(wf, task)

    def _workflow_actor_type(self, wf: Workflow) -> str:
        return self._capability_gate.workflow_actor_type(wf)

    def _capability_name_for_task(self, task: Task) -> str:
        return self._capability_gate.capability_name_for_task(task)

    def _capability_args_for_task(self, task: Task) -> dict[str, Any]:
        return self._capability_gate.capability_args_for_task(task)

    def _current_attempt_policy_audit_path(self, task: Task) -> Optional[Path]:
        return self._capability_gate.current_attempt_policy_audit_path(task)

    def _policy_audit_path(self, wf: Workflow, task: Task, attempt_id: int) -> Path:
        return self._capability_gate.policy_audit_path(wf, task, attempt_id)

    def _write_policy_audit(
        self,
        audit_path: Path,
        *,
        capability_name: str,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        decision: dict[str, Any],
        execution_result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Path:
        return self._capability_gate.write_policy_audit(
            audit_path,
            capability_name=capability_name,
            request=request,
            metadata=metadata,
            decision=decision,
            execution_result=execution_result,
            error=error,
        )

    def _finalize_policy_audit(
        self,
        audit_path: Path,
        *,
        execution_result: dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        self._capability_gate.finalize_policy_audit(
            audit_path,
            execution_result=execution_result,
            error=error,
        )

    def _append_policy_audit_index(
        self,
        *,
        actor_id: str,
        audit_id: str,
        audit_path: Path,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        decision: dict[str, Any],
        execution_status: str,
        error: Optional[str] = None,
        approval_request_id: Optional[str] = None,
    ) -> None:
        self._capability_gate.append_policy_audit_index(
            actor_id=actor_id,
            audit_id=audit_id,
            audit_path=audit_path,
            request=request,
            metadata=metadata,
            decision=decision,
            execution_status=execution_status,
            error=error,
            approval_request_id=approval_request_id,
        )

    def _append_policy_audit_index_from_path(
        self,
        audit_path: Path,
        *,
        execution_status: str,
        error: Optional[str] = None,
        approval_request_id: Optional[str] = None,
    ) -> None:
        self._capability_gate.append_policy_audit_index_from_path(
            audit_path,
            execution_status=execution_status,
            error=error,
            approval_request_id=approval_request_id,
        )

    def _policy_audit_id(self, audit_path: Path) -> str:
        return self._capability_gate.policy_audit_id(audit_path)

    def _evaluate_task_policy(
        self,
        wf: Workflow,
        task: Task,
        attempt_id: int,
    ) -> tuple[CapabilityRequest, CapabilityMetadata, dict[str, Any], Path]:
        return self._capability_gate.evaluate_task_policy(wf, task, attempt_id)

    def _policy_block_result_payload(
        self,
        wf: Workflow,
        task: Task,
        request: CapabilityRequest,
        decision: dict[str, Any],
        *,
        target_status: TaskStatus,
        approval_request_id: Optional[str] = None,
        audit_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        return self._capability_gate.policy_block_result_payload(
            wf,
            task,
            request,
            decision,
            target_status=target_status,
            approval_request_id=approval_request_id,
            audit_path=audit_path,
        )

    def _handle_policy_block_for_task(
        self,
        wf: Workflow,
        task: Task,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        decision: dict[str, Any],
        audit_path: Path,
    ) -> None:
        result_payload, approval_request_id, message = self._capability_gate.handle_policy_block_for_task(
            wf,
            task,
            request,
            metadata,
            decision,
            audit_path,
        )
        result_path = str(self._store.write_attempt_result(
            wf.workflow_id,
            task.task_id,
            task.current_attempt,
            result_payload,
        ))
        extra_event_type = ev.TOOL_FAILED if task.task_kind == "tool" else ev.WATCHER_FAILED
        target_status = (
            TaskStatus.BLOCKED
            if decision["status"] == "requires_approval" else
            TaskStatus.FAILED
        )
        error_type = "approval_required" if target_status is TaskStatus.BLOCKED else "policy_block"
        extra_metadata: dict[str, Any] = {
            "policy_status": decision["status"],
            "reason": decision["reason"],
            "failure_type": error_type,
            "retryable": False,
        }
        if approval_request_id is not None:
            extra_metadata["approval_request_id"] = approval_request_id
        self._finish_attempt(
            wf,
            task,
            target_status,
            event=ev.TASK_BLOCKED if target_status is TaskStatus.BLOCKED else ev.TASK_FAILED,
            message=message,
            error=message,
            error_type=error_type,
            blocked_by=[] if target_status is TaskStatus.BLOCKED else None,
            blocked_reason=message if target_status is TaskStatus.BLOCKED else _UNSET,
            blocked_at=self._clock.now_iso() if target_status is TaskStatus.BLOCKED else _UNSET,
            result_path=result_path,
            result_summary=message,
            extra_events=[(
                extra_event_type,
                message,
                extra_metadata,
            )],
        )

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

    def _semantic_expectation_for_task(
        self,
        wf: Workflow,
        task: Task,
    ) -> Optional[dict[str, Any]]:
        return self._semantic_validator.expectation_for_task(wf, task)

    def _semantic_validation_text(
        self,
        *,
        summary: Optional[str],
        text: Optional[str],
        data: Optional[dict[str, Any]],
    ) -> str:
        return self._semantic_validator.validation_text(
            summary=summary,
            text=text,
            data=data,
        )

    def _semantic_target_path(self, raw_path: Optional[str]) -> Optional[Path]:
        return self._semantic_validator.semantic_target_path(raw_path)

    def _semantic_mismatch_reason(
        self,
        wf: Workflow,
        task: Task,
        expectation: dict[str, Any],
        *,
        summary: Optional[str],
        text: Optional[str],
        data: Optional[dict[str, Any]],
    ) -> Optional[str]:
        return self._semantic_validator.mismatch_reason(
            wf,
            task,
            expectation,
            summary=summary,
            text=text,
            data=data,
        )

    def _apply_semantic_validation(
        self,
        wf: Workflow,
        task: Task,
        *,
        target_status: TaskStatus,
        event_type: str,
        summary: Optional[str],
        text: Optional[str],
        data: Optional[dict[str, Any]],
        result_payload: dict[str, Any],
    ) -> tuple[TaskStatus, str, Optional[str], Optional[str], Optional[str], dict[str, Any]]:
        return self._semantic_validator.apply(
            wf,
            task,
            target_status=target_status,
            event_type=event_type,
            summary=summary,
            text=text,
            data=data,
            result_payload=result_payload,
        )

    def _finalize_task(
        self,
        wf: Workflow,
        task: Task,
        handle: RunHandle,
        result: RunResult,
    ) -> bool:
        self._drop_handle(task, reason="terminal_result")

        status_map = {
            RunStatus.SUCCEEDED: (TaskStatus.SUCCEEDED, ev.TASK_SUCCEEDED),
            RunStatus.FAILED: (TaskStatus.FAILED, ev.TASK_FAILED),
            RunStatus.TIMED_OUT: (TaskStatus.TIMED_OUT, ev.TASK_TIMED_OUT),
        }
        target_status, event_type = status_map.get(
            result.status, (TaskStatus.FAILED, ev.TASK_FAILED),
        )
        error_type = self._normalize_error_type(result, target_status)
        result_summary = (result.summary or "")[:500] if result.summary else None
        result_payload = dict(result.result_payload or {})

        extra_events: list[tuple[str, str, dict[str, Any]]] = []
        output_path: Optional[str] = None
        dataflow_error: Optional[str] = None
        output = None
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
                        result_summary=result_summary,
                        exit_code=result.exit_code,
                    ),
                    result_payload,
                )
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
        target_status, event_type, result_summary, semantic_error, semantic_error_type, result_payload = self._apply_semantic_validation(
            wf,
            task,
            target_status=target_status,
            event_type=event_type,
            summary=output.summary if output is not None else result_summary,
            text=output.text if output is not None else result_payload.get("text"),
            data=output.data if output is not None else result_payload.get("data"),
            result_payload=result_payload,
        )
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
        result_path = str(self._store.write_attempt_result(
            wf.workflow_id,
            task.task_id,
            task.current_attempt,
            result_payload,
        ))

        self._finish_attempt(
            wf,
            task,
            target_status,
            event=event_type,
            message=semantic_error or dataflow_error or result.error or result.summary or "",
            exit_code=result.exit_code,
            result_summary=result_summary,
            error=semantic_error or dataflow_error or result.error,
            error_type=semantic_error_type or error_type,
            log_stdout_path=str(result.stdout_path) if result.stdout_path else None,
            log_stderr_path=str(result.stderr_path) if result.stderr_path else None,
            result_path=result_path,
            output_path=output_path,
            dataflow_error=dataflow_error,
            artifacts=task.artifacts,
            extra_events=extra_events,
        )
        return True

    def _reconcile_queued_tasks(self, wf: Workflow) -> bool:
        changed = False
        for task in wf.tasks.values():
            if task.status in {
                TaskStatus.RUNNING,
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.TIMED_OUT,
                TaskStatus.CANCELLED,
            }:
                continue
            if (
                task.status is TaskStatus.BLOCKED
                and task.last_error_type == "approval_required"
            ):
                continue
            gate = _evaluate_dependency_gate(wf, task)
            if task.status is TaskStatus.SKIPPED:
                if gate.state == "skip":
                    continue
                if gate.state == "pass":
                    continue
                if gate.state == "wait":
                    self._commit(
                        wf,
                        task,
                        TaskStatus.WAITING,
                        event=None,
                        message=gate.reason,
                        pid=None,
                        finished_at=None,
                        blocked_by=[],
                        blocked_reason=None,
                        blocked_at=None,
                        skip_reason=None,
                        condition_result=None,
                    )
                else:
                    self._commit_blocked(wf, task, gate)
                changed = True
                continue
            if gate.state == "block":
                if task.status is not TaskStatus.BLOCKED or task.blocked_by != gate.blocked_by:
                    self._commit_blocked(wf, task, gate)
                    changed = True
                continue
            if gate.state == "skip":
                if task.status is not TaskStatus.SKIPPED or task.skip_reason != gate.reason:
                    self._commit_skipped(wf, task, gate.reason)
                    changed = True
                continue
            if gate.state == "wait":
                if task.status is TaskStatus.BLOCKED:
                    self._commit(
                        wf,
                        task,
                        TaskStatus.WAITING,
                        event=ev.TASK_UNBLOCKED,
                        message="dependencies recovered",
                        pid=None,
                        finished_at=None,
                        blocked_by=[],
                        blocked_reason=None,
                        blocked_at=None,
                        skip_reason=None,
                        condition_result=None,
                    )
                    changed = True
                elif task.status in {TaskStatus.READY, TaskStatus.WAITING}:
                    if task.status is not TaskStatus.WAITING:
                        self._commit(
                            wf,
                            task,
                            TaskStatus.WAITING,
                            event=None,
                            message=gate.reason,
                            pid=None,
                            finished_at=None,
                            blocked_by=[],
                            blocked_reason=None,
                            blocked_at=None,
                            skip_reason=None,
                            condition_result=None,
                        )
                        changed = True
                continue
            if task.status is TaskStatus.READY and (
                task.run_if is None or task.condition_result is not None
            ):
                continue
            changed |= self._promote_task_ready(wf, task, gate.reason)
        return changed

    def _promote_task_ready(
        self,
        wf: Workflow,
        task: Task,
        dependency_reason: str,
    ) -> bool:
        extra_events: list[tuple[str, str, dict[str, Any]]] = []
        ready_condition_result: Optional[dict[str, Any]] = None
        if task.run_if is not None:
            evaluation = evaluate_condition(task.run_if, wf, task, self._store)
            extra_events.append(_condition_event_payload(task, evaluation))
            if not evaluation.passed:
                self._commit_skipped(
                    wf,
                    task,
                    _condition_message(evaluation),
                    condition_result=evaluation,
                    extra_events=extra_events,
                )
                return True
            dependency_reason = _condition_message(evaluation)
            ready_condition_result = evaluation.to_dict()
        if task.status is TaskStatus.READY and task.condition_result == ready_condition_result:
            return False
        self._commit(
            wf,
            task,
            TaskStatus.READY,
            event=ev.TASK_READY,
            message=dependency_reason,
            pid=None,
            finished_at=None,
            blocked_by=[],
            blocked_reason=None,
            blocked_at=None,
            skip_reason=None,
            condition_result=ready_condition_result,
            extra_events=extra_events,
        )
        return True

    def _commit_blocked(
        self,
        wf: Workflow,
        task: Task,
        gate: DependencyGateDecision,
    ) -> None:
        self._commit(
            wf,
            task,
            TaskStatus.BLOCKED,
            event=ev.TASK_BLOCKED,
            message=f"blocked by {gate.reason}",
            finished=True,
            pid=None,
            blocked_by=gate.blocked_by,
            blocked_reason=gate.reason,
            blocked_at=self._clock.now_iso(),
            skip_reason=None,
            condition_result=None,
        )

    def _commit_skipped(
        self,
        wf: Workflow,
        task: Task,
        skip_reason: str,
        *,
        condition_result: Optional[ConditionEvaluation] = None,
        extra_events: Optional[list[tuple[str, str, dict[str, Any]]]] = None,
    ) -> None:
        self._commit(
            wf,
            task,
            TaskStatus.SKIPPED,
            event=ev.TASK_SKIPPED,
            message=skip_reason,
            finished=True,
            pid=None,
            blocked_by=[],
            blocked_reason=None,
            blocked_at=None,
            skip_reason=skip_reason,
            condition_result=condition_result.to_dict() if condition_result is not None else None,
            extra_events=extra_events,
        )

    def _transition_workflow_status(self, wf: Workflow) -> bool:
        """Flip `pending → running → succeeded/failed` based on tasks."""
        original_wf = wf
        with self._store.locked():
            wf = self._store.load_workflow(wf.workflow_id)
            if wf is None:
                return False
            all_terminal = all(t.is_terminal() for t in wf.tasks.values())
            any_running_or_live = any(
                t.status in (TaskStatus.RUNNING, TaskStatus.READY, TaskStatus.WAITING)
                for t in wf.tasks.values()
            )
            pending_approval_blocks = self._pending_approval_block_task_ids(wf)

            target: Optional[WorkflowStatus] = None
            event_type: Optional[str] = None
            message = ""

            if wf.status is WorkflowStatus.PENDING and any_running_or_live:
                target = WorkflowStatus.RUNNING
            elif wf.status is WorkflowStatus.PENDING and pending_approval_blocks:
                target = WorkflowStatus.RUNNING

            if all_terminal:
                if pending_approval_blocks:
                    if target is None:
                        return False
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
                    message = "all tasks completed without failure"

            if target is None or target is wf.status:
                return False

            wf.status = target
            if target in (WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED):
                wf.finished_at = self._clock.now_iso()
            self._scoped_agents.normalize_workflow_ownership(wf)
            self._store.save_workflow(wf)
            if target is WorkflowStatus.RUNNING:
                self._timeline.emit(
                    event_type="workflow_started",
                    actor_id=self._agent_id,
                    actor_type="scheduler",
                    target_id=wf.workflow_id,
                    target_type="workflow",
                    status=wf.status.value,
                    summary=f"workflow started: {wf.title}",
                    workflow_id=wf.workflow_id,
                    record_path=str(self._store.workflow_json_path(wf.workflow_id)),
                    metadata={"title": wf.title},
                )
            if event_type is not None:
                self._events.emit(
                    event_type, wf.workflow_id,
                    agent_id=self._agent_id, message=message,
                )
                self._timeline.emit(
                    event_type=(
                        "workflow_completed"
                        if target is WorkflowStatus.SUCCEEDED else
                        "workflow_failed"
                    ),
                    actor_id=self._agent_id,
                    actor_type="scheduler",
                    target_id=wf.workflow_id,
                    target_type="workflow",
                    status=wf.status.value,
                    summary=message,
                    workflow_id=wf.workflow_id,
                    record_path=str(self._store.workflow_json_path(wf.workflow_id)),
                    metadata={"title": wf.title},
                )
            sync_workflow_view(original_wf, wf)
        if target in (WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED):
            report_path = self._scoped_agents.write_workflow_report(wf, self._store)
            if report_path is not None:
                self._send_workflow_report_message(wf, report_path)
        return True

    def _pending_approval_block_task_ids(self, wf: Workflow) -> set[str]:
        pending = {
            (approval.workflow_id, approval.task_id)
            for approval in self._approval_store.list_requests()
            if approval.status == "pending"
            and approval.workflow_id == wf.workflow_id
            and approval.task_id
        }
        return {
            task.task_id
            for task in wf.tasks.values()
            if task.status is TaskStatus.BLOCKED
            and task.last_error_type == "approval_required"
            and (wf.workflow_id, task.task_id) in pending
        }

    def _send_workflow_report_message(self, wf: Workflow, report_path: Path) -> None:
        self._reporter.send_workflow_report_message(wf, report_path)

    def _begin_attempt(
        self,
        wf: Workflow,
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
        return self._attempts.begin_attempt(
            wf,
            task,
            message=message,
            pid=pid,
            watch_started=watch_started,
            tool_started=tool_started,
            policy_audit_path=policy_audit_path,
            extra_events=extra_events,
            run_handle=run_handle,
        )

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
        blocked_reason: Any = _UNSET,
        blocked_at: Any = _UNSET,
        watch_satisfied_at: Any = _UNSET,
        last_checked_at: Any = _UNSET,
        last_check_result: Any = _UNSET,
        condition: Any = _UNSET,
        tool_finished_at: Any = _UNSET,
        tool_error: Any = _UNSET,
        extra_events: Optional[list[tuple[str, str, dict[str, Any]]]] = None,
    ) -> None:
        self._attempts.finish_attempt(
            wf,
            task,
            new_status,
            event=event,
            message=message,
            exit_code=exit_code,
            result_summary=result_summary,
            error=error,
            error_type=error_type,
            log_stdout_path=log_stdout_path,
            log_stderr_path=log_stderr_path,
            result_path=result_path,
            output_path=output_path,
            inputs_path=inputs_path,
            materialized_prompt_path=materialized_prompt_path,
            artifacts=artifacts,
            dataflow_error=dataflow_error,
            blocked_by=blocked_by,
            blocked_reason=blocked_reason,
            blocked_at=blocked_at,
            watch_satisfied_at=watch_satisfied_at,
            last_checked_at=last_checked_at,
            last_check_result=last_check_result,
            condition=condition,
            tool_finished_at=tool_finished_at,
            tool_error=tool_error,
            extra_events=extra_events,
        )

    def _launch_ready(self, wf: Workflow) -> None:
        slots = available_agent_slots(wf, self._concurrency)
        for task in wf.tasks.values():
            if task.status is not TaskStatus.READY:
                continue
            if task.task_kind == "watcher":
                attempt_id = task.attempt_count + 1
                audit_path = self._policy_audit_path(wf, task, attempt_id)
                task.current_attempt = attempt_id
                self._begin_attempt(
                    wf,
                    task,
                    message="watcher started",
                    watch_started=True,
                    policy_audit_path=str(audit_path),
                    extra_events=[(
                        ev.WATCHER_STARTED,
                        "watcher started",
                        {
                            "watcher_type": task.watcher_type,
                            "status": TaskStatus.RUNNING.value,
                        },
                    )],
                )
                request, metadata, decision, audit_path = self._evaluate_task_policy(
                    wf,
                    task,
                    attempt_id,
                )
                if not decision["allowed"]:
                    self._handle_policy_block_for_task(
                        wf,
                        task,
                        request,
                        metadata,
                        decision,
                        audit_path,
                    )
                continue
            if task.task_kind == "tool":
                attempt_id = task.attempt_count + 1
                audit_path = self._policy_audit_path(wf, task, attempt_id)
                task.current_attempt = attempt_id
                self._begin_attempt(
                    wf,
                    task,
                    message="tool started",
                    tool_started=True,
                    policy_audit_path=str(audit_path),
                    extra_events=[(
                        ev.TOOL_STARTED,
                        "tool started",
                        {
                            "tool_type": task.tool_type,
                            "status": TaskStatus.RUNNING.value,
                        },
                    )],
                )
                request, metadata, decision, audit_path = self._evaluate_task_policy(
                    wf,
                    task,
                    attempt_id,
                )
                if not decision["allowed"]:
                    self._handle_policy_block_for_task(
                        wf,
                        task,
                        request,
                        metadata,
                        decision,
                        audit_path,
                    )
                    continue
                self._run_tool_task(wf, task, request, metadata, decision, audit_path)
                continue
            if slots <= 0:
                break
            if task.inputs:
                if self._materialize_inputs_for_task(wf, task):
                    continue
            previous_attempt = task.current_attempt
            next_attempt = task.attempt_count + 1
            self._log_scheduler_task(
                task,
                "launch_request",
                metadata={"attempt_id": next_attempt},
            )
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
            try:
                self._begin_attempt(
                    wf,
                    task,
                    message="task started",
                    pid=handle.pid,
                    run_handle=handle,
                )
                self._log_scheduler_task(task, "handle_registered", handle=handle)
            except Exception:
                # _begin_attempt failed before saving RUNNING; cancel the
                # orphaned subprocess so it doesn't run untracked.
                try:
                    self._runner.cancel(handle)
                except Exception:
                    pass
                raise
            slots -= 1
            immediate_result = self._runner.poll(handle)
            if immediate_result is not None:
                self._log_scheduler_task(
                    task,
                    "immediate_poll_terminal",
                    handle=handle,
                    metadata={"terminal_status": immediate_result.status.value},
                )
                self._handle_terminal_result(wf, task, handle, immediate_result)

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
        original_wf = wf
        original_task = task
        with self._store.locked():
            live_wf = self._store.load_workflow(wf.workflow_id)
            if live_wf is None:
                raise RuntimeError(f"workflow not found during input materialization: {wf.workflow_id}")
            live_task = live_wf.tasks.get(task.task_id)
            if live_task is None:
                raise RuntimeError(f"task not found during input materialization: {task.task_id}")
            live_task.inputs_path = str(inputs_path)
            live_task.materialized_prompt_path = str(prompt_path)
            live_task.dataflow_error = None
            self._store.save_workflow(live_wf)
            for event_type, event_message, event_metadata in extra_events:
                self._events.emit(
                    event_type,
                    live_wf.workflow_id,
                    task_id=live_task.task_id,
                    agent_id=self._agent_id,
                    message=event_message,
                    metadata=dict(event_metadata),
                )
            sync_workflow_view(original_wf, live_wf)
            sync_task_view(original_task, live_task)
        return False

    def _run_tool_task(
        self,
        wf: Workflow,
        task: Task,
        request: CapabilityRequest,
        metadata: CapabilityMetadata,
        decision: dict[str, Any],
        audit_path: Path,
    ) -> None:
        return self._tool_task_runner.run(wf, task, request, metadata, decision, audit_path)

    # ------------------------------------------------------------------
    # Atomic commit helper
    # ------------------------------------------------------------------

    def _commit(
        self,
        wf: Workflow,
        task: Task,
        new_status: TaskStatus,
        *,
        event: Optional[str],
        message: str = "",
        started: bool = False,
        finished: bool = False,
        attempt_id: Optional[int] = None,
        started_at: Any = _UNSET,
        finished_at: Any = _UNSET,
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
        skip_reason: Any = _UNSET,
        condition_result: Any = _UNSET,
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
        original_wf = wf
        original_task = task
        with self._store.locked():
            task_id = task.task_id
            wf = self._store.load_workflow(wf.workflow_id)
            if wf is None:
                raise RuntimeError(f"workflow not found during commit: {task.workflow_id}")
            task = wf.tasks.get(task_id)
            if task is None:
                raise RuntimeError(f"task not found during commit: {task_id}")
            active_attempt_id = attempt_id
            if active_attempt_id is None and task.current_attempt > 0:
                active_attempt_id = task.current_attempt
            task.status = new_status
            if started:
                task.started_at = self._clock.now_iso()
            elif started_at is not _UNSET:
                task.started_at = started_at
            if finished:
                task.finished_at = self._clock.now_iso()
            elif finished_at is not _UNSET:
                task.finished_at = finished_at
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
            if skip_reason is not _UNSET:
                task.skip_reason = skip_reason
            if condition_result is not _UNSET:
                task.condition_result = (
                    dict(condition_result)
                    if condition_result is not None else None
                )
            if watch_started and task.watch_started_at is None:
                task.watch_started_at = self._clock.now_iso()
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
                task.tool_started_at = self._clock.now_iso()
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
            if task.skip_reason:
                metadata["skip_reason"] = task.skip_reason
            if task.condition_result:
                metadata["condition_result"] = dict(task.condition_result)
            if event is not None:
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
            sync_workflow_view(original_wf, wf)
            sync_task_view(original_task, task)

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_workflow(self, workflow_id: str) -> bool:
        wf = self._store.load_workflow(workflow_id)
        if wf is None:
            return False
        if wf.status is WorkflowStatus.CANCELLED:
            raise WorkflowSpecError(f"workflow already cancelled: {workflow_id}")
        if wf.is_terminal():
            raise WorkflowSpecError(
                f"workflow already terminal: {workflow_id} ({wf.status.value})"
            )
        cancel_workflow_on_disk(
            self._store,
            workflow_id,
            agent_id=self._agent_id,
        )
        self._approval_store.expire_requests_for_workflow(
            workflow_id,
            reason="workflow_cancelled",
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
    return compute_ancestor_labels(depends_on_by_label)
