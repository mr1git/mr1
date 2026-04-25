"""
Workflow authoring and compilation helpers for Phase 5.

This module sits above the scheduler/runtime. It decides when a user
request should become a workflow, asks a compiler model for JSON, validates
and fixes the result, renders previews, and submits or rewrites workflows
without changing scheduler behavior.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from mr1.capabilities import CapabilityRegistry
from mr1.dataflow import TaskInputSpec
from mr1.scheduler import WorkflowSpecError, validate_spec
from mr1.tools import ToolRegistry, default_tool_registry
from mr1.watchers import WatcherRegistry, default_watcher_registry
from mr1.workflow_schema import (
    WorkflowSchemaRegistry,
    default_workflow_schema_registry,
)
from mr1.workflow_events import WorkflowEventLog
from mr1.workflow_models import (
    Provenance,
    Task,
    TaskStatus,
    Workflow,
    WorkflowStatus,
    new_task_id,
)
from mr1.workflow_store import WorkflowStore


MODIFIABLE_TASK_STATUSES = frozenset({
    TaskStatus.CREATED,
    TaskStatus.WAITING,
    TaskStatus.READY,
})

CONFIRM_PREVIEW_INPUTS = frozenset({
    "yes",
    "confirm",
    "run it",
    "execute",
    "submit",
})

CANCEL_PREVIEW_INPUTS = frozenset({
    "cancel",
    "stop",
    "nevermind",
})

JSON_PREVIEW_PATTERNS = (
    "show json",
    "workflow json",
    "show the json",
    "raw json",
    "json preview",
)

WORKFLOW_ID_PATTERN = re.compile(r"\bwf-\d{8}T\d{6}-[0-9a-f]{6}\b")

_DEFAULT_COMPILER_SYSTEM_PROMPT = """\
You compile natural language requests into MR1 workflow JSON.

Output rules:
- Return exactly one JSON object and nothing else.
- Do not use markdown fences.
- Use the provided Workflow Schema metadata for exact field shapes.
- Use the provided Capability metadata for available tools, watchers, agents, config shapes, and output references.
- Do not guess field shapes.
- Generate JSON only.

Allowed values:
- task_kind: "agent" | "tool" | "watcher"
- agent_type: "kazi" only

Workflow rules:
- Use tools whenever a step is deterministic.
- Use agents only for reasoning, synthesis, or generation.
- Use depends_on only with labels that exist in the same workflow.
- ALWAYS use inputs to pass upstream data.
- NEVER inline upstream outputs into downstream prompts.
- Agent prompts should describe reasoning work only.
- Tool configs must contain the real deterministic action.
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if not part or part == "json":
                continue
            raw = part.removeprefix("json").strip()
            break
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise WorkflowSpecError("workflow compiler must return a JSON object")
    return data


def _spec_task_dict(raw: dict[str, Any]) -> dict[str, Any]:
    task_kind = raw.get("task_kind", "agent")
    task = {
        "label": raw["label"],
        "title": raw.get("title", raw["label"]),
        "task_kind": task_kind,
    }
    depends_on = list(raw.get("depends_on") or [])
    if depends_on:
        task["depends_on"] = depends_on
    inputs = raw.get("inputs") or []
    if inputs:
        task["inputs"] = [
            {
                "name": item["name"],
                "from": item["from"],
            }
            for item in inputs
        ]
    if raw.get("timeout_s") is not None:
        task["timeout_s"] = raw["timeout_s"]
    if task_kind == "agent":
        task["agent_type"] = raw.get("agent_type", "kazi")
        task["prompt"] = raw.get("prompt", "")
    elif task_kind == "tool":
        task["tool_type"] = raw.get("tool_type")
        task["tool_config"] = dict(raw.get("tool_config", {}))
    elif task_kind == "watcher":
        task["watcher_type"] = raw.get("watcher_type")
        task["watch_config"] = dict(raw.get("watch_config", {}))
        if raw.get("condition") is not None:
            task["condition"] = dict(raw["condition"])
    return task


def workflow_to_spec(workflow: Workflow) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for label, task_id in workflow.label_to_task_id.items():
        task = workflow.tasks.get(task_id)
        if task is None:
            continue
        task_spec: dict[str, Any] = {
            "label": task.label,
            "title": task.title,
            "task_kind": task.task_kind,
        }
        dep_labels = [
            _label_for_task_id(workflow, parent_id)
            for parent_id in task.depends_on
        ]
        dep_labels = [label for label in dep_labels if label]
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
        tasks.append(task_spec)
    return {
        "title": workflow.title,
        "tasks": tasks,
    }


def _label_for_task_id(workflow: Workflow, task_id: str) -> Optional[str]:
    for label, candidate in workflow.label_to_task_id.items():
        if candidate == task_id:
            return label
    return None


def _task_spec_for_comparison(workflow: Workflow, task: Task) -> dict[str, Any]:
    raw = workflow_to_spec(workflow)
    for item in raw["tasks"]:
        if item["label"] == task.label:
            return item
    raise KeyError(task.label)


@dataclass(frozen=True)
class PendingWorkflowDraft:
    original_request: str
    mode: str
    spec: dict[str, Any]
    target_workflow_id: Optional[str] = None
    preview_text: str = ""
    complexity: str = "complex"
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_request": self.original_request,
            "mode": self.mode,
            "spec": self.spec,
            "target_workflow_id": self.target_workflow_id,
            "preview_text": self.preview_text,
            "complexity": self.complexity,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingWorkflowDraft":
        return cls(
            original_request=data["original_request"],
            mode=data["mode"],
            spec=dict(data["spec"]),
            target_workflow_id=data.get("target_workflow_id"),
            preview_text=data.get("preview_text", ""),
            complexity=data.get("complexity", "complex"),
            created_at=data.get("created_at", _now_iso()),
        )


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    spec: Optional[dict[str, Any]]
    error: Optional[str] = None
    corrected: bool = False


@dataclass(frozen=True)
class SubmissionResult:
    workflow_id: str
    message: str
    in_place: bool = False


CompilerFn = Callable[[str, str], str]


class WorkflowAuthoringService:
    def __init__(
        self,
        scheduler: Any,
        store: WorkflowStore,
        *,
        compiler: Optional[CompilerFn] = None,
        tool_registry: Optional[ToolRegistry] = None,
        watcher_registry: Optional[WatcherRegistry] = None,
        capability_registry: Optional[CapabilityRegistry] = None,
        workflow_schema_registry: Optional[WorkflowSchemaRegistry] = None,
        compiler_system_prompt: str = _DEFAULT_COMPILER_SYSTEM_PROMPT,
    ):
        self._scheduler = scheduler
        self._store = store
        self._compiler = compiler
        self._tool_registry = tool_registry or default_tool_registry()
        self._watcher_registry = watcher_registry or default_watcher_registry()
        self._capability_registry = capability_registry or CapabilityRegistry(
            tool_registry=self._tool_registry,
            watcher_registry=self._watcher_registry,
        )
        self._workflow_schema_registry = (
            workflow_schema_registry or default_workflow_schema_registry()
        )
        self._compiler_system_prompt_base = compiler_system_prompt

    def classify_request(
        self,
        user_input: str,
        pending_draft: Optional[PendingWorkflowDraft] = None,
    ) -> str:
        normalized = _normalize_text(user_input)
        if pending_draft is not None:
            if normalized in CONFIRM_PREVIEW_INPUTS:
                return "confirm_preview"
            if normalized in CANCEL_PREVIEW_INPUTS:
                return "cancel_preview"
            if any(pattern in normalized for pattern in JSON_PREVIEW_PATTERNS):
                return "show_json_preview"
            return "modify_workflow"

        workflow_id = self.extract_workflow_id(user_input)
        if workflow_id and any(token in normalized for token in ("modify", "update", "change", "edit", "add", "remove")):
            return "modify_workflow"
        if workflow_id:
            return "modify_workflow"

        tool_markers = (
            "read file",
            "write file",
            "shell command",
            "run ",
            "check ",
            "summarize",
            "save ",
            "create file",
            "python version",
            "workflow",
            "depends on",
            "then ",
        )
        action_words = re.findall(
            r"\b(read|write|run|check|summarize|create|generate|inspect|list|save|wait|trigger|search)\b",
            normalized,
        )
        if any(marker in normalized for marker in tool_markers):
            return "create_workflow"
        if len(action_words) >= 2 and any(joiner in normalized for joiner in (" and ", ",", " then ")):
            return "create_workflow"
        if re.fullmatch(r"what is [\d\s+*/().-?]+", normalized):
            return "direct_answer"
        if normalized.startswith(("hi", "hello", "hey", "thanks", "thank you")):
            return "direct_answer"
        if normalized.endswith("?") and not any(token in normalized for token in ("file", "command", "workflow", "run", "write", "read")):
            return "direct_answer"
        return "direct_answer"

    def extract_workflow_id(self, user_input: str) -> Optional[str]:
        match = WORKFLOW_ID_PATTERN.search(user_input)
        return match.group(0) if match else None

    def coerce_pending_draft(self, raw: Optional[dict[str, Any]]) -> Optional[PendingWorkflowDraft]:
        if not raw:
            return None
        try:
            return PendingWorkflowDraft.from_dict(raw)
        except KeyError:
            return None

    def generate_spec(
        self,
        user_input: str,
        *,
        mode: str = "create",
        baseline_spec: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        compiler = self._require_compiler()
        prompt = self._build_generation_prompt(
            user_input,
            mode=mode,
            baseline_spec=baseline_spec,
        )
        raw = compiler(self._build_compiler_system_prompt(), prompt)
        return _extract_json_object(raw)

    def validate_and_maybe_fix(self, spec: dict[str, Any]) -> ValidationResult:
        try:
            validate_spec(
                spec,
                watcher_registry=self._watcher_registry,
                tool_registry=self._tool_registry,
            )
            return ValidationResult(ok=True, spec=spec)
        except WorkflowSpecError as exc:
            first_error = str(exc)

        compiler = self._require_compiler()
        fix_prompt = self._build_fix_prompt(spec, first_error)
        try:
            fixed = _extract_json_object(
                compiler(self._build_compiler_system_prompt(), fix_prompt)
            )
        except (json.JSONDecodeError, WorkflowSpecError, ValueError) as exc:
            return ValidationResult(
                ok=False,
                spec=None,
                error=str(exc),
                corrected=True,
            )

        try:
            validate_spec(
                fixed,
                watcher_registry=self._watcher_registry,
                tool_registry=self._tool_registry,
            )
        except WorkflowSpecError as exc:
            return ValidationResult(
                ok=False,
                spec=None,
                error=str(exc),
                corrected=True,
            )
        return ValidationResult(ok=True, spec=fixed, corrected=True)

    def preview(self, spec: dict[str, Any]) -> tuple[str, str]:
        tasks = list(spec.get("tasks") or [])
        complexity = "simple" if self._is_simple_workflow(spec) else "complex"
        lines = ["This workflow will:"]
        for idx, task in enumerate(tasks, start=1):
            lines.append(f"{idx}. {self._describe_task(task)}")
        dependency_lines = self._describe_dependencies(tasks)
        if dependency_lines:
            lines.append("")
            lines.append("Dependencies:")
            lines.extend(dependency_lines)
        dataflow_lines = self._describe_dataflow(tasks)
        if dataflow_lines:
            lines.append("")
            lines.append("Dataflow:")
            lines.extend(dataflow_lines)
        if complexity == "complex":
            lines.append("")
            lines.append("Reply with `yes`, `confirm`, `run it`, `execute`, or `submit` to run it.")
            lines.append("Reply with `show json` to inspect the exact workflow JSON.")
        return "\n".join(lines), complexity

    def submit(
        self,
        spec: dict[str, Any],
        *,
        created_by: Provenance,
        target_workflow_id: Optional[str] = None,
    ) -> SubmissionResult:
        if target_workflow_id:
            workflow = self._store.load_workflow(target_workflow_id)
            if workflow is not None:
                rewritten = self._rewrite_workflow_in_place(
                    workflow,
                    spec,
                    created_by=created_by,
                )
                if rewritten is not None:
                    return SubmissionResult(
                        workflow_id=rewritten.workflow_id,
                        message=self._build_submission_message(rewritten, in_place=True),
                        in_place=True,
                    )

        workflow_id = self._scheduler.submit_workflow(spec, created_by)
        workflow = self._store.load_workflow(workflow_id)
        if workflow is None:
            raise RuntimeError(f"submitted workflow not found: {workflow_id}")
        return SubmissionResult(
            workflow_id=workflow_id,
            message=self._build_submission_message(workflow, in_place=False),
            in_place=False,
        )

    def clarify_message(
        self,
        error: str,
        *,
        mode: str = "create",
        target_workflow_id: Optional[str] = None,
    ) -> str:
        if mode == "modify" and target_workflow_id is None:
            return "I need a workflow id to modify. Reference the workflow as `wf-...` or reply to a pending preview."
        return f"I couldn't produce a valid workflow yet: {error}"

    def _require_compiler(self) -> CompilerFn:
        if self._compiler is None:
            raise RuntimeError("workflow compiler is not configured")
        return self._compiler

    def _build_compiler_system_prompt(self) -> str:
        workflow_schema = _json_dumps(self._workflow_schema_registry.describe_all())
        capabilities = _json_dumps(self._capability_registry.describe_all())
        return "\n\n".join([
            self._compiler_system_prompt_base.strip(),
            "Available workflow schema:",
            workflow_schema,
            "Available capabilities:",
            capabilities,
        ])

    def _build_generation_prompt(
        self,
        user_input: str,
        *,
        mode: str,
        baseline_spec: Optional[dict[str, Any]],
    ) -> str:
        lines = [
            f"Mode: {mode}",
            "Return JSON only.",
            f"User request:\n{user_input.strip()}",
        ]
        if baseline_spec is not None:
            lines.append("Existing workflow JSON:")
            lines.append(_json_dumps(baseline_spec))
            lines.append("Preserve labels when possible.")
        return "\n\n".join(lines)

    def _build_fix_prompt(self, spec: dict[str, Any], error: str) -> str:
        return "\n\n".join([
            "Fix only schema/config errors in the workflow JSON below.",
            "Preserve user intent.",
            "Return JSON only.",
            f"Validation error:\n{error}",
            "Available workflow schema:",
            _json_dumps(self._workflow_schema_registry.describe_all()),
            "Available capabilities:",
            _json_dumps(self._capability_registry.describe_all()),
            "Current JSON:",
            _json_dumps(spec),
        ])

    def _is_simple_workflow(self, spec: dict[str, Any]) -> bool:
        tasks = list(spec.get("tasks") or [])
        if not tasks or len(tasks) > 3:
            return False
        if any(task.get("task_kind") == "watcher" for task in tasks):
            return False
        if any(task.get("tool_type") == "write_file" for task in tasks):
            return False
        labels = {task["label"] for task in tasks if isinstance(task.get("label"), str)}
        outdegree = {label: 0 for label in labels}
        for task in tasks:
            for dep in task.get("depends_on") or []:
                if dep in outdegree:
                    outdegree[dep] += 1
        sinks = [label for label, degree in outdegree.items() if degree == 0]
        return len(sinks) == 1

    def _describe_task(self, task: dict[str, Any]) -> str:
        task_kind = task.get("task_kind", "agent")
        title = task.get("title") or task.get("label", "task")
        if task_kind == "tool":
            tool_type = task.get("tool_type")
            config = dict(task.get("tool_config", {}))
            if tool_type == "read_file":
                return f"Read a file at {config.get('path', '?')}."
            if tool_type == "write_file":
                return f"Write a file at {config.get('path', '?')}."
            if tool_type == "shell_command":
                argv = config.get("argv") or []
                rendered = " ".join(argv) if isinstance(argv, list) else "command"
                return f"Run `{rendered}`."
            return f"Run tool task `{title}`."
        if task_kind == "watcher":
            return f"Wait on watcher `{task.get('watcher_type', title)}`."
        return f"Use an agent to {task.get('prompt', title)}"

    def _describe_dependencies(self, tasks: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for task in tasks:
            deps = list(task.get("depends_on") or [])
            if not deps:
                continue
            lines.append(f"- {task['label']} waits for: {', '.join(deps)}")
        return lines

    def _describe_dataflow(self, tasks: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for task in tasks:
            for item in task.get("inputs") or []:
                lines.append(f"- {task['label']}.{item['name']} <- {item['from']}")
        return lines

    def _build_submission_message(self, workflow: Workflow, *, in_place: bool) -> str:
        prefix = "updated workflow in place" if in_place else "submitted workflow"
        lines = [
            f"{prefix}: {workflow.workflow_id}",
            f"/workflow {workflow.workflow_id}",
            "/jobs",
            f"/events {workflow.workflow_id}",
            f"/artifacts {workflow.workflow_id}",
        ]
        for label, task_id in workflow.label_to_task_id.items():
            lines.append(f"/result {task_id}  # {label}")
            lines.append(f"/inputs {task_id}  # {label}")
        return "\n".join(lines)

    def _rewrite_workflow_in_place(
        self,
        workflow: Workflow,
        spec: dict[str, Any],
        *,
        created_by: Provenance,
    ) -> Optional[Workflow]:
        if workflow.is_terminal():
            return None

        new_tasks_raw = [_spec_task_dict(raw) for raw in list(spec.get("tasks") or [])]
        old_by_label = {
            label: workflow.tasks[task_id]
            for label, task_id in workflow.label_to_task_id.items()
            if task_id in workflow.tasks
        }
        new_labels = {task["label"] for task in new_tasks_raw}

        for label, task in old_by_label.items():
            if label in new_labels:
                continue
            if task.status not in MODIFIABLE_TASK_STATUSES:
                return None

        new_workflow = Workflow(
            workflow_id=workflow.workflow_id,
            title=spec.get("title") or workflow.title,
            status=workflow.status,
            created_by=workflow.created_by or created_by,
            created_at=workflow.created_at,
            finished_at=workflow.finished_at,
        )

        label_to_new_task: dict[str, Task] = {}
        for raw in new_tasks_raw:
            label = raw["label"]
            old_task = old_by_label.get(label)
            old_raw = None
            if old_task is not None:
                old_raw = _task_spec_for_comparison(workflow, old_task)
                if old_task.status not in MODIFIABLE_TASK_STATUSES and old_raw != raw:
                    return None

            if old_task is not None and old_raw == raw and old_task.status not in MODIFIABLE_TASK_STATUSES:
                task = Task.from_dict(old_task.to_dict())
            else:
                reuse_task_id = old_task is not None and old_raw == raw
                task = Task(
                    task_id=old_task.task_id if reuse_task_id else new_task_id(),
                    workflow_id=workflow.workflow_id,
                    label=label,
                    title=raw.get("title", label),
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
                    inputs=[],
                    status=TaskStatus.CREATED,
                )
            label_to_new_task[label] = task

        for raw in new_tasks_raw:
            label = raw["label"]
            task = label_to_new_task[label]
            task.workflow_id = workflow.workflow_id
            task.label = label
            task.title = raw.get("title", label)
            task.task_kind = raw.get("task_kind", "agent")
            task.agent_type = raw.get("agent_type", "kazi") if task.task_kind == "agent" else None
            task.prompt = raw.get("prompt", "") if task.task_kind == "agent" else ""
            task.watcher_type = raw.get("watcher_type")
            task.watch_config = dict(raw.get("watch_config", {}))
            task.tool_type = raw.get("tool_type")
            task.tool_config = dict(raw.get("tool_config", {}))
            task.condition = dict(raw["condition"]) if raw.get("condition") is not None else None
            task.timeout_s = raw.get("timeout_s")
            task.inputs = []
            for item in raw.get("inputs") or []:
                task.inputs.append(TaskInputSpec.from_dict(item))
            task.depends_on = [
                label_to_new_task[label].task_id
                for label in (raw.get("depends_on") or [])
            ]
            new_workflow.tasks[task.task_id] = task
            new_workflow.label_to_task_id[label] = task.task_id

        if any(task.status is TaskStatus.RUNNING for task in new_workflow.tasks.values()):
            new_workflow.status = WorkflowStatus.RUNNING
            new_workflow.finished_at = None
        else:
            new_workflow.status = WorkflowStatus.PENDING
            new_workflow.finished_at = None

        event_log = WorkflowEventLog(self._store, default_agent_id=created_by.id)
        with self._store.locked():
            self._store.save_workflow(new_workflow)
            event_log.workflow_submitted(
                new_workflow.workflow_id,
                agent_id=created_by.id,
                message=f"updated in place '{new_workflow.title}' with {len(new_workflow.tasks)} task(s)",
                metadata={
                    "task_count": len(new_workflow.tasks),
                    "updated_in_place": True,
                },
            )
        return new_workflow
