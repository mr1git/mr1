"""
Deterministic workflow dataflow helpers and models.

Phase 3 adds:
  * normalized task outputs,
  * registered artifacts,
  * explicit input references between workflow tasks,
  * reproducible prompt materialization.

This module intentionally avoids scheduler/store orchestration logic so
the scheduler can stay focused on task lifecycle transitions.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mr1.workflow_store import WorkflowStore


LOG_INPUT_LIMIT = 4096
_ARTIFACT_KINDS = frozenset({
    "text",
    "json",
    "csv",
    "image",
    "pdf",
    "log",
    "binary",
    "directory",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def new_artifact_id() -> str:
    return f"art-{_ts_compact()}-{uuid.uuid4().hex[:6]}"


class DataflowError(ValueError):
    """Raised for deterministic dataflow validation or normalization errors."""


@dataclass(frozen=True)
class ParsedTaskReference:
    label: str
    root: str
    path: tuple[str, ...] = ()


@dataclass
class Artifact:
    artifact_id: str
    workflow_id: str
    task_id: str
    name: str
    kind: str
    path: str
    created_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        kind = data["kind"]
        if kind not in _ARTIFACT_KINDS:
            raise DataflowError(f"unsupported artifact kind '{kind}'")
        return cls(
            artifact_id=data["artifact_id"],
            workflow_id=data["workflow_id"],
            task_id=data["task_id"],
            name=data["name"],
            kind=kind,
            path=data["path"],
            created_at=data.get("created_at", _now_iso()),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TaskOutput:
    task_id: str
    workflow_id: str
    status: str
    summary: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskOutput":
        return cls(
            task_id=data["task_id"],
            workflow_id=data["workflow_id"],
            status=data["status"],
            summary=data.get("summary", ""),
            text=data.get("text", ""),
            data=dict(data.get("data", {})),
            metrics=dict(data.get("metrics", {})),
            artifacts=[
                Artifact.from_dict(item)
                for item in data.get("artifacts", [])
            ],
            created_at=data.get("created_at", _now_iso()),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TaskInputSpec:
    name: str
    from_ref: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "from": self.from_ref}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskInputSpec":
        return cls(name=data["name"], from_ref=data["from"])


@dataclass
class ResolvedTaskInput:
    name: str
    source: str
    resolved_task_id: Optional[str]
    resolved_type: str
    value: Any = None
    artifact_path: Optional[str] = None
    materialized_at: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResolvedTaskInput":
        return cls(
            name=data["name"],
            source=data["source"],
            resolved_task_id=data.get("resolved_task_id"),
            resolved_type=data["resolved_type"],
            value=data.get("value"),
            artifact_path=data.get("artifact_path"),
            materialized_at=data.get("materialized_at", _now_iso()),
            metadata=dict(data.get("metadata", {})),
        )


def parse_input_reference(source: str) -> ParsedTaskReference:
    if not isinstance(source, str) or not source:
        raise DataflowError("input reference must be a non-empty string")
    parts = source.split(".")
    if len(parts) < 2:
        raise DataflowError(
            "input reference must start with '<label>.<root>'"
        )
    label, root = parts[0], parts[1]
    if not label:
        raise DataflowError("input reference missing source task label")
    if root not in {"result", "stdout", "stderr", "artifact", "status", "condition_result", "skip_reason"}:
        raise DataflowError(f"unsupported input reference root '{root}'")
    path = tuple(parts[2:])
    if root in {"stdout", "stderr", "status", "condition_result", "skip_reason"} and path:
        raise DataflowError(f"unsupported input reference path '{source}'")
    if root == "artifact":
        if len(path) != 1 or not path[0]:
            raise DataflowError(
                "artifact references must look like '<label>.artifact.<name>'"
            )
    if root == "result":
        if not path:
            return ParsedTaskReference(label=label, root=root, path=path)
        head = path[0]
        if head not in {"summary", "text", "data", "metrics"}:
            raise DataflowError(f"unsupported input reference path '{source}'")
        if head in {"summary", "text"} and len(path) != 1:
            raise DataflowError(f"unsupported input reference path '{source}'")
    return ParsedTaskReference(label=label, root=root, path=path)


def validate_artifact_uniqueness(artifacts: list[Artifact]) -> None:
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.name in seen:
            raise DataflowError(
                f"duplicate artifact name '{artifact.name}' in task '{artifact.task_id}'"
            )
        seen.add(artifact.name)


def normalize_artifacts(
    raw_artifacts: Any,
    *,
    workflow_id: str,
    task_id: str,
) -> list[Artifact]:
    if raw_artifacts in (None, []):
        return []
    if not isinstance(raw_artifacts, list):
        raise DataflowError("artifacts payload must be a list")
    artifacts: list[Artifact] = []
    for idx, raw in enumerate(raw_artifacts):
        if isinstance(raw, Artifact):
            data = raw.to_dict()
        elif isinstance(raw, dict):
            data = dict(raw)
        else:
            raise DataflowError(f"artifact[{idx}] must be an object")
        data.setdefault("artifact_id", new_artifact_id())
        data.setdefault("workflow_id", workflow_id)
        data.setdefault("task_id", task_id)
        artifacts.append(Artifact.from_dict(data))
    validate_artifact_uniqueness(artifacts)
    return artifacts


def register_artifacts(
    task: Any,
    store: "WorkflowStore",
    raw_artifacts: Any,
) -> list[Artifact]:
    if raw_artifacts is None:
        existing = getattr(task, "artifacts", []) or []
        artifacts = [
            item if isinstance(item, Artifact) else Artifact.from_dict(item)
            for item in existing
        ]
        validate_artifact_uniqueness(artifacts)
        task.artifacts = artifacts
        return artifacts
    artifacts = normalize_artifacts(
        raw_artifacts,
        workflow_id=task.workflow_id,
        task_id=task.task_id,
    )
    if not artifacts:
        task.artifacts = []
        return []
    store.task_artifacts_dir(task.workflow_id, task.task_id)
    task.artifacts = artifacts
    return artifacts


def build_agent_task_output(task: Any, result_payload: Optional[dict[str, Any]]) -> TaskOutput:
    payload = dict(result_payload or {})
    artifacts = normalize_artifacts(
        payload.get("artifacts") if "artifacts" in payload else getattr(task, "artifacts", []),
        workflow_id=task.workflow_id,
        task_id=task.task_id,
    )
    summary = ""
    if isinstance(payload.get("summary"), str):
        summary = payload["summary"]
    elif isinstance(getattr(task, "result_summary", None), str):
        summary = task.result_summary or ""
    text = payload.get("text") if isinstance(payload.get("text"), str) else summary
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metadata = {
        "task_kind": getattr(task, "task_kind", None),
        "agent_type": getattr(task, "agent_type", None),
        "exit_code": getattr(task, "exit_code", None),
    }
    for key in ("error", "error_type", "kazi_status", "pid"):
        if key in payload:
            metadata[key] = payload.get(key)
    return TaskOutput(
        task_id=task.task_id,
        workflow_id=task.workflow_id,
        status=getattr(getattr(task, "status", None), "value", getattr(task, "status", "succeeded")),
        summary=summary,
        text=text,
        data=data,
        metrics=metrics,
        artifacts=artifacts,
        metadata=metadata,
    )


def build_watcher_task_output(task: Any) -> TaskOutput:
    message = ""
    if isinstance(getattr(task, "last_check_result", None), dict):
        message = str(task.last_check_result.get("message", ""))
    return TaskOutput(
        task_id=task.task_id,
        workflow_id=task.workflow_id,
        status=getattr(getattr(task, "status", None), "value", getattr(task, "status", "succeeded")),
        summary=message,
        text=message,
        data=dict(getattr(task, "last_check_result", {}) or {}),
        metrics={},
        artifacts=[
            Artifact.from_dict(item.to_dict())
            if hasattr(item, "to_dict") else Artifact.from_dict(item)
            for item in getattr(task, "artifacts", [])
        ],
        metadata={
            "task_kind": getattr(task, "task_kind", None),
            "watcher_type": getattr(task, "watcher_type", None),
        },
    )


def build_tool_task_output(task: Any, tool_result: Any) -> TaskOutput:
    artifacts = normalize_artifacts(
        getattr(tool_result, "artifacts", []),
        workflow_id=task.workflow_id,
        task_id=task.task_id,
    )
    metadata = {
        "task_kind": getattr(task, "task_kind", None),
        "tool_type": getattr(task, "tool_type", None),
    }
    if getattr(tool_result, "error", None):
        metadata["error"] = tool_result.error
    metadata.update(dict(getattr(tool_result, "metadata", {}) or {}))
    return TaskOutput(
        task_id=task.task_id,
        workflow_id=task.workflow_id,
        status=getattr(getattr(task, "status", None), "value", getattr(task, "status", getattr(tool_result, "state", "succeeded"))),
        summary=getattr(tool_result, "summary", "") or "",
        text=getattr(tool_result, "text", "") or "",
        data=dict(getattr(tool_result, "data", {}) or {}),
        metrics=dict(getattr(tool_result, "metrics", {}) or {}),
        artifacts=artifacts,
        metadata=metadata,
    )


def resolve_task_input(
    workflow: Any,
    input_spec: TaskInputSpec,
    store: "WorkflowStore",
    *,
    log_limit: int = LOG_INPUT_LIMIT,
) -> ResolvedTaskInput:
    parsed = parse_input_reference(input_spec.from_ref)
    resolved_task_id = workflow.label_to_task_id.get(parsed.label)
    if not resolved_task_id:
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=None,
            resolved_type="missing",
            metadata={"reason": "unknown_label"},
        )
    source_task = workflow.tasks.get(resolved_task_id)
    if source_task is None:
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=resolved_task_id,
            resolved_type="missing",
            metadata={"reason": "missing_task"},
        )
    if parsed.root in {"status", "condition_result", "skip_reason"}:
        return _resolve_from_task_field(source_task, input_spec, parsed.root)
    if parsed.root == "result":
        return _resolve_from_output(source_task, input_spec, parsed, store)
    if parsed.root in {"stdout", "stderr"}:
        return _resolve_from_log(
            source_task,
            input_spec,
            parsed.root,
            store,
            log_limit=log_limit,
        )
    return _resolve_from_artifact(source_task, input_spec, parsed, store)


def _resolve_from_task_field(
    task: Any,
    input_spec: TaskInputSpec,
    field_name: str,
) -> ResolvedTaskInput:
    value = getattr(task, field_name, None)
    if field_name == "status":
        value = getattr(value, "value", value)
    nullable_fields = {"condition_result", "skip_reason"}
    if value is None and field_name not in nullable_fields:
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=task.task_id,
            resolved_type="missing",
            metadata={"reason": f"missing_{field_name}"},
        )
    resolved_type = "text" if isinstance(value, str) else "json"
    if isinstance(value, dict):
        value = dict(value)
    return ResolvedTaskInput(
        name=input_spec.name,
        source=input_spec.from_ref,
        resolved_task_id=task.task_id,
        resolved_type=resolved_type,
        value=value,
    )


def materialize_task_inputs(
    workflow: Any,
    task: Any,
    store: "WorkflowStore",
    *,
    log_limit: int = LOG_INPUT_LIMIT,
) -> list[ResolvedTaskInput]:
    resolved: list[ResolvedTaskInput] = []
    for raw in getattr(task, "inputs", []) or []:
        spec = raw if isinstance(raw, TaskInputSpec) else TaskInputSpec.from_dict(raw)
        resolved.append(resolve_task_input(workflow, spec, store, log_limit=log_limit))
    return resolved


def build_materialized_prompt(raw_prompt: str, resolved_inputs: list[ResolvedTaskInput]) -> str:
    lines = [raw_prompt or "", "", "[Workflow Inputs]", ""]
    for item in resolved_inputs:
        lines.append(f"Input: {item.name}")
        lines.append(f"Source: {item.source}")
        lines.append(f"Type: {item.resolved_type}")
        lines.append("Value:")
        lines.append(_render_input_value(item))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _resolve_from_output(
    task: Any,
    input_spec: TaskInputSpec,
    parsed: ParsedTaskReference,
    store: "WorkflowStore",
) -> ResolvedTaskInput:
    output = store.load_task_output(task.workflow_id, task.task_id)
    if output is None:
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=task.task_id,
            resolved_type="missing",
            metadata={"reason": "missing_output"},
        )
    if not parsed.path:
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=task.task_id,
            resolved_type="json",
            value=output.to_dict(),
        )
    head = parsed.path[0]
    if head == "summary":
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=task.task_id,
            resolved_type="text",
            value=output.summary,
        )
    if head == "text":
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=task.task_id,
            resolved_type="text",
            value=output.text,
        )
    if head == "data":
        return _resolve_nested_mapping(
            mapping=output.data,
            raw_source=input_spec.from_ref,
            name=input_spec.name,
            task_id=task.task_id,
            path=parsed.path[1:],
        )
    if head == "metrics":
        return _resolve_nested_mapping(
            mapping=output.metrics,
            raw_source=input_spec.from_ref,
            name=input_spec.name,
            task_id=task.task_id,
            path=parsed.path[1:],
        )
    return ResolvedTaskInput(
        name=input_spec.name,
        source=input_spec.from_ref,
        resolved_task_id=task.task_id,
        resolved_type="missing",
        metadata={"reason": "unsupported_output_path"},
    )


def _resolve_nested_mapping(
    *,
    mapping: dict[str, Any],
    raw_source: str,
    name: str,
    task_id: str,
    path: tuple[str, ...],
) -> ResolvedTaskInput:
    if not path:
        return ResolvedTaskInput(
            name=name,
            source=raw_source,
            resolved_task_id=task_id,
            resolved_type="json",
            value=dict(mapping),
        )
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return ResolvedTaskInput(
                name=name,
                source=raw_source,
                resolved_task_id=task_id,
                resolved_type="missing",
                metadata={"reason": "missing_key", "missing_key": key},
            )
        current = current[key]
    if isinstance(current, str):
        resolved_type = "text"
    else:
        resolved_type = "json"
    return ResolvedTaskInput(
        name=name,
        source=raw_source,
        resolved_task_id=task_id,
        resolved_type=resolved_type,
        value=current,
    )


def _resolve_from_log(
    task: Any,
    input_spec: TaskInputSpec,
    root: str,
    store: "WorkflowStore",
    *,
    log_limit: int,
) -> ResolvedTaskInput:
    path_value = getattr(task, "log_stdout_path" if root == "stdout" else "log_stderr_path", None)
    if not path_value:
        default_stdout, default_stderr = store.task_log_paths(task.workflow_id, task.task_id)
        path_value = str(default_stdout if root == "stdout" else default_stderr)
    path = Path(path_value)
    if not path.exists():
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=task.task_id,
            resolved_type="missing",
            metadata={"reason": f"missing_{root}_path", "path": str(path)},
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    original_size = len(text)
    truncated = original_size > log_limit
    value = text[:log_limit]
    return ResolvedTaskInput(
        name=input_spec.name,
        source=input_spec.from_ref,
        resolved_task_id=task.task_id,
        resolved_type="text",
        value=value,
        metadata={
            "path": str(path),
            "truncated": truncated,
            "original_size": original_size,
            "truncated_size": len(value),
        },
    )


def _resolve_from_artifact(
    task: Any,
    input_spec: TaskInputSpec,
    parsed: ParsedTaskReference,
    store: "WorkflowStore",
) -> ResolvedTaskInput:
    name = parsed.path[0]
    raw_artifacts = getattr(task, "artifacts", []) or []
    artifacts = [
        item if isinstance(item, Artifact) else Artifact.from_dict(item)
        for item in raw_artifacts
    ]
    try:
        validate_artifact_uniqueness(artifacts)
    except DataflowError as exc:
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=task.task_id,
            resolved_type="missing",
            metadata={"reason": "duplicate_artifact_names", "error": str(exc)},
        )
    for artifact in artifacts:
        if artifact.name != name:
            continue
        artifact_path = Path(artifact.path)
        if not artifact_path.exists():
            return ResolvedTaskInput(
                name=input_spec.name,
                source=input_spec.from_ref,
                resolved_task_id=task.task_id,
                resolved_type="missing",
                metadata={"reason": "missing_artifact_path", "path": str(artifact_path)},
            )
        return ResolvedTaskInput(
            name=input_spec.name,
            source=input_spec.from_ref,
            resolved_task_id=task.task_id,
            resolved_type="artifact",
            value=str(artifact_path),
            artifact_path=str(artifact_path),
            metadata={"name": artifact.name, "kind": artifact.kind},
        )
    return ResolvedTaskInput(
        name=input_spec.name,
        source=input_spec.from_ref,
        resolved_task_id=task.task_id,
        resolved_type="missing",
        metadata={"reason": "missing_artifact", "name": name},
    )


def _render_input_value(item: ResolvedTaskInput) -> str:
    if item.resolved_type == "artifact":
        return item.artifact_path or ""
    if item.value is None:
        return ""
    if isinstance(item.value, str):
        return item.value
    return json.dumps(item.value, indent=2, sort_keys=True)
