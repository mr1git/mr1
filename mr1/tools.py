"""
Deterministic workflow tool runners.

Tools are bounded, non-LLM capabilities that execute synchronously in
the scheduler and emit normalized Phase 3 outputs.
"""

from __future__ import annotations

import os
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from mr1.capability_policy import normalize_path
from mr1.dataflow import Artifact, new_artifact_id
from mr1.memory_queries import (
    memory_graph_agent_summary,
    memory_graph_capabilities,
    memory_graph_failures,
    memory_graph_top_workflows,
    memory_insight_show,
    memory_insights_search,
)
from mr1.workflow_models import Task, Workflow
from mr1.workflow_store import WorkflowStore


class ToolConfigError(ValueError):
    """Raised when a tool type or config is invalid."""


@dataclass
class ToolResult:
    state: str  # succeeded | failed | timed_out
    summary: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ToolRunner(Protocol):
    def validate_config(self, config: dict[str, Any]) -> None: ...

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult: ...


@dataclass(frozen=True)
class ToolDefinition:
    tool_type: str
    description: str
    config_shape: str
    config_schema: dict[str, Any]
    outputs: dict[str, str]
    examples: list[dict[str, Any]]
    runner: ToolRunner


def _resolve_path(raw_path: str) -> Path:
    return normalize_path(raw_path)


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    if b"\x00" in data:
        return False
    text_bytes = set(range(32, 127)) | {9, 10, 13, 8, 12}
    non_text = sum(1 for byte in data if byte not in text_bytes)
    return (non_text / len(data)) <= 0.30


def _decode_bytes(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _tail_truncate_bytes(data: bytes, limit: int) -> tuple[bytes, bool]:
    if len(data) <= limit:
        return data, False
    return data[-limit:], True


def _artifact(
    *,
    workflow_id: str,
    task_id: str,
    name: str,
    kind: str,
    path: Path,
    metadata: Optional[dict[str, Any]] = None,
) -> Artifact:
    return Artifact(
        artifact_id=new_artifact_id(),
        workflow_id=workflow_id,
        task_id=task_id,
        name=name,
        kind=kind,
        path=str(path),
        metadata=dict(metadata or {}),
    )


def _write_text_artifact(
    store: WorkflowStore,
    task: Task,
    name: str,
    text: str,
) -> Artifact:
    artifacts_dir = store.task_artifacts_dir(task.workflow_id, task.task_id)
    path = artifacts_dir / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return _artifact(
        workflow_id=task.workflow_id,
        task_id=task.task_id,
        name=name,
        kind="text",
        path=path,
    )


def _read_file_pure(path_str: str, max_bytes: int) -> dict[str, Any]:
    """Pure read-file logic; no workflow task or store required.

    Returns keys: state, text, data, artifact_path, artifact_kind, error.
    """
    path = _resolve_path(path_str)
    if not path.exists():
        return {
            "state": "failed",
            "text": "",
            "data": {"path": str(path)},
            "artifact_path": None,
            "artifact_kind": None,
            "error": f"path does not exist: {path}",
        }
    if not path.is_file():
        return {
            "state": "failed",
            "text": "",
            "data": {"path": str(path)},
            "artifact_path": None,
            "artifact_kind": None,
            "error": f"path is not a file: {path}",
        }
    raw = path.read_bytes()
    captured = raw[:max_bytes]
    text = _decode_bytes(captured)
    return {
        "state": "succeeded",
        "text": text,
        "data": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "truncated": len(raw) > max_bytes,
        },
        "artifact_path": path,
        "artifact_kind": "text" if _looks_like_text(captured) else "binary",
        "error": None,
    }


class ReadFileTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        path = config.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ToolConfigError("read_file requires tool_config.path")
        max_bytes = config.get("max_bytes", 65536)
        if not isinstance(max_bytes, int) or max_bytes < 1:
            raise ToolConfigError("read_file max_bytes must be an integer >= 1")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        del store, workflow
        result = _read_file_pure(task.tool_config["path"], task.tool_config.get("max_bytes", 65536))
        path_display = result["data"]["path"]
        if result["state"] == "succeeded":
            summary = f"read file: {path_display}"
        else:
            summary = f"read file failed: {path_display}"
        artifacts = []
        if result["artifact_path"] is not None:
            artifacts.append(_artifact(
                workflow_id=task.workflow_id,
                task_id=task.task_id,
                name="file",
                kind=result["artifact_kind"],
                path=result["artifact_path"],
            ))
        return ToolResult(
            state=result["state"],
            summary=summary,
            text=result["text"],
            data=result["data"],
            artifacts=artifacts,
            error=result["error"],
        )


class WriteFileTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        path = config.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ToolConfigError("write_file requires tool_config.path")
        if not isinstance(config.get("content"), str):
            raise ToolConfigError("write_file requires tool_config.content")
        create_dirs = config.get("create_dirs", False)
        if not isinstance(create_dirs, bool):
            raise ToolConfigError("write_file create_dirs must be a boolean")
        overwrite = config.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise ToolConfigError("write_file overwrite must be a boolean")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        del workflow
        path = _resolve_path(task.tool_config["path"])
        content = task.tool_config["content"]
        create_dirs = task.tool_config.get("create_dirs", False)
        overwrite = task.tool_config.get("overwrite", False)
        parent = path.parent
        if not parent.exists():
            if create_dirs:
                parent.mkdir(parents=True, exist_ok=True)
            else:
                return ToolResult(
                    state="failed",
                    summary=f"write file failed: {path}",
                    text="",
                    data={"path": str(path)},
                    error=f"parent directory does not exist: {parent}",
                )
        if path.exists() and not overwrite:
            return ToolResult(
                state="failed",
                summary=f"write file failed: {path}",
                text="",
                data={"path": str(path)},
                error=f"file exists and overwrite is false: {path}",
            )
        payload = content.encode("utf-8")
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            state="succeeded",
            summary=f"wrote file: {path}",
            text=str(path),
            data={"path": str(path), "bytes_written": len(payload)},
            artifacts=[
                _artifact(
                    workflow_id=task.workflow_id,
                    task_id=task.task_id,
                    name="written_file",
                    kind="text",
                    path=path,
                )
            ],
        )


class ShellCommandTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        argv = config.get("argv")
        if not isinstance(argv, list) or not argv:
            raise ToolConfigError("shell_command requires non-empty tool_config.argv")
        if any(not isinstance(item, str) or not item for item in argv):
            raise ToolConfigError("shell_command argv must be a non-empty list of strings")
        cwd = config.get("cwd")
        if not isinstance(cwd, str) or not cwd.strip():
            raise ToolConfigError("shell_command requires non-empty tool_config.cwd")
        cwd_path = _resolve_path(cwd)
        if not cwd_path.exists():
            raise ToolConfigError(f"shell_command cwd does not exist: {cwd_path}")
        if not cwd_path.is_dir():
            raise ToolConfigError(f"shell_command cwd is not a directory: {cwd_path}")
        timeout_s = config.get("timeout_s", 30)
        if not isinstance(timeout_s, int) or timeout_s < 1 or timeout_s > 300:
            raise ToolConfigError("shell_command timeout_s must be an integer between 1 and 300")
        capture_max_bytes = config.get("capture_max_bytes", 65536)
        if not isinstance(capture_max_bytes, int) or capture_max_bytes < 1:
            raise ToolConfigError("shell_command capture_max_bytes must be an integer >= 1")
        env = config.get("env", {})
        if not isinstance(env, dict):
            raise ToolConfigError("shell_command env must be a JSON object")
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ToolConfigError("shell_command env keys and values must be strings")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        del workflow
        argv = list(task.tool_config["argv"])
        cwd = _resolve_path(task.tool_config["cwd"])
        timeout_s = task.tool_config.get("timeout_s", 30)
        capture_max_bytes = task.tool_config.get("capture_max_bytes", 65536)
        env_overrides = dict(task.tool_config.get("env", {}))
        command_env = os.environ.copy()
        command_env.update(env_overrides)

        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                shell=False,
                input=None,
                capture_output=True,
                text=False,
                cwd=str(cwd),
                env=command_env,
                timeout=timeout_s,
            )
            duration_s = time.monotonic() - started
            stdout_bytes, stdout_truncated = _tail_truncate_bytes(proc.stdout or b"", capture_max_bytes)
            stderr_bytes, stderr_truncated = _tail_truncate_bytes(proc.stderr or b"", capture_max_bytes)
            stdout_text = _decode_bytes(stdout_bytes)
            stderr_text = _decode_bytes(stderr_bytes)
            state = "succeeded" if proc.returncode == 0 else "failed"
            artifacts: list[Artifact] = []
            if stdout_text:
                artifacts.append(_write_text_artifact(store, task, "stdout", stdout_text))
            if stderr_text:
                artifacts.append(_write_text_artifact(store, task, "stderr", stderr_text))
            return ToolResult(
                state=state,
                summary=f"command exited {proc.returncode}: {' '.join(argv)}",
                text=stdout_text,
                data={
                    "argv": argv,
                    "cwd": str(cwd),
                    "exit_code": proc.returncode,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "duration_s": duration_s,
                },
                metrics={"duration_s": duration_s},
                artifacts=artifacts,
                metadata={"capture_max_bytes": capture_max_bytes},
                error=None if proc.returncode == 0 else f"command exited with code {proc.returncode}",
            )
        except subprocess.TimeoutExpired as exc:
            duration_s = time.monotonic() - started
            stdout_bytes, stdout_truncated = _tail_truncate_bytes(exc.stdout or b"", capture_max_bytes)
            stderr_bytes, stderr_truncated = _tail_truncate_bytes(exc.stderr or b"", capture_max_bytes)
            stdout_text = _decode_bytes(stdout_bytes)
            stderr_text = _decode_bytes(stderr_bytes)
            artifacts = []
            if stdout_text:
                artifacts.append(_write_text_artifact(store, task, "stdout", stdout_text))
            if stderr_text:
                artifacts.append(_write_text_artifact(store, task, "stderr", stderr_text))
            return ToolResult(
                state="timed_out",
                summary=f"command timed out after {timeout_s}s: {' '.join(argv)}",
                text=stdout_text,
                data={
                    "argv": argv,
                    "cwd": str(cwd),
                    "exit_code": None,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                    "duration_s": duration_s,
                },
                metrics={"duration_s": duration_s},
                artifacts=artifacts,
                metadata={"capture_max_bytes": capture_max_bytes},
                error=f"command timed out after {timeout_s}s",
            )
        except OSError as exc:
            duration_s = time.monotonic() - started
            return ToolResult(
                state="failed",
                summary=f"command failed to start: {' '.join(argv)}",
                text="",
                data={
                    "argv": argv,
                    "cwd": str(cwd),
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                    "duration_s": duration_s,
                },
                metrics={"duration_s": duration_s},
                error=f"failed to start command: {exc}",
            )


class MemoryInsightsSearchTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        query = config.get("query")
        if query is not None and not isinstance(query, str):
            raise ToolConfigError("memory_insights_search query must be a string")
        types = config.get("types")
        if types is not None:
            if not isinstance(types, list) or any(not isinstance(item, str) or not item.strip() for item in types):
                raise ToolConfigError("memory_insights_search types must be a list of non-empty strings")
        status = config.get("status")
        if status is not None and (not isinstance(status, str) or not status.strip()):
            raise ToolConfigError("memory_insights_search status must be a non-empty string")
        limit = config.get("limit")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
            raise ToolConfigError("memory_insights_search limit must be an integer >= 0")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        del workflow
        result = memory_insights_search(
            store.root.parent,
            query=task.tool_config.get("query"),
            types=task.tool_config.get("types"),
            status=task.tool_config.get("status"),
            limit=task.tool_config.get("limit"),
        )
        return ToolResult(
            state="succeeded",
            summary=f"found {result['count']} insight(s)",
            text="",
            data=result,
        )


class MemoryInsightShowTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        insight_id = config.get("insight_id")
        if not isinstance(insight_id, str) or not insight_id.strip():
            raise ToolConfigError("memory_insight_show requires tool_config.insight_id")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        del workflow
        result = memory_insight_show(
            store.root.parent,
            insight_id=task.tool_config.get("insight_id"),
        )
        return ToolResult(
            state="succeeded",
            summary="memory insight loaded" if result["found"] else "memory insight not found",
            text="",
            data=result,
        )


class MemoryGraphTopWorkflowsTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        limit = config.get("limit")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
            raise ToolConfigError("memory_graph_top_workflows limit must be an integer >= 0")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        del workflow
        result = memory_graph_top_workflows(
            store.root.parent,
            limit=task.tool_config.get("limit"),
        )
        return ToolResult(
            state="succeeded",
            summary=f"found {result['count']} workflow template(s)",
            text="",
            data=result,
        )


class MemoryGraphCapabilitiesTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        limit = config.get("limit")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
            raise ToolConfigError("memory_graph_capabilities limit must be an integer >= 0")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        del workflow
        result = memory_graph_capabilities(
            store.root.parent,
            limit=task.tool_config.get("limit"),
        )
        return ToolResult(
            state="succeeded",
            summary=f"found {result['count']} capability node(s)",
            text="",
            data=result,
        )


class MemoryGraphFailuresTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        limit = config.get("limit")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
            raise ToolConfigError("memory_graph_failures limit must be an integer >= 0")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        del workflow
        result = memory_graph_failures(
            store.root.parent,
            limit=task.tool_config.get("limit"),
        )
        return ToolResult(
            state="succeeded",
            summary=f"found {result['count']} failure mode(s)",
            text="",
            data=result,
        )


class MemoryGraphAgentSummaryTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        agent_id = config.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ToolConfigError("memory_graph_agent_summary requires tool_config.agent_id")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        del workflow
        result = memory_graph_agent_summary(
            store.root.parent,
            agent_id=task.tool_config.get("agent_id"),
        )
        return ToolResult(
            state="succeeded",
            summary="agent summary loaded" if result["found"] else "agent summary not found",
            text="",
            data=result,
        )


class MemoryGraphUpdateTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise ToolConfigError("memory_graph_update tool_config must be a JSON object")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        from mr1.event_log import EventLog
        from mr1.memory_feedback import run_memory_graph_update
        from mr1.memory_graph import MemoryGraphStore

        del task, workflow
        result = run_memory_graph_update(
            event_log=EventLog(store.root.parent / "events"),
            graph_store=MemoryGraphStore(store.root.parent / "graph"),
        )
        return ToolResult(
            state="succeeded",
            summary=f"processed {result['processed_events']} event(s) into graph memory",
            text="",
            data=result,
        )


class MemoryCurateTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise ToolConfigError("memory_curate tool_config must be a JSON object")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        from mr1.event_log import EventLog
        from mr1.memory_curator import InsightStore
        from mr1.memory_feedback import run_memory_curate
        from mr1.memory_graph import MemoryGraphStore

        del workflow
        result = run_memory_curate(
            event_log=EventLog(store.root.parent / "events"),
            graph_store=MemoryGraphStore(store.root.parent / "graph"),
            insight_store=InsightStore(store.root.parent / "insights"),
            trigger_reason=f"workflow:{task.workflow_id}:memory_curate",
        )
        return ToolResult(
            state="succeeded",
            summary=f"memory curate run status: {result['status']}",
            text="",
            data=result,
        )


class MemoryFeedbackUpdateTool:
    def validate_config(self, config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise ToolConfigError("memory_feedback_update tool_config must be a JSON object")

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        from mr1.event_log import EventLog
        from mr1.memory_curator import InsightStore
        from mr1.memory_feedback import update_insight_feedback

        del task, workflow
        result = update_insight_feedback(
            event_log=EventLog(store.root.parent / "events"),
            insight_store=InsightStore(store.root.parent / "insights"),
            workflow_store=store,
        ).to_dict()
        return ToolResult(
            state="succeeded",
            summary=f"created {result['feedback_created']} insight feedback record(s)",
            text="",
            data=result,
        )


class ToolRegistry:
    def __init__(self):
        self._definitions: dict[str, ToolDefinition] = {}

    def register(
        self,
        tool_type: str,
        runner: ToolRunner,
        *,
        description: str,
        config_shape: str,
        config_schema: dict[str, Any],
        outputs: dict[str, str],
        examples: list[dict[str, Any]],
    ) -> None:
        self._definitions[tool_type] = ToolDefinition(
            tool_type=tool_type,
            description=description,
            config_shape=config_shape,
            config_schema=deepcopy(config_schema),
            outputs=deepcopy(outputs),
            examples=deepcopy(examples),
            runner=runner,
        )

    def is_registered(self, tool_type: Optional[str]) -> bool:
        return bool(tool_type) and tool_type in self._definitions

    def validate_spec(
        self,
        tool_type: Optional[str],
        config: Optional[dict[str, Any]],
    ) -> None:
        if not isinstance(tool_type, str) or not tool_type:
            raise ToolConfigError("tool task requires tool_type")
        definition = self._definitions.get(tool_type)
        if definition is None:
            raise ToolConfigError(f"unknown tool_type '{tool_type}'")
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ToolConfigError("tool_config must be a JSON object")
        definition.runner.validate_config(config)

    def run(self, task: Task, store: WorkflowStore, workflow: Workflow) -> ToolResult:
        definition = self._definitions.get(task.tool_type or "")
        if definition is None:
            raise ToolConfigError(f"unknown tool_type '{task.tool_type}'")
        return definition.runner.run(task, store, workflow)

    def list_tools(self) -> list[ToolDefinition]:
        return [
            self._definitions[key]
            for key in sorted(self._definitions)
        ]

    def describe_tool(self, tool_type: str) -> dict[str, Any]:
        definition = self._definitions.get(tool_type)
        if definition is None:
            raise ValueError(f"tool not found: {tool_type}")
        return {
            "name": definition.tool_type,
            "type": "tool",
            "description": definition.description,
            "inputs": {
                field: details.get("type", "unknown")
                for field, details in definition.config_schema.items()
            },
            "outputs": deepcopy(definition.outputs),
            "examples": deepcopy(definition.examples),
            "config_schema": deepcopy(definition.config_schema),
        }

    def describe_all_tools(self) -> list[dict[str, Any]]:
        return [self.describe_tool(tool.tool_type) for tool in self.list_tools()]


_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def default_tool_registry() -> ToolRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = ToolRegistry()
        registry.register(
            "read_file",
            ReadFileTool(),
            description="Read a file deterministically and expose contents as normalized output.",
            config_shape='{"path": "notes.txt", "max_bytes": 65536}',
            config_schema={
                "path": {"type": "string", "required": True},
                "max_bytes": {"type": "int", "required": False, "default": 65536},
            },
            outputs={
                "result.text": "file contents as UTF-8 text",
                "result.data.path": "resolved file path",
                "result.data.size_bytes": "file size in bytes",
                "result.data.truncated": "true when content exceeded max_bytes",
                "artifact.file": "artifact path for the source file",
            },
            examples=[
                {
                    "label": "read_notes",
                    "title": "Read notes",
                    "task_kind": "tool",
                    "tool_type": "read_file",
                    "tool_config": {"path": "notes.txt"},
                }
            ],
        )
        registry.register(
            "write_file",
            WriteFileTool(),
            description="Write UTF-8 text to a file deterministically.",
            config_shape='{"path": "outputs/summary.txt", "content": "hello", "create_dirs": true, "overwrite": false}',
            config_schema={
                "path": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
                "create_dirs": {"type": "bool", "required": False, "default": False},
                "overwrite": {"type": "bool", "required": False, "default": False},
            },
            outputs={
                "result.text": "written file path",
                "result.data.path": "resolved written file path",
                "result.data.bytes_written": "number of bytes written",
                "artifact.written_file": "artifact path for the written file",
            },
            examples=[
                {
                    "label": "write_report",
                    "title": "Write report",
                    "task_kind": "tool",
                    "tool_type": "write_file",
                    "tool_config": {
                        "path": "outputs/report.txt",
                        "content": "hello",
                    },
                }
            ],
        )
        registry.register(
            "shell_command",
            ShellCommandTool(),
            description="Run a bounded argv command with shell=False and structured captured output.",
            config_shape='{"argv": ["python", "--version"], "cwd": ".", "timeout_s": 10, "capture_max_bytes": 65536, "env": {}}',
            config_schema={
                "argv": {"type": "list[string]", "required": True},
                "cwd": {"type": "string", "required": True},
                "timeout_s": {"type": "int", "required": False, "default": 30},
                "capture_max_bytes": {"type": "int", "required": False, "default": 65536},
                "env": {"type": "object[string,string]", "required": False, "default": {}},
            },
            outputs={
                "result.text": "stdout text",
                "result.data.argv": "executed argv list",
                "result.data.cwd": "resolved working directory",
                "result.data.exit_code": "process exit code or null on timeout/start failure",
                "result.data.stdout": "captured stdout text",
                "result.data.stderr": "captured stderr text",
                "result.data.stdout_truncated": "true when stdout exceeded capture_max_bytes",
                "result.data.stderr_truncated": "true when stderr exceeded capture_max_bytes",
                "result.data.duration_s": "command duration in seconds",
                "artifact.stdout": "artifact path for stdout when present",
                "artifact.stderr": "artifact path for stderr when present",
            },
            examples=[
                {
                    "label": "python_version",
                    "title": "Python version",
                    "task_kind": "tool",
                    "tool_type": "shell_command",
                    "tool_config": {
                        "argv": ["python3", "--version"],
                        "cwd": ".",
                    },
                }
            ],
        )
        registry.register(
            "memory_insights_search",
            MemoryInsightsSearchTool(),
            description="Search curated MR1 memory insights using deterministic substring matching and ranking.",
            config_shape='{"query": "file access approval friction", "types": ["capability_friction"], "status": "active", "limit": 5}',
            config_schema={
                "query": {"type": "string", "required": False},
                "types": {"type": "list[string]", "required": False},
                "status": {"type": "string", "required": False, "default": "active"},
                "limit": {"type": "int", "required": False, "default": 5},
            },
            outputs={
                "result.data.items": "compact matching insight records",
                "result.data.count": "number of returned insights",
                "result.data.query": "normalized search query",
                "result.data.types": "requested insight type filters",
                "result.data.status": "status filter used",
                "result.data.limit": "applied result limit",
            },
            examples=[
                {
                    "label": "search_memory_insights",
                    "title": "Search memory insights",
                    "task_kind": "tool",
                    "tool_type": "memory_insights_search",
                    "tool_config": {
                        "query": "file access approval friction",
                        "limit": 5,
                    },
                }
            ],
        )
        registry.register(
            "memory_insight_show",
            MemoryInsightShowTool(),
            description="Show one curated MR1 memory insight by id.",
            config_shape='{"insight_id": "insight:capability_friction:capability:read_file"}',
            config_schema={
                "insight_id": {"type": "string", "required": False},
            },
            outputs={
                "result.data.found": "true when the requested insight exists",
                "result.data.insight": "full insight object or null when not found",
            },
            examples=[
                {
                    "label": "show_memory_insight",
                    "title": "Show memory insight",
                    "task_kind": "tool",
                    "tool_type": "memory_insight_show",
                    "tool_config": {
                        "insight_id": "insight:capability_friction:capability:read_file",
                    },
                }
            ],
        )
        registry.register(
            "memory_graph_top_workflows",
            MemoryGraphTopWorkflowsTool(),
            description="List top workflow templates from deterministic graph memory.",
            config_shape='{"limit": 5}',
            config_schema={
                "limit": {"type": "int", "required": False, "default": 5},
            },
            outputs={
                "result.data.items": "workflow template summaries ordered by credibility",
                "result.data.count": "number of returned workflow templates",
            },
            examples=[
                {
                    "label": "top_workflow_templates",
                    "title": "Top workflow templates",
                    "task_kind": "tool",
                    "tool_type": "memory_graph_top_workflows",
                    "tool_config": {"limit": 5},
                }
            ],
        )
        registry.register(
            "memory_graph_capabilities",
            MemoryGraphCapabilitiesTool(),
            description="List capability usage statistics from deterministic graph memory.",
            config_shape='{"limit": 10}',
            config_schema={
                "limit": {"type": "int", "required": False, "default": 10},
            },
            outputs={
                "result.data.items": "capability node summaries ordered by request count",
                "result.data.count": "number of returned capability summaries",
            },
            examples=[
                {
                    "label": "memory_capabilities",
                    "title": "Memory capability stats",
                    "task_kind": "tool",
                    "tool_type": "memory_graph_capabilities",
                    "tool_config": {"limit": 10},
                }
            ],
        )
        registry.register(
            "memory_graph_failures",
            MemoryGraphFailuresTool(),
            description="List common failure modes from deterministic graph memory.",
            config_shape='{"limit": 10}',
            config_schema={
                "limit": {"type": "int", "required": False, "default": 10},
            },
            outputs={
                "result.data.items": "failure mode node summaries ordered by occurrence count",
                "result.data.count": "number of returned failure summaries",
            },
            examples=[
                {
                    "label": "memory_failures",
                    "title": "Memory failure stats",
                    "task_kind": "tool",
                    "tool_type": "memory_graph_failures",
                    "tool_config": {"limit": 10},
                }
            ],
        )
        registry.register(
            "memory_graph_agent_summary",
            MemoryGraphAgentSummaryTool(),
            description="Show one agent summary from deterministic graph memory.",
            config_shape='{"agent_id": "ag-123"}',
            config_schema={
                "agent_id": {"type": "string", "required": False},
            },
            outputs={
                "result.data.found": "true when the agent exists in graph memory",
                "result.data.summary": "full graph summary object or null when not found",
            },
            examples=[
                {
                    "label": "memory_agent_summary",
                    "title": "Memory agent summary",
                    "task_kind": "tool",
                    "tool_type": "memory_graph_agent_summary",
                    "tool_config": {"agent_id": "ag-123"},
                }
            ],
        )
        registry.register(
            "memory_graph_update",
            MemoryGraphUpdateTool(),
            description="Update deterministic graph memory from timeline events.",
            config_shape='{}',
            config_schema={},
            outputs={
                "result.data.processed_events": "number of timeline events processed",
                "result.data.last_processed_event_index": "last processed graph cursor index",
                "result.data.nodes_created": "number of graph nodes created",
                "result.data.nodes_updated": "number of graph nodes updated",
                "result.data.edges_created": "number of graph edges created",
                "result.data.edges_updated": "number of graph edges updated",
            },
            examples=[
                {
                    "label": "memory_graph_update",
                    "title": "Update graph memory",
                    "task_kind": "tool",
                    "tool_type": "memory_graph_update",
                    "tool_config": {},
                }
            ],
        )
        registry.register(
            "memory_curate",
            MemoryCurateTool(),
            description="Run one bounded memory curation pass.",
            config_shape='{}',
            config_schema={},
            outputs={
                "result.data.status": "curation run status",
                "result.data.run_id": "curation run id",
                "result.data.output_insight_ids": "curated insight ids written by the run",
                "result.data.errors": "non-fatal or fatal curation errors",
            },
            examples=[
                {
                    "label": "memory_curate",
                    "title": "Curate memory insights",
                    "task_kind": "tool",
                    "tool_type": "memory_curate",
                    "tool_config": {},
                }
            ],
        )
        registry.register(
            "memory_feedback_update",
            MemoryFeedbackUpdateTool(),
            description="Evaluate workflow outcomes against memory insights and update feedback stats.",
            config_shape='{}',
            config_schema={},
            outputs={
                "result.data.processed_events": "number of relevant events processed",
                "result.data.feedback_created": "number of feedback records created",
                "result.data.insights_updated": "number of insights updated",
                "result.data.last_evaluated_event_index": "last feedback cursor index",
                "result.data.errors": "non-fatal feedback processing errors",
            },
            examples=[
                {
                    "label": "memory_feedback_update",
                    "title": "Update insight feedback",
                    "task_kind": "tool",
                    "tool_type": "memory_feedback_update",
                    "tool_config": {},
                }
            ],
        )
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
