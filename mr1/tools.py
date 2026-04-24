"""
Deterministic workflow tool runners.

Tools are bounded, non-LLM capabilities that execute synchronously in
the scheduler and emit normalized Phase 3 outputs.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from mr1.dataflow import Artifact, new_artifact_id
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
    runner: ToolRunner


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


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
        path = _resolve_path(task.tool_config["path"])
        max_bytes = task.tool_config.get("max_bytes", 65536)
        if not path.exists():
            return ToolResult(
                state="failed",
                summary=f"read file failed: {path}",
                text="",
                data={"path": str(path)},
                error=f"path does not exist: {path}",
            )
        if not path.is_file():
            return ToolResult(
                state="failed",
                summary=f"read file failed: {path}",
                text="",
                data={"path": str(path)},
                error=f"path is not a file: {path}",
            )
        raw = path.read_bytes()
        captured = raw[:max_bytes]
        text = _decode_bytes(captured)
        artifact_kind = "text" if _looks_like_text(captured) else "binary"
        return ToolResult(
            state="succeeded",
            summary=f"read file: {path}",
            text=text,
            data={
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "truncated": len(raw) > max_bytes,
            },
            artifacts=[
                _artifact(
                    workflow_id=task.workflow_id,
                    task_id=task.task_id,
                    name="file",
                    kind=artifact_kind,
                    path=path,
                )
            ],
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
        if cwd is not None:
            if not isinstance(cwd, str) or not cwd.strip():
                raise ToolConfigError("shell_command cwd must be a non-empty string")
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
        cwd = _resolve_path(task.tool_config.get("cwd", "."))
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
    ) -> None:
        self._definitions[tool_type] = ToolDefinition(
            tool_type=tool_type,
            description=description,
            config_shape=config_shape,
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
        )
        registry.register(
            "write_file",
            WriteFileTool(),
            description="Write UTF-8 text to a file deterministically.",
            config_shape='{"path": "outputs/summary.txt", "content": "hello", "create_dirs": true, "overwrite": false}',
        )
        registry.register(
            "shell_command",
            ShellCommandTool(),
            description="Run a bounded argv command with shell=False and structured captured output.",
            config_shape='{"argv": ["python", "--version"], "cwd": ".", "timeout_s": 10, "capture_max_bytes": 65536, "env": {}}',
        )
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
