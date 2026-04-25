"""
Non-blocking execution adapters for workflow tasks.

The scheduler never calls `subprocess.Popen` itself. It talks to a
`Runner` implementation: `start(task) -> RunHandle`, `poll(handle) ->
RunStatus`, `cancel(handle)`.

Three implementations are provided:

- `KaziAsyncRunner` — real Kazi runs driven via `subprocess.Popen`, with
  stdout/stderr redirected directly to log files so the scheduler never
  has to drain pipes.
- `KaziBlockingRunner` — wraps the existing synchronous `kazi.run()` for
  the rare caller that wants blocking semantics (tests, CLI one-shots).
- `MockRunner` — test-only. Tasks transition through status callbacks
  provided by the test harness; no subprocess is launched.
"""

from __future__ import annotations

import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from mr1.agents import (
    AgentRuntimeError,
    build_agent_command,
    is_auth_error_text,
    load_agent_runtime_config,
    parse_agent_json_envelope,
)
from mr1.core import Dispatcher, PermissionDenied, Logger
from mr1.workflow_models import Task
from mr1.workflow_store import WorkflowStore


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass
class RunResult:
    """Terminal result the scheduler persists on the Task."""
    status: RunStatus
    exit_code: Optional[int] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    result_payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunHandle:
    """Opaque handle each Runner hands back from start()."""
    task_id: str
    workflow_id: str
    pid: Optional[int] = None
    started_monotonic: float = field(default_factory=time.monotonic)
    timeout_s: Optional[int] = None
    payload: dict[str, Any] = field(default_factory=dict)


class Runner(ABC):
    """Interface the scheduler uses to launch/poll/cancel tasks."""

    @abstractmethod
    def start(self, task: Task) -> RunHandle: ...

    @abstractmethod
    def poll(self, handle: RunHandle) -> Optional[RunResult]:
        """Return None if still running; otherwise a terminal RunResult."""

    @abstractmethod
    def cancel(self, handle: RunHandle) -> None: ...


# ---------------------------------------------------------------------------
# Kazi async runner
# ---------------------------------------------------------------------------


_DEFAULT_KAZI_TIMEOUT_S = 300


def _result_payload_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": parsed["text"],
        "text": parsed["text"],
        "data": {
            "raw": parsed["raw"],
            "is_error": parsed["is_error"],
            "metadata": parsed["metadata"],
        },
        "metrics": {
            "usage": parsed["usage"],
        },
    }


def _parse_claude_json_envelope(raw: str) -> dict[str, Any]:
    return parse_agent_json_envelope(raw)


def _classify_envelope_error(parsed: dict[str, Any]) -> str:
    return "auth_error" if is_auth_error_text(parsed.get("text")) else "cli_error"


class KaziAsyncRunner(Runner):
    """
    Runs each Kazi task as a non-blocking subprocess.

    stdout/stderr are redirected directly to the task's log files so
    the scheduler's poll loop only has to check `process.poll()` —
    there is no pipe drain thread.
    """

    def __init__(
        self,
        store: WorkflowStore,
        *,
        dispatcher: Optional[Dispatcher] = None,
        logger: Optional[Logger] = None,
        claude_binary: str = "claude",
    ):
        self._store = store
        self._dispatcher = dispatcher or Dispatcher()
        self._logger = logger or Logger()
        self._claude_binary = claude_binary
        self._config = load_agent_runtime_config("kazi")

    def start(self, task: Task) -> RunHandle:
        prompt = task.prompt or task.title
        tools = list(self._config.get("allowed_tools", []))
        timeout_s = task.timeout_s or self._config.get("timeout_s") or _DEFAULT_KAZI_TIMEOUT_S
        cmd = build_agent_command(
            "kazi",
            prompt,
            config=self._config,
            binary_override=self._claude_binary,
        )

        cli_flags = [tok for tok in cmd[1:] if tok.startswith("-")]
        try:
            self._dispatcher.validate_full_spawn("kazi", cli_flags, tools)
        except PermissionDenied as e:
            self._logger.log_denied(task.task_id, "kazi", str(e))
            raise

        stdout_path, stderr_path = self._store.task_log_paths(
            task.workflow_id, task.task_id,
        )
        stdout_fh = open(stdout_path, "wb")
        stderr_fh = open(stderr_path, "wb")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_fh,
                stderr=stderr_fh,
            )
        except OSError as exc:
            stdout_fh.close()
            stderr_fh.close()
            stderr_path.write_text(str(exc), encoding="utf-8")
            return RunHandle(
                task_id=task.task_id,
                workflow_id=task.workflow_id,
                pid=None,
                timeout_s=timeout_s,
                payload={
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "startup_error": str(exc),
                    "cmd": cmd,
                },
            )
        self._logger.log_spawn(task.task_id, "kazi", proc.pid, cmd)

        return RunHandle(
            task_id=task.task_id,
            workflow_id=task.workflow_id,
            pid=proc.pid,
            timeout_s=timeout_s,
            payload={
                "process": proc,
                "stdout_fh": stdout_fh,
                "stderr_fh": stderr_fh,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "cmd": cmd,
            },
        )

    def poll(self, handle: RunHandle) -> Optional[RunResult]:
        if handle.payload.get("startup_error") is not None:
            return self._build_result(
                handle,
                RunStatus.FAILED,
                exit_code=None,
                error=str(handle.payload["startup_error"]),
                error_type="cli_error",
            )

        proc: subprocess.Popen = handle.payload["process"]
        returncode = proc.poll()

        if returncode is None:
            # Still running — check timeout.
            if (
                handle.timeout_s
                and (time.monotonic() - handle.started_monotonic) > handle.timeout_s
            ):
                self._terminate(proc)
                self._close_handles(handle)
                self._logger.log_exit(handle.task_id, "kazi", proc.pid, -9)
                return self._build_result(
                    handle,
                    RunStatus.TIMED_OUT,
                    exit_code=proc.returncode,
                    error=f"exceeded {handle.timeout_s}s timeout",
                    error_type="timeout",
                )
            return None

        self._close_handles(handle)
        self._logger.log_exit(handle.task_id, "kazi", proc.pid, returncode)
        stdout_text = self._read_log(handle.payload["stdout_path"])
        stderr_text = self._read_log(handle.payload["stderr_path"])

        try:
            parsed = parse_agent_json_envelope(stdout_text)
        except AgentRuntimeError as exc:
            return self._build_result(
                handle,
                RunStatus.FAILED,
                exit_code=returncode,
                error=str(exc),
                error_type="parse_error",
            )

        payload = _result_payload_from_parsed(parsed)
        if parsed["is_error"]:
            error_type = _classify_envelope_error(parsed)
            return self._build_result(
                handle,
                RunStatus.FAILED,
                exit_code=returncode,
                summary=parsed["text"],
                error=parsed["text"] or "agent returned is_error=true",
                error_type=error_type,
                result_payload=payload,
            )

        if returncode == 0:
            return self._build_result(
                handle,
                RunStatus.SUCCEEDED,
                exit_code=0,
                summary=parsed["text"],
                result_payload=payload,
            )

        return self._build_result(
            handle,
            RunStatus.FAILED,
            exit_code=returncode,
            summary=parsed["text"],
            error=f"exit {returncode}: {stderr_text[:300]}",
            error_type="cli_error",
            result_payload=payload,
        )

    def cancel(self, handle: RunHandle) -> None:
        proc: Optional[subprocess.Popen] = handle.payload.get("process")
        if proc is None:
            return
        self._terminate(proc)
        self._close_handles(handle)
        self._logger.log_kill(handle.task_id, "kazi", handle.pid or -1, "cancel")

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except OSError:
            pass

    @staticmethod
    def _close_handles(handle: RunHandle) -> None:
        for key in ("stdout_fh", "stderr_fh"):
            fh = handle.payload.get(key)
            if fh is not None and not fh.closed:
                try:
                    fh.close()
                except OSError:
                    pass

    @staticmethod
    def _read_log(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _build_result(
        handle: RunHandle,
        status: RunStatus,
        *,
        exit_code: Optional[int] = None,
        summary: Optional[str] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        result_payload: Optional[dict[str, Any]] = None,
    ) -> RunResult:
        payload = dict(result_payload or {})
        payload.setdefault("status", status.value)
        payload.setdefault("exit_code", exit_code)
        payload.setdefault("summary", summary)
        payload.setdefault("error", error)
        payload.setdefault("pid", handle.pid)
        if error_type is not None:
            payload["error_type"] = error_type
        return RunResult(
            status=status,
            exit_code=exit_code,
            summary=summary,
            error=error,
            error_type=error_type,
            stdout_path=handle.payload.get("stdout_path"),
            stderr_path=handle.payload.get("stderr_path"),
            result_payload=payload,
        )


# ---------------------------------------------------------------------------
# Kazi blocking runner
# ---------------------------------------------------------------------------


class KaziBlockingRunner(Runner):
    """
    Blocking adapter around the legacy `kazi.run()` entry point.

    `start()` fully executes the Kazi job inline, then `poll()`
    immediately returns its terminal `RunResult`. Useful for tests
    and any caller that wants synchronous semantics without spinning
    up the async runner.
    """

    def __init__(
        self,
        store: WorkflowStore,
        *,
        logger: Optional[Logger] = None,
        kazi_run: Optional[Callable[..., Any]] = None,
    ):
        self._store = store
        self._logger = logger or Logger()
        if kazi_run is None:
            from mr1 import kazi as _kazi
            kazi_run = _kazi.run
        self._kazi_run = kazi_run

    def start(self, task: Task) -> RunHandle:
        stdout_path, stderr_path = self._store.task_log_paths(
            task.workflow_id, task.task_id,
        )
        context = {
            "task_id": task.task_id,
            "instructions": task.prompt or task.title,
            "description": task.title,
            "timeout": task.timeout_s,
        }
        result = self._kazi_run(context, logger=self._logger)
        stdout_path.write_text(result.output or "", encoding="utf-8")
        if result.error:
            stderr_path.write_text(result.error, encoding="utf-8")

        status = {
            "completed": RunStatus.SUCCEEDED,
            "timeout": RunStatus.TIMED_OUT,
            "denied": RunStatus.FAILED,
            "invalid": RunStatus.FAILED,
            "context_exceeded": RunStatus.FAILED,
            "failed": RunStatus.FAILED,
        }.get(result.status, RunStatus.FAILED)

        run_result = RunResult(
            status=status,
            exit_code=0 if status is RunStatus.SUCCEEDED else 1,
            summary=result.output,
            error=result.error,
            error_type=getattr(result, "error_type", None),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result_payload=dict(getattr(result, "payload", {}) or {
                "status": status.value,
                "kazi_status": result.status,
                "summary": result.output,
                "error": result.error,
                "pid": result.pid,
            }),
        )
        return RunHandle(
            task_id=task.task_id,
            workflow_id=task.workflow_id,
            pid=result.pid,
            timeout_s=task.timeout_s,
            payload={"result": run_result},
        )

    def poll(self, handle: RunHandle) -> Optional[RunResult]:
        return handle.payload.get("result")

    def cancel(self, handle: RunHandle) -> None:
        # Blocking runner already finished before returning from start().
        return


# ---------------------------------------------------------------------------
# Mock runner (test-only)
# ---------------------------------------------------------------------------


class MockRunner(Runner):
    """
    In-memory runner driven by test callbacks.

    Tests can either:
      * call `complete(task_id, status, ...)` directly to end a task,
      * or pass `on_start` / `on_poll` callbacks to shape behaviour.

    `start()` records the task as running and does nothing else.
    `poll()` first consults the pending-terminal map, then the optional
    `on_poll` callback.
    """

    def __init__(
        self,
        *,
        on_start: Optional[Callable[[Task], None]] = None,
        on_poll: Optional[Callable[[RunHandle], Optional[RunResult]]] = None,
    ):
        self._on_start = on_start
        self._on_poll = on_poll
        self._running: dict[str, RunHandle] = {}
        self._terminal: dict[str, RunResult] = {}
        self._start_log: list[str] = []

    @property
    def started_task_ids(self) -> list[str]:
        return list(self._start_log)

    def start(self, task: Task) -> RunHandle:
        handle = RunHandle(
            task_id=task.task_id,
            workflow_id=task.workflow_id,
            pid=None,
        )
        self._running[task.task_id] = handle
        self._start_log.append(task.task_id)
        if self._on_start:
            self._on_start(task)
        return handle

    def poll(self, handle: RunHandle) -> Optional[RunResult]:
        if handle.task_id in self._terminal:
            result = self._terminal.pop(handle.task_id)
            self._running.pop(handle.task_id, None)
            return result
        if self._on_poll:
            result = self._on_poll(handle)
            if result is not None:
                self._running.pop(handle.task_id, None)
                return result
        return None

    def cancel(self, handle: RunHandle) -> None:
        self._running.pop(handle.task_id, None)
        self._terminal.pop(handle.task_id, None)

    def complete(
        self,
        task_id: str,
        status: RunStatus = RunStatus.SUCCEEDED,
        *,
        summary: Optional[str] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        exit_code: Optional[int] = None,
        result_payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """Flag a running task to return `status` on the next poll."""
        self._terminal[task_id] = RunResult(
            status=status,
            exit_code=exit_code if exit_code is not None else (0 if status is RunStatus.SUCCEEDED else 1),
            summary=summary,
            error=error,
            error_type=error_type,
            result_payload=result_payload or {
                "status": status.value,
                "summary": summary,
                "error": error,
                "error_type": error_type,
            },
        )
