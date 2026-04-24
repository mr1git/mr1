"""
Deterministic watcher evaluators for workflow tasks.

Watchers are condition gates only. They never invoke an LLM and do not
launch through the agent runner abstraction.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol

from mr1.workflow_models import Task


class WatcherConfigError(ValueError):
    """Raised when a watcher type or config is invalid."""


@dataclass
class WatchEvaluation:
    state: str  # satisfied | not_satisfied | failed | timed_out
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


class WatcherEvaluator(Protocol):
    def validate_config(self, watch_config: dict[str, Any]) -> None: ...

    def evaluate(self, task: Task, now: datetime) -> WatchEvaluation: ...


def _local_tzinfo():
    return datetime.now().astimezone().tzinfo


def _parse_iso_timestamp(raw: Any) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise WatcherConfigError("watch_config.at must be a non-empty ISO timestamp string")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise WatcherConfigError(f"invalid ISO timestamp: {raw}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_local_tzinfo())
    return dt


def _truncate(text: str, limit: int = 500) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class FileExistsWatcher:
    def validate_config(self, watch_config: dict[str, Any]) -> None:
        path = watch_config.get("path")
        if not isinstance(path, str) or not path.strip():
            raise WatcherConfigError("file_exists requires watch_config.path")

    def evaluate(self, task: Task, now: datetime) -> WatchEvaluation:
        path = Path(task.watch_config["path"])
        exists = path.exists()
        return WatchEvaluation(
            state="satisfied" if exists else "not_satisfied",
            message=f"path {'exists' if exists else 'does not exist'}: {path}",
            metadata={"path": str(path), "exists": exists},
        )


class TimeReachedWatcher:
    def validate_config(self, watch_config: dict[str, Any]) -> None:
        _parse_iso_timestamp(watch_config.get("at"))

    def evaluate(self, task: Task, now: datetime) -> WatchEvaluation:
        target = _parse_iso_timestamp(task.watch_config["at"])
        satisfied = now >= target
        return WatchEvaluation(
            state="satisfied" if satisfied else "not_satisfied",
            message=f"time {'reached' if satisfied else 'not reached'}: {target.isoformat()}",
            metadata={"at": target.isoformat(), "now": now.isoformat()},
        )


class ManualEventWatcher:
    def validate_config(self, watch_config: dict[str, Any]) -> None:
        event = watch_config.get("event")
        if not isinstance(event, str) or not event.strip():
            raise WatcherConfigError("manual_event requires watch_config.event")

    def evaluate(self, task: Task, now: datetime) -> WatchEvaluation:
        condition = task.condition or {}
        triggered = bool(condition.get("triggered"))
        event_name = task.watch_config.get("event")
        if triggered:
            return WatchEvaluation(
                state="satisfied",
                message=f"manual event triggered: {event_name}",
                metadata={
                    "event": event_name,
                    "triggered": True,
                    "triggered_at": condition.get("triggered_at"),
                    "trigger_metadata": dict(condition.get("metadata") or {}),
                },
            )
        return WatchEvaluation(
            state="not_satisfied",
            message=f"waiting for manual event: {event_name}",
            metadata={"event": event_name, "triggered": False},
        )


def _resolve_condition_script_path(raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise WatcherConfigError("condition_script requires watch_config.path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise WatcherConfigError(f"condition_script path does not exist: {path}")
    if not path.is_file():
        raise WatcherConfigError(f"condition_script path is not a file: {path}")
    return path


class ConditionScriptWatcher:
    def validate_config(self, watch_config: dict[str, Any]) -> None:
        _resolve_condition_script_path(watch_config.get("path"))
        timeout_s = watch_config.get("timeout_s", 10)
        if not isinstance(timeout_s, int) or timeout_s < 1:
            raise WatcherConfigError("condition_script timeout_s must be an integer >= 1")

    def evaluate(self, task: Task, now: datetime) -> WatchEvaluation:
        path = _resolve_condition_script_path(task.watch_config.get("path"))
        timeout_s = task.watch_config.get("timeout_s", 10)
        cmd = [sys.executable, str(path)] if path.suffix == ".py" else [str(path)]
        try:
            proc = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(Path.cwd()),
            )
        except subprocess.TimeoutExpired as exc:
            return WatchEvaluation(
                state="timed_out",
                message=f"condition script timed out after {timeout_s}s",
                metadata={
                    "path": str(path),
                    "timeout_s": timeout_s,
                    "stdout": _truncate(exc.stdout or ""),
                    "stderr": _truncate(exc.stderr or ""),
                },
            )
        except OSError as exc:
            return WatchEvaluation(
                state="failed",
                message=f"condition script failed to start: {exc}",
                metadata={"path": str(path), "error": str(exc)},
            )

        state = {
            0: "satisfied",
            1: "not_satisfied",
        }.get(proc.returncode, "failed")
        if state == "satisfied":
            message = f"condition script satisfied: {path.name}"
        elif state == "not_satisfied":
            message = f"condition script not satisfied: {path.name}"
        else:
            message = f"condition script failed with exit {proc.returncode}"
        return WatchEvaluation(
            state=state,
            message=message,
            metadata={
                "path": str(path),
                "exit_code": proc.returncode,
                "stdout": _truncate(proc.stdout or ""),
                "stderr": _truncate(proc.stderr or ""),
                "command": cmd,
            },
        )


class WatcherRegistry:
    def __init__(self):
        self._evaluators: dict[str, WatcherEvaluator] = {}

    def register(self, watcher_type: str, evaluator: WatcherEvaluator) -> None:
        self._evaluators[watcher_type] = evaluator

    def is_registered(self, watcher_type: Optional[str]) -> bool:
        return bool(watcher_type) and watcher_type in self._evaluators

    def validate_spec(
        self,
        watcher_type: Optional[str],
        watch_config: Optional[dict[str, Any]],
    ) -> None:
        if not isinstance(watcher_type, str) or not watcher_type:
            raise WatcherConfigError("watcher task requires watcher_type")
        evaluator = self._evaluators.get(watcher_type)
        if evaluator is None:
            raise WatcherConfigError(f"unknown watcher_type '{watcher_type}'")
        if watch_config is None:
            watch_config = {}
        if not isinstance(watch_config, dict):
            raise WatcherConfigError("watch_config must be a JSON object")
        evaluator.validate_config(watch_config)

    def evaluate(self, task: Task, now: datetime) -> WatchEvaluation:
        evaluator = self._evaluators.get(task.watcher_type or "")
        if evaluator is None:
            raise WatcherConfigError(f"unknown watcher_type '{task.watcher_type}'")
        return evaluator.evaluate(task, now)


_DEFAULT_REGISTRY: Optional[WatcherRegistry] = None


def default_watcher_registry() -> WatcherRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = WatcherRegistry()
        registry.register("file_exists", FileExistsWatcher())
        registry.register("time_reached", TimeReachedWatcher())
        registry.register("manual_event", ManualEventWatcher())
        registry.register("condition_script", ConditionScriptWatcher())
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
