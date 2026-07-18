"""MR1 persistent orchestrator state.

`StateManager` owns the JSON document at `memory/active/mr1_state.json`:
the current session, active/completed tasks, running agent PIDs, and
a rolling window of recent orchestration decisions and conversation
turns. It is shared by the root MR1 instance and any test harness that
needs a clean state file.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from .identity import _PKG_ROOT, _now_iso


_STATE_PATH = _PKG_ROOT / "memory" / "active" / "mr1_state.json"

_MAX_DECISIONS = 50
_MAX_CONVERSATION = 80
_MAX_RUNTIME_ERRORS = 20

_TERMINAL_TASK_STATUSES = {
    "completed",
    "failed",
    "timeout",
    "context_exceeded",
    "denied",
    "killed",
}


class StateCorruptionError(RuntimeError):
    def __init__(self, path: Path, reason: str):
        self.path = Path(path)
        self.reason = reason
        super().__init__(f"runtime state is corrupted at {self.path}: {reason}")


class StateManager:
    """
    Manages MR1's persistent state at memory/active/mr1_state.json.

    Tracks the current session, active/completed tasks, running agent
    PIDs, and a rolling window of recent orchestration decisions.
    """

    def __init__(self, state_path: Path = _STATE_PATH):
        self._path = state_path
        self._lock = threading.RLock()
        self._state = self._load_or_init()

    def _load_or_init(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise TypeError("top-level JSON value must be an object")
                self._ensure_reference_defaults(data)
                return data
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise StateCorruptionError(self._path, str(exc)) from exc
        data = {
            "session_id": uuid.uuid4().hex[:12],
            "started_at": _now_iso(),
            "claude_session_id": None,
            "tasks": {},
            "decisions": [],
            "agent_pids": [],
            "conversation": [],
            "pending_workflow": None,
            "runtime_errors": [],
        }
        self._ensure_reference_defaults(data)
        return data

    @staticmethod
    def _ensure_reference_defaults(state: dict[str, Any]) -> None:
        state.setdefault("last_created_agent_id", None)
        state.setdefault("last_referenced_agent_id", None)
        state.setdefault("last_created_workflow_id", None)
        state.setdefault("last_referenced_workflow_id", None)
        aliases = state.setdefault("reference_aliases", {})
        if not isinstance(aliases, dict):
            aliases = {}
            state["reference_aliases"] = aliases
        agents = aliases.get("agents")
        workflows = aliases.get("workflows")
        aliases["agents"] = dict(agents) if isinstance(agents, dict) else {}
        aliases["workflows"] = dict(workflows) if isinstance(workflows, dict) else {}
        runtime_errors = state.get("runtime_errors", [])
        state["runtime_errors"] = list(runtime_errors) if isinstance(runtime_errors, list) else []

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self._path)
        self._fsync_directory(self._path.parent)

    def _snapshot_locked(self) -> dict[str, Any]:
        return deepcopy(self._state)

    @staticmethod
    def _active_tasks_from_state(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            tid: deepcopy(task)
            for tid, task in state["tasks"].items()
            if task["status"] == "running"
        }

    def set_claude_session_id(self, session_id: Optional[str]) -> None:
        with self._lock:
            self._state["claude_session_id"] = session_id
            self._save_locked()

    # -- Tasks -------------------------------------------------------------

    def begin_task(
        self,
        task_id: str,
        agent_type: str,
        description: str,
        pid: Optional[int] = None,
        parent_task_id: str = "mr1",
        lane: str = "conversation",
    ) -> None:
        with self._lock:
            existing = self._state["tasks"].get(task_id, {})
            task = {
                "agent_type": agent_type,
                "status": existing.get("status", "running"),
                "pid": existing.get("pid"),
                "description": description[:300],
                "started_at": existing.get("started_at", _now_iso()),
                "parent_task_id": existing.get("parent_task_id", parent_task_id),
                "lane": existing.get("lane", lane),
            }
            if pid is not None:
                task["pid"] = pid
            if task["status"] not in _TERMINAL_TASK_STATUSES:
                task["status"] = "running"
            self._state["tasks"][task_id] = task
            self._save_locked()

    def add_task(
        self,
        task_id: str,
        agent_type: str,
        description: str,
        pid: Optional[int],
    ) -> None:
        self.begin_task(task_id, agent_type, description, pid=pid)

    def update_task_pid(self, task_id: str, pid: int) -> None:
        with self._lock:
            if task_id not in self._state["tasks"]:
                return
            self._state["tasks"][task_id]["pid"] = pid
            self._save_locked()

    def complete_task(self, task_id: str, status: str = "completed") -> None:
        with self._lock:
            if task_id in self._state["tasks"]:
                self._state["tasks"][task_id]["status"] = status
                self._state["tasks"][task_id]["finished_at"] = _now_iso()
                self._save_locked()

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            task = self._state["tasks"].get(task_id)
            if task is None:
                return None
            return deepcopy(task)

    @property
    def active_tasks(self) -> dict:
        with self._lock:
            return self._active_tasks_from_state(self._state)

    # -- Decisions ---------------------------------------------------------

    def add_decision(
        self,
        user_input: str,
        action: str,
        task_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._state["decisions"].append({
                "timestamp": _now_iso(),
                "input_summary": user_input[:200],
                "action": action,
                "task_id": task_id,
            })
            # Rolling window.
            if len(self._state["decisions"]) > _MAX_DECISIONS:
                self._state["decisions"] = self._state["decisions"][-_MAX_DECISIONS:]
            self._save_locked()

    def add_conversation(
        self,
        role: str,
        text: str,
        kind: str = "message",
        task_id: Optional[str] = None,
        lane: str = "conversation",
    ) -> dict[str, Any]:
        with self._lock:
            entry = {
                "timestamp": _now_iso(),
                "role": role,
                "text": text[:3000],
                "kind": kind,
                "task_id": task_id,
                "lane": lane,
            }
            self._state.setdefault("conversation", []).append(entry)
            if len(self._state["conversation"]) > _MAX_CONVERSATION:
                self._state["conversation"] = self._state["conversation"][-_MAX_CONVERSATION:]
            self._save_locked()
            return deepcopy(entry)

    def set_pending_workflow(self, draft: Optional[dict[str, Any]]) -> None:
        with self._lock:
            self._state["pending_workflow"] = deepcopy(draft) if draft is not None else None
            self._save_locked()

    def clear_pending_workflow(self) -> None:
        self.set_pending_workflow(None)

    def set_reference_state(self, key: str, value: Optional[str]) -> None:
        with self._lock:
            self._state[key] = value
            self._save_locked()

    def set_reference_alias(self, kind: str, alias: str, target_id: str) -> None:
        normalized_alias = alias.strip().lower()
        if not normalized_alias:
            return
        with self._lock:
            aliases = self._state.setdefault("reference_aliases", {})
            bucket = aliases.setdefault(kind, {})
            bucket[normalized_alias] = target_id
            self._save_locked()

    def record_runtime_error(
        self,
        source: str,
        error: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            entry = {
                "timestamp": _now_iso(),
                "source": source,
                "error": error[:1000],
                "details": deepcopy(details) if isinstance(details, dict) else {},
            }
            self._state.setdefault("runtime_errors", []).append(entry)
            if len(self._state["runtime_errors"]) > _MAX_RUNTIME_ERRORS:
                self._state["runtime_errors"] = self._state["runtime_errors"][-_MAX_RUNTIME_ERRORS:]
            self._save_locked()
            return deepcopy(entry)

    @property
    def reference_aliases(self) -> dict[str, dict[str, str]]:
        with self._lock:
            aliases = self._state.get("reference_aliases", {})
            return {
                "agents": dict(aliases.get("agents", {})),
                "workflows": dict(aliases.get("workflows", {})),
            }

    @property
    def last_created_agent_id(self) -> Optional[str]:
        with self._lock:
            return self._state.get("last_created_agent_id")

    @property
    def last_referenced_agent_id(self) -> Optional[str]:
        with self._lock:
            return self._state.get("last_referenced_agent_id")

    @property
    def last_created_workflow_id(self) -> Optional[str]:
        with self._lock:
            return self._state.get("last_created_workflow_id")

    @property
    def last_referenced_workflow_id(self) -> Optional[str]:
        with self._lock:
            return self._state.get("last_referenced_workflow_id")

    # -- Agent PIDs --------------------------------------------------------

    def add_agent_pid(self, pid: int) -> None:
        with self._lock:
            if pid not in self._state.get("agent_pids", []):
                self._state.setdefault("agent_pids", []).append(pid)
                self._save_locked()

    def remove_agent_pid(self, pid: int) -> None:
        with self._lock:
            pids = self._state.get("agent_pids", [])
            if pid in pids:
                pids.remove(pid)
                self._save_locked()

    # -- Accessors ---------------------------------------------------------

    @property
    def session_id(self) -> str:
        with self._lock:
            return self._state["session_id"]

    @property
    def conversation(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._state.get("conversation", []))

    @property
    def claude_session_id(self) -> Optional[str]:
        with self._lock:
            return self._state.get("claude_session_id")

    @property
    def pending_workflow(self) -> Optional[dict[str, Any]]:
        with self._lock:
            value = self._state.get("pending_workflow")
            return deepcopy(value) if isinstance(value, dict) else None

    @property
    def runtime_errors(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._state.get("runtime_errors", []))

    @property
    def decisions(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._state.get("decisions", []))

    @property
    def tasks(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return deepcopy(self._state.get("tasks", {}))

    def format_status(self) -> str:
        """Human-readable status block."""
        with self._lock:
            state = self._snapshot_locked()
        active = self._active_tasks_from_state(state)
        recent = state["decisions"][-5:]
        lines = [
            f"Session:  {state['session_id']}",
            f"Started:  {state['started_at']}",
            f"Active tasks: {len(active)}",
        ]
        for tid, t in active.items():
            lines.append(
                f"  {tid}  [{t['agent_type']}]  pid={t['pid']}  {t['description'][:60]}"
            )
        if recent:
            lines.append("Recent decisions:")
            for d in recent:
                lines.append(
                    f"  {d['timestamp'][:19]}  {d['action']}"
                    + (f"  ({d['task_id']})" if d.get("task_id") else "")
                )
        runtime_errors = state.get("runtime_errors", [])
        if runtime_errors:
            latest_error = runtime_errors[-1]
            lines.append(
                f"Last runtime error: {latest_error['timestamp'][:19]} "
                f"{latest_error['source']} {latest_error['error'][:120]}"
            )
        return "\n".join(lines)

    def format_tasks(self) -> str:
        """Human-readable task list."""
        with self._lock:
            state = self._snapshot_locked()
        if not state["tasks"]:
            return "No tasks."
        lines = []
        for tid, t in state["tasks"].items():
            status_icon = {
                "running": "~",
                "completed": "+",
                "failed": "!",
                "killed": "x",
            }.get(t["status"], "?")
            lines.append(
                f"  [{status_icon}] {tid}  {t['agent_type']}  "
                f"{t['status']}  {t['description'][:50]}"
            )
        return "\n".join(lines)

    def format_for_prompt(self) -> str:
        """Compact state summary."""
        with self._lock:
            state = self._snapshot_locked()
        active = self._active_tasks_from_state(state)
        if not active:
            return "No active tasks."
        parts = []
        for tid, t in active.items():
            parts.append(f"{tid} [{t['agent_type']}]: {t['description'][:80]}")
        return "\n".join(parts)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            dir_fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
