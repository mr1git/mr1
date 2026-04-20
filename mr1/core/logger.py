"""
MR1 Logger — Structured JSON Task Logging
==========================================
Fully deterministic. No LLM logic.

Writes structured JSON log entries to tasks/{task_id}/logs/ with:
  - timestamp (ISO 8601 UTC)
  - agent_type
  - action
  - result
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# tasks/ directory lives at the mr1 package root.
_TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"


class Logger:
    """
    Writes one JSON object per log entry, one entry per line (JSONL format).
    Log file: tasks/{task_id}/logs/{agent_type}.jsonl
    """

    def __init__(self, tasks_dir: Optional[str] = None):
        self._tasks_dir = Path(tasks_dir) if tasks_dir else _TASKS_DIR

    def _ensure_log_dir(self, task_id: str) -> Path:
        log_dir = self._tasks_dir / task_id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def log(
        self,
        task_id: str,
        agent_type: str,
        action: str,
        result: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Write a structured log entry.

        Returns the entry dict for convenience (e.g. testing, chaining).
        """
        log_dir = self._ensure_log_dir(task_id)
        log_file = log_dir / f"{agent_type}.jsonl"

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "agent_type": agent_type,
            "action": action,
            "result": result,
        }
        if metadata:
            entry["metadata"] = metadata

        with open(log_file, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

        return entry

    def log_spawn(self, task_id: str, agent_type: str, pid: int, cmd: list[str]) -> dict:
        """Convenience: log a process spawn event."""
        return self.log(
            task_id=task_id,
            agent_type=agent_type,
            action="spawn",
            result="ok",
            metadata={"pid": pid, "cmd": cmd},
        )

    def log_kill(self, task_id: str, agent_type: str, pid: int, reason: str) -> dict:
        """Convenience: log a process kill event."""
        return self.log(
            task_id=task_id,
            agent_type=agent_type,
            action="kill",
            result="ok",
            metadata={"pid": pid, "reason": reason},
        )

    def log_denied(self, task_id: str, agent_type: str, detail: str) -> dict:
        """Convenience: log a permission denial."""
        return self.log(
            task_id=task_id,
            agent_type=agent_type,
            action="permission_check",
            result="denied",
            metadata={"detail": detail},
        )

    def log_exit(self, task_id: str, agent_type: str, pid: int, returncode: int) -> dict:
        """Convenience: log a process exit."""
        return self.log(
            task_id=task_id,
            agent_type=agent_type,
            action="exit",
            result="ok" if returncode == 0 else "error",
            metadata={"pid": pid, "returncode": returncode},
        )

    def read_logs(self, task_id: str, agent_type: Optional[str] = None) -> list[dict]:
        """
        Read back log entries for a task.
        If agent_type is given, reads only that agent's log file.
        Otherwise reads all log files for the task.
        """
        log_dir = self._tasks_dir / task_id / "logs"
        if not log_dir.exists():
            return []

        entries = []
        if agent_type:
            files = [log_dir / f"{agent_type}.jsonl"]
        else:
            files = sorted(log_dir.glob("*.jsonl"))

        for log_file in files:
            if not log_file.exists():
                continue
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))

        # Sort by timestamp for consistent ordering across agent files.
        entries.sort(key=lambda e: e.get("timestamp", ""))
        return entries
