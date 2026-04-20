"""
MR1 Spawner — Subprocess Lifecycle Manager
===========================================
Spawns claude CLI processes, tracks them by agent name and task_id,
and can kill them by name or task_id.

CRITICAL: Always calls Dispatcher before spawning. If the dispatcher
rejects the command, the process is never created.

Completed processes are cleaned up from tracking when their result
is retrieved — no process accumulation.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .dispatcher import Dispatcher, PermissionDenied
from .logger import Logger


@dataclass
class ProcessRecord:
    """Tracks a single spawned subprocess."""
    pid: int
    agent_type: str
    task_id: str
    process: subprocess.Popen
    cmd: list[str]


class Spawner:
    """
    Manages the lifecycle of claude CLI subprocesses.

    Usage:
        spawner = Spawner()
        record = spawner.spawn(
            agent_type="kazi",
            task_id="task-001",
            prompt="Summarize the README",
            model="claude-3-5-haiku-20241022",
            tools=["Read", "Glob"],
        )
        spawner.kill_by_task("task-001")
    """

    def __init__(
        self,
        dispatcher: Optional[Dispatcher] = None,
        logger: Optional[Logger] = None,
        claude_binary: str = "claude",
    ):
        self._dispatcher = dispatcher or Dispatcher()
        self._logger = logger or Logger()
        self._claude_binary = claude_binary

        # Active processes indexed two ways for fast lookup.
        self._by_pid: dict[int, ProcessRecord] = {}
        self._by_task: dict[str, list[ProcessRecord]] = {}

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    def spawn(
        self,
        agent_type: str,
        task_id: str,
        prompt: str,
        model: Optional[str] = None,
        tools: Optional[list[str]] = None,
        extra_flags: Optional[list[str]] = None,
        cwd: Optional[str] = None,
    ) -> ProcessRecord:
        """
        Build a claude CLI command, validate it through the dispatcher,
        then spawn the subprocess.

        Raises PermissionDenied if the dispatcher rejects any part of the command.
        """
        tools = tools or []
        extra_flags = extra_flags or []

        # Build the CLI argument list.
        cmd = [self._claude_binary, "-p", prompt]

        if model:
            cmd.extend(["--model", model])

        if tools:
            cmd.extend(["--allowedTools", ",".join(tools)])

        cmd.extend(extra_flags)

        # Extract just the flag tokens for validation.
        cli_flags = [tok for tok in cmd[1:] if tok.startswith("-")]

        # --- DISPATCHER GATE ---
        # If this raises, we never touch subprocess.
        try:
            self._dispatcher.validate_full_spawn(agent_type, cli_flags, tools)
        except PermissionDenied as e:
            self._logger.log_denied(task_id, agent_type, str(e))
            raise

        # --- SPAWN ---
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )

        record = ProcessRecord(
            pid=proc.pid,
            agent_type=agent_type,
            task_id=task_id,
            process=proc,
            cmd=cmd,
        )

        # Index the record.
        self._by_pid[proc.pid] = record
        self._by_task.setdefault(task_id, []).append(record)

        self._logger.log_spawn(task_id, agent_type, proc.pid, cmd)
        return record

    # ------------------------------------------------------------------
    # Kill
    # ------------------------------------------------------------------

    def kill_by_pid(self, pid: int, reason: str = "manual") -> bool:
        """Kill a specific process by PID. Returns True if it was tracked."""
        record = self._by_pid.pop(pid, None)
        if record is None:
            return False

        self._terminate(record, reason)
        self._remove_from_task_index(record)
        return True

    def kill_by_task(self, task_id: str, reason: str = "task_cancel") -> int:
        """Kill all processes for a task. Returns count of processes killed."""
        records = self._by_task.pop(task_id, [])
        for record in records:
            self._by_pid.pop(record.pid, None)
            self._terminate(record, reason)
        return len(records)

    def kill_by_agent_type(self, agent_type: str, reason: str = "agent_type_cancel") -> int:
        """Kill all processes of a given agent type. Returns count killed."""
        to_kill = [r for r in self._by_pid.values() if r.agent_type == agent_type]
        for record in to_kill:
            self._by_pid.pop(record.pid, None)
            self._remove_from_task_index(record)
            self._terminate(record, reason)
        return len(to_kill)

    def kill_all(self, reason: str = "shutdown") -> int:
        """Kill every tracked process. Returns count killed."""
        count = len(self._by_pid)
        for record in list(self._by_pid.values()):
            self._terminate(record, reason)
        self._by_pid.clear()
        self._by_task.clear()
        return count

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def list_active(self) -> list[dict]:
        """Return a snapshot of all tracked processes."""
        self._reap_finished()
        return [
            {
                "pid": r.pid,
                "agent_type": r.agent_type,
                "task_id": r.task_id,
                "running": r.process.poll() is None,
            }
            for r in self._by_pid.values()
        ]

    def is_alive(self, pid: int) -> bool:
        """Check if a tracked process is still running."""
        record = self._by_pid.get(pid)
        if record is None:
            return False
        return record.process.poll() is None

    def get_result(self, pid: int) -> Optional[dict]:
        """
        Get stdout/stderr from a finished process.
        Returns None if the process is still running or not tracked.

        IMPORTANT: Cleans up the process record after returning the result.
        This prevents zombie records from accumulating.
        """
        record = self._by_pid.get(pid)
        if record is None:
            return None
        if record.process.poll() is None:
            return None  # Still running.

        stdout, stderr = record.process.communicate()

        # Clean up tracking — the process is done.
        self._by_pid.pop(pid, None)
        self._remove_from_task_index(record)

        return {
            "pid": pid,
            "returncode": record.process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _terminate(self, record: ProcessRecord, reason: str) -> None:
        """Send SIGTERM, then SIGKILL if the process doesn't exit quickly."""
        try:
            record.process.terminate()
            try:
                record.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                record.process.kill()
        except OSError:
            pass  # Already dead.

        returncode = record.process.poll()
        self._logger.log_kill(record.task_id, record.agent_type, record.pid, reason)
        if returncode is not None:
            self._logger.log_exit(
                record.task_id, record.agent_type, record.pid, returncode
            )

    def _remove_from_task_index(self, record: ProcessRecord) -> None:
        """Remove a record from the task index."""
        task_records = self._by_task.get(record.task_id, [])
        self._by_task[record.task_id] = [
            r for r in task_records if r.pid != record.pid
        ]
        if not self._by_task[record.task_id]:
            del self._by_task[record.task_id]

    def _reap_finished(self) -> None:
        """Log and clean up any processes that have exited on their own."""
        finished_pids = []
        for pid, record in self._by_pid.items():
            returncode = record.process.poll()
            if returncode is not None:
                self._logger.log_exit(
                    record.task_id, record.agent_type, pid, returncode
                )
                finished_pids.append(pid)

        for pid in finished_pids:
            record = self._by_pid.pop(pid)
            self._remove_from_task_index(record)
