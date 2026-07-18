"""
The control plane.

Control is a file, not a socket: crash-safe, inspectable, cross-process, and
it needs no new transport. `control.json` is written with the same
fsync + atomic-replace discipline as `mr1_state.json` and read on every
supervisor tick.

Modes
-----
running   normal operation; planning and execution both allowed
paused    no new autonomous work; in-flight work keeps draining
stopping  no new autonomous work; drain in-flight, then exit
halted    emergency stop; cancel running work, revoke every consent grant,
          pause every objective, exit

The semantic that matters: `pause` stops new work, `halt` removes MR1's
authority. A stop that leaves standing consent in place has not stopped
anything — which is why halt revokes grants.

A missing control file means `running` (absent = never configured). An
*unreadable* one means `paused`: a corrupt control plane must never be read
as permission to act.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from mr1.clock import Clock, default_clock
from mr1.event_log import EventLog


CONTROL_FILE_NAME = "control.json"
PID_FILE_NAME = "supervisor.pid"

MODE_RUNNING = "running"
MODE_PAUSED = "paused"
MODE_STOPPING = "stopping"
MODE_HALTED = "halted"

CONTROL_MODES = (MODE_RUNNING, MODE_PAUSED, MODE_STOPPING, MODE_HALTED)


class ServiceLockError(RuntimeError):
    """Raised when a second supervisor tries to start on the same runtime root."""


@dataclass(frozen=True)
class ControlState:
    mode: str
    reason: str = ""
    requested_by: str = ""
    requested_at: str = ""
    corrupt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
        }

    @property
    def planning_allowed(self) -> bool:
        """May new autonomous work be created?"""
        return self.mode == MODE_RUNNING

    @property
    def draining(self) -> bool:
        """May in-flight work keep running to completion?"""
        return self.mode in {MODE_RUNNING, MODE_PAUSED, MODE_STOPPING}

    @property
    def should_exit(self) -> bool:
        return self.mode in {MODE_STOPPING, MODE_HALTED}


class ControlPlane:
    def __init__(self, runtime_root: Path, *, clock: Optional[Clock] = None):
        self._runtime_root = Path(runtime_root)
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or default_clock()
        self._event_log = EventLog(self._runtime_root / "events")

    @property
    def path(self) -> Path:
        return self._runtime_root / CONTROL_FILE_NAME

    @property
    def runtime_root(self) -> Path:
        return self._runtime_root

    def read(self) -> ControlState:
        path = self.path
        if not path.exists():
            return ControlState(mode=MODE_RUNNING, reason="no control file")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            mode = payload["mode"]
            if mode not in CONTROL_MODES:
                raise ValueError(f"unknown control mode: {mode}")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            # Fail closed: an unreadable control plane must not be read as
            # permission to create new autonomous work.
            return ControlState(
                mode=MODE_PAUSED,
                reason=f"control file unreadable: {exc}",
                corrupt=True,
            )
        return ControlState(
            mode=mode,
            reason=str(payload.get("reason") or ""),
            requested_by=str(payload.get("requested_by") or ""),
            requested_at=str(payload.get("requested_at") or ""),
        )

    def set_mode(
        self,
        mode: str,
        *,
        reason: str = "",
        requested_by: str = "operator",
    ) -> ControlState:
        if mode not in CONTROL_MODES:
            raise ValueError(f"unknown control mode: {mode}")
        previous = self.read()
        state = ControlState(
            mode=mode,
            reason=reason,
            requested_by=requested_by,
            requested_at=self._clock.now_iso(),
        )
        self._write(state)
        if previous.mode != mode:
            self._emit_mode_change(previous, state)
        return state

    def _write(self, state: ControlState) -> None:
        path = self.path
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        _fsync_directory(path.parent)

    def _emit_mode_change(self, previous: ControlState, state: ControlState) -> None:
        try:
            self._event_log.emit(
                event_type="control_mode_changed",
                actor_id=state.requested_by or "operator",
                actor_type="root_orchestrator",
                target_id="supervisor",
                target_type="supervisor",
                status=state.mode,
                summary=f"control mode {previous.mode} -> {state.mode}",
                record_path=str(self.path),
                metadata={
                    "previous_mode": previous.mode,
                    "mode": state.mode,
                    "reason": state.reason,
                    "requested_by": state.requested_by,
                },
            )
        except Exception:
            # The control plane must remain usable even if the timeline is not.
            pass


class ServiceLock:
    """
    Cross-process singleton for `mr1 serve`.

    An exclusive `fcntl.flock` on the pidfile. The lock dies with the process,
    so a crashed supervisor never leaves a stale lock behind — unlike a bare
    pidfile, which is why the pid is only advisory content here.
    """

    def __init__(self, runtime_root: Path):
        self._path = Path(runtime_root) / PID_FILE_NAME
        self._handle = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> int:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a+")
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                handle.close()
                running_pid = self.read_pid()
                raise ServiceLockError(
                    "another mr1 supervisor is already running on this runtime root"
                    + (f" (pid {running_pid})" if running_pid else "")
                ) from exc
        pid = os.getpid()
        handle.seek(0)
        handle.truncate()
        handle.write(str(pid))
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return pid

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
        try:
            self._path.unlink()
        except OSError:
            pass

    def read_pid(self) -> Optional[int]:
        try:
            raw = self._path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def is_held_by_live_process(self) -> bool:
        """True when some other live process holds the lock."""
        if not self._path.exists():
            return False
        if fcntl is None:  # pragma: no cover - non-POSIX fallback
            return self.read_pid() is not None
        try:
            with open(self._path, "a+") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return True
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            return False
        return False

    def __enter__(self) -> "ServiceLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()


def _fsync_directory(path: Path) -> None:
    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
