"""
Execution ownership — one execution authority per runtime root.

`Scheduler._tick_lock` is a `threading.Lock`: it makes a tick atomic *inside*
one process and says nothing about any other. `mr1 serve` is a singleton, but
nothing stopped an independently opened REPL — or a CLI that ticks once — from
driving `start()` on the very same task the service was already running. Two
processes launching one task is not a race that resolves itself: it is the same
side effect executed twice.

The lock is `fcntl.flock` on `<runtime_root>/execution.lock`, for two reasons:

  * It is held by the *open file description*, not by the PID. A crashed owner
    releases it the moment the kernel closes its files, so ownership never goes
    stale and there is no "is pid 4711 still alive, and is it still *mine*"
    guesswork. A PID file cannot express that.
  * flock conflicts across separate `open()` calls even within one process, so
    two `Scheduler`s in a single interpreter exclude each other exactly as two
    processes do. The semantics do not change with the deployment shape.

`execution_owner.json` alongside it is *advisory only* — it exists so an
operator can be told who holds the lock. The lock is the authority; the JSON is
the label. A reader that finds the JSON but no live lock reports "not owned",
which is why a leftover file after `kill -9` is harmless.

A process that does not hold the lock may read every store, list workflows,
submit specs to disk, and issue control commands. What it may not do is
launch, poll, or finalize a task: `Scheduler.tick()` returns immediately and
counts a delegated tick instead.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from mr1.clock import Clock, default_clock


EXECUTION_LOCK_NAME = "execution.lock"
EXECUTION_OWNER_NAME = "execution_owner.json"

ROLE_SERVICE = "service"
ROLE_REPL = "repl"
ROLE_CLI = "cli"


@dataclass(frozen=True)
class ExecutionOwner:
    """Who holds execution authority. Advisory metadata, not the authority."""

    pid: int
    host: str
    role: str
    acquired_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "host": self.host,
            "role": self.role,
            "acquired_at": self.acquired_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Optional["ExecutionOwner"]:
        try:
            return cls(
                pid=int(payload["pid"]),
                host=str(payload.get("host") or ""),
                role=str(payload.get("role") or "unknown"),
                acquired_at=str(payload.get("acquired_at") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None


class ExecutionOwnership:
    """
    The execution authority for one runtime root.

    Not acquired on construction: a process decides deliberately whether it
    wants to be the executor. `acquire()` is non-blocking and idempotent, so a
    follower may call it on every tick and will take over the moment the owner
    dies.
    """

    def __init__(
        self,
        runtime_root: Path,
        *,
        role: str = ROLE_CLI,
        clock: Optional[Clock] = None,
    ):
        self._runtime_root = Path(runtime_root)
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._role = role
        self._clock = clock or default_clock()
        self._handle = None
        self._acquired_at: Optional[str] = None

    # -- identity ------------------------------------------------------

    @property
    def runtime_root(self) -> Path:
        return self._runtime_root

    @property
    def role(self) -> str:
        return self._role

    @property
    def lock_path(self) -> Path:
        return self._runtime_root / EXECUTION_LOCK_NAME

    @property
    def owner_path(self) -> Path:
        return self._runtime_root / EXECUTION_OWNER_NAME

    @property
    def owned(self) -> bool:
        """True when *this* object holds the lock."""
        return self._handle is not None

    # -- acquisition ---------------------------------------------------

    def acquire(self) -> bool:
        """Take execution authority if it is free. Never blocks, never raises."""
        if self._handle is not None:
            return True
        try:
            handle = open(self.lock_path, "a+")
        except OSError:
            return False
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                return False
        self._handle = handle
        self._acquired_at = self._clock.now_iso()
        self._write_owner_file()
        return True

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        self._acquired_at = None
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
        # Best-effort: the lock is what matters, and a leftover owner file
        # without a live lock already reads as "not owned".
        try:
            self.owner_path.unlink()
        except OSError:
            pass

    # -- observation ---------------------------------------------------

    def is_held(self) -> bool:
        """True when anyone — this object or another process — holds the lock."""
        if self._handle is not None:
            return True
        return self._probe_locked()

    def is_held_elsewhere(self) -> bool:
        """True when some *other* holder has execution authority."""
        if self._handle is not None:
            return False
        return self._probe_locked()

    def current_owner(self) -> Optional[ExecutionOwner]:
        """
        Who holds the lock right now, or None when it is free.

        The lock probe gates the answer, so a stale `execution_owner.json` left
        behind by a killed process is never reported as an owner.
        """
        if self._handle is not None:
            return self._self_owner()
        if not self._probe_locked():
            return None
        return read_owner_file(self._runtime_root)

    def describe(self) -> dict[str, Any]:
        """The status surface: is execution mine, someone else's, or nobody's?"""
        owner = self.current_owner()
        return {
            "owned_by_me": self.owned,
            "held": owner is not None,
            "role": self._role,
            "owner": owner.to_dict() if owner else None,
            "lock_path": str(self.lock_path),
        }

    # -- internals -----------------------------------------------------

    def _self_owner(self) -> ExecutionOwner:
        return ExecutionOwner(
            pid=os.getpid(),
            host=_hostname(),
            role=self._role,
            acquired_at=self._acquired_at or self._clock.now_iso(),
        )

    def _probe_locked(self) -> bool:
        path = self.lock_path
        if not path.exists():
            return False
        if fcntl is None:  # pragma: no cover - non-POSIX fallback
            return read_owner_file(self._runtime_root) is not None
        try:
            with open(path, "a+") as probe:
                try:
                    fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return True
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        except OSError:
            return False
        return False

    def _write_owner_file(self) -> None:
        path = self.owner_path
        tmp = path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._self_owner().to_dict(), handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
        except OSError:
            # Advisory only. Losing the label must not lose the authority.
            pass

    def __enter__(self) -> "ExecutionOwnership":
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()


def read_owner_file(runtime_root: Path) -> Optional[ExecutionOwner]:
    """The advisory owner label. Says nothing about whether the lock is held."""
    path = Path(runtime_root) / EXECUTION_OWNER_NAME
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return ExecutionOwner.from_dict(payload)


def execution_owner(runtime_root: Path) -> Optional[ExecutionOwner]:
    """Read-only ownership query for CLI/status callers. Acquires nothing."""
    return ExecutionOwnership(runtime_root).current_owner()


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover - defensive
        return "unknown"
