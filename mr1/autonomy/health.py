"""
Health and heartbeat.

`DoctorReport` is already a health-check system — categorised, rolled up to
ok/warning/error, already JSON. This module does not rebuild it. It adds the
one thing the doctor cannot know: *is the autonomy loop itself alive*, and
the supervisor-computed gauges that describe unattended operation.

`health.json` is rewritten every supervisor tick. A `supervisor_heartbeat_at`
older than a few tick intervals means MR1 is dead or wedged — today, nothing
else would tell you.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from mr1.clock import Clock, default_clock, parse_iso


HEALTH_FILE_NAME = "health.json"

DoctorFn = Callable[[Path], Any]

# Distinguishes "use the real doctor" from an explicit `doctor_fn=None`, which
# disables the disk scan entirely (the soak harness relies on that).
_USE_DEFAULT_DOCTOR = object()


@dataclass
class HealthSnapshot:
    heartbeat_at: str
    pid: int
    mode: str
    started_at: str
    uptime_s: float
    doctor_status: str = "unknown"
    doctor_summary: dict[str, Any] = field(default_factory=dict)
    doctor_checked_at: Optional[str] = None
    gauges: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.doctor_status

    def to_dict(self) -> dict[str, Any]:
        return {
            "supervisor_heartbeat_at": self.heartbeat_at,
            "pid": self.pid,
            "mode": self.mode,
            "started_at": self.started_at,
            "uptime_s": round(self.uptime_s, 3),
            "doctor_status": self.doctor_status,
            "doctor_summary": dict(self.doctor_summary),
            "doctor_checked_at": self.doctor_checked_at,
            "gauges": dict(self.gauges),
        }


class HealthReporter:
    """
    Owns `health.json`.

    The doctor is a directory scan, so it runs on its own cadence
    (`doctor_interval_s`) rather than on every tick; the rollup from the last
    run is carried forward in between. Pass `doctor_fn=None` to disable the
    doctor entirely (the soak harness does this — it asserts on gauges, not on
    disk scans).
    """

    def __init__(
        self,
        runtime_root: Path,
        *,
        clock: Optional[Clock] = None,
        doctor_fn: Optional[DoctorFn] | object = _USE_DEFAULT_DOCTOR,
        doctor_interval_s: float = 300.0,
    ):
        self._runtime_root = Path(runtime_root)
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._clock = clock or default_clock()
        self._doctor_fn = (
            _default_doctor
            if doctor_fn is _USE_DEFAULT_DOCTOR else
            doctor_fn
        )
        self._doctor_interval_s = float(doctor_interval_s)
        self._last_doctor_at: Optional[float] = None
        self._last_doctor_status = "unknown"
        self._last_doctor_summary: dict[str, Any] = {}
        self._last_doctor_checked_at: Optional[str] = None

    @property
    def path(self) -> Path:
        return self._runtime_root / HEALTH_FILE_NAME

    def refresh_doctor(self, *, force: bool = False) -> str:
        if self._doctor_fn is None:
            return self._last_doctor_status
        now = self._clock.monotonic()
        due = (
            force
            or self._last_doctor_at is None
            or (now - self._last_doctor_at) >= self._doctor_interval_s
        )
        if not due:
            return self._last_doctor_status
        try:
            report = self._doctor_fn(self._runtime_root)
            self._last_doctor_status = getattr(report, "status", "unknown")
            summary = getattr(report, "summary", {}) or {}
            self._last_doctor_summary = dict(summary)
        except Exception as exc:
            self._last_doctor_status = "error"
            self._last_doctor_summary = {"error": f"{type(exc).__name__}: {exc}"}
        self._last_doctor_at = now
        self._last_doctor_checked_at = self._clock.now_iso()
        return self._last_doctor_status

    def write(
        self,
        *,
        pid: int,
        mode: str,
        started_at: str,
        uptime_s: float,
        gauges: dict[str, Any],
    ) -> HealthSnapshot:
        snapshot = HealthSnapshot(
            heartbeat_at=self._clock.now_iso(),
            pid=pid,
            mode=mode,
            started_at=started_at,
            uptime_s=uptime_s,
            doctor_status=self._last_doctor_status,
            doctor_summary=dict(self._last_doctor_summary),
            doctor_checked_at=self._last_doctor_checked_at,
            gauges=dict(gauges),
        )
        _atomic_write_json(self.path, snapshot.to_dict())
        return snapshot

    def read(self) -> Optional[dict[str, Any]]:
        return read_health(self._runtime_root)


def read_health(runtime_root: Path) -> Optional[dict[str, Any]]:
    path = Path(runtime_root) / HEALTH_FILE_NAME
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def heartbeat_age_s(
    payload: Optional[dict[str, Any]],
    *,
    clock: Optional[Clock] = None,
) -> Optional[float]:
    if not payload:
        return None
    beat = parse_iso(payload.get("supervisor_heartbeat_at"))
    if beat is None:
        return None
    now = (clock or default_clock()).now()
    return max(0.0, (now - beat).total_seconds())


def heartbeat_is_stale(
    payload: Optional[dict[str, Any]],
    *,
    tick_interval_s: float,
    max_missed_ticks: int = 3,
    clock: Optional[Clock] = None,
) -> bool:
    age = heartbeat_age_s(payload, clock=clock)
    if age is None:
        return True
    return age > tick_interval_s * max_missed_ticks


def disk_free_bytes(path: Path) -> int:
    try:
        return int(shutil.disk_usage(Path(path)).free)
    except OSError:
        return -1


def events_jsonl_bytes(runtime_root: Path) -> int:
    path = Path(runtime_root) / "events" / "events.jsonl"
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _default_doctor(runtime_root: Path):
    from mr1.doctor import run_doctor

    return run_doctor(Path(runtime_root))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
