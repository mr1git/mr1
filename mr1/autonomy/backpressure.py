"""
Backpressure and adaptive degradation (B4).

Phase A capped concurrency, plans per hour, and workflows per objective per day,
and stopped planning when the doctor reported an error. Those are the caps that
keep MR1 from spending too much. This module adds the ones that keep it from
*hurting itself* — and, just as importantly, makes every refusal visible without
turning a 60-second tick into a 60-per-hour log flood.

Three principles.

**Stop creating, keep draining.** Every signal here blocks *planning*. None of
them cancels work, kills a task, or revokes authority. A runtime under pressure
finishes what it started and declines to start more. The alternative — shedding
in-flight work when the disk gets tight — turns a resource problem into a
correctness problem.

**Refuse before the wall, not at it.** Disk is the clearest case: a runtime that
plans until `write()` fails leaves torn workflow state behind. So planning stops
while there is still room to finish the work in flight, write its results, and
run retention. The threshold is not where MR1 dies; it is where MR1 stops
digging.

**Deterministic, and configurable.** No adaptive rate controller, no learned
thresholds, nothing that makes "why did it not run" a research question. Each
signal is a comparison against a number an operator set.

Observability is the subtle part. A supervisor ticking every 60 seconds under
sustained pressure would emit 1 440 identical "disk is low" events a day, which
is indistinguishable from having no events at all. So the reporter emits on the
*edges*: once when a signal starts applying, once when it lifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# Codes are stable: `mr1 status`, the timeline, and any external monitor key off
# them. Add, do not rename.
DISK_PRESSURE = "disk_pressure"
CONCURRENCY_CAP = "concurrency_cap"
SUPERVISOR_DEGRADED = "supervisor_degraded"
SCHEDULER_DEGRADED = "scheduler_degraded"
HEALTH_DEGRADED = "health_degraded"
PLAN_BUDGET = "plan_budget_exhausted"


@dataclass(frozen=True)
class BackpressureLimits:
    max_concurrent_workflows: int = 4
    # Planning stops here — with room left to drain, persist, and run retention.
    min_disk_free_bytes: int = 512 * 1024 * 1024  # 512 MiB
    # Repeated failure of either loop means MR1 does not understand its own
    # state well enough to be creating more work in it.
    max_consecutive_supervisor_errors: int = 3
    max_consecutive_scheduler_errors: int = 5

    def validate(self) -> "BackpressureLimits":
        if self.max_concurrent_workflows < 1:
            raise ValueError("max_concurrent_workflows must be >= 1")
        if self.min_disk_free_bytes < 0:
            raise ValueError("min_disk_free_bytes must be >= 0")
        if self.max_consecutive_supervisor_errors < 1:
            raise ValueError("max_consecutive_supervisor_errors must be >= 1")
        if self.max_consecutive_scheduler_errors < 1:
            raise ValueError("max_consecutive_scheduler_errors must be >= 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_concurrent_workflows": self.max_concurrent_workflows,
            "min_disk_free_bytes": self.min_disk_free_bytes,
            "max_consecutive_supervisor_errors": self.max_consecutive_supervisor_errors,
            "max_consecutive_scheduler_errors": self.max_consecutive_scheduler_errors,
        }


@dataclass(frozen=True)
class BackpressureSignal:
    code: str
    detail: str
    observed: Any = None
    limit: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "observed": self.observed,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class RuntimePressure:
    """What the supervisor observed this tick. Pure input; no I/O in here."""

    active_workflows: int = 0
    disk_free_bytes: int = -1
    consecutive_supervisor_errors: int = 0
    consecutive_scheduler_errors: int = 0
    health_status: str = "ok"


def evaluate_backpressure(
    pressure: RuntimePressure,
    limits: BackpressureLimits,
) -> list[BackpressureSignal]:
    """
    Every reason planning should stop right now. Empty means go.

    Deliberately returns *all* of them rather than the first: an operator whose
    disk is full and whose scheduler is crashing should be told both, not told
    one, fix it, and be told the other.
    """
    signals: list[BackpressureSignal] = []

    if pressure.health_status == "error":
        signals.append(BackpressureSignal(
            code=HEALTH_DEGRADED,
            detail="runtime health is in error; MR1 will not create work it cannot trust",
        ))

    # -1 means "could not read the volume" — not "no space". Treating an unknown
    # as pressure would wedge planning on any OSError from statvfs.
    if 0 <= pressure.disk_free_bytes < limits.min_disk_free_bytes:
        signals.append(BackpressureSignal(
            code=DISK_PRESSURE,
            detail=(
                f"only {pressure.disk_free_bytes / 1024**2:.0f} MiB free "
                f"(floor {limits.min_disk_free_bytes / 1024**2:.0f} MiB); "
                "planning stops while there is still room to finish and archive"
            ),
            observed=pressure.disk_free_bytes,
            limit=limits.min_disk_free_bytes,
        ))

    if pressure.consecutive_supervisor_errors >= limits.max_consecutive_supervisor_errors:
        signals.append(BackpressureSignal(
            code=SUPERVISOR_DEGRADED,
            detail=(
                f"the supervisor tick has failed {pressure.consecutive_supervisor_errors} "
                "times in a row; entering degraded mode and creating no new work"
            ),
            observed=pressure.consecutive_supervisor_errors,
            limit=limits.max_consecutive_supervisor_errors,
        ))

    if pressure.consecutive_scheduler_errors >= limits.max_consecutive_scheduler_errors:
        signals.append(BackpressureSignal(
            code=SCHEDULER_DEGRADED,
            detail=(
                f"the scheduler tick has failed {pressure.consecutive_scheduler_errors} "
                "times in a row; work in flight may not be advancing"
            ),
            observed=pressure.consecutive_scheduler_errors,
            limit=limits.max_consecutive_scheduler_errors,
        ))

    if pressure.active_workflows >= limits.max_concurrent_workflows:
        signals.append(BackpressureSignal(
            code=CONCURRENCY_CAP,
            detail=(
                f"{pressure.active_workflows} workflows already active "
                f"(cap {limits.max_concurrent_workflows})"
            ),
            observed=pressure.active_workflows,
            limit=limits.max_concurrent_workflows,
        ))

    return signals


class BackpressureReporter:
    """
    Emit on the edges, not on the ticks.

    A signal that applies for six hours produces two events — one when it starts,
    one when it stops — not 360. Anything else and the timeline becomes a wall of
    identical lines nobody reads, which is the same as having no signal at all.
    """

    def __init__(self, emit: Any = None):
        self._emit = emit
        self._applied: dict[str, BackpressureSignal] = {}

    @property
    def active(self) -> list[BackpressureSignal]:
        return list(self._applied.values())

    @property
    def active_codes(self) -> set[str]:
        return set(self._applied)

    def observe(self, signals: list[BackpressureSignal]) -> dict[str, list[BackpressureSignal]]:
        """
        Fold this tick's signals into the standing set.

        Returns what changed: the signals that newly started applying, and the
        ones that just lifted. Unchanged signals are returned in neither — they
        are still active, and still uninteresting.
        """
        current = {signal.code: signal for signal in signals}
        started = [signal for code, signal in current.items() if code not in self._applied]
        lifted = [signal for code, signal in self._applied.items() if code not in current]

        for signal in started:
            self._publish("backpressure_applied", signal, status="blocked")
        for signal in lifted:
            self._publish("backpressure_lifted", signal, status="ok")

        self._applied = current
        return {"started": started, "lifted": lifted}

    def _publish(self, event_type: str, signal: BackpressureSignal, *, status: str) -> None:
        if self._emit is None:
            return
        try:
            self._emit(
                event_type,
                status=status,
                summary=(
                    f"backpressure: {signal.detail}"
                    if event_type == "backpressure_applied" else
                    f"backpressure lifted: {signal.code}"
                ),
                metadata=signal.to_dict(),
            )
        except Exception:  # noqa: BLE001 - observability must never break the loop
            pass
