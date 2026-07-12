"""
The soak harness (A8).

Everything an autonomous runtime does over months, compressed into
milliseconds and made deterministic:

  * `VirtualClock`      — time only moves when the soak says so; no real waiting
  * `FakeBrain`         — a planner that counts calls and never spends a token
  * `FaultInjector`     — seeded, reproducible failures of every class
  * `SoakRuntime`       — supervisor + scheduler + stores over one runtime root,
                          restartable in place so crash recovery is testable

No real LLM, no real subprocess, no `sleep`. A 10 000-tick soak covering a week
of simulated uptime runs in seconds, which is the only way an assertion like
"tick cost does not grow with tick number" can actually be checked.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mr1.autonomy.budget import BudgetLedger, BudgetLimits
from mr1.autonomy.health import HealthReporter, read_health
from mr1.autonomy.objectives import KIND_ONCE, PARKED_STATUSES, TERMINAL_STATUSES
from mr1.autonomy.service import Supervisor, SupervisorConfig
from mr1.clock import VirtualClock
from mr1.event_log import EventLog
from mr1.kazi_runner import MockRunner, RunHandle, RunResult, RunStatus
from mr1.workflow_models import TaskStatus


class FakeBrain:
    """A planner with no LLM behind it. Records every call it is asked to make."""

    def __init__(self, *, spec_factory=None, fail_every: int = 0):
        self.calls: list[dict[str, Any]] = []
        self.failures = 0
        self._spec_factory = spec_factory
        self._fail_every = fail_every

    def plan(self, objective, context):
        self.calls.append(dict(context))
        if self._fail_every and len(self.calls) % self._fail_every == 0:
            from mr1.autonomy.planner import PlanningError

            self.failures += 1
            raise PlanningError("the planner is having a bad day")
        if self._spec_factory is not None:
            return self._spec_factory(objective, context)
        return {
            "title": f"plan for {objective.title}",
            "tasks": [{"label": "work", "prompt": "do the thing"}],
        }

    @property
    def call_count(self) -> int:
        return len(self.calls)


@dataclass
class FaultProfile:
    """What fraction of runs fail, and how."""

    timeout: float = 0.0
    infrastructure: float = 0.0
    merit_failure: float = 0.0        # failed on its merits -> planning class
    identical_failure: float = 0.0    # the *same* failure, over and over
    crash_mid_run: float = 0.0        # the task never reports; the handle is lost

    @property
    def total(self) -> float:
        return (
            self.timeout
            + self.infrastructure
            + self.merit_failure
            + self.identical_failure
            + self.crash_mid_run
        )


class FaultInjector(MockRunner):
    """
    A runner that fails on purpose, reproducibly.

    Seeded from an int, so a soak that finds a bug can be replayed exactly.
    """

    def __init__(self, profile: FaultProfile, *, seed: int = 1234):
        super().__init__()
        self._profile = profile
        self._rng = random.Random(seed)
        self.outcomes: dict[str, int] = {}
        self.abandoned_tasks: set[str] = set()

    def poll(self, handle: RunHandle) -> Optional[RunResult]:
        pending = super().poll(handle)
        if pending is not None:
            return pending
        if handle.task_id in self.abandoned_tasks:
            return None

        roll = self._rng.random()
        profile = self._profile
        threshold = 0.0

        threshold += profile.timeout
        if roll < threshold:
            return self._record(RunResult(
                status=RunStatus.TIMED_OUT,
                exit_code=None,
                summary="timed out",
                error="the task timed out",
                error_type="timeout",
            ), "timeout")

        threshold += profile.infrastructure
        if roll < threshold:
            return self._record(RunResult(
                status=RunStatus.FAILED,
                exit_code=1,
                summary="infrastructure failure",
                error="run handle lost: no recoverable persisted result",
                error_type="infrastructure_failure",
            ), "infrastructure")

        threshold += profile.merit_failure
        if roll < threshold:
            return self._record(RunResult(
                status=RunStatus.FAILED,
                exit_code=2,
                summary="failed",
                error=f"assertion failed in check {self._rng.randint(1, 9)}",
                error_type="unknown",
            ), "merit_failure")

        threshold += profile.identical_failure
        if roll < threshold:
            return self._record(RunResult(
                status=RunStatus.FAILED,
                exit_code=2,
                summary="failed",
                error="cannot find module widget",  # deliberately identical
                error_type="unknown",
            ), "identical_failure")

        threshold += profile.crash_mid_run
        if roll < threshold:
            # The task simply never reports again — as if the process vanished.
            self.abandoned_tasks.add(handle.task_id)
            self._count("crash_mid_run")
            return None

        return self._record(RunResult(
            status=RunStatus.SUCCEEDED,
            exit_code=0,
            summary="ok",
        ), "succeeded")

    def _record(self, result: RunResult, kind: str) -> RunResult:
        self._count(kind)
        return result

    def _count(self, kind: str) -> None:
        self.outcomes[kind] = self.outcomes.get(kind, 0) + 1


class _StubDoctor:
    """The doctor is a disk scan; a soak asserts on gauges, not on `du`."""

    def __init__(self, status: str = "ok"):
        self.status = status
        self.summary: dict[str, Any] = {}
        self.calls = 0

    def __call__(self, _runtime_root):
        self.calls += 1
        return self


@dataclass
class TickSample:
    tick: int
    wall_ms: float
    planned: int
    escalated: int
    recovered: int


class SoakRuntime:
    """
    One runtime root, one supervisor, restartable in place.

    `restart()` throws away every in-memory structure and rebuilds the
    supervisor from disk — which is exactly what a crash and a `mr1 serve`
    afterwards look like.
    """

    def __init__(
        self,
        runtime_root: Path,
        *,
        workspace_root: Optional[Path] = None,
        planner: Optional[FakeBrain] = None,
        runner: Optional[MockRunner] = None,
        config: Optional[SupervisorConfig] = None,
        limits: Optional[BudgetLimits] = None,
        clock: Optional[VirtualClock] = None,
        seed: int = 7,
    ):
        self.runtime_root = Path(runtime_root)
        self.workspace_root = Path(workspace_root or runtime_root)
        self.clock = clock or VirtualClock(start=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.planner = planner or FakeBrain()
        self.runner = runner or MockRunner(
            on_poll=lambda _handle: RunResult(status=RunStatus.SUCCEEDED, exit_code=0, summary="ok")
        )
        self.config = config or SupervisorConfig(
            tick_interval_s=60.0,
            doctor_interval_s=3600.0,
            max_concurrent_workflows=8,
            max_plans_per_hour=1_000_000,
            max_workflows_per_objective_per_day=1_000_000,
        )
        self.limits = limits or BudgetLimits(
            max_plans_per_hour=1_000_000,
            max_actions_per_hour=1_000_000,
            max_workflows_per_objective_per_day=1_000_000,
        )
        self.doctor = _StubDoctor()
        self.rng = random.Random(seed)
        self.samples: list[TickSample] = []
        self.supervisor = self._build()

    def _build(self) -> Supervisor:
        return Supervisor(
            self.runtime_root,
            config=self.config,
            clock=self.clock,
            runner=self.runner,
            planner=self.planner,
            budget=BudgetLedger(self.runtime_root, clock=self.clock, limits=self.limits),
            health_reporter=HealthReporter(
                self.runtime_root,
                clock=self.clock,
                doctor_fn=self.doctor,
                doctor_interval_s=self.config.doctor_interval_s,
            ),
            workspace_root=self.workspace_root,
            auto_scheduler_tick=False,
            rng=self.rng,
        )

    def restart(self) -> Supervisor:
        """Simulate a crash and a fresh `mr1 serve` over the same disk."""
        self.supervisor.scheduler.shutdown()
        self.supervisor = self._build()
        return self.supervisor

    # -- driving -------------------------------------------------------

    @property
    def objectives(self):
        return self.supervisor.objectives

    @property
    def scheduler(self):
        return self.supervisor.scheduler

    @property
    def store(self):
        return self.supervisor.scheduler._store

    @property
    def agents(self):
        return self.supervisor.scheduler._scoped_agents

    def create_objective(self, **overrides):
        payload = {
            "title": "soak",
            "statement": "keep the lights on",
            "owner_agent_id": self.agents.root_agent_id,
            "kind": KIND_ONCE,
            "idempotent": True,
        }
        payload.update(overrides)
        return self.objectives.create(**payload)

    def tick(self, *, scheduler_ticks: int = 2, advance_s: Optional[float] = None) -> TickSample:
        """One supervisor tick, plus the scheduler ticks that execute the work."""
        started = time.perf_counter()
        outcome = self.supervisor.tick()
        for _ in range(scheduler_ticks):
            self.scheduler.tick()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.clock.advance(
            self.config.tick_interval_s if advance_s is None else advance_s
        )
        sample = TickSample(
            tick=len(self.samples) + 1,
            wall_ms=elapsed_ms,
            planned=int(outcome.get("planned", 0) or 0),
            escalated=int(outcome.get("escalated", 0) or 0),
            recovered=int(outcome.get("recovered", 0) or 0),
        )
        self.samples.append(sample)
        return sample

    def run(self, ticks: int, **kwargs) -> list[TickSample]:
        for _ in range(ticks):
            self.tick(**kwargs)
        return self.samples

    # -- assertions the soaks share -------------------------------------

    def health(self) -> dict[str, Any]:
        return read_health(self.runtime_root) or {}

    def gauges(self) -> dict[str, Any]:
        return dict(self.health().get("gauges") or {})

    def events(self, **kwargs):
        return EventLog(self.runtime_root / "events").filter_events(**kwargs)

    def event_bytes(self) -> int:
        path = self.runtime_root / "events" / "events.jsonl"
        return path.stat().st_size if path.exists() else 0

    def runtime_bytes(self) -> int:
        return sum(
            item.stat().st_size
            for item in self.runtime_root.rglob("*")
            if item.is_file()
        )

    def stuck_objectives(self) -> list[Any]:
        """
        An objective that is neither finished, nor parked for a human, nor
        making progress. This is the thing a soak exists to catch.
        """
        stuck = []
        for objective in self.objectives.list_objectives():
            if objective.status in TERMINAL_STATUSES or objective.status in PARKED_STATUSES:
                continue
            if objective.status == "executing" and objective.current_workflow_id:
                workflow = self.store.load_workflow(objective.current_workflow_id)
                if workflow is not None and not workflow.is_terminal():
                    continue  # legitimately in flight
                stuck.append(objective)
                continue
            if objective.status in {"active", "recovering", "planning"}:
                continue  # will be picked up by a later tick
            stuck.append(objective)
        return stuck

    def unresolved_objectives(self) -> list[Any]:
        """Objectives that never reached *any* explicit end state."""
        return [
            objective
            for objective in self.objectives.list_objectives()
            if objective.status not in TERMINAL_STATUSES
            and objective.status not in PARKED_STATUSES
        ]

    def live_task_count(self) -> int:
        return sum(
            1
            for workflow in self.store.list_active_workflows()
            for task in workflow.tasks.values()
            if task.status is TaskStatus.RUNNING
        )
