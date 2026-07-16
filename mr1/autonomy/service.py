"""
The headless supervisor.

Two loops at different frequencies. The fast one already exists and does not
change: `Scheduler._run_loop()` reconciles active workflows every second and
executes everything. The slow one is here: `Supervisor.tick()` decides what
work *exists*, and executes nothing itself.

    observe -> gate -> sweep -> reconcile -> plan -> recover -> escalate

The ordering is a safety property: every gate runs before any work is created.

Invariant, load-bearing for cost: a steady-state tick with nothing to do makes
zero brain calls. The brain is called from PLAN and nowhere else, and PLAN only
runs on a state change. OBSERVE, GATE, and SWEEP must never call it.

Phase A2 ships this host with the control plane and the scheduler drain loop.
A5 extends the same tick with objectives, recovery, and escalation.
"""

from __future__ import annotations

import json
import signal
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from mr1.autonomy.control import (
    MODE_HALTED,
    MODE_PAUSED,
    MODE_RUNNING,
    MODE_STOPPING,
    ControlPlane,
    ControlState,
    ServiceLock,
)
from mr1.autonomy.backpressure import (
    HEALTH_DEGRADED,
    BackpressureLimits,
    BackpressureReporter,
    RuntimePressure,
    evaluate_backpressure,
)
from mr1.autonomy.budget import BudgetLedger, BudgetLimits
from mr1.autonomy.consent import ConsentGrantStore
from mr1.autonomy.escalation import (
    REASON_AMBIGUOUS,
    REASON_APPROVAL_EXPIRED,
    REASON_BLOCKED,
    REASON_BUDGET_EXHAUSTED,
    REASON_CONSENT_MISSING,
    REASON_HEALTH_DEGRADED,
    REASON_PLAN_FAILED,
    REASON_QUARANTINED,
    REASON_REPEATED_FAILURE,
    REASON_STUCK,
    Escalator,
)
from mr1.autonomy.health import (
    HealthReporter,
    disk_free_bytes,
    events_jsonl_bytes,
)
from mr1.autonomy.objectives import (
    STATUS_ACTIVE,
    STATUS_EXECUTING,
    STATUS_PAUSED,
    STATUS_RECOVERING,
    STATUS_SATISFIED,
    Attempt,
    Objective,
    ObjectiveStore,
    evaluate_trigger,
)
from mr1.autonomy.triggers import TriggerDecision
from mr1.autonomy.notify import Notifier
from mr1.autonomy.ownership import ExecutionOwnership
from mr1.autonomy.retention import RetentionManager, RetentionPolicy
from mr1.autonomy.planner import (
    AmbiguousObjective,
    Planner,
    PlanningError,
    build_planning_context,
    preflight_authority,
)
from mr1.autonomy.recovery import (
    FailureClass,
    RecoveryKind,
    classify,
    decide,
)
from mr1.autonomy.triage import GovernedTriage
from mr1.inbox_triage import InboxTriagePolicy
from mr1.capability_policy import CapabilityApprovalStore
from mr1.clock import Clock, default_clock, parse_iso
from mr1.event_log import EventLog
from mr1.kazi_runner import KaziAsyncRunner, Runner
from mr1.messages import MessageStore
from mr1.scheduler import Scheduler, WorkflowSpecError, validate_spec
from mr1.scoped_agents import PersistentAgentStore
from mr1.watchers import default_watcher_registry
from mr1.workflow_models import Provenance, WorkflowStatus
from mr1.workflow_store import WorkflowStore


DEFAULT_TICK_INTERVAL_S = 60.0


@dataclass
class SupervisorConfig:
    tick_interval_s: float = DEFAULT_TICK_INTERVAL_S
    scheduler_tick_interval_s: float = 1.0
    scheduler_concurrency: int = 4
    doctor_interval_s: float = 300.0
    drain_grace_ticks: int = 0
    max_concurrent_workflows: int = 4
    max_plans_per_hour: int = 20
    max_workflows_per_objective_per_day: int = 24
    max_actions_per_hour: int = 60
    stuck_workflow_after_s: float = 86_400.0
    approval_ttl_s: float = 86_400.0
    tick_event_interval: int = 60
    # B4. Planning stops while there is still room to drain, persist, and
    # archive — not at the point where a write would fail.
    min_disk_free_bytes: int = 512 * 1024 * 1024
    max_consecutive_supervisor_errors: int = 3
    max_consecutive_scheduler_errors: int = 5
    # B1. Retention runs on its own slow cadence inside the supervisor. It is a
    # deterministic filesystem sweep — no brain, no network — and it archives
    # rather than deletes, so running it unattended is safe. 0 disables it and
    # leaves retention entirely to `mr1 maintenance run`.
    retention_interval_s: float = 21_600.0  # 6h

    def validate(self) -> "SupervisorConfig":
        if self.tick_interval_s <= 0:
            raise ValueError("tick_interval_s must be > 0")
        if self.scheduler_concurrency < 1:
            raise ValueError("scheduler_concurrency must be >= 1")
        if self.max_concurrent_workflows < 1:
            raise ValueError("max_concurrent_workflows must be >= 1")
        if self.max_plans_per_hour < 0:
            raise ValueError("max_plans_per_hour must be >= 0")
        return self


class Supervisor:
    """
    The autonomy host. Owns the slow loop; the scheduler stays the executor.

    Constructed against a runtime root (the directory that already holds
    `workflows/`, `agents/`, `events/`, `messages/`, `capability_approvals/`).
    """

    def __init__(
        self,
        runtime_root: Path,
        *,
        config: Optional[SupervisorConfig] = None,
        clock: Optional[Clock] = None,
        runner: Optional[Runner] = None,
        scheduler: Optional[Scheduler] = None,
        workflow_store: Optional[WorkflowStore] = None,
        scoped_agent_store: Optional[PersistentAgentStore] = None,
        message_store: Optional[MessageStore] = None,
        approval_store: Optional[CapabilityApprovalStore] = None,
        control_plane: Optional[ControlPlane] = None,
        health_reporter: Optional[HealthReporter] = None,
        workspace_root: Optional[Path] = None,
        auto_scheduler_tick: bool = True,
        objective_store: Optional[ObjectiveStore] = None,
        consent_store: Optional[ConsentGrantStore] = None,
        budget: Optional[BudgetLedger] = None,
        escalator: Optional[Escalator] = None,
        planner: Optional[Planner] = None,
        rng: Optional[Any] = None,
        triage: Optional[Any] = None,
        triage_policy: Optional[Any] = None,
        enable_triage: bool = False,
        execution_ownership: Optional[ExecutionOwnership] = None,
        retention: Optional[RetentionManager] = None,
        retention_policy: Optional[RetentionPolicy] = None,
        notifier: Optional[Notifier] = None,
    ):
        self._runtime_root = Path(runtime_root)
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._config = (config or SupervisorConfig()).validate()
        self._clock = clock or default_clock()
        self._workspace_root = Path(workspace_root) if workspace_root else Path.cwd()

        self._store = workflow_store or WorkflowStore(root=self._runtime_root / "workflows")
        self._scoped_agents = scoped_agent_store or PersistentAgentStore(
            root=self._runtime_root / "agents"
        )
        self._message_store = message_store or MessageStore(
            root=self._runtime_root / "messages",
            scoped_agent_store=self._scoped_agents,
        )
        self._approval_store = approval_store or CapabilityApprovalStore(
            self._runtime_root / "capability_approvals",
            clock=self._clock,
        )
        self._timeline = EventLog(self._runtime_root / "events")
        self._control = control_plane or ControlPlane(self._runtime_root, clock=self._clock)
        self._health = health_reporter or HealthReporter(
            self._runtime_root,
            clock=self._clock,
            doctor_interval_s=self._config.doctor_interval_s,
        )
        self._root_agent_id = self._scoped_agents.root_agent_id

        self._objectives = objective_store or ObjectiveStore(
            self._runtime_root,
            clock=self._clock,
        )
        self._consent = consent_store or ConsentGrantStore(
            self._runtime_root,
            clock=self._clock,
            scoped_agent_store=self._scoped_agents,
        )
        self._budget = budget or BudgetLedger(
            self._runtime_root,
            clock=self._clock,
            limits=BudgetLimits(
                max_plans_per_hour=self._config.max_plans_per_hour,
                max_actions_per_hour=self._config.max_actions_per_hour,
                max_workflows_per_objective_per_day=(
                    self._config.max_workflows_per_objective_per_day
                ),
            ),
        )
        self._notifier = notifier or Notifier(
            self._runtime_root,
            clock=self._clock,
            event_log=self._timeline,
        )
        self._escalator = escalator or Escalator(
            self._runtime_root,
            objective_store=self._objectives,
            message_store=self._message_store,
            scoped_agent_store=self._scoped_agents,
            clock=self._clock,
            notifier=self._notifier,
        )
        self._planner = planner
        self._watchers = default_watcher_registry()
        self._rng = rng
        self._triage = triage if triage is not None else (
            self._build_triage() if enable_triage else None
        )
        self._triage_policy = triage_policy or InboxTriagePolicy(
            max_messages=10,
            max_actions=5,
        )

        self._backpressure_limits = BackpressureLimits(
            max_concurrent_workflows=self._config.max_concurrent_workflows,
            min_disk_free_bytes=self._config.min_disk_free_bytes,
            max_consecutive_supervisor_errors=self._config.max_consecutive_supervisor_errors,
            max_consecutive_scheduler_errors=self._config.max_consecutive_scheduler_errors,
        ).validate()
        self._backpressure = BackpressureReporter(emit=self._emit_backpressure)

        self._ownership = execution_ownership
        self._retention = retention or RetentionManager(
            self._runtime_root,
            policy=retention_policy,
            clock=self._clock,
            event_log=self._timeline,
        )
        self._last_retention_at: Optional[float] = None
        self._scheduler = scheduler or Scheduler(
            self._store,
            runner or KaziAsyncRunner(self._store),
            concurrency=self._config.scheduler_concurrency,
            auto_tick=auto_scheduler_tick,
            tick_interval_s=self._config.scheduler_tick_interval_s,
            agent_id="supervisor",
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
            workspace_root=self._workspace_root,
            clock=self._clock,
            runtime_error_sink=self._record_runtime_error,
            consent_store=self._consent,
            execution_ownership=self._ownership,
        )

        self._stop_event = threading.Event()
        self._pid = 0
        self._started_at = self._clock.now_iso()
        self._started_monotonic = self._clock.monotonic()
        self._tick_count = 0
        self._tick_errors = 0
        self._consecutive_tick_errors = 0
        self._last_tick_duration_ms = 0.0
        self._last_tick_error: Optional[str] = None
        self._runtime_errors: list[dict[str, Any]] = []
        self._exit_requested = False
        self._exit_reason = ""
        self._health_error_escalated = False
        self._last_gauges: dict[str, Any] = {}

    def _build_triage(self) -> Optional[GovernedTriage]:
        """The same InboxTriageRunner the REPL uses, behind the same gates."""
        try:
            return GovernedTriage(
                self._runtime_root,
                message_store=self._message_store,
                scoped_agent_store=self._scoped_agents,
                runner_factory=self._make_triage_runner,
                control=self._control,
                budget=self._budget,
                clock=self._clock,
                event_log=self._timeline,
                error_sink=self._record_runtime_error,
                escalate=self._escalate_triage_failure,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._record_runtime_error("supervisor_triage", f"triage unavailable: {exc}")
            return None

    def _make_triage_runner(self) -> Any:
        from mr1.inbox_triage import InboxTriageRunner

        return InboxTriageRunner(
            workflow_store=self._store,
            scoped_agent_store=self._scoped_agents,
            message_store=self._message_store,
        )

    # ------------------------------------------------------------------
    # Accessors used by the CLI, tests, and the A5 objective layer
    # ------------------------------------------------------------------

    @property
    def runtime_root(self) -> Path:
        return self._runtime_root

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    @property
    def control(self) -> ControlPlane:
        return self._control

    @property
    def config(self) -> SupervisorConfig:
        return self._config

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    def metrics(self) -> dict[str, Any]:
        return {
            "supervisor_tick_count": self._tick_count,
            "supervisor_tick_errors": self._tick_errors,
            "supervisor_consecutive_tick_errors": self._consecutive_tick_errors,
            "supervisor_tick_duration_ms": round(self._last_tick_duration_ms, 3),
            "supervisor_last_tick_error": self._last_tick_error,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def serve(self, *, install_signal_handlers: bool = True) -> int:
        """
        Run headless until stopped. Returns the process exit code.

        Singleton: a second `mr1 serve` on the same runtime root raises
        `ServiceLockError` rather than double-driving the same workflows.
        """
        lock = ServiceLock(self._runtime_root)
        self._pid = lock.acquire()
        self._claim_execution_authority()
        if install_signal_handlers:
            self._install_signal_handlers()
        state = self._control.read()
        if state.mode in {MODE_STOPPING, MODE_HALTED}:
            # A previous run left a terminal mode behind; start clean rather
            # than exiting immediately on a stale instruction.
            self._control.set_mode(
                MODE_RUNNING,
                reason="supervisor restarted",
                requested_by="supervisor",
            )
        self._emit("supervisor_started", status="running", summary="supervisor started")
        try:
            while not self._stop_event.is_set():
                self.tick()
                if self._exit_requested:
                    break
                self._clock.wait(self._stop_event, self._config.tick_interval_s)
        finally:
            self.shutdown()
            lock.release()
        return 0

    def _claim_execution_authority(self) -> None:
        """
        B8. The service is the execution authority for its runtime root.

        `ServiceLock` already stops a second supervisor. This stops everything
        *else* — an open REPL, a ticking CLI — from launching the same task.
        Claiming it here rather than lazily in the scheduler means a supervisor
        that starts while a REPL holds the root is visible at startup instead of
        silently idling.
        """
        if self._ownership is None:
            return
        if self._ownership.acquire():
            self._emit(
                "execution_ownership_acquired",
                status="ok",
                summary=f"execution authority acquired by {self._ownership.role}",
                metadata=self._ownership.describe(),
            )
            return
        owner = self._ownership.current_owner()
        self._emit(
            "execution_ownership_delegated",
            status="warning",
            summary=(
                "another process holds execution authority for this runtime root"
                + (f" (pid {owner.pid}, {owner.role})" if owner else "")
            ),
            metadata=self._ownership.describe(),
        )

    @property
    def execution_ownership(self) -> Optional[ExecutionOwnership]:
        return self._ownership

    def request_shutdown(self, reason: str = "signal") -> None:
        """Ask the loop to leave after the current tick. Signal-handler safe."""
        self._exit_requested = True
        self._exit_reason = reason
        self._stop_event.set()

    def shutdown(self, *, cancel_running: bool = False) -> None:
        self._scheduler.shutdown(cancel_running=cancel_running)
        if self._ownership is not None:
            self._ownership.release()
        self._emit(
            "supervisor_stopped",
            status="stopped",
            summary=f"supervisor stopped: {self._exit_reason or 'shutdown'}",
            metadata={"reason": self._exit_reason or "shutdown"},
        )

    def _install_signal_handlers(self) -> None:
        def _handle(signum, _frame):
            name = signal.Signals(signum).name
            self._control.set_mode(
                MODE_STOPPING,
                reason=f"received {name}",
                requested_by="signal",
            )
            self.request_shutdown(reason=name.lower())

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, _handle)
            except ValueError:
                # Not on the main thread (tests, embedded use) — skip.
                pass

    # ------------------------------------------------------------------
    # The ordered tick
    # ------------------------------------------------------------------

    def tick(self) -> dict[str, Any]:
        """
        One pass. The order is the safety property: every gate runs before any
        work is created, and the brain is reachable only from PLAN.
        """
        started = self._clock.monotonic()
        outcome: dict[str, Any] = {"planned": 0, "escalated": 0, "recovered": 0}
        try:
            state = self._observe()                      # 1. OBSERVE
            outcome["mode"] = state.mode
            gate = self._gate(state)                     # 2. GATE
            outcome["gate"] = gate
            if gate == "exit":
                return outcome
            self._sweep(state, outcome)                  # 3. SWEEP
            self._maybe_run_retention(outcome)           # 3b. RETENTION (B1)
            self._reconcile(state, outcome)              # 4. RECONCILE (6. RECOVER)
            if gate == "planning":
                self._plan_phase(state, outcome)         # 5. PLAN
                self._triage_phase(outcome)              # A9: governed triage
            self._consecutive_tick_errors = 0
        except Exception as exc:
            self._record_tick_failure(exc)
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self._tick_count += 1
            self._last_tick_duration_ms = (self._clock.monotonic() - started) * 1000.0
            self._write_heartbeat()
            self._maybe_emit_tick_event(outcome)
        return outcome

    def _maybe_emit_tick_event(self, outcome: dict[str, Any]) -> None:
        """
        The per-tick pulse lives in health.json, which is *rewritten*.

        Appending a `supervisor_tick` event on every tick would grow
        events.jsonl without bound and make each tick a little more expensive
        than the last — the exact creep the soak asserts against. So the
        timeline only sees a tick when something happened, or once every
        `tick_event_interval` ticks as a liveness marker.
        """
        did_something = any(
            int(outcome.get(key, 0) or 0)
            for key in ("planned", "escalated", "recovered")
        )
        periodic = (
            self._config.tick_event_interval > 0
            and self._tick_count % self._config.tick_event_interval == 0
        )
        if not (did_something or periodic):
            return
        self._emit(
            "supervisor_tick",
            status="ok" if "error" not in outcome else "error",
            summary=f"supervisor tick {self._tick_count}",
            metadata={
                "tick": self._tick_count,
                "mode": outcome.get("mode"),
                "gate": outcome.get("gate"),
                "planned": outcome.get("planned", 0),
                "recovered": outcome.get("recovered", 0),
                "escalated": outcome.get("escalated", 0),
                "duration_ms": round(self._last_tick_duration_ms, 3),
            },
        )

    def _emit_backpressure(
        self,
        event_type: str,
        *,
        status: str,
        summary: str,
        metadata: dict[str, Any],
    ) -> None:
        self._emit(event_type, status=status, summary=summary, metadata=metadata)

    @property
    def backpressure(self) -> BackpressureReporter:
        return self._backpressure

    # -- 1. OBSERVE ----------------------------------------------------

    def _observe(self) -> ControlState:
        state = self._control.read()
        self._health.refresh_doctor()
        return state

    # -- 2. GATE -------------------------------------------------------

    def _gate(self, state: ControlState) -> str:
        """
        Returns one of: "planning" | "draining" | "exit".

        Every backpressure condition stops PLANNING, never draining — with the
        single exception of halt, which stops everything and revokes authority.
        """
        if state.mode == MODE_HALTED:
            self._apply_halt(state)
            self.request_shutdown(reason="halted")
            return "exit"

        if state.mode == MODE_STOPPING:
            if self._is_idle():
                self.request_shutdown(reason="stopped")
                return "exit"
            return "draining"

        if state.mode == MODE_PAUSED:
            return "draining"

        # B4. Every reason to stop creating work, evaluated together, so an
        # operator with two problems is told about two problems.
        signals = evaluate_backpressure(
            RuntimePressure(
                active_workflows=self._active_workflow_count(),
                disk_free_bytes=disk_free_bytes(self._runtime_root),
                consecutive_supervisor_errors=self._consecutive_tick_errors,
                consecutive_scheduler_errors=int(
                    self._scheduler.metrics().get("consecutive_tick_errors", 0) or 0
                ),
                health_status=self._health.refresh_doctor(),
            ),
            self._backpressure_limits,
        )
        self._backpressure.observe(signals)

        if any(signal.code == HEALTH_DEGRADED for signal in signals):
            self._on_health_error(state)
            return "draining"
        self._health_error_escalated = False

        if signals:
            # Draining, not stopping: in-flight work still runs to completion,
            # the scheduler still ticks, retention still reclaims disk. MR1 has
            # only stopped digging.
            return "draining"

        return "planning"

    def _on_health_error(self, state: ControlState) -> None:
        if self._health_error_escalated:
            return
        self._health_error_escalated = True
        self.on_health_error(state)

    def on_health_error(self, state: ControlState) -> None:
        """Overridden in A5 to escalate to a human. Base host only records it."""
        self._record_runtime_error(
            "supervisor_health",
            "runtime health is in error; planning halted",
            details={"mode": state.mode},
        )

    # -- 3. SWEEP ------------------------------------------------------

    def _sweep(self, state: ControlState, outcome: dict[str, Any]) -> None:
        """
        Time-based lifecycle. Nothing else in the runtime does this.

        Never calls the brain: SWEEP only reclassifies what time has already
        made true. Expiring an approval or a grant *removes* authority, it never
        grants it — blocked work stays blocked, and the owning objective becomes
        eligible for escalation.
        """
        for approval_request_id in self._approval_store.expire_stale_requests():
            self._escalate_expired_approval(approval_request_id, outcome)

        for grant_id in self._consent.expire_stale():
            self._note_expired_grant(grant_id)

        self._sweep_stuck_workflows(outcome)
        self.sweep(state)

    def sweep(self, state: ControlState) -> None:
        """Extension point for additional time-based lifecycle."""
        return None

    # -- 3b. RETENTION (B1) --------------------------------------------

    def _maybe_run_retention(self, outcome: dict[str, Any]) -> None:
        """
        Reclaim disk on a slow cadence.

        Deliberately *not* behind the planning gate: retention creates no work
        and spends no authority, so a paused or draining supervisor should still
        keep the disk from filling. It only archives — nothing is deleted unless
        an operator configured `purge_archives_after_days` — so an unattended run
        is recoverable by construction. And it never calls the brain.
        """
        interval = self._config.retention_interval_s
        if interval <= 0 or self._retention is None:
            return
        now = self._clock.monotonic()
        if self._last_retention_at is not None and (now - self._last_retention_at) < interval:
            return
        self._last_retention_at = now
        try:
            report = self._retention.run()
        except Exception as exc:  # noqa: BLE001 - retention must never kill the loop
            self._record_runtime_error("supervisor_retention", f"{type(exc).__name__}: {exc}")
            return
        if report.changed:
            outcome["retention"] = report.summary()
        for error in report.errors:
            self._record_runtime_error("supervisor_retention", error)

    @property
    def retention(self) -> RetentionManager:
        return self._retention

    def _escalate_expired_approval(
        self,
        approval_request_id: str,
        outcome: dict[str, Any],
    ) -> None:
        approval = self._approval_store.load(approval_request_id)
        if approval is None:
            return
        objective = self._objective_for_workflow(approval.workflow_id)
        if objective is None:
            # Not autonomous work; the human owns it, and the approval_expired
            # event already tells them. Nothing to park.
            return
        self._escalate(
            objective,
            reason=REASON_APPROVAL_EXPIRED,
            problem=(
                f"the approval for {approval.capability_name} expired after "
                f"{self._config.approval_ttl_s:.0f}s with no decision"
            ),
            attempted=[
                f"requested approval for {approval.capability_name}",
                "waited for a human until the request timed out",
            ],
            needs=(
                f"a decision on {approval.capability_name}, or standing consent so this "
                "objective stops asking every run"
            ),
            capability=approval.capability_name,
            workflow_id=approval.workflow_id,
            approval_request_id=approval_request_id,
            outcome=outcome,
        )

    def _note_expired_grant(self, grant_id: str) -> None:
        grant = self._consent.load(grant_id)
        if grant is None:
            return
        self._record_runtime_error(
            "consent_expired",
            f"consent grant {grant_id} for {grant.grantee_id} expired",
            details={"capability": grant.capability_name, "use_count": grant.use_count},
        )

    def _sweep_stuck_workflows(self, outcome: dict[str, Any]) -> None:
        """A workflow running past its threshold is stuck, not working."""
        now = self._clock.now()
        for workflow in self._store.list_active_workflows():
            if workflow.status is not WorkflowStatus.RUNNING:
                continue
            started = parse_iso(workflow.created_at)
            if started is None:
                continue
            if (now - started).total_seconds() < self._config.stuck_workflow_after_s:
                continue
            objective = self._objective_for_workflow(workflow.workflow_id)
            if objective is None:
                continue
            self._escalate(
                objective,
                reason=REASON_STUCK,
                problem=(
                    f"workflow {workflow.workflow_id} has been running for longer than "
                    f"{self._config.stuck_workflow_after_s:.0f}s"
                ),
                attempted=[f"submitted workflow {workflow.workflow_id}"],
                needs="a decision: cancel the stuck workflow, or investigate why it hangs",
                workflow_id=workflow.workflow_id,
                outcome=outcome,
            )

    # -- 4. RECONCILE + 6. RECOVER -------------------------------------

    def _reconcile(self, state: ControlState, outcome: dict[str, Any]) -> None:
        """
        Per live objective: has the world moved? No brain calls here.

        The supervisor decides what work *exists*. The scheduler is already
        driving anything in flight; this only notices when it finished.
        """
        for objective in self._objectives.list_live():
            if not objective.current_workflow_id:
                continue
            workflow = self._store.load_workflow(objective.current_workflow_id)
            if workflow is None:
                self._objectives.update(
                    objective.objective_id,
                    current_workflow_id=None,
                    status=STATUS_ACTIVE,
                    status_reason="workflow vanished; will replan",
                )
                continue
            if not workflow.is_terminal():
                if objective.status != STATUS_EXECUTING:
                    self._objectives.set_status(
                        objective.objective_id,
                        STATUS_EXECUTING,
                        reason=f"workflow {workflow.workflow_id} in flight",
                    )
                continue
            if workflow.status is WorkflowStatus.SUCCEEDED:
                self._record_success(objective, workflow.workflow_id)
            else:
                self._recover(objective, workflow, outcome)

    def _record_success(self, objective: Objective, workflow_id: str) -> None:
        now = self._clock.now_iso()
        objective.record_attempt(Attempt(
            workflow_id=workflow_id,
            outcome="succeeded",
            at=now,
            action="none",
        ))
        satisfied = objective.kind == "once"
        self._objectives.update(
            objective.objective_id,
            status=STATUS_SATISFIED if satisfied else STATUS_ACTIVE,
            status_reason="objective satisfied" if satisfied else "awaiting next trigger",
            current_workflow_id=None,
            consecutive_failures=0,
            retries_used=0,
            replans_used=0,
            fallbacks_used=0,
            use_fallback_next=False,
            last_failure=None,
            next_attempt_at=None,
            success_count=objective.success_count + 1,
            last_completed_at=now,
            history=objective.history,
        )
        self._emit_objective(
            objective,
            "objective_satisfied" if satisfied else "objective_succeeded",
            status=STATUS_SATISFIED if satisfied else STATUS_ACTIVE,
            summary=(
                f"objective satisfied: {objective.title}"
                if satisfied else
                f"objective run succeeded: {objective.title}"
            ),
            workflow_id=workflow_id,
        )

    def _recover(
        self,
        objective: Objective,
        workflow: Any,
        outcome: dict[str, Any],
    ) -> None:
        """The ladder. Exactly one level per exhausted budget; every level ends."""
        signal = classify(workflow)
        state = objective.recovery_state(self._clock.now())
        action = decide(signal, state, objective.failure_policy, rng=self._rng)

        now = self._clock.now_iso()
        objective.record_attempt(Attempt(
            workflow_id=workflow.workflow_id,
            outcome="failed",
            at=now,
            classification=signal.classification.value,
            signature=signal.signature,
            action=action.kind.value,
            detail=signal.detail,
        ))
        changes: dict[str, Any] = {
            "consecutive_failures": objective.consecutive_failures + 1,
            "failure_count": objective.failure_count + 1,
            "history": objective.history,
            "last_failure": {
                **signal.to_dict(),
                "workflow_id": workflow.workflow_id,
                "at": now,
                "action": action.kind.value,
            },
        }
        outcome["recovered"] = int(outcome.get("recovered", 0)) + 1
        self._emit_objective(
            objective,
            "objective_recovery",
            status=objective.status,
            summary=f"recovery: {action.kind.value} ({signal.classification.value})",
            workflow_id=workflow.workflow_id,
            metadata={
                "classification": signal.classification.value,
                "action": action.kind.value,
                "reason": action.reason,
                "delay_s": action.delay_s,
                "consecutive_failures": changes["consecutive_failures"],
            },
        )

        if action.kind is RecoveryKind.RETRY:
            due = self._clock.now() + timedelta(seconds=action.delay_s)
            self._objectives.update(
                objective.objective_id,
                status=STATUS_RECOVERING,
                status_reason=action.reason,
                current_workflow_id=None,
                retries_used=objective.retries_used + 1,
                next_attempt_at=due.isoformat(),
                **changes,
            )
            return

        if action.kind is RecoveryKind.REPLAN:
            self._objectives.update(
                objective.objective_id,
                status=STATUS_ACTIVE,
                status_reason=action.reason,
                current_workflow_id=None,
                replans_used=objective.replans_used + 1,
                last_spec=None,
                next_attempt_at=None,
                **changes,
            )
            return

        if action.kind is RecoveryKind.FALLBACK:
            self._objectives.update(
                objective.objective_id,
                status=STATUS_ACTIVE,
                status_reason=action.reason,
                current_workflow_id=None,
                fallbacks_used=objective.fallbacks_used + 1,
                use_fallback_next=True,
                last_spec=None,
                next_attempt_at=None,
                **changes,
            )
            return

        # ESCALATE / QUARANTINE — both park the objective and tell a human.
        self._objectives.update(objective.objective_id, **changes)
        refreshed = self._objectives.require(objective.objective_id)
        reason = (
            REASON_BLOCKED
            if signal.classification is FailureClass.BLOCKED else
            REASON_REPEATED_FAILURE
            if "same failure repeated" in action.reason else
            REASON_BUDGET_EXHAUSTED
            if "budget" in action.reason else
            REASON_QUARANTINED
        )
        self._escalate(
            refreshed,
            reason=reason,
            problem=f"{signal.reason}: {signal.detail}"[:400],
            attempted=self._attempt_summary(refreshed),
            needs=(
                "authority or a decision MR1 cannot give itself"
                if signal.classification is FailureClass.BLOCKED else
                "a human to look at why this keeps failing"
            ),
            capability=signal.error_type,
            workflow_id=workflow.workflow_id,
            status=(
                None
                if action.kind is RecoveryKind.ESCALATE else
                "quarantined"
            ),
            outcome=outcome,
        )
        if action.kind is RecoveryKind.QUARANTINE:
            self._emit_objective(
                refreshed,
                "objective_quarantined",
                status="quarantined",
                summary=f"objective quarantined: {action.reason}"[:300],
                workflow_id=workflow.workflow_id,
                metadata={"reason": action.reason},
            )

    @staticmethod
    def _attempt_summary(objective: Objective) -> list[str]:
        summary = []
        if objective.retries_used:
            summary.append(f"retried {objective.retries_used} time(s) with backoff")
        if objective.replans_used:
            summary.append(f"replanned {objective.replans_used} time(s)")
        if objective.fallbacks_used:
            summary.append("tried the objective's fallback")
        if not summary:
            summary.append("submitted the workflow once; it did not succeed")
        return summary

    # -- 5. PLAN -------------------------------------------------------

    def _plan_phase(self, state: ControlState, outcome: dict[str, Any]) -> None:
        """
        The ONLY place the brain is called, and only on a state change.

        A retry resubmits a spec MR1 already has, so it costs zero tokens. Only
        a new plan, a replan, or a fallback reaches the planner.
        """
        now = self._clock.now()
        in_flight = self._active_workflow_count()
        for objective in self._objectives.list_live():
            if objective.current_workflow_id:
                continue
            if in_flight >= self._config.max_concurrent_workflows:
                # The cap binds per submission, not per tick: one tick must not
                # be able to submit its way past the concurrency limit.
                outcome["gate"] = "draining"
                return

            if objective.status == STATUS_RECOVERING:
                if not objective.ready_to_retry(now):
                    continue
                if objective.last_spec:
                    if self._resubmit(objective, outcome):
                        in_flight += 1
                    continue
                # No spec to replay: fall through and plan afresh.

            elif objective.status == STATUS_ACTIVE:
                decision = evaluate_trigger(
                    objective,
                    now=now,
                    watcher_registry=self._watchers,
                )
                if not decision.ready:
                    # A trigger that is not ready can still have moved: `skip`
                    # realigns past a backlog it will never run. Persist that,
                    # or the same backlog is rediscovered on every tick.
                    self._record_trigger_state(objective, decision, fired=False)
                    continue
                outcome.setdefault("trigger", decision.reason)
                if decision.missed:
                    outcome["missed_runs"] = decision.missed
                # Record the fire *before* planning. A crash between deciding to
                # fire and finishing the plan must not fire the same occurrence
                # again on restart — at-most-once beats at-least-once for
                # anything holding standing consent.
                self._record_trigger_state(objective, decision, fired=True, now=now)
                objective = self._objectives.require(objective.objective_id)
            else:
                continue

            if self._plan_objective(objective, outcome):
                in_flight += 1

    def _record_trigger_state(
        self,
        objective: Objective,
        decision: TriggerDecision,
        *,
        fired: bool,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Persist where the recurrence stands. This is what survives a restart.

        Deriving the next due time from the last *completion* — as Phase A did —
        loses the distinction between "never ran" and "ran, and we crashed before
        it finished". Writing `last_fired_at` at the moment of firing closes that
        window: a restarted supervisor sees the occurrence as spent.
        """
        changes: dict[str, Any] = {}
        if decision.next_due_at != objective.next_due_at:
            changes["next_due_at"] = decision.next_due_at
        if decision.catch_up_remaining != objective.catch_up_remaining:
            changes["catch_up_remaining"] = decision.catch_up_remaining
        if fired:
            changes["last_fired_at"] = (now or self._clock.now()).isoformat()
        if not changes:
            return
        try:
            self._objectives.update(objective.objective_id, **changes)
        except Exception as exc:  # noqa: BLE001 - never let bookkeeping kill the tick
            self._record_runtime_error(
                "supervisor_trigger",
                f"could not persist trigger state for {objective.objective_id}: {exc}",
            )

    def _resubmit(self, objective: Objective, outcome: dict[str, Any]) -> bool:
        """Retry: the same work, again, without asking the brain anything."""
        decision = self._budget.try_consume_objective_workflow(objective.objective_id)
        if not decision.allowed:
            self._on_budget_exhausted(objective, decision, outcome)
            return False
        workflow_id = self._submit(objective, objective.last_spec)
        self._objectives.update(
            objective.objective_id,
            status=STATUS_EXECUTING,
            status_reason="retrying the same workflow after backoff",
            current_workflow_id=workflow_id,
            next_attempt_at=None,
        )
        outcome["planned"] = int(outcome.get("planned", 0))  # a retry is not a plan
        self._emit_objective(
            objective,
            "objective_workflow_submitted",
            status=STATUS_EXECUTING,
            summary=f"retry submitted: {objective.title}",
            workflow_id=workflow_id,
            metadata={"retry": True, "retries_used": objective.retries_used},
        )
        return True

    def _plan_objective(self, objective: Objective, outcome: dict[str, Any]) -> bool:
        if self._planner is None:
            self._record_runtime_error(
                "supervisor_plan",
                f"no planner configured; objective {objective.objective_id} cannot plan",
            )
            return False

        decision = self._budget.try_consume_plan(objective.objective_id)
        if not decision.allowed:
            self._on_budget_exhausted(objective, decision, outcome)
            return False

        self._objectives.set_status(objective.objective_id, "planning", reason="calling the planner")
        context = build_planning_context(objective)
        try:
            spec = self._planner.plan(objective, context)
            validate_spec(spec)
        except AmbiguousObjective as exc:
            self._escalate(
                objective,
                reason=REASON_AMBIGUOUS,
                problem=str(exc)[:400],
                attempted=["asked the planner to turn this objective into a workflow"],
                needs="a more concrete objective statement, or a decision to abandon it",
                outcome=outcome,
            )
            return False
        except (PlanningError, WorkflowSpecError, ValueError) as exc:
            self._emit_objective(
                objective,
                "objective_plan_failed",
                status="active",
                summary=f"planning failed: {type(exc).__name__}",
                metadata={"error": str(exc)[:300]},
            )
            self._escalate(
                objective,
                reason=REASON_PLAN_FAILED,
                problem=f"{type(exc).__name__}: {exc}"[:400],
                attempted=["asked the planner to turn this objective into a workflow"],
                needs="a look at why planning fails for this objective",
                outcome=outcome,
            )
            return False

        missing = preflight_authority(
            spec,
            objective,
            workflow_store=self._store,
            scoped_agents=self._scoped_agents,
            consent_store=self._consent,
            workspace_root=self._workspace_root,
            now=self._clock.now(),
        )
        if missing:
            # Do NOT submit. MR1 asks for the authority it lacks; it never
            # creates work it already knows it cannot finish, and it never
            # grants itself the missing consent.
            first = missing[0]
            self._escalate(
                objective,
                reason=REASON_CONSENT_MISSING,
                problem=(
                    f"the plan needs {first.capability_name} (risk {first.risk_score}), "
                    f"which this objective is not authorized to use: {first.reason}"
                ),
                attempted=[
                    "planned a workflow for this objective",
                    "checked every capability it needs against the objective's consent grants",
                ],
                needs=(
                    f"standing consent for {first.capability_name}, or a narrower objective"
                ),
                capability=first.capability_name,
                outcome=outcome,
                extra={
                    "missing_authority": json.dumps(
                        [item.to_dict() for item in missing],
                        sort_keys=True,
                    )[:800],
                },
            )
            return False

        workflow_id = self._submit(objective, spec)
        now = self._clock.now_iso()
        self._objectives.update(
            objective.objective_id,
            status=STATUS_EXECUTING,
            status_reason=f"workflow {workflow_id} submitted",
            current_workflow_id=workflow_id,
            last_spec=spec,
            last_planned_at=now,
            first_attempt_at=objective.first_attempt_at or now,
            use_fallback_next=False,
            next_attempt_at=None,
        )
        outcome["planned"] = int(outcome.get("planned", 0)) + 1
        self._emit_objective(
            objective,
            "objective_planned",
            status=STATUS_EXECUTING,
            summary=f"planned: {objective.title}",
            workflow_id=workflow_id,
            metadata={
                "task_count": len(spec.get("tasks", [])),
                "is_fallback": bool(context.get("is_fallback")),
            },
        )
        self._emit_objective(
            objective,
            "objective_workflow_submitted",
            status=STATUS_EXECUTING,
            summary=f"workflow submitted: {workflow_id}",
            workflow_id=workflow_id,
            metadata={"retry": False},
        )
        return True

    # -- A9: governed inbox triage -------------------------------------

    def _triage_phase(self, outcome: dict[str, Any]) -> None:
        """
        Triage the root inbox while serving headless — under the same gates.

        Opt-in (`enable_triage=True`). The REPL's triage loop is the one A9
        governs; the supervisor can also run it, but it is off by default
        because triage acting on MR1's own escalations is a loop nobody wants:
        the human's inbox is not MR1's to tidy. `GovernedTriage` refuses to act
        on self-escalations either way — this is belt and braces.

        Only reached when the gate said "planning", so pause/stop/halt already
        stopped it. With no actionable unread mail it makes no LLM call, which
        is what keeps the zero-cost steady-state tick true.
        """
        if self._triage is None:
            return
        result = self._triage.run_pass(
            self._triage_policy,
            caller_agent_id=self._root_agent_id,
        )
        if result.ran:
            outcome["triage_actions"] = result.actions_executed

    def _escalate_triage_failure(self, detail: str, problem: str) -> None:
        """Repeated triage failure is a real failure: tell a human."""
        objective = next(
            (item for item in self._objectives.list_live()),
            None,
        )
        if objective is None:
            self._record_runtime_error("inbox_triage", f"{problem}: {detail}")
            return
        self._escalator.escalate(
            objective,
            reason=REASON_PLAN_FAILED,
            problem=f"{problem}: {detail}"[:400],
            attempted=["ran the governed inbox triage pass"],
            needs="a look at why inbox triage keeps failing",
            status=STATUS_PAUSED,
        )

    def _submit(self, objective: Objective, spec: dict[str, Any]) -> str:
        """
        Every workflow an objective creates carries its objective_id.

        That stamp is what makes the side effects attributable — and it is what
        the capability gate reads back to decide which consent grants apply.
        """
        return self._scheduler.submit_workflow(
            dict(spec),
            Provenance(type="agent", id="supervisor"),
            owner_agent_id=objective.owner_agent_id,
            workflow_metadata={
                "objective_id": objective.objective_id,
                "objective_title": objective.title,
                "autonomous": True,
            },
        )

    def _on_budget_exhausted(
        self,
        objective: Objective,
        decision: Any,
        outcome: dict[str, Any],
    ) -> None:
        outcome["budget_blocked"] = decision.reason
        self._emit(
            "budget_exhausted",
            status="blocked",
            summary=f"budget exhausted: {decision.reason}",
            metadata={
                "objective_id": objective.objective_id,
                "reason": decision.reason,
                "used": decision.used,
                "limit": decision.limit,
            },
        )
        if decision.reason == "objective_daily_workflow_limit":
            # A rate cap is backpressure, not a failure — but an objective that
            # keeps hitting its daily ceiling needs a human to know.
            self._escalate(
                objective,
                reason=REASON_BUDGET_EXHAUSTED,
                problem=(
                    f"this objective hit its daily workflow limit "
                    f"({decision.used}/{decision.limit})"
                ),
                attempted=[f"submitted {decision.used} workflow(s) in the last 24h"],
                needs="a decision: raise the limit, or fix why it keeps resubmitting",
                outcome=outcome,
            )

    # -- 7. ESCALATE ---------------------------------------------------

    def _escalate(
        self,
        objective: Objective,
        *,
        reason: str,
        problem: str,
        needs: str,
        attempted: Optional[list[str]] = None,
        capability: Optional[str] = None,
        workflow_id: Optional[str] = None,
        approval_request_id: Optional[str] = None,
        status: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
        outcome: Optional[dict[str, Any]] = None,
    ) -> None:
        self._escalator.escalate(
            objective,
            reason=reason,
            problem=problem,
            attempted=attempted,
            needs=needs,
            capability=capability,
            workflow_id=workflow_id,
            approval_request_id=approval_request_id,
            status=status,
            extra=extra,
        )
        if outcome is not None:
            outcome["escalated"] = int(outcome.get("escalated", 0)) + 1

    def on_health_error(self, state: ControlState) -> None:
        """Degraded health stops planning — and says so, once."""
        self._record_runtime_error(
            "supervisor_health",
            "runtime health is in error; planning halted",
            details={"mode": state.mode},
        )
        for objective in self._objectives.list_live():
            if objective.status == STATUS_EXECUTING:
                continue
            self._escalator.escalate(
                objective,
                reason=REASON_HEALTH_DEGRADED,
                problem="runtime health is in error, so MR1 has stopped creating new work",
                attempted=["ran the runtime health checks"],
                needs="`mr1 doctor` to be run and the failing check fixed",
                status=STATUS_PAUSED,
            )
            break  # one message, not one per objective

    # -- halt ----------------------------------------------------------

    def _apply_halt(self, state: ControlState) -> None:
        """
        Emergency stop: cancel running work AND remove standing authority.

        A stop that leaves standing consent in place has not actually stopped
        anything — the next supervisor to start would pick the objectives back
        up with their grants intact.
        """
        for workflow in self._store.list_active_workflows():
            if workflow.status is WorkflowStatus.CANCELLED:
                continue
            try:
                self._scheduler.cancel_workflow(workflow.workflow_id)
            except Exception as exc:
                self._record_runtime_error(
                    "supervisor_halt",
                    f"failed to cancel {workflow.workflow_id}: {exc}",
                )
        try:
            self._consent.revoke_all(
                revoked_by="supervisor",
                reason=f"halt: {state.reason or 'emergency stop'}",
            )
            self._objectives.pause_all(reason=f"halt: {state.reason or 'emergency stop'}")
        except Exception as exc:
            self._record_runtime_error("supervisor_halt", f"halt cleanup failed: {exc}")

    def _objective_for_workflow(self, workflow_id: Optional[str]) -> Optional[Objective]:
        if not workflow_id:
            return None
        workflow = self._store.load_workflow(workflow_id)
        if workflow is None:
            return None
        objective_id = (workflow.metadata or {}).get("objective_id")
        if not isinstance(objective_id, str) or not objective_id:
            return None
        return self._objectives.load(objective_id)

    def _emit_objective(
        self,
        objective: Objective,
        event_type: str,
        *,
        status: str,
        summary: str,
        workflow_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            self._timeline.emit(
                event_type=event_type,
                actor_id=objective.owner_agent_id or self._root_agent_id,
                actor_type="mr1",
                target_id=objective.objective_id,
                target_type="objective",
                status=status,
                summary=summary[:300],
                workflow_id=workflow_id,
                record_path=str(self._objectives.objective_path(objective.objective_id)),
                metadata={
                    "objective_id": objective.objective_id,
                    **dict(metadata or {}),
                },
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Gauges, heartbeat, failure recording
    # ------------------------------------------------------------------

    def _active_workflow_count(self) -> int:
        return len(self._store.list_active_workflows())

    def _is_idle(self) -> bool:
        """No in-flight work left to drain."""
        return self._active_workflow_count() == 0

    def _oldest_pending_approval_age_s(self) -> Optional[float]:
        from mr1.clock import parse_iso

        now = self._clock.now()
        ages = [
            (now - created).total_seconds()
            for created in (
                parse_iso(approval.created_at)
                for approval in self._approval_store.list_requests()
                if approval.status == "pending"
            )
            if created is not None
        ]
        return max(ages) if ages else None

    def gauges(self) -> dict[str, Any]:
        scheduler_metrics = self._scheduler.metrics()
        counts = self._objectives.counts_by_status()
        budget = self._budget.snapshot()
        active_grants = self._consent.list_active()
        gauges: dict[str, Any] = {
            "active_workflows": self._active_workflow_count(),
            "oldest_pending_approval_age_s": self._oldest_pending_approval_age_s(),
            "events_jsonl_bytes": events_jsonl_bytes(self._runtime_root),
            "disk_free_bytes": disk_free_bytes(self._runtime_root),
            "scheduler_tick_count": scheduler_metrics["tick_count"],
            "scheduler_tick_errors": scheduler_metrics["tick_errors"],
            "scheduler_tick_duration_ms": scheduler_metrics["last_tick_duration_ms"],
            "scheduler_last_tick_error": scheduler_metrics["last_tick_error"],
            "runtime_errors": len(self._runtime_errors),
            "objectives_total": sum(counts.values()),
            "objectives_active": counts.get(STATUS_ACTIVE, 0),
            "objectives_executing": counts.get(STATUS_EXECUTING, 0),
            "objectives_recovering": counts.get(STATUS_RECOVERING, 0),
            "objectives_waiting_human": counts.get("waiting_human", 0),
            "objectives_quarantined": counts.get("quarantined", 0),
            "objectives_satisfied": counts.get(STATUS_SATISFIED, 0),
            "objectives_paused": counts.get(STATUS_PAUSED, 0),
            "consecutive_failures_max": max(
                [item.consecutive_failures for item in self._objectives.list_objectives()] or [0]
            ),
            "grants_active": len(active_grants),
            "grants_expiring_soon": len(self._consent.expiring_within(86_400.0)),
            "unattended_executions": self._consent.unattended_executions(),
            "plans_this_hour": budget["plans_this_hour"],
            "actions_this_hour": budget["actions_this_hour"],
            "execution_owned": scheduler_metrics["execution_owned"],
            "scheduler_delegated_ticks": scheduler_metrics["delegated_ticks"],
            "backpressure": sorted(self._backpressure.active_codes),
            "scheduler_consecutive_tick_errors": scheduler_metrics["consecutive_tick_errors"],
            **self.metrics(),
        }
        gauges.update(self.extra_gauges())
        self._last_gauges = gauges
        return gauges

    def extra_gauges(self) -> dict[str, Any]:
        return {}

    @property
    def objectives(self) -> ObjectiveStore:
        return self._objectives

    @property
    def consent(self) -> ConsentGrantStore:
        return self._consent

    @property
    def budget(self) -> BudgetLedger:
        return self._budget

    @property
    def escalator(self) -> Escalator:
        return self._escalator

    @property
    def notifier(self) -> Notifier:
        return self._notifier

    @property
    def triage(self) -> Optional[GovernedTriage]:
        return self._triage

    def _write_heartbeat(self) -> None:
        try:
            self._health.write(
                pid=self._pid,
                mode=self._control.read().mode,
                started_at=self._started_at,
                uptime_s=self._clock.monotonic() - self._started_monotonic,
                gauges=self.gauges(),
            )
        except Exception as exc:
            # A failed heartbeat must not kill the loop, but it must be visible.
            self._record_runtime_error(
                "supervisor_heartbeat",
                f"{type(exc).__name__}: {exc}",
            )

    def _record_tick_failure(self, exc: BaseException) -> None:
        detail = f"{type(exc).__name__}: {exc}"
        self._tick_errors += 1
        self._consecutive_tick_errors += 1
        self._last_tick_error = detail
        self._record_runtime_error("supervisor_tick", detail)
        self._emit(
            "supervisor_tick_failed",
            status="error",
            summary=f"supervisor tick failed: {detail}"[:300],
            metadata={
                "error": detail,
                "error_type": type(exc).__name__,
                "tick_errors": self._tick_errors,
                "consecutive_tick_errors": self._consecutive_tick_errors,
            },
        )

    def _record_runtime_error(
        self,
        source: str,
        error: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self._runtime_errors.append({
            "timestamp": self._clock.now_iso(),
            "source": source,
            "error": str(error)[:1000],
            "details": dict(details or {}),
        })
        if len(self._runtime_errors) > 50:
            self._runtime_errors = self._runtime_errors[-50:]

    @property
    def runtime_errors(self) -> list[dict[str, Any]]:
        return list(self._runtime_errors)

    def _emit(
        self,
        event_type: str,
        *,
        status: str,
        summary: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            self._timeline.emit(
                event_type=event_type,
                actor_id=self._root_agent_id,
                actor_type="mr1",
                target_id="supervisor",
                target_type="supervisor",
                status=status,
                summary=summary,
                metadata=dict(metadata or {}),
            )
        except Exception:
            pass
