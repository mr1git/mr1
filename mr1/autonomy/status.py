"""
Operator status: what is MR1 doing, and is it all right? (B3)

Everything here already existed somewhere — `DoctorReport`, `health.json`,
scheduler metrics, supervisor gauges, the objective/approval/grant/budget
stores. What did not exist was one place that answers the question an operator
actually asks, which is never "what is the value of `oldest_pending_approval_age_s`"
but "is anything stuck, and do I need to do something about it".

So this module does two jobs and no more:

  * assemble one **stable, machine-readable payload** from the stores, and
  * roll it up to **ok / warning / error** with a plain-language reason.

The rollup is the load-bearing part. A gauge nobody reads is not monitoring. The
rules below are the ones that would have caught every stall this runtime has
actually had: a dead supervisor, an approval nobody answered, an objective that
quarantined itself at 3am, a disk that filled.

Nothing here calls the brain, and nothing here mutates anything. `mr1 status` is
safe to run in a loop, from cron, from a shell prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mr1.clock import Clock, default_clock, parse_iso


STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_ERROR = "error"

# The exit codes `mr1 status` returns, so a cron job or a shell `&&` can branch
# on health without parsing anything.
EXIT_OK = 0
EXIT_WARNING = 1
EXIT_ERROR = 2

SCHEMA_VERSION = 1

_SEVERITY_ORDER = {STATUS_OK: 0, STATUS_WARNING: 1, STATUS_ERROR: 2}

# Defaults for the rollup thresholds. Every one is overridable, because "when is
# an approval too old" is a judgement about a particular runtime, not a fact.
DEFAULT_MAX_MISSED_HEARTBEATS = 3
DEFAULT_APPROVAL_WARN_AGE_S = 3_600.0        # an hour unanswered: mention it
DEFAULT_APPROVAL_ERROR_AGE_S = 86_400.0      # a day unanswered: MR1 is stalled
DEFAULT_GRANT_EXPIRY_WARN_S = 86_400.0
DEFAULT_DISK_WARN_BYTES = 2 * 1024**3        # 2 GiB
DEFAULT_DISK_ERROR_BYTES = 512 * 1024**2     # 512 MiB


@dataclass
class StatusThresholds:
    max_missed_heartbeats: int = DEFAULT_MAX_MISSED_HEARTBEATS
    approval_warn_age_s: float = DEFAULT_APPROVAL_WARN_AGE_S
    approval_error_age_s: float = DEFAULT_APPROVAL_ERROR_AGE_S
    grant_expiry_warn_s: float = DEFAULT_GRANT_EXPIRY_WARN_S
    disk_warn_bytes: int = DEFAULT_DISK_WARN_BYTES
    disk_error_bytes: int = DEFAULT_DISK_ERROR_BYTES
    tick_interval_s: float = 60.0


@dataclass
class Finding:
    """One reason the rollup is not `ok`. Says what, and what to do."""

    level: str
    code: str
    detail: str
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "detail": self.detail,
            "action": self.action,
        }


@dataclass
class RuntimeStatus:
    schema_version: int = SCHEMA_VERSION
    generated_at: str = ""
    health: str = STATUS_OK
    findings: list[Finding] = field(default_factory=list)
    service: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    objectives: dict[str, Any] = field(default_factory=dict)
    workflows: dict[str, Any] = field(default_factory=dict)
    approvals: dict[str, Any] = field(default_factory=dict)
    grants: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, Any] = field(default_factory=dict)
    autonomy: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    triage: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return {
            STATUS_OK: EXIT_OK,
            STATUS_WARNING: EXIT_WARNING,
            STATUS_ERROR: EXIT_ERROR,
        }[self.health]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "health": self.health,
            "findings": [item.to_dict() for item in self.findings],
            "service": self.service,
            "scheduler": self.scheduler,
            "objectives": self.objectives,
            "workflows": self.workflows,
            "approvals": self.approvals,
            "grants": self.grants,
            "budgets": self.budgets,
            "autonomy": self.autonomy,
            "storage": self.storage,
            "triage": self.triage,
        }


def collect_status(
    runtime_root: Path,
    *,
    clock: Optional[Clock] = None,
    thresholds: Optional[StatusThresholds] = None,
) -> RuntimeStatus:
    """
    Read every store and roll the answer up. Never raises: a status command that
    dies on a corrupt file is a status command that fails exactly when it is
    needed most, so every section degrades to a recorded error instead.
    """
    from mr1.autonomy.control import MODE_HALTED, MODE_PAUSED, MODE_STOPPING, ControlPlane, ServiceLock
    from mr1.autonomy.health import disk_free_bytes, heartbeat_age_s, read_health
    from mr1.autonomy.ownership import ExecutionOwnership

    runtime_root = Path(runtime_root)
    clock = clock or default_clock()
    limits = thresholds or StatusThresholds()

    status = RuntimeStatus(generated_at=clock.now_iso())
    findings: list[Finding] = []

    # -- service, ownership, heartbeat ---------------------------------
    control = ControlPlane(runtime_root, clock=clock).read()
    health = read_health(runtime_root) or {}
    gauges = dict(health.get("gauges") or {})
    lock = ServiceLock(runtime_root)
    owner = ExecutionOwnership(runtime_root).current_owner()
    age = heartbeat_age_s(health, clock=clock)
    supervisor_running = lock.is_held_by_live_process()

    status.service = {
        "mode": control.mode,
        "reason": control.reason,
        "control_file_corrupt": control.corrupt,
        "supervisor_running": supervisor_running,
        "supervisor_pid": lock.read_pid(),
        "heartbeat_at": health.get("supervisor_heartbeat_at"),
        "heartbeat_age_s": round(age, 1) if age is not None else None,
        "uptime_s": health.get("uptime_s"),
        "doctor_status": health.get("doctor_status", "unknown"),
        "doctor_summary": dict(health.get("doctor_summary") or {}),
        "execution_owner": owner.to_dict() if owner else None,
        "execution_owned": owner is not None,
    }

    if control.corrupt:
        findings.append(Finding(
            STATUS_ERROR,
            "control_file_corrupt",
            f"control.json is unreadable ({control.reason}); MR1 is failing closed as paused",
            "repair or delete control.json, then `mr1 resume`",
        ))
    if control.mode == MODE_HALTED:
        findings.append(Finding(
            STATUS_ERROR,
            "halted",
            f"MR1 is halted: {control.reason or 'no reason recorded'}. All consent is revoked.",
            "`mr1 resume` to allow work again; grants must be re-issued",
        ))
    elif control.mode == MODE_PAUSED:
        findings.append(Finding(
            STATUS_WARNING,
            "paused",
            f"MR1 is paused: {control.reason or 'no reason recorded'}. No new work is being created.",
            "`mr1 resume`",
        ))
    elif control.mode == MODE_STOPPING:
        findings.append(Finding(
            STATUS_WARNING,
            "stopping",
            "MR1 is draining and will exit when idle.",
            "`mr1 resume` to cancel the stop",
        ))

    stale_after = limits.tick_interval_s * limits.max_missed_heartbeats
    if supervisor_running and age is not None and age > stale_after:
        # The most important signal there is: the process is alive but the loop
        # is not turning. Nothing else in the runtime would tell you.
        findings.append(Finding(
            STATUS_ERROR,
            "stale_heartbeat",
            f"the supervisor is running but has not ticked for {age:.0f}s "
            f"(> {stale_after:.0f}s); the autonomy loop is wedged",
            "check `mr1 timeline recent` for supervisor_tick_failed, then restart the service",
        ))
    elif supervisor_running and age is None:
        findings.append(Finding(
            STATUS_WARNING,
            "no_heartbeat",
            "a supervisor holds the service lock but has never written a heartbeat",
            "it may still be starting; if this persists, restart it",
        ))

    doctor_status = str(health.get("doctor_status") or "unknown")
    if doctor_status == STATUS_ERROR:
        findings.append(Finding(
            STATUS_ERROR,
            "doctor_error",
            "the runtime health checks are failing; planning is halted",
            "`mr1 doctor` and fix the failing check",
        ))
    elif doctor_status == STATUS_WARNING:
        findings.append(Finding(
            STATUS_WARNING,
            "doctor_warning",
            "the runtime health checks report a warning",
            "`mr1 doctor`",
        ))

    # -- scheduler -----------------------------------------------------
    status.scheduler = {
        "tick_count": gauges.get("scheduler_tick_count", 0),
        "tick_errors": gauges.get("scheduler_tick_errors", 0),
        "last_tick_duration_ms": gauges.get("scheduler_tick_duration_ms"),
        "last_tick_error": gauges.get("scheduler_last_tick_error"),
        "last_successful_tick_at": health.get("supervisor_heartbeat_at"),
        "delegated_ticks": gauges.get("scheduler_delegated_ticks", 0),
    }
    if int(gauges.get("scheduler_tick_errors", 0) or 0):
        findings.append(Finding(
            STATUS_WARNING,
            "scheduler_tick_errors",
            f"the scheduler has failed {gauges['scheduler_tick_errors']} tick(s): "
            f"{gauges.get('scheduler_last_tick_error')}",
            "`mr1 timeline recent` and `mr1 doctor`",
        ))

    # -- objectives ----------------------------------------------------
    status.objectives = _objectives_section(runtime_root, clock, findings)

    # -- workflows -----------------------------------------------------
    status.workflows = _workflows_section(runtime_root, findings)

    # -- approvals -----------------------------------------------------
    status.approvals = _approvals_section(runtime_root, clock, limits, findings)

    # -- grants --------------------------------------------------------
    status.grants = _grants_section(runtime_root, clock, limits, findings)

    # -- budgets -------------------------------------------------------
    status.budgets = _budgets_section(runtime_root, clock, findings)

    # -- autonomy: recent failures and escalations ---------------------
    status.autonomy = _autonomy_section(runtime_root, gauges)

    # -- storage: events, archive, disk, retention ---------------------
    status.storage = _storage_section(runtime_root, clock, limits, findings, disk_free_bytes)

    # -- triage --------------------------------------------------------
    status.triage = {
        "enabled": bool(gauges.get("triage_enabled", False)),
        "actions_this_hour": gauges.get("actions_this_hour", 0),
        "last_skip_reason": gauges.get("triage_last_skip_reason"),
        "failures": gauges.get("triage_failures", 0),
    }

    status.findings = sorted(
        findings,
        key=lambda item: _SEVERITY_ORDER[item.level],
        reverse=True,
    )
    status.health = _rollup(status.findings)
    return status


def _rollup(findings: list[Finding]) -> str:
    if any(item.level == STATUS_ERROR for item in findings):
        return STATUS_ERROR
    if any(item.level == STATUS_WARNING for item in findings):
        return STATUS_WARNING
    return STATUS_OK


def _objectives_section(runtime_root, clock, findings) -> dict[str, Any]:
    try:
        from mr1.autonomy.objectives import ObjectiveStore

        store = ObjectiveStore(runtime_root, clock=clock)
        objectives = store.list_objectives()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    by_status: dict[str, int] = {}
    for objective in objectives:
        by_status[objective.status] = by_status.get(objective.status, 0) + 1

    quarantined = [o for o in objectives if o.status == "quarantined"]
    waiting = [o for o in objectives if o.status == "waiting_human"]

    if quarantined:
        findings.append(Finding(
            STATUS_ERROR,
            "objectives_quarantined",
            f"{len(quarantined)} objective(s) gave up: "
            + ", ".join(f"{o.objective_id} ({o.status_reason or 'no reason'})" for o in quarantined[:3]),
            "`mr1 objective show <id>` to see what it tried, then fix and `mr1 objective resume <id>`",
        ))
    if waiting:
        findings.append(Finding(
            STATUS_WARNING,
            "objectives_waiting_human",
            f"{len(waiting)} objective(s) are blocked on a human: "
            + ", ".join(o.objective_id for o in waiting[:3]),
            "`mr1 inbox` — MR1 has said what it needs",
        ))

    return {
        "total": len(objectives),
        "by_status": by_status,
        "active": by_status.get("active", 0),
        "executing": by_status.get("executing", 0),
        "recovering": by_status.get("recovering", 0),
        "waiting_human": by_status.get("waiting_human", 0),
        "quarantined": by_status.get("quarantined", 0),
        "paused": by_status.get("paused", 0),
        "satisfied": by_status.get("satisfied", 0),
        "next_due": sorted(
            [
                {
                    "objective_id": o.objective_id,
                    "title": o.title,
                    "next_due_at": o.next_due_at,
                }
                for o in objectives
                if o.next_due_at and o.status == "active"
            ],
            key=lambda item: item["next_due_at"],
        )[:5],
        "blocked": [
            {
                "objective_id": o.objective_id,
                "title": o.title,
                "status": o.status,
                "reason": o.status_reason,
                "consecutive_failures": o.consecutive_failures,
            }
            for o in objectives
            if o.status in {"waiting_human", "quarantined"}
        ],
    }


def _workflows_section(runtime_root, findings) -> dict[str, Any]:
    try:
        from mr1.workflow_models import TaskStatus, WorkflowStatus
        from mr1.workflow_store import WorkflowStore

        store = WorkflowStore(root=Path(runtime_root) / "workflows")
        active = store.list_active_workflows()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    blocked = []
    for workflow in active:
        for task in workflow.tasks.values():
            if task.status is TaskStatus.BLOCKED:
                blocked.append({
                    "workflow_id": workflow.workflow_id,
                    "task_id": task.task_id,
                    "label": task.label,
                    "reason": task.blocked_reason or task.last_error or "blocked",
                    "error_type": task.last_error_type,
                })

    if blocked:
        findings.append(Finding(
            STATUS_WARNING,
            "blocked_tasks",
            f"{len(blocked)} task(s) are blocked and cannot proceed without a human",
            "`mr1 approvals` — the task is waiting on a decision or authority it does not have",
        ))

    return {
        "active": len(active),
        "running": sum(1 for w in active if w.status is WorkflowStatus.RUNNING),
        "blocked_tasks": blocked,
    }


def _approvals_section(runtime_root, clock, limits, findings) -> dict[str, Any]:
    try:
        from mr1.capability_policy import CapabilityApprovalStore

        store = CapabilityApprovalStore(
            Path(runtime_root) / "capability_approvals",
            clock=clock,
        )
        requests = store.list_requests()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    now = clock.now()
    pending = [item for item in requests if item.status == "pending"]

    aged: list[tuple[float, Any]] = []
    for approval in pending:
        created = parse_iso(approval.created_at)
        if created is None:
            continue
        aged.append(((now - created).total_seconds(), approval))
    aged.sort(key=lambda item: item[0], reverse=True)

    oldest_age, oldest = (aged[0] if aged else (None, None))

    if oldest_age is not None:
        if oldest_age >= limits.approval_error_age_s:
            findings.append(Finding(
                STATUS_ERROR,
                "approval_stalled",
                f"an approval has been pending for {oldest_age / 3600:.1f}h "
                f"({oldest.capability_name}); MR1 has been waiting that long",
                f"`mr1 approvals approve {oldest.approval_request_id}` or deny it",
            ))
        elif oldest_age >= limits.approval_warn_age_s:
            findings.append(Finding(
                STATUS_WARNING,
                "approval_pending",
                f"an approval has been pending for {oldest_age / 60:.0f}m "
                f"({oldest.capability_name})",
                "`mr1 approvals`",
            ))

    return {
        "pending": len(pending),
        "oldest_pending_age_s": round(oldest_age, 1) if oldest_age is not None else None,
        "oldest_pending": (
            {
                "approval_request_id": oldest.approval_request_id,
                "capability": oldest.capability_name,
                "workflow_id": oldest.workflow_id,
                "created_at": oldest.created_at,
                "expires_at": oldest.expires_at,
            }
            if oldest else None
        ),
        "expiring_soon": [
            {
                "approval_request_id": item.approval_request_id,
                "expires_at": item.expires_at,
            }
            for item in pending
            if item.expires_at
        ][:5],
    }


def _grants_section(runtime_root, clock, limits, findings) -> dict[str, Any]:
    try:
        from mr1.autonomy.consent import ConsentGrantStore

        store = ConsentGrantStore(runtime_root, clock=clock)
        active = store.list_active()
        expiring = store.expiring_within(limits.grant_expiry_warn_s)
        unattended = store.unattended_executions()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    if expiring:
        findings.append(Finding(
            STATUS_WARNING,
            "grants_expiring",
            f"{len(expiring)} consent grant(s) expire within "
            f"{limits.grant_expiry_warn_s / 3600:.0f}h; the objectives holding them will stall",
            "`mr1 grant list` and re-issue the ones still needed",
        ))

    return {
        "active": len(active),
        "expiring_soon": len(expiring),
        "unattended_executions": unattended,
        "detail": [
            {
                "grant_id": grant.grant_id,
                "objective_id": grant.grantee_id,
                "capability": grant.capability_name,
                "expires_at": grant.expires_at,
                "use_count": grant.use_count,
            }
            for grant in active
        ],
    }


def _budgets_section(runtime_root, clock, findings) -> dict[str, Any]:
    try:
        from mr1.autonomy.budget import BudgetLedger

        snapshot = BudgetLedger(runtime_root, clock=clock).snapshot()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    plans = int(snapshot.get("plans_this_hour", 0) or 0)
    max_plans = int(snapshot.get("max_plans_per_hour", 0) or 0)
    if max_plans and plans >= max_plans:
        findings.append(Finding(
            STATUS_WARNING,
            "plan_budget_exhausted",
            f"the planning budget is spent ({plans}/{max_plans} this hour); "
            "no new work is being created until the window rolls",
            "raise --max-plans-per-hour, or find out why it is planning so much",
        ))
    return snapshot


def _autonomy_section(runtime_root, gauges) -> dict[str, Any]:
    """Recent autonomous failures and escalations, from the timeline."""
    try:
        from mr1.event_log import EventLog

        # Recency, not completeness — this is "what just went wrong", and it must
        # not read a million archived events to answer.
        recent = EventLog(Path(runtime_root) / "events").recent_events(limit=500)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    failures = [
        {
            "at": event.timestamp,
            "type": event.event_type,
            "summary": event.summary,
        }
        for event in recent
        if event.event_type in {
            "objective_plan_failed",
            "objective_quarantined",
            "supervisor_tick_failed",
            "scheduler_tick_failed",
            "inbox_triage_failed",
        }
    ]
    escalations = [
        {
            "at": event.timestamp,
            "summary": event.summary,
            "objective_id": event.target_id,
        }
        for event in recent
        if event.event_type == "escalation_raised"
    ]

    return {
        "recent_failures": failures[-10:],
        "recent_escalations": escalations[-10:],
        "consecutive_failures_max": gauges.get("consecutive_failures_max", 0),
        "runtime_errors": gauges.get("runtime_errors", 0),
    }


def _storage_section(runtime_root, clock, limits, findings, disk_free_bytes) -> dict[str, Any]:
    try:
        from mr1.autonomy.retention import RetentionManager

        retention = RetentionManager(runtime_root, clock=clock).status()
    except Exception as exc:  # noqa: BLE001
        retention = {"error": f"{type(exc).__name__}: {exc}"}

    free = disk_free_bytes(Path(runtime_root))

    if 0 <= free < limits.disk_error_bytes:
        findings.append(Finding(
            STATUS_ERROR,
            "disk_critical",
            f"only {free / 1024**2:.0f} MiB of disk left; MR1 has stopped creating work",
            "`mr1 maintenance run` to archive, or free space on the volume",
        ))
    elif 0 <= free < limits.disk_warn_bytes:
        findings.append(Finding(
            STATUS_WARNING,
            "disk_low",
            f"{free / 1024**3:.1f} GiB of disk left",
            "`mr1 maintenance run --dry-run` to see what can be archived",
        ))

    events = retention.get("events", {}) if isinstance(retention, dict) else {}
    if retention.get("events_rotation_due"):
        findings.append(Finding(
            STATUS_WARNING,
            "event_rotation_due",
            f"events.jsonl is {int(events.get('live_bytes', 0)) / 1024**2:.0f} MiB "
            "and past its rotation threshold",
            "`mr1 maintenance run`",
        ))

    return {
        "disk_free_bytes": free,
        "events": events,
        "archive_bytes": retention.get("archive_bytes", 0),
        "rotation_due": retention.get("events_rotation_due", False),
        "last_retention_report": retention.get("last_report"),
        "retention_runs": retention.get("reports", 0),
    }
