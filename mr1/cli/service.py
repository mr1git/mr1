"""CLI handlers for the autonomy service and its control plane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mr1.autonomy.control import (
    MODE_HALTED,
    MODE_PAUSED,
    MODE_RUNNING,
    MODE_STOPPING,
    ControlPlane,
    ServiceLock,
    ServiceLockError,
)
from mr1.autonomy.health import heartbeat_age_s, read_health
from mr1.autonomy.ownership import ROLE_SERVICE, ExecutionOwnership
from mr1.autonomy.service import Supervisor, SupervisorConfig
from mr1.autonomy.notify import Notifier, build_sinks
from mr1.autonomy.status import collect_status
from mr1.clock import default_clock


def _runtime_root(store) -> Path:
    return Path(store.root).parent


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _config_from_args(args) -> SupervisorConfig:
    config = SupervisorConfig()
    for attr in (
        "tick_interval_s",
        "max_concurrent_workflows",
        "max_plans_per_hour",
        "max_workflows_per_objective_per_day",
        "max_actions_per_hour",
        "min_disk_free_bytes",
        "max_consecutive_supervisor_errors",
        "max_consecutive_scheduler_errors",
        "retention_interval_s",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(config, attr, value)
    return config.validate()


def _build_planner(scoped_agent_store):
    """
    The production planner: the existing workflow-compiler agent.

    An objective's plan goes through the same validated-envelope path a human's
    `compile-workflow` does — same validation, same governance, same audit.
    """
    from mr1.autonomy.planner import CompilerPlanner
    from mr1.workflow_compiler import WorkflowCompilerClient

    return CompilerPlanner(
        WorkflowCompilerClient(scoped_agent_store=scoped_agent_store)
    )


def _cmd_serve(args, store, caller_agent_id, scoped_agent_store) -> int:
    runtime_root = _runtime_root(store)
    holder = ExecutionOwnership(runtime_root).current_owner()
    if holder is not None:
        # Not fatal: the supervisor still observes, sweeps, plans, and submits.
        # It simply will not execute while another process holds the root, and
        # it takes over automatically the moment that process leaves.
        print(
            f"[mr1] warning: execution authority is held by pid {holder.pid} "
            f"({holder.role}) — this supervisor will not launch tasks until it exits"
        )
    supervisor = Supervisor(
        runtime_root,
        config=_config_from_args(args),
        workflow_store=store,
        scoped_agent_store=scoped_agent_store,
        workspace_root=Path(getattr(args, "workspace_root", None) or Path.cwd()),
        planner=(
            None
            if getattr(args, "no_planner", False) else
            _build_planner(scoped_agent_store)
        ),
        enable_triage=bool(getattr(args, "triage", False)),
        execution_ownership=ExecutionOwnership(runtime_root, role=ROLE_SERVICE),
        notifier=Notifier(
            runtime_root,
            sinks=build_sinks(getattr(args, "notify", None) or [], runtime_root),
        ),
    )
    print(f"[mr1] supervisor serving runtime_root={runtime_root}")
    print(
        f"[mr1] tick_interval={supervisor.config.tick_interval_s}s"
        "  (ctrl-c, or `mr1 stop` from another shell, to exit)"
    )
    try:
        return supervisor.serve()
    except ServiceLockError as exc:
        print(f"error: {exc}")
        return 1


def _set_mode(args, store, mode: str, *, default_reason: str) -> int:
    control = ControlPlane(_runtime_root(store))
    state = control.set_mode(
        mode,
        reason=getattr(args, "reason", None) or default_reason,
        requested_by=getattr(args, "requested_by", None) or "operator",
    )
    if getattr(args, "json", False):
        _print_json(state.to_dict())
    else:
        print(f"mode: {state.mode}  ({state.reason})")
    return 0


def _cmd_pause(args, store, caller_agent_id, scoped_agent_store) -> int:
    return _set_mode(args, store, MODE_PAUSED, default_reason="paused by operator")


def _cmd_resume(args, store, caller_agent_id, scoped_agent_store) -> int:
    return _set_mode(args, store, MODE_RUNNING, default_reason="resumed by operator")


def _cmd_stop(args, store, caller_agent_id, scoped_agent_store) -> int:
    return _set_mode(args, store, MODE_STOPPING, default_reason="graceful stop requested")


def halt_runtime(
    runtime_root: Path,
    *,
    reason: str,
    requested_by: str,
    clock=None,
) -> dict[str, Any]:
    """
    Emergency stop, applied from wherever the halt was requested.

    Authority is revoked here rather than only inside a running supervisor: a
    halt that takes effect only if a supervisor happens to be alive to read it
    is not a halt. A live supervisor also acts on the mode on its next tick —
    cancelling in-flight tasks and exiting — and both paths are idempotent.
    """
    clock = clock or default_clock()
    state = ControlPlane(runtime_root, clock=clock).set_mode(
        MODE_HALTED,
        reason=reason,
        requested_by=requested_by,
    )
    payload: dict[str, Any] = {
        **state.to_dict(),
        "revoked_grants": [],
        "paused_objectives": [],
        "supervisor_running": ServiceLock(runtime_root).is_held_by_live_process(),
    }
    payload.update(
        _revoke_standing_authority(
            runtime_root,
            reason=reason,
            requested_by=requested_by,
            clock=clock,
        )
    )
    return payload


def _revoke_standing_authority(
    runtime_root: Path,
    *,
    reason: str,
    requested_by: str,
    clock,
) -> dict[str, Any]:
    """Revoke every consent grant and park every objective. A4/A5 surfaces."""
    revoked: list[str] = []
    paused: list[str] = []
    try:
        from mr1.autonomy.consent import ConsentGrantStore

        revoked = ConsentGrantStore(runtime_root, clock=clock).revoke_all(
            revoked_by=requested_by,
            reason=f"halt: {reason}",
        )
    except ImportError:  # pragma: no cover - consent lands in A4
        pass
    try:
        from mr1.autonomy.objectives import ObjectiveStore

        paused = ObjectiveStore(runtime_root, clock=clock).pause_all(
            reason=f"halt: {reason}",
        )
    except ImportError:  # pragma: no cover - objectives land in A5
        pass
    return {"revoked_grants": revoked, "paused_objectives": paused}


def _cmd_halt(args, store, caller_agent_id, scoped_agent_store) -> int:
    payload = halt_runtime(
        _runtime_root(store),
        reason=getattr(args, "reason", None) or "halted by operator",
        requested_by=getattr(args, "requested_by", None) or "operator",
    )
    if getattr(args, "json", False):
        _print_json(payload)
        return 0
    print(f"mode: halted  ({payload['reason']})")
    print(f"consent grants revoked: {len(payload['revoked_grants'])}")
    print(f"objectives paused: {len(payload['paused_objectives'])}")
    if payload["supervisor_running"]:
        print("a supervisor is live; it will cancel running tasks and exit on its next tick")
    return 0


def _cmd_status(args, store, caller_agent_id, scoped_agent_store) -> int:
    """
    The operator's one command. Human-readable by default, JSON on demand, and
    an exit code either way so automation can branch without parsing anything:
    0 = ok, 1 = warning, 2 = error.
    """
    runtime_root = _runtime_root(store)
    report = collect_status(runtime_root)

    if getattr(args, "json", False):
        _print_json(report.to_dict())
        return report.exit_code

    _render_status(report)
    return report.exit_code


_HEALTH_MARK = {"ok": "ok", "warning": "WARNING", "error": "ERROR"}


def _render_status(report) -> None:
    service = report.service
    print(f"health:            {_HEALTH_MARK.get(report.health, report.health).upper()}")
    print(
        f"mode:              {service['mode']}"
        + (f"  ({service['reason']})" if service.get("reason") else "")
    )
    print(
        "supervisor:        "
        + (
            f"running (pid {service['supervisor_pid']})"
            if service["supervisor_running"] else
            "not running"
        )
    )
    owner = service.get("execution_owner")
    print(
        "execution:         "
        + (
            f"owned by pid {owner['pid']} ({owner['role']})"
            if owner else
            "unowned — no process is advancing workflows"
        )
    )
    age = service.get("heartbeat_age_s")
    print("heartbeat:         " + (f"{age:.0f}s ago" if age is not None else "none"))

    objectives = report.objectives
    if not objectives.get("error"):
        print(
            "objectives:        "
            f"{objectives.get('active', 0)} active, "
            f"{objectives.get('executing', 0)} executing, "
            f"{objectives.get('waiting_human', 0)} waiting-human, "
            f"{objectives.get('quarantined', 0)} quarantined, "
            f"{objectives.get('paused', 0)} paused"
        )
        for item in objectives.get("next_due", [])[:3]:
            print(f"  next due:        {item['next_due_at']}  {item['title'][:40]}")

    workflows = report.workflows
    if not workflows.get("error"):
        print(
            f"workflows:         {workflows.get('active', 0)} active, "
            f"{len(workflows.get('blocked_tasks', []))} blocked task(s)"
        )

    approvals = report.approvals
    if not approvals.get("error"):
        oldest = approvals.get("oldest_pending")
        print(
            f"approvals:         {approvals.get('pending', 0)} pending"
            + (
                f"  (oldest {approvals['oldest_pending_age_s'] / 60:.0f}m: "
                f"{oldest['capability']}, expires {oldest['expires_at']})"
                if oldest else ""
            )
        )

    grants = report.grants
    if not grants.get("error"):
        print(
            f"consent grants:    {grants.get('active', 0)} active, "
            f"{grants.get('expiring_soon', 0)} expiring soon"
        )

    budgets = report.budgets
    if not budgets.get("error"):
        print(
            f"budgets:           plans {budgets.get('plans_this_hour', 0)}/"
            f"{budgets.get('max_plans_per_hour', 0)} this hour, "
            f"actions {budgets.get('actions_this_hour', 0)}/"
            f"{budgets.get('max_actions_per_hour', 0)}"
        )

    storage = report.storage
    events = storage.get("events") or {}
    print(
        f"storage:           events {int(events.get('live_bytes', 0)) / 1024**2:.1f} MiB live, "
        f"{int(storage.get('archive_bytes', 0)) / 1024**2:.1f} MiB archived; "
        f"disk free {int(storage.get('disk_free_bytes', 0)) / 1024**3:.1f} GiB"
        + ("  [rotation due]" if storage.get("rotation_due") else "")
    )

    if report.findings:
        print()
        for finding in report.findings:
            print(f"[{finding.level.upper()}] {finding.detail}")
            if finding.action:
                print(f"         -> {finding.action}")
    else:
        print("\nnothing needs you.")
